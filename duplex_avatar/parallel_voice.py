"""MiniCPM's original TTS on a dedicated GPU with its own ordered state owner.

The token generation sequence matches DuplexCapability.streaming_generate.
Only the GPU placement and scheduling differ; no replacement TTS is used.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
from copy import copy
import logging
import time

import numpy as np
import torch


def reset_speech_turn(state):
    """Discard an interrupted utterance without flushing its remaining speech."""
    state.tts_text_start_pos = 0
    state.tts_past_key_values = None
    state.tts_current_turn_start_time = None
    state._reset_token2wav_for_new_turn()


class VoicePipeline:
    def __init__(self, model, device, prompt_path, compile_vocoder=False):
        self.model, self.device, self.prompt_path = model, torch.device(device), prompt_path
        model.tts.to(self.device)
        tokenizer = model.tts.audio_tokenizer
        for name in ("audio_tokenizer", "flow", "hift"):
            getattr(tokenizer, name).to(self.device)
        tokenizer.speech_window = tokenizer.speech_window.to(self.device)
        tokenizer.cache = None
        tokenizer.stream_cache = None
        tokenizer.hift_cache_dict = {}
        self.state = copy(model.duplex)
        self.state.device = self.device
        self.state.tts_eos_token = self.state.tts_eos_token.to(self.device)
        self.state.session_logs = []
        self.state._reset_streaming_state()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="minicpm-voice",
                                          initializer=lambda: torch.cuda.set_device(self.device))
        self.last = None
        self.generation = 0
        self.compile_vocoder = compile_vocoder
        if compile_vocoder:
            tokenizer.flow.decoder.estimator.forward_chunk = torch.compile(
                tokenizer.flow.decoder.estimator.forward_chunk, dynamic=True)
            tokenizer.hift.forward = torch.compile(tokenizer.hift.forward, dynamic=True)

    def prepare(self):
        self.generation = 0
        def reset():
            with torch.inference_mode(), torch.cuda.device(self.device):
                self.state._reset_streaming_state()
                self.state.session_logs = []
                self.state.session_start_time = time.time()
                self.state._init_token2wav_cache(self.prompt_path)
                self.state._reset_token2wav_for_new_turn()
        self.last = self.executor.submit(reset)
        return self.last

    def interrupt(self):
        # Invalidate queued speech immediately. The running CUDA job finishes on
        # this same worker before the cache reset and any new speech can execute.
        self.generation += 1
        generation = self.generation
        self.last = self.executor.submit(self._reset_turn, generation)
        return self.last

    @torch.inference_mode()
    def _reset_turn(self, generation):
        with torch.cuda.device(self.device):
            if generation != self.generation:
                return
            previous_position = self.state.tts_text_start_pos
            previous_tokens = len(self.state.token2wav_buffer)
            reset_speech_turn(self.state)
            logging.getLogger("uvicorn.error").info(
                "voice_reset generation=%s discarded_tts_position=%s discarded_buffer_tokens=%s remaining_prefix_tokens=%s",
                generation, previous_position, previous_tokens, len(self.state.token2wav_buffer))

    def submit(self, hidden, end_of_turn, unit):
        # An explicit copy makes the hand-off independent of the LLM's next step.
        copied = [(token, tensor.detach().to(self.device, copy=True), end) for token, tensor, end in hidden]
        queued = time.perf_counter()
        self.last = self.executor.submit(self.synthesize, copied, end_of_turn, unit, queued, self.generation)
        return self.last

    def drain(self):
        if self.last is not None:
            self.last.result()

    @torch.inference_mode()
    def synthesize(self, hidden, end_of_turn, unit, queued=None, generation=None):
        generation = self.generation if generation is None else generation
        if generation != self.generation:
            return {"discarded": True, "generation": generation}
        with torch.cuda.device(self.device):
            state = self.state
            started = time.perf_counter()
            text_position = state.tts_text_start_pos
            buffer_tokens = len(state.token2wav_buffer)
            fresh_cache = state.tts_past_key_values is None
            condition = state._convert_results_to_tts_input(hidden)
            prepared = time.perf_counter()
            minimum = 0 if end_of_turn or state.tts_text_start_pos == 0 else 26
            new_tokens, old_kv = self.model.tts.generate_chunk(
                inputs_embeds=condition, temperature=state.tts_temperature,
                repetition_penalty=state.tts_repetition_penalty, eos_token=state.tts_eos_token,
                force_no_stop=False, max_new_token=26, min_new_tokens=minimum,
                past_key_values=state.tts_past_key_values, logits_processors=state.tts_logits_processors,
                text_start_pos=state.tts_text_start_pos)
            generated = time.perf_counter()
            if end_of_turn:
                state.tts_text_start_pos = 0
                state.tts_past_key_values = None
                state.tts_current_turn_start_time = None
            else:
                state.tts_past_key_values = old_kv
                state.tts_text_start_pos += condition.shape[1] + new_tokens.shape[1]
            samples = state._generate_waveform_from_tokens(new_tokens, self.prompt_path, end_of_turn, force_flush=True)
            synthetic_silence = samples is None
            if synthetic_silence:
                samples = state._generate_silence_waveform()
            if end_of_turn:
                state._reset_token2wav_for_new_turn()
            finished = time.perf_counter()
            if generation != self.generation:
                return {"discarded": True, "generation": generation}
            return {"audio": base64.b64encode(samples.astype("<f4").tobytes()).decode(),
                    "metrics": {"tts_unit": unit, "tts_generation": generation,
                                "tts_text_position_before": text_position,
                                "token2wav_buffer_before": buffer_tokens, "tts_fresh_cache": fresh_cache,
                                "tts_started": started, "tts_finished": finished,
                                "tts_queue_ms": round((started - (queued or started)) * 1000, 2),
                                "cost_tts_prep_ms": round((prepared - started) * 1000, 2),
                                "cost_tts_ms": round((generated - prepared) * 1000, 2),
                                "cost_token2wav_ms": round((finished - generated) * 1000, 2),
                                "tts_total_ms": round((finished - started) * 1000, 2),
                                "n_tts_tokens": new_tokens.numel(), "synthetic_silence": synthetic_silence}}

    def warmup(self):
        self.prepare().result()
        tokens = self.model.duplex.tokenizer.encode("你好，我正在听你说话。", add_special_tokens=False)[:4]
        dtype = self.model.tts.emb_text.weight.dtype
        hidden = [(token, torch.zeros(1, 1, self.model.llm.config.hidden_size, device=self.device, dtype=dtype), False)
                  for token in tokens]
        for unit in range(3):
            self.submit(hidden, unit == 2, unit).result()
