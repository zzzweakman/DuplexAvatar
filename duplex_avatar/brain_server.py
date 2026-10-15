"""Run the installed MiniCPM backend with an explicit per-input acknowledgement."""
import os
import logging
import sys
import time

from duplex_avatar.settings import CONFIG, WORKSPACE
from duplex_avatar.vision import configure_window

MINICPM = WORKSPACE / "MiniCPM-o-Demo"
sys.path.insert(0, str(MINICPM))
os.chdir(MINICPM)

import uvicorn
from py_backend import server
from core.processors.pytorch_backend import PyTorchBackend

PARALLEL_VOICE = CONFIG.get("tts_gpu", CONFIG["brain_gpu"]) != CONFIG["brain_gpu"]


class DeploymentBackend(PyTorchBackend):
    def set_duplex_config(self, config):
        super().set_duplex_config(config)
        if config and self.processor:
            self.set_chunk_ms(int(config.get("chunk_ms", 1000)))
            configure_window(self.processor.model.duplex.decoder, bool(config.get("vision_input")),
                             CONFIG.get("vision_context_units", 24), CONFIG.get("vision_previous_tokens", 500))

    def set_chunk_ms(self, chunk_ms):
        if not 200 <= chunk_ms <= 2000 or chunk_ms % 10:
            raise ValueError("MiniCPM decision unit must be 200–2000 ms, in 10 ms steps")
        model = self.processor.model
        # The upstream view stores chunk_ms but does not apply it to the streaming
        # mel processor. Both owners must agree before prepare() initializes it.
        model.CHUNK_MS = model.duplex.CHUNK_MS = chunk_ms
        model.FIRST_CHUNK_MS = model.duplex.FIRST_CHUNK_MS = chunk_ms + 35

    def load_model(self):
        want_compile = self.compile
        if PARALLEL_VOICE:
            self.compile = False  # Place modules first, then compile on their actual devices.
        super().load_model()
        self.compile = want_compile
        self.voice = None
        self.voice_future = None
        if PARALLEL_VOICE:
            from duplex_avatar.parallel_voice import VoicePipeline
            self.processor.compile = want_compile
            self.processor.set_duplex_mode()
            model = self.processor.model
            model.duplex.generate_audio = False
            if want_compile:
                model.apply_torch_compile(mode="default", dynamic=True)
            self.voice = VoicePipeline(model, "cuda:1", self.ref_audio_path,
                                       compile_vocoder=CONFIG.get("compile_vocoder", False))
            self.status = "warming_voice"
            self.voice.warmup()
            logging.getLogger("uvicorn.error").info(
                "Parallel devices: understanding=%s original_tts=%s vocoder=%s",
                next(model.llm.parameters()).device, next(model.tts.parameters()).device,
                next(model.tts.audio_tokenizer.hift.parameters()).device)
        self.status = "warming_audio"
        if self.compile:
            import librosa
            import numpy as np
            model = self.processor.model
            reference, _ = librosa.load(self.ref_audio_path, sr=16000, mono=True)
            from PIL import Image
            image = Image.new("RGB", (448, 336), "red")
            cases = [(1000, None), (960, None), (960, [image]),
                     (960, [image] * 4), (960, [image] * 5), (960, [image] * 6), (1000, [image] * 5)]
            for chunk_ms, frames in cases:
                logging.getLogger("uvicorn.error").info("Warming input: chunk_ms=%s video_frames=%s batch_vision=%s",
                                                       chunk_ms, len(frames or []), len(frames or []) > 1)
                self.set_chunk_ms(chunk_ms)
                configure_window(model.duplex.decoder, bool(frames), CONFIG.get("vision_context_units", 24),
                                 CONFIG.get("vision_previous_tokens", 500))
                model.duplex.prepare(
                    prefix_system_prompt="<|im_start|>system\nStreaming Omni Conversation.\n<|audio_start|>",
                    suffix_system_prompt="<|audio_end|><|im_end|>",
                    ref_audio=reference, prompt_wav_path=self.ref_audio_path)
                # Exercise context eviction before the first long camera session.
                steps = CONFIG.get("vision_context_units", 24) + 3 if chunk_ms == 960 and len(frames or []) == 5 else 3
                for step in range(steps):
                    result = model.duplex.streaming_prefill(audio_waveform=np.zeros(chunk_ms * 16, dtype=np.float32),
                                                            frame_list=frames, max_slice_nums=1,
                                                            batch_vision_feed=len(frames or []) > 1)
                    if result.get("success") is False:
                        raise RuntimeError(f"Decision-unit warmup failed: {result}")
                    model.duplex.streaming_generate(force_listen_override=True)
                    model.duplex.finalize_unit()
                    if steps > 3 and step < 3:
                        # This discarded warmup history must contain text too:
                        # rotating an all-listen window never exercises the LLM
                        # forward that preserves previous replies.
                        entry = model.duplex.decoder._unit_history[-1]
                        entry.update(is_listen=False, generated_text="这是启动预热用的上下文。",
                                     generated_tokens=model.duplex.tokenizer.encode("这是启动预热用的上下文。", add_special_tokens=False))
            self.set_chunk_ms(1000)
            configure_window(model.duplex.decoder, False)
            model.reset_session(reset_token2wav_cache=True)
        self.status = "ready"

    def duplex_prefill(self, audio_waveform=None, frame_list=None, max_slice_nums=1):
        self.last_input_vision = {}
        if frame_list and len(frame_list) > 1:
            self.processor.set_duplex_mode()
            # The upstream capability already batches image encoding. Also feed
            # its ordered visual embeddings in one causal LLM pass (not 3*N).
            result = self.processor.model.duplex.streaming_prefill(
                audio_waveform=audio_waveform, frame_list=frame_list,
                max_slice_nums=max_slice_nums, batch_vision_feed=True)
        else:
            result = super().duplex_prefill(audio_waveform=audio_waveform, frame_list=frame_list,
                                           max_slice_nums=max_slice_nums)
        decoder = self.processor.model.duplex.decoder
        self.last_input_vision = {
            "vision_slices": result.get("n_vision_slices", result.get("n_vision_images", 0)),
            "vision_tokens": (result.get("usage") or {}).get("input_vision_tokens", 0),
            "vision_context_tokens": decoder.get_cache_length(),
            "vision_context_units": len(decoder._unit_history),
            "vision_window_mode": decoder._window_config.sliding_window_mode,
        }
        return result

    def duplex_prepare(self, *args, **kwargs):
        self._force_listen_active = False
        if self.voice is None:
            return super().duplex_prepare(*args, **kwargs)
        self.voice.drain()
        self.processor.set_duplex_mode()
        ready = self.voice.prepare()
        result = super().duplex_prepare(*args, **kwargs)
        ready.result()
        return result

    def duplex_generate(self, force_listen=False):
        self.voice_future = None
        if force_listen and not getattr(self, "_force_listen_active", False):
            if self.voice is not None:
                self.voice.interrupt()
            else:
                # The single-GPU fallback also needs its TTS KV state cleared;
                # upstream force-listen only resets the waveform caches.
                from duplex_avatar.parallel_voice import reset_speech_turn
                reset_speech_turn(self.processor.model.duplex)
        self._force_listen_active = bool(force_listen)
        result = super().duplex_generate(force_listen=force_listen)
        if self.voice is not None:
            result.audio_data = None
            if not result.is_listen:
                state = self.processor.model.duplex
                self.voice_future = self.voice.submit(state.total_hidden[-1], result.end_of_turn, result.current_time)
        return result

    def duplex_cleanup(self):
        if self.voice is not None:
            self.voice.drain()
            self.voice.state._reset_streaming_state()
        super().duplex_cleanup()


class AcknowledgedSession(server.BackendProtocolSession):
    async def send_output_delta(self, kind, **fields):
        self._input_vision = {key: fields.get("metrics", {}).get(key, 0)
                              for key in ("vision_slices", "vision_tokens")}
        await super().send_output_delta(kind, **fields)

    async def push(self, message):
        started = time.perf_counter()
        self._input_vision = {}
        await super().push(message)
        await self.send("input.processed", input_id=message.get("input", {}).get("input_id"),
                        processing_started=started, processing_finished=time.perf_counter(),
                        **getattr(self.backend, "last_input_vision", self._input_vision))


if PARALLEL_VOICE:
    from duplex_avatar.parallel_session import ParallelSession
    server.BackendProtocolSession = ParallelSession
else:
    server.BackendProtocolSession = AcknowledgedSession
server.create_backend = lambda config: DeploymentBackend(**config)
server.SERVER_CONFIG.update({
    "model_path": str(MINICPM / "models/MiniCPM-o-4_5"),
    "gpu_id": 0,
    "pt_path": None,
    "ref_audio_path": str(MINICPM / "assets/ref_audio/ref_minicpm_signature.wav"),
    "duplex_pause_timeout": 1800,
    "compile": CONFIG["compile_brain"],
    "chat_vocoder": "token2wav",
    "attn_implementation": "sdpa",
})


@server.app.get("/acceleration")
async def acceleration():
    processor = getattr(server._backend, "processor", None)
    model = getattr(processor, "model", None)
    return {"compile": CONFIG["compile_brain"],
            "decision_ms": getattr(model, "CHUNK_MS", None), "configurable_decision_unit": True,
            "compiled": bool(getattr(model, "_compiled", False)),
            "active": bool(getattr(model, "_compile_active", False)),
            "targets": ["llm.model", "tts.model"] if CONFIG["compile_brain"] else [],
            "gpu": CONFIG["brain_gpu"], "tts_gpu": CONFIG.get("tts_gpu", CONFIG["brain_gpu"]),
            "supported_modes": ["full_duplex"] if PARALLEL_VOICE else ["full_duplex", "turn_based"],
            "parallel_voice": PARALLEL_VOICE, "compile_vocoder": CONFIG.get("compile_vocoder", False)}

if __name__ == "__main__":
    uvicorn.run(server.app, host="127.0.0.1", port=CONFIG["brain_port"], ws_max_size=16 * 1024 * 1024)
