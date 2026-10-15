import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from duplex_avatar.parallel_voice import VoicePipeline, reset_speech_turn


class SpeechState:
    """Observable old utterance, prompt caches, and untouched conversation state."""
    def __init__(self, entered=None, release=None):
        self.tts_text_start_pos = 99
        self.tts_past_key_values = object()
        self.tts_current_turn_start_time = 7
        self.token2wav_buffer = [777, 888]
        self.flow_cache_base, self.hift_cache_base = {"prompt": [1]}, {"prompt": [2]}
        self.stream_cache, self.hift_cache_dict = {"old": 3}, {"old": 4}
        self.decoder = object()
        self.audio_buffer = np.array([.1, .2])
        self.tts_temperature, self.tts_repetition_penalty = .8, 1.1
        self.tts_eos_token, self.tts_logits_processors = None, None
        self.entered, self.release = entered, release
        self.reset_threads = []

    def _convert_results_to_tts_input(self, hidden):
        if self.entered is not None:
            self.entered.set()
            if not self.release.wait(3):
                raise TimeoutError("Test worker was not released")
            self.entered = None
        return torch.zeros(1, 1, 1)

    def _reset_token2wav_for_new_turn(self):
        self.stream_cache = deepcopy(self.flow_cache_base)
        self.hift_cache_dict = deepcopy(self.hift_cache_base)
        self.token2wav_buffer = [4218] * 3
        self.reset_threads.append(threading.get_ident())

    def _generate_waveform_from_tokens(self, tokens, *args, **kwargs):
        self.token2wav_buffer += tokens.flatten().tolist()
        return np.array(self.token2wav_buffer, dtype=np.float32)


class VoiceInterruptionTests(unittest.TestCase):
    def test_reset_discards_speech_but_preserves_prompt_and_conversation(self):
        state = SpeechState()
        decoder, audio = state.decoder, state.audio_buffer
        reset_speech_turn(state)
        self.assertEqual(state.tts_text_start_pos, 0)
        self.assertIsNone(state.tts_past_key_values)
        self.assertIsNone(state.tts_current_turn_start_time)
        self.assertEqual(state.token2wav_buffer, [4218] * 3)
        self.assertEqual(state.stream_cache, state.flow_cache_base)
        self.assertEqual(state.hift_cache_dict, state.hift_cache_base)
        self.assertIsNot(state.stream_cache, state.flow_cache_base)
        self.assertIs(state.decoder, decoder)
        self.assertIs(state.audio_buffer, audio)

    def exercise(self, repeated):
        entered, release = threading.Event(), threading.Event()
        state = SpeechState(entered, release)
        calls = []
        def generate(**kwargs):
            calls.append(kwargs)
            return torch.tensor([[7]]), object()
        pipeline = VoicePipeline.__new__(VoicePipeline)
        pipeline.state, pipeline.model = state, SimpleNamespace(tts=SimpleNamespace(generate_chunk=generate))
        pipeline.device, pipeline.prompt_path = torch.device("cpu"), "reference.wav"
        pipeline.generation, pipeline.last = 0, None
        with ThreadPoolExecutor(max_workers=1) as executor, patch("torch.cuda.device", lambda _: nullcontext()):
            pipeline.executor = executor
            try:
                running = pipeline.submit([], False, 0)
                self.assertTrue(entered.wait(2))
                queued = pipeline.submit([], False, 1)
                reset = pipeline.interrupt()
                intermediate = pipeline.submit([], False, 2) if repeated else None
                if repeated:
                    reset = pipeline.interrupt()
                fresh = pipeline.submit([], False, 3)
                self.assertFalse(reset.done(), "Reset must not race the running GPU job")
                self.assertEqual(state.tts_text_start_pos, 99)
            finally:
                release.set()
            self.assertTrue(running.result(timeout=3)["discarded"])
            self.assertTrue(queued.result(timeout=3)["discarded"])
            if intermediate is not None:
                self.assertTrue(intermediate.result(timeout=3)["discarded"])
            result = fresh.result(timeout=3)
            self.assertEqual(len(calls), 2, "Queued obsolete jobs must not synthesize speech")
            self.assertEqual(calls[-1]["text_start_pos"], 0)
            self.assertIsNone(calls[-1]["past_key_values"])
            self.assertEqual(result["metrics"]["token2wav_buffer_before"], 3)
            self.assertTrue(result["metrics"]["tts_fresh_cache"])
            self.assertEqual(result["metrics"]["tts_generation"], 2 if repeated else 1)
            samples = np.frombuffer(base64.b64decode(result["audio"]), dtype="<f4")
            np.testing.assert_array_equal(samples, [4218, 4218, 4218, 7])
            self.assertEqual(len(state.reset_threads), 1)
            self.assertNotEqual(state.reset_threads[0], threading.get_ident())

    def test_running_job_finishes_before_reset_and_next_speech_starts_clean(self):
        self.exercise(repeated=False)

    def test_repeated_interrupts_skip_intermediate_jobs_and_reset_on_owner(self):
        self.exercise(repeated=True)
