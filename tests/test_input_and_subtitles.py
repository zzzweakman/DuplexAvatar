import asyncio
import json
import unittest

import numpy as np

from duplex_avatar.gateway import Session
from duplex_avatar.media import AudioChunks, encode_pcm
from duplex_avatar.settings import CONFIG
from duplex_avatar.transcription import Utterances
from duplex_avatar.transport import InputQueue


class InputAndSubtitlesTests(unittest.IsolatedAsyncioTestCase):
    async def test_visual_confirmation_requires_model_tokens_and_keeps_camera_timestamp(self):
        session = Session(None)
        async def brain():
            for index, (frames, tokens) in enumerate([(["jpeg"], 0), (["jpeg"], 64), ([], 0)], 1):
                session.inflight_input = {"video_frames": frames, "client_input_id": index,
                                          "capture_start_ms": 1000 * index, "video_capture_ms": 1000 * index + 950}
                yield json.dumps({"type": "input.processed", "input_id": f"0:{index}",
                                  "vision_tokens": tokens, "vision_slices": int(tokens > 0)})
        session.brain = brain()
        await session.read_brain()
        events = [await session.outbox.get() for _ in range(3)]
        self.assertEqual([event["processed_video_frames"] for event in events], [0, 1, 1])
        self.assertEqual(events[-1]["last_vision_client_input_id"], 2)
        self.assertEqual(events[-1]["last_vision_capture_ms"], 2950)

    async def test_batch_confirmation_requires_all_five_visual_embeddings(self):
        session = Session(None)
        async def brain():
            for index, (slices, tokens) in enumerate([(1, 64), (5, 320)], 1):
                session.inflight_input = {"video_frames": ["jpeg"] * 5, "client_input_id": index,
                                          "video_capture_ms": 1800, "video_frame_times_ms": [1000, 1200, 1400, 1600, 1800]}
                yield json.dumps({"type": "input.processed", "input_id": f"0:{index}",
                                  "vision_tokens": tokens, "vision_slices": slices})
        session.brain = brain()
        await session.read_brain()
        events = [await session.outbox.get() for _ in range(2)]
        self.assertEqual([e["processed_video_frames"] for e in events], [0, 5])
        self.assertEqual(events[-1]["last_vision_capture_ms"], 1800)
        self.assertEqual(events[-1]["video_frames"], 5)
        self.assertEqual(events[-1]["video_frame_times_ms"], [1000, 1200, 1400, 1600, 1800])

    async def test_slow_brain_preserves_voice_beyond_old_six_second_limit(self):
        queue = InputQueue(limit=30)
        for index in range(24):
            self.assertEqual(queue.put({"quiet": False, "id": index}), [])
        self.assertEqual([(await queue.get())["id"] for _ in range(24)], list(range(24)))
        for index in range(30):
            queue.put({"quiet": False, "id": index})
        with self.assertRaises(RuntimeError):
            queue.put({"quiet": False, "id": 31})

    async def test_silence_is_coalesced_without_crossing_voice_boundaries(self):
        queue = InputQueue()
        for index, quiet in enumerate([False, True, True, True, False, True, True]):
            queue.put({"quiet": quiet, "id": index})
        self.assertEqual([(await queue.get())["id"] for _ in range(queue.qsize())], [0, 3, 4, 6])

    async def test_final_video_completes_blocks_and_minimum_postroll_without_changing_speech(self):
        session = Session(None)
        session.output_mode, session.diagnostics = "avatar", False
        session.chunks = AudioChunks(23040)
        speech = np.linspace(-.2, .2, 2 * 23040 + 20000, dtype=np.float32)

        async def brain():
            yield json.dumps({"type": "response.output.delta", "kind": "audio", "response_id": "reply",
                              "input_id": "0:1", "audio": encode_pcm(speech)})
            yield json.dumps({"type": "response.output.delta", "kind": "listen", "response_id": "reply", "input_id": "0:1"})

        session.brain = brain()
        await session.read_brain()
        items = [session.outputs.get_nowait() for _ in range(session.outputs.qsize())]
        self.assertEqual(items[-1]["kind"], "end")
        chunks = items[:-1]
        np.testing.assert_array_equal(np.concatenate([c["samples"][:c["speech_samples"]] for c in chunks]), speech)
        tail = np.concatenate([c["samples"][c["speech_samples"]:] for c in chunks])
        self.assertTrue(all(len(c["samples"]) == 23040 for c in chunks))
        self.assertGreaterEqual(len(tail), CONFIG["speech_postroll_ms"] * 24)
        self.assertLess(len(tail), CONFIG["speech_postroll_ms"] * 24 + 23040)
        self.assertFalse(tail.any())

    async def test_slow_asr_does_not_block_input_and_stamps_reply(self):
        session = Session(None)
        session.output_mode, session.diagnostics = "audio", False
        session.chunks = AudioChunks(24000)
        session.asr_jobs.put_nowait({"pending": True})

        async def brain():
            yield json.dumps({"type": "response.output.delta", "kind": "text", "response_id": "r", "input_id": "0:1", "text": "你好"})
            yield json.dumps({"type": "input.processed", "input_id": "0:1"})

        session.brain = brain()
        await asyncio.wait_for(session.read_brain(), 1)
        text = await session.outbox.get()
        self.assertGreater(text["generated_at_ms"], 0)
        self.assertTrue(session.ack.is_set())
        self.assertEqual(session.asr_jobs.qsize(), 1)


class UtteranceTests(unittest.TestCase):
    def test_silence_never_produces_a_transcription(self):
        segmenter = Utterances()
        for second in range(60):
            self.assertEqual(segmenter.append(np.zeros(16000, dtype=np.float32), second * 1000), [])
        self.assertIsNone(segmenter.flush())

    def test_speech_timestamp_survives_silence_and_capture_gap(self):
        segmenter = Utterances()
        silence = np.zeros(16000, dtype=np.float32)
        speech = np.ones(16000, dtype=np.float32) * .1
        segmenter.append(silence, 0)
        segmenter.append(speech, 1000)
        result = segmenter.append(silence, 2000)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["start_ms"], 800)
        self.assertEqual(result[0]["end_ms"], 2200)
        segmenter.append(speech, 8000)
        after_gap = segmenter.append(speech, 14000)
        self.assertEqual(len(after_gap), 1)
        self.assertGreaterEqual(after_gap[0]["start_ms"], 7800)

    def test_long_speech_is_split_into_bounded_asr_jobs(self):
        segmenter = Utterances()
        jobs = []
        for second in range(37):
            jobs.extend(segmenter.append(np.ones(16000, dtype=np.float32) * .1, second * 1000))
        self.assertEqual(len(jobs), 3)
        self.assertTrue(all(len(job["pcm"]) <= 12 * 16000 * 2 for job in jobs))
