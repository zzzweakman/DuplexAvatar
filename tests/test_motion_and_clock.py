import json
import unittest
import numpy as np

from duplex_avatar.gateway import Session
from duplex_avatar.media import AudioChunks
from duplex_avatar.motion import MotionCache
from duplex_avatar.timing import BlockClock


class MotionAndClockTests(unittest.TestCase):
    def test_frame_anchor_uses_played_position_instead_of_latest_generated_chunk(self):
        cache = MotionCache(limit=3)
        frames = np.arange(33, dtype=np.uint8)[:, None, None, None] * np.ones((33, 2, 2, 3), dtype=np.uint8)
        history = np.arange(100, 110, dtype=np.float32)
        driving = np.arange(24, dtype=np.float32)
        for sequence in (1, 2, 3):
            cache.remember(sequence, frames + sequence, history, driving, 9, 25, 25)
        motion, audio, anchor = cache.resume({"sequence": 1, "frame_index": 8})
        np.testing.assert_array_equal(motion, (frames + 1)[9:18])
        np.testing.assert_array_equal(audio, np.concatenate((history[-1:], driving[:9])))
        self.assertEqual(anchor, {"sequence": 1, "frame_index": 8})
        self.assertIsNone(cache.resume({"sequence": 1, "frame_index": 24}))
        cache.remember(4, frames, history, driving, 9, 25, 25)
        self.assertIsNone(cache.resume({"sequence": 1, "frame_index": 8}))
        self.assertEqual(len(cache.clips), 3)
        cache.clear()
        self.assertIsNone(cache.resume({"sequence": 4, "frame_index": 8}))

    def test_clock_uses_actual_worker_shape_and_exact_sample_accounting(self):
        clock = BlockClock.from_worker({"fps": 25, "frames_per_chunk": 24, "chunk_samples": 23040})
        self.assertEqual((clock.duration_ms, clock.input_samples, clock.driving_samples), (960, 15360, 23040))
        self.assertEqual(clock.input_samples * 1500, 16000 * 1440)  # 24 minutes, zero drift
        alternate = BlockClock.from_worker({"fps": 25, "frames_per_chunk": 32, "chunk_samples": 30720})
        self.assertEqual(alternate.duration_ms, 1280)
        self.assertEqual(BlockClock.from_worker().duration_ms, 1000)
        with self.assertRaises(ValueError):
            BlockClock.from_worker({"fps": 25, "frames_per_chunk": 24, "chunk_samples": 24000})


class InterruptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_interrupt_prioritizes_silence_and_forwards_actual_frame_anchor(self):
        anchor = {"sequence": 12, "frame_index": 7}

        class Browser:
            messages = iter([{"type": "interrupt", "anchor": anchor}, {"type": "session.stop"}])
            async def receive_json(self):
                return next(self.messages)

        class Avatar:
            async def send(self, value):
                self.request = json.loads(value)

        session = Session(Browser())
        session.avatar = Avatar()
        session.clock = BlockClock.from_worker({"fps": 25, "frames_per_chunk": 24, "chunk_samples": 23040})
        session.chunks = AudioChunks(23040)
        session.outputs.put_nowait({"kind": "chunk", "epoch": 0})
        await session.read_browser()
        self.assertEqual(session.avatar.request["anchor"], anchor)
        transition = session.outputs.get_nowait()
        self.assertEqual((transition["kind"], transition["epoch"]), ("transition", 1))
        self.assertEqual(len(transition["samples"]), 23040)
        self.assertFalse(transition["samples"].any())
        self.assertTrue(session.outputs.empty())
