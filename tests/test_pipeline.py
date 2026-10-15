import asyncio
import json
import time
import unittest
import numpy as np

from duplex_avatar.gateway import Session
from duplex_avatar.media import AudioChunks, encode_pcm
from duplex_avatar.transport import Outbox, MediaWindow


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_unfinished_reply_does_not_insert_unplayed_idle_motion(self):
        session = Session(None)
        session.chunks = AudioChunks(23040)
        session.current_response = "waiting-for-more-speech"
        session.chunks.append(np.ones(100, dtype=np.float32))
        rendered = asyncio.Queue()
        class Avatar:
            async def send(self, raw):
                rendered.put_nowait(json.loads(raw))
        session.avatar = Avatar()
        # First speech block is playing while TTS is preparing the next one.
        session.playback_until = time.perf_counter() + .8
        writer = asyncio.create_task(session.write_avatar())
        try:
            await asyncio.sleep(.16)
            self.assertTrue(rendered.empty(), "Idle would advance the pose before the next speech block")
            self.assertEqual(len(session.chunks.pending), 100, "Partial original speech is untouched")
            session.chunks.clear()
            await asyncio.sleep(.12)
            self.assertTrue(rendered.empty(), "An empty audio buffer is still part of the unfinished reply")
            # Closing the reply releases idle prefetch, even before postroll ends.
            session.current_response = None
            request = await asyncio.wait_for(rendered.get(), .3)
            self.assertEqual(request["type"], "render")
            self.assertEqual(session.avatar_pending[request["sequence"]]["kind"], "idle")
            await asyncio.sleep(.1)
            self.assertTrue(rendered.empty(), "Only one idle render may be in flight")
        finally:
            writer.cancel()
            await asyncio.gather(writer, return_exceptions=True)

    async def test_idle_prefetch_is_bounded_and_speech_keeps_priority(self):
        session = Session(None)
        session.chunks = AudioChunks(23040)
        session.idle_playback_until = time.perf_counter() + 5
        rendered = asyncio.Queue()
        class Avatar:
            async def send(self, raw):
                rendered.put_nowait(json.loads(raw))
        session.avatar = Avatar()
        writer = asyncio.create_task(session.write_avatar())
        try:
            await asyncio.sleep(.12)
            self.assertTrue(rendered.empty(), "Do not generate excessive idle lookahead")
            await session.enqueue(np.ones(23040, dtype=np.float32), "reply")
            request = await asyncio.wait_for(rendered.get(), .3)
            self.assertEqual(session.avatar_pending[request["sequence"]]["kind"], "chunk")
        finally:
            writer.cancel()
            await asyncio.gather(writer, return_exceptions=True)

    async def test_media_window_waits_for_browser_and_ignores_duplicate_ack(self):
        window = MediaWindow(limit=2)
        await window.acquire(1)
        await window.acquire(2)
        pending = asyncio.create_task(window.acquire(3))
        await asyncio.sleep(.01)
        self.assertFalse(pending.done())
        self.assertIsNone(window.acknowledge(999))
        await asyncio.sleep(.01)
        self.assertFalse(pending.done())
        window.acknowledge(1)
        await asyncio.wait_for(pending, 1)
        self.assertEqual(set(window.pending), {2, 3})

    async def test_blocked_media_socket_does_not_delay_input_ack(self):
        blocked, entered, acknowledged = asyncio.Event(), asyncio.Event(), asyncio.Event()
        class SlowMedia:
            async def send_text(self, raw):
                entered.set()
                await blocked.wait()
        class Control:
            async def send_text(self, raw):
                if json.loads(raw)["type"] == "input.processed":
                    acknowledged.set()
        session = Session(Control())
        session.separate_media, session.media_browser = True, SlowMedia()
        session.media_ready.set()
        tasks = [asyncio.create_task(session.write_browser()), asyncio.create_task(session.write_media())]
        try:
            await session.send({"type": "media", "epoch": 0, "sequence": 1})
            await asyncio.wait_for(entered.wait(), 1)
            await session.send({"type": "input.processed", "client_input_id": 1})
            await asyncio.wait_for(acknowledged.wait(), 1)
            self.assertFalse(blocked.is_set())
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_control_priority_speech_order_and_latest_idle(self):
        box = Outbox()
        box.put({"type": "idle", "sequence": 1})
        box.put({"type": "idle", "sequence": 2})
        box.put({"type": "media", "sequence": 3})
        box.put({"type": "response.end", "sequence": 4})
        box.put({"type": "input.processed", "sequence": 5})
        self.assertEqual([(await box.get())["sequence"] for _ in range(4)], [5, 3, 4, 2])

    async def test_interrupt_discards_pending_old_audio_video_and_text(self):
        box = Outbox()
        for kind in ("idle", "media", "text"):
            box.put({"type": kind, "epoch": 0})
        box.put({"type": "input.processed"})
        box.discard_before(1)
        box.put({"type": "interrupted", "epoch": 1})
        self.assertEqual((await box.get())["type"], "input.processed")
        self.assertEqual((await box.get())["type"], "interrupted")
        self.assertFalse(box.media)
        self.assertIsNone(box.idle)

    async def test_backpressure_is_bounded(self):
        box = Outbox(media_limit=2)
        for _ in range(2):
            box.put({"type": "media"})
        with self.assertRaises(RuntimeError):
            box.put({"type": "media"})
        self.assertEqual(len(box.media), 2)

    async def test_slow_browser_and_pending_avatar_do_not_block_brain_reader(self):
        blocked = asyncio.Event()

        class Browser:
            async def send_text(self, raw):
                await blocked.wait()

        session = Session(Browser())
        session.output_mode = "avatar"
        session.diagnostics = True
        session.chunks = AudioChunks(23040)
        session.avatar_pending[1] = {"kind": "idle"}

        async def brain():
            for _ in range(3):
                yield json.dumps({"type": "response.output.delta", "kind": "audio", "response_id": "reply",
                                  "input_id": "0:1", "audio": encode_pcm(np.zeros(23040, dtype=np.float32))})
            yield json.dumps({"type": "input.processed", "input_id": "0:1"})

        session.brain = brain()
        writer = asyncio.create_task(session.write_browser())
        try:
            await asyncio.wait_for(session.read_brain(), 2)
            self.assertTrue(session.ack.is_set())
            self.assertEqual(session.outputs.qsize(), 3)
            self.assertFalse(writer.done())
        finally:
            writer.cancel()
            await asyncio.gather(writer, return_exceptions=True)

    async def test_audio_output_bypasses_avatar_chunking_and_preserves_pcm(self):
        session = Session(None)
        session.output_mode, session.diagnostics = "audio", False
        session.chunks = AudioChunks(23040)
        original = encode_pcm(np.linspace(-.3, .3, 7901, dtype=np.float32))

        async def brain():
            yield json.dumps({"type": "response.output.delta", "kind": "audio", "response_id": "reply",
                              "input_id": "0:1", "audio": original})

        session.brain = brain()
        await session.read_brain()
        media = await session.outbox.get()
        self.assertEqual(media["audio"], original)
        self.assertEqual(media["frames"], [])
        self.assertEqual(media["valid_samples"], 7901)
        self.assertTrue(session.outputs.empty())

    async def test_stale_playback_report_does_not_start_idle_early(self):
        class Browser:
            def __init__(self):
                self.messages = iter([
                    {"type": "playback", "epoch": 0, "sequence": 4, "speech_remaining_ms": 0},
                    {"type": "session.stop"},
                ])

            async def receive_json(self):
                return next(self.messages)

        session = Session(Browser())
        session.last_speech_sequence = 5
        session.playback_until = 12345
        await session.read_browser()
        self.assertEqual(session.playback_until, 12345)
