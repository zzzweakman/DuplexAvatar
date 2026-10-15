import asyncio
from concurrent.futures import Future
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MiniCPM-o-Demo"))
from duplex_avatar.parallel_session import ParallelSession
from py_backend.server import BackendProtocolSession, BackendServerState


class ParallelSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_interrupted_future_and_previously_completed_old_audio_are_not_delivered(self):
        session, events = self.session()
        session.backend.voice.generation = 1
        session._voice_queue = asyncio.Queue(maxsize=4)
        for result in ({"discarded": True, "generation": 0},
                       {**self.voice_result("old"), "metrics": {**self.voice_result("old")["metrics"], "tts_generation": 0}},
                       {**self.voice_result("new"), "metrics": {**self.voice_result("new")["metrics"], "tts_generation": 1}}):
            future = Future()
            future.set_result(result)
            session._voice_queue.put_nowait((future, {"response_id": "reply", "input_id": "1:3"}, []))
        task = asyncio.create_task(session._deliver_voice())
        try:
            await asyncio.wait_for(session._voice_queue.join(), 1)
            self.assertEqual([event["audio"] for event in events], ["new"])
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_close_waits_for_understanding_before_cancelling_voice_and_resetting(self):
        session, events = self.session()
        entered, finish = asyncio.Event(), asyncio.Event()
        cleaned = []
        async def processing(owner, payload):
            entered.set()
            await finish.wait()
        async def cleanup(owner, **kwargs):
            cleaned.append(True)
            owner.closed = True
        with patch.object(BackendProtocolSession, "_push_full_duplex", processing), patch.object(BackendProtocolSession, "close", cleanup):
            pushing = asyncio.create_task(session.push({"input": {"input_id": "0:1"}}))
            await entered.wait()
            closing = asyncio.create_task(session.close())
            await asyncio.sleep(.01)
            self.assertFalse(cleaned)
            finish.set()
            await asyncio.wait_for(asyncio.gather(pushing, closing), 1)
            self.assertEqual(cleaned, [True])
            self.assertFalse(events, "Closing must suppress output from the last in-flight decision")

    def session(self):
        events = []
        async def send_json(message):
            events.append(message)
        backend = SimpleNamespace(voice_future=None, metrics=lambda: {}, voice=SimpleNamespace(drain=lambda: None))
        session = ParallelSession(session_id="test", mode="full_duplex", backend=backend,
                                  ws=SimpleNamespace(send_json=send_json), state=BackendServerState(backend))
        session.initialized = True
        return session, events

    @staticmethod
    def voice_result(audio):
        return {"audio": audio, "metrics": {"cost_tts_ms": 3, "cost_token2wav_ms": 2, "tts_queue_ms": 0}}

    async def test_ack_and_next_decision_do_not_wait_for_voice_but_audio_keeps_order(self):
        session, events = self.session()
        first, second = Future(), Future()
        futures = iter((first, second))

        async def decision(owner, payload):
            owner.backend.voice_future = next(futures)
            vision = {"vision_slices": 1, "vision_tokens": 64} if payload["input_id"] == "0:1" else {"vision_slices": 0, "vision_tokens": 0}
            await owner.send_output_delta("text", response_id="reply", input_id=payload["input_id"], text="字", metrics=vision)
            if payload["input_id"] == "0:2":
                await owner.send_output_delta("listen", response_id="reply", input_id="0:2", metrics=vision)

        with patch.object(BackendProtocolSession, "_push_full_duplex", decision):
            try:
                for index in (1, 2):
                    await asyncio.wait_for(session.push({"input": {"input_id": f"0:{index}"}}), 1)
                self.assertEqual(sum(e["type"] == "input.processed" for e in events), 2)
                self.assertEqual([(e["vision_slices"], e["vision_tokens"]) for e in events if e["type"] == "input.processed"],
                                 [(1, 64), (0, 0)], "Visual consumption is acknowledged before pending TTS, without stale metrics")
                self.assertFalse(any(e.get("kind") == "audio" for e in events))
                second.set_result(self.voice_result("second"))
                await asyncio.sleep(.01)
                self.assertFalse(any(e.get("kind") in ("audio", "listen") for e in events))
                first.set_result(self.voice_result("first"))
                await asyncio.wait_for(session._voice_queue.join(), 1)
                delivered = [(e["kind"], e.get("audio"), e["input_id"]) for e in events
                             if e.get("kind") in ("audio", "listen")]
                self.assertEqual(delivered, [("audio", "first", "0:1"), ("audio", "second", "0:2"),
                                             ("listen", None, "0:2")])
            finally:
                session._voice_task.cancel()
                await asyncio.gather(session._voice_task, return_exceptions=True)

    async def test_vision_ack_does_not_require_a_text_or_listen_event(self):
        session, events = self.session()
        async def decision(owner, payload):
            owner.backend.last_input_vision = {"vision_slices": 5, "vision_tokens": 320,
                                              "vision_context_tokens": 8500, "vision_context_units": 24,
                                              "vision_window_mode": "context"}
        with patch.object(BackendProtocolSession, "_push_full_duplex", decision):
            try:
                await session.push({"input": {"input_id": "0:1"}})
                ack = next(event for event in events if event["type"] == "input.processed")
                self.assertEqual(ack["vision_tokens"], 320)
                self.assertEqual(ack["vision_context_units"], 24)
            finally:
                session._voice_task.cancel()
                await asyncio.gather(session._voice_task, return_exceptions=True)

    async def test_close_waits_for_gpu_job_before_upstream_cache_cleanup(self):
        session, events = self.session()
        future, drained = Future(), threading.Event()
        def drain():
            future.result()
            drained.set()
        session.backend.voice.drain = drain
        session._voice_queue = asyncio.Queue(maxsize=4)
        session._voice_queue.put_nowait((future, {"response_id": "old", "input_id": "0:1"}, []))
        session._voice_task = asyncio.create_task(session._deliver_voice())
        await asyncio.sleep(.01)

        async def base_close(owner, **kwargs):
            self.assertTrue(drained.is_set())
        with patch.object(BackendProtocolSession, "close", base_close):
            closing = asyncio.create_task(session.close())
            await asyncio.sleep(.02)
            self.assertFalse(closing.done())
            self.assertFalse(future.cancelled())
            future.set_result(self.voice_result("obsolete"))
            await asyncio.wait_for(closing, 1)
        self.assertFalse(any(e.get("kind") == "audio" for e in events))

    async def test_rejects_unsupported_chat_mode_before_touching_shared_gpu_state(self):
        session, _ = self.session()
        session.mode = "turn_based"
        with self.assertRaisesRegex(ValueError, "full_duplex"):
            await session.init({})
