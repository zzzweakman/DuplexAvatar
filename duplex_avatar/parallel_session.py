"""Return MiniCPM decisions promptly while an ordered task delivers original TTS."""
import asyncio
import contextlib
import logging
import time

from py_backend import server


class ParallelSession(server.BackendProtocolSession):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lifecycle_lock = asyncio.Lock()
        self._closing = False

    async def send(self, event_type, **fields):
        if self._closing and event_type != "session.closed":
            return
        await super().send(event_type, **fields)

    async def init(self, params):
        if self.mode != "full_duplex":
            raise ValueError("The separate-GPU voice deployment supports full_duplex sessions only")
        await super().init(params)

    async def send_output_delta(self, kind, **fields):
        if getattr(self, "_step_events", None) is not None:
            self._step_events.append((kind, fields))
            if kind != "text":
                return
        await super().send_output_delta(kind, **fields)

    async def _push_full_duplex(self, payload):
        if not hasattr(self, "_voice_queue"):
            self._voice_queue = asyncio.Queue(maxsize=4)
            self._voice_task = asyncio.create_task(self._deliver_voice())
        if self._voice_task.done():
            self._voice_task.result()
        self._step_events = []
        try:
            await super()._push_full_duplex(payload)
            events = self._step_events
        finally:
            self._step_events = None
        fields = dict(events[-1][1]) if events else {
            "session_id": self.session_id, "response_id": self._active_response_id,
            "input_id": payload.get("input_id"), "metrics": self._safe_metrics()}
        self._input_vision = dict(getattr(self.backend, "last_input_vision", {
            key: fields.get("metrics", {}).get(key, 0) for key in ("vision_slices", "vision_tokens")}))
        enqueue = asyncio.create_task(self._voice_queue.put(
            (self.backend.voice_future, fields, [(kind, data) for kind, data in events if kind != "text"])))
        try:
            await asyncio.wait((enqueue, self._voice_task), return_when=asyncio.FIRST_COMPLETED)
            if self._voice_task.done():
                self._voice_task.result()
            await enqueue
        finally:
            enqueue.cancel()
            await asyncio.gather(enqueue, return_exceptions=True)

    async def _deliver_voice(self):
        try:
            while True:
                future, fields, events = await self._voice_queue.get()
                try:
                    if future is not None:
                        result = await asyncio.shield(asyncio.wrap_future(future))
                        generation = result.get("metrics", {}).get("tts_generation")
                        if result.get("discarded") or (generation is not None and generation != self.backend.voice.generation):
                            logging.getLogger("uvicorn.error").info(
                                "voice_discarded input=%s generation=%s", fields.get("input_id"),
                                result.get("generation", generation))
                            continue
                        metrics = {**fields.get("metrics", {}), **result["metrics"]}
                        await super().send_output_delta(
                            "audio", session_id=self.session_id, response_id=fields["response_id"],
                            input_id=fields.get("input_id"), audio=result["audio"], metrics=metrics)
                        logging.getLogger("uvicorn.error").info(
                            "voice_done input=%s tts_ms=%s token2wav_ms=%s queue_ms=%s",
                            fields.get("input_id"), metrics["cost_tts_ms"], metrics["cost_token2wav_ms"], metrics["tts_queue_ms"])
                    for kind, data in events:
                        await super().send_output_delta(kind, **data)
                finally:
                    self._voice_queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logging.exception("Asynchronous MiniCPM speech failed")
            with contextlib.suppress(Exception):
                await self.send("session.closed", reason="voice_error", message=str(exc))
            raise

    async def push(self, message):
        async with self._lifecycle_lock:
            if self._closing:
                return
            started = time.perf_counter()
            self._input_vision = {}
            await super().push(message)
            await self.send("input.processed", input_id=message.get("input", {}).get("input_id"),
                            processing_started=started, processing_finished=time.perf_counter(),
                            processing_scope="input_and_llm_voice_runs_asynchronously",
                            **self._input_vision)

    async def close(self, **kwargs):
        self._closing = True
        async with self._lifecycle_lock:
            if self.closed:
                return
            task = getattr(self, "_voice_task", None)
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            # In-flight understanding and speech must both finish before cleanup.
            await asyncio.to_thread(self.backend.voice.drain)
            await super().close(**kwargs)
