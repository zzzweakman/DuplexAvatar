"""One HTTPS entry; brain and avatar inference run concurrently on separate GPUs."""
import asyncio
import contextlib
import json
import math
import logging
import time
import uuid

import httpx
import numpy as np
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from duplex_avatar.media import AudioChunks, decode_pcm, encode_pcm
from duplex_avatar.transport import Outbox, InputQueue, MediaWindow
from duplex_avatar.transcription import Utterances
from duplex_avatar.timing import BlockClock
from duplex_avatar.vision import frame_limit, parse_frames
from duplex_avatar.settings import ROOT, WORKSPACE, CONFIG, BRAIN_URL, BRAIN_WS, AVATAR_URL, ASR_URL, image_path, tls_paths

app = FastAPI(title="DuplexAvatar")
lease = asyncio.Lock()
sessions = {}


@app.get("/health")
async def health():
    async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
        async def probe(url):
            try:
                response = await client.get(url + "/health")
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                return {"status": "loading", "detail": str(exc)}
        brain, avatar, asr = await asyncio.gather(probe(BRAIN_URL), probe(AVATAR_URL), probe(ASR_URL))
        asr["gpu"] = CONFIG.get("asr_gpu", CONFIG["avatar_gpu"])
        if brain.get("status") == "ready":
            with contextlib.suppress(Exception):
                brain["acceleration"] = (await client.get(BRAIN_URL + "/acceleration")).json()
    return {"status": "ready" if brain.get("status") == avatar.get("status") == "ready" else "loading",
            "brain": brain, "avatar": avatar, "asr": asr, "busy": lease.locked(),
            "audio_ready": brain.get("status") == "ready",
            "gpus": sorted({CONFIG["brain_gpu"], CONFIG["avatar_gpu"],
                            CONFIG.get("tts_gpu", CONFIG["brain_gpu"]), CONFIG.get("asr_gpu", CONFIG["avatar_gpu"])})}


@app.get("/")
async def index():
    return FileResponse(ROOT / "frontend/index.html")


@app.get("/avatar.jpg")
async def avatar_image():
    return FileResponse(image_path())


@app.get("/demo/audio.wav")
async def demo_audio():
    fixture = ROOT / "runtime/fixtures/question.wav"
    if fixture.exists():
        return FileResponse(fixture)
    return FileResponse(WORKSPACE / "MiniCPM-o-Demo/examples/realtime/assets/test.wav")


class Session:
    def __init__(self, browser):
        self.browser = browser
        self.id = uuid.uuid4().hex
        self.epoch = 0
        self.sequence = 0
        self.sample_cursor = 0
        self.input_count = 0
        self.force_listen = 0
        self.brain_id = None
        self.inputs = InputQueue(limit=CONFIG["input_buffer_seconds"])
        self.started_ms = round(time.time() * 1000)
        self.received_count = self.processed_count = self.skipped_count = 0
        self.received_video_frames = self.processed_video_frames = 0
        self.last_vision_capture_ms = None
        self.last_vision_client_input_id = None
        self.last_received_client_input_id = None
        self.inflight_input = None
        self.clock = BlockClock.from_worker()
        self.video_input_fps = CONFIG.get("vision_input_fps", 5)
        self.asr_jobs = asyncio.Queue(maxsize=4)
        self.utterances = Utterances()
        self.response_times = {}
        self.outputs = asyncio.Queue(maxsize=CONFIG["max_pending_chunks"])
        self.ack = asyncio.Event()
        self.outbox = Outbox(media_limit=CONFIG["max_pending_chunks"])
        self.media_outbox = Outbox(media_limit=CONFIG["max_pending_chunks"])
        self.separate_media = False
        self.media_browser = None
        self.media_token = uuid.uuid4().hex
        self.media_ready, self.media_lost, self.done = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self.media_ack = False
        self.media_window = MediaWindow()
        self.avatar = None
        self.avatar_pending = {}
        self.avatar_slots = asyncio.Semaphore(2)
        self.generation = 0
        self.playback_until = 0
        self.last_speech_sequence = 0
        self.idle_playback_until = 0
        self.last_idle_sequence = 0
        self.next_idle = 0
        self.current_response = None
        self.tasks = []

    async def send(self, payload):
        box = self.media_outbox if self.separate_media and payload["type"] in ("media", "idle", "response.end") else self.outbox
        box.put(payload)

    async def write_browser(self):
        await self.write_channel(self.browser, self.outbox, "control")

    async def write_media(self):
        await asyncio.wait_for(self.media_ready.wait(), 15)
        await self.write_channel(self.media_browser, self.media_outbox, "media")

    async def watch_media(self):
        await self.media_lost.wait()
        raise RuntimeError("视频连接已断开，请重新开始会话。")

    async def write_channel(self, browser, box, channel):
        while True:
            payload = await box.get()
            tracked = self.media_ack and payload["type"] in ("media", "idle")
            if tracked:
                await self.media_window.acquire(payload["sequence"])
            # JPEG serialization must not hold up input acknowledgements or GPU dispatch.
            raw = await asyncio.to_thread(json.dumps, payload)
            if payload.get("epoch", self.epoch) != self.epoch:
                if tracked:
                    self.media_window.acknowledge(payload["sequence"])
                continue
            if payload["type"] == "idle" and payload["generation"] != self.generation:
                if tracked:
                    self.media_window.acknowledge(payload["sequence"])
                continue
            started = time.perf_counter()
            await browser.send_text(raw)
            if payload["type"] in ("media", "idle"):
                logging.getLogger("uvicorn.error").info(
                    "media_sent session=%s channel=%s sequence=%s bytes=%s send_ms=%.2f",
                    self.id, channel, payload.get("sequence"), len(raw), (time.perf_counter() - started) * 1000)

    async def run(self, options):
        self.input_mode = options.get("input_mode", "audiovisual")
        self.output_mode = options.get("output_mode", "avatar")
        self.separate_media = options.get("media_channel") == "separate" and self.output_mode == "avatar"
        self.media_ack = self.separate_media and options.get("media_ack") is True
        if self.input_mode not in ("audio", "audiovisual") or self.output_mode not in ("audio", "avatar"):
            raise ValueError("Unsupported input/output mode")
        default_prompt = CONFIG.get("vision_system_prompt", CONFIG["system_prompt"]) if self.input_mode == "audiovisual" else CONFIG["system_prompt"]
        system_prompt = str(options.get("system_prompt") or default_prompt)[:3500]
        if self.input_mode == "audio":
            system_prompt += ("\n当前是仅音频模式，没有提供任何图片或摄像头画面。你只能听到用户的声音。"
                              "如果用户询问图片内容或颜色，请明确说明当前看不到图片，建议开启摄像头；不要猜测图片内容。")
        self.diagnostics = bool(options.get("diagnostics", False))
        async with contextlib.AsyncExitStack() as stack:
            self.http = await stack.enter_async_context(httpx.AsyncClient(timeout=180, trust_env=False))
            avatar = None
            if self.output_mode == "avatar":
                self.avatar = await stack.enter_async_context(websockets.connect(
                    AVATAR_URL.replace("http:", "ws:") + "/stream", max_size=32 * 1024 * 1024,
                    ping_interval=20, ping_timeout=120, compression=None))
                await self.avatar.send(json.dumps({"type": "session.start", "session_id": self.id, "epoch": 0,
                                                   "video_codec": options.get("video_codec", "jpeg")}))
                avatar = json.loads(await asyncio.wait_for(self.avatar.recv(), 180))
                if avatar.get("type") != "session.ready":
                    raise RuntimeError(f"Avatar initialization failed: {avatar}")
            self.clock = BlockClock.from_worker(avatar)
            self.media_window.unit_ms = self.clock.duration_ms
            self.inputs.limit = math.ceil(CONFIG["input_buffer_seconds"] * 1000 / self.clock.duration_ms)
            self.chunks = AudioChunks(self.clock.driving_samples)
            self.brain = await stack.enter_async_context(websockets.connect(
                BRAIN_WS, max_size=16 * 1024 * 1024, ping_interval=20, ping_timeout=120))
            await self.brain.send(json.dumps({"type": "session.init", "payload": {
                    "mode": "full_duplex",
                    "system_prompt": system_prompt,
                    "config": {"force_listen_count": 1, "length_penalty": 1.0, "chunk_ms": self.clock.duration_ms,
                               "vision_input": self.input_mode == "audiovisual"},
                }}))
            created = json.loads(await asyncio.wait_for(self.brain.recv(), 180))
            if created.get("type") != "session.created":
                raise RuntimeError(f"MiniCPM initialization failed: {created}")
            self.brain_id = created["session_id"]
            self.started_ms = round(time.time() * 1000)
            if self.separate_media:
                sessions[self.id] = self
            await self.send({"type": "session.ready", "session_id": self.id, "epoch": self.epoch,
                             "session_started_ms": self.started_ms, "input_window": 2,
                             "input_window_max": 8, "media_ack": self.media_ack,
                             "video_codec": avatar.get("video_codec", "jpeg") if avatar else None,
                             "timing": self.clock.metadata(),
                             "input_buffer_seconds": CONFIG["input_buffer_seconds"],
                             "input_sample_rate": 16000, "output_sample_rate": 24000,
                             "video_input_fps": self.video_input_fps,
                             "max_video_frames_per_input": frame_limit(self.clock.duration_ms, self.video_input_fps),
                             "input_mode": self.input_mode, "output_mode": self.output_mode, "avatar": avatar,
                             "media_channel": {"path": "/media", "token": self.media_token} if self.separate_media else None})
            coroutines = [self.read_browser(), self.write_brain(), self.read_brain(), self.write_browser(), self.transcribe()]
            if self.avatar:
                coroutines += [self.write_avatar(), self.read_avatar()]
            if self.separate_media:
                coroutines += [self.write_media(), self.watch_media()]
            self.tasks = [asyncio.create_task(coro) for coro in coroutines]
            try:
                done, _ = await asyncio.wait(self.tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                for task in self.tasks:
                    task.cancel()
                await asyncio.gather(*self.tasks, return_exceptions=True)
                # Official backend closes via its HTTP control channel.
                if self.brain_id:
                    with contextlib.suppress(Exception):
                        await self.http.post(BRAIN_URL + f"/sessions/{self.brain_id}/close",
                                             json={"reason": "avatar_session_closed"})

    async def read_browser(self):
        while True:
            message = await self.browser.receive_json()
            kind = message.get("type")
            if kind == "session.stop":
                return
            if kind == "interrupt":
                self.epoch += 1
                self.generation += 1
                self.playback_until = 0
                self.last_speech_sequence = 0
                self.idle_playback_until = 0
                self.last_idle_sequence = 0
                self.next_idle = 0
                self.outbox.discard_before(self.epoch)
                self.media_outbox.discard_before(self.epoch)
                self.chunks.clear()
                self.current_response = None
                self.sample_cursor = 0
                self.force_listen = 2
                while not self.outputs.empty():
                    self.outputs.get_nowait()
                    self.outputs.task_done()
                await self.send({"type": "interrupted", "epoch": self.epoch})
                if self.avatar:
                    await self.avatar.send(json.dumps({"type": "interrupt", "epoch": self.epoch,
                                                       "anchor": message.get("anchor")}))
                    # Submit a silence block immediately, even while stale work is
                    # draining. It has priority over routine idle cadence.
                    now = time.perf_counter()
                    self.outputs.put_nowait({"kind": "transition", "epoch": self.epoch, "generation": self.generation,
                                             "samples": np.zeros(self.chunks.size, dtype=np.float32), "queued": now})
                    self.next_idle = now
            elif kind == "playback":
                remaining = float(message.get("speech_remaining_ms", 0))
                if not math.isfinite(remaining) or not 0 <= remaining <= 10000:
                    raise ValueError("Invalid playback buffer length")
                if message.get("epoch") == self.epoch and int(message.get("sequence", -1)) >= self.last_speech_sequence:
                    self.playback_until = time.perf_counter() + remaining / 1000
                idle_remaining = float(message.get("idle_remaining_ms", 0))
                if not math.isfinite(idle_remaining) or not 0 <= idle_remaining <= 10000:
                    raise ValueError("Invalid idle playback buffer length")
                if (message.get("epoch") == self.epoch and "idle_sequence" in message
                        and int(message["idle_sequence"]) >= self.last_idle_sequence):
                    self.idle_playback_until = time.perf_counter() + idle_remaining / 1000
            elif kind == "media.ack":
                elapsed = self.media_window.acknowledge(message.get("sequence"))
                if elapsed is not None:
                    logging.getLogger("uvicorn.error").info(
                        "media_ack session=%s sequence=%s delivery_and_decode_ms=%.2f pending=%s",
                        self.id, message.get("sequence"), elapsed, len(self.media_window.pending))
            elif kind == "input":
                pcm = decode_pcm(message.get("audio", ""), max_samples=self.clock.input_samples)
                if len(pcm) != self.clock.input_samples:
                    raise ValueError(f"Each input must contain {self.clock.input_samples} samples ({self.clock.duration_ms} ms)")
                start_ms = float(message.get("capture_start_ms", self.received_count * self.clock.duration_ms))
                if not math.isfinite(start_ms) or not 0 <= start_ms <= 48 * 3600 * 1000:
                    raise ValueError("Invalid capture timestamp")
                frames, frame_times, video_capture_ms = parse_frames(message, self.clock.duration_ms, self.video_input_fps, start_ms)
                if self.input_mode == "audio" and frames:
                    raise ValueError("Audio input mode does not accept camera frames")
                self.received_count += 1
                self.received_video_frames += len(frames)
                self.last_received_client_input_id = message.get("client_input_id")
                # ASR sees every received input, before any silence coalescing for MiniCPM.
                for job in self.utterances.append(pcm, start_ms):
                    if self.asr_jobs.full():
                        await self.send({"type": "asr.status", "state": "error", "message": "字幕识别积压，本段字幕未处理；语音对话继续。"})
                    else:
                        self.asr_jobs.put_nowait(job)
                        await self.send({"type": "asr.status", "state": "processing"})
                frames_rms = np.array([np.sqrt(np.mean(pcm[i:i + 1600] ** 2)) for i in range(0, len(pcm), 1600)])
                quiet = float(frames_rms.max()) < .001 and float(np.abs(pcm).max()) < .005
                dropped = self.inputs.put({"audio": message["audio"], "video_frames": frames,
                                          "client_input_id": message.get("client_input_id", self.received_count),
                                          "capture_start_ms": start_ms, "video_capture_ms": video_capture_ms,
                                          "video_frame_times_ms": frame_times, "quiet": quiet,
                                          "received": time.perf_counter()})
                self.skipped_count += len(dropped)
                logging.getLogger("uvicorn.error").info(
                    "input_received session=%s client_input=%s capture_ms=%.1f received_at_ms=%s pending=%s video_frames=%s",
                    self.id, message.get("client_input_id"), start_ms, round(time.time() * 1000), self.inputs.qsize(), len(frames))
                await self.send({"type": "input.received", "client_input_id": message.get("client_input_id"),
                                 **self.input_status()})
            elif kind == "ping":
                await self.send({"type": "pong"})
            else:
                raise ValueError(f"Unknown client message: {kind}")

    async def write_brain(self):
        import json
        while True:
            item = await self.inputs.get()
            self.input_count += 1
            self.inflight_input = item
            logging.getLogger("uvicorn.error").info(
                "input_dispatched session=%s client_input=%s queue_ms=%.2f",
                self.id, item["client_input_id"], (time.perf_counter() - item["received"]) * 1000)
            payload = {k: item[k] for k in ("audio", "video_frames")}
            payload.update(input_id=f"{self.epoch}:{self.input_count}", max_slice_nums=1,
                           force_listen=self.force_listen > 0)
            self.force_listen = max(0, self.force_listen - 1)
            self.ack.clear()
            await self.brain.send(json.dumps({"type": "input.append", "input": payload}))
            await asyncio.wait_for(self.ack.wait(), timeout=180)

    def input_status(self):
        return {"received_inputs": self.received_count, "processed_inputs": self.processed_count,
                "last_received_client_input_id": self.last_received_client_input_id,
                "skipped_silence_inputs": self.skipped_count,
                "pending_inputs": self.inputs.qsize() + int(self.inflight_input is not None),
                "input_unit_ms": self.clock.duration_ms,
                "received_audio_ms": self.received_count * self.clock.duration_ms,
                "received_video_frames": self.received_video_frames,
                "processed_video_frames": self.processed_video_frames,
                "last_vision_capture_ms": self.last_vision_capture_ms,
                "last_vision_client_input_id": self.last_vision_client_input_id,
                "skipped_silence_ms": self.skipped_count * self.clock.duration_ms}

    async def transcribe(self):
        while True:
            job = await self.asr_jobs.get()
            started = time.perf_counter()
            try:
                response = await self.http.post(ASR_URL + "/transcribe", content=job["pcm"],
                                               params={"start_ms": round(job["start_ms"]), "language": "zh"},
                                               headers={"Content-Type": "application/octet-stream"}, timeout=45)
                response.raise_for_status()
                result = response.json()
                if result.get("text"):
                    segments = result.get("segments") or []
                    start = max(job["start_ms"], segments[0]["start"] * 1000) if segments else job["start_ms"]
                    end = min(job["end_ms"], segments[-1]["end"] * 1000) if segments else job["end_ms"]
                    await self.send({"type": "transcript", "role": "user", "utterance_id": job["utterance_id"],
                                     "text": result["text"], "start_ms": round(start), "end_ms": round(max(start, end)),
                                     "captured_at_ms": self.started_ms + round(start),
                                     "completed_at_ms": round(time.time() * 1000),
                                     "asr_ms": round((time.perf_counter() - started) * 1000), "final": True})
                await self.send({"type": "asr.status", "state": "ready"})
            except Exception:
                logging.exception("ASR failed for %s", job["utterance_id"])
                await self.send({"type": "asr.status", "state": "error", "message": "本段字幕识别失败；语音对话继续。"})
            finally:
                self.asr_jobs.task_done()

    async def enqueue(self, samples, response_id, speech_samples=None):
        self.outputs.put_nowait({"kind": "chunk", "epoch": self.epoch,
                                "pts_samples": self.sample_cursor, "samples": samples,
                                "speech_samples": len(samples) if speech_samples is None else speech_samples,
                                "response_id": response_id, "queued": time.perf_counter()})
        self.sample_cursor += len(samples)

    async def read_brain(self):
        import json
        async for raw in self.brain:
            message = json.loads(raw)
            kind = message.get("type")
            if kind == "input.processed":
                self.ack.set()
                item = self.inflight_input or {}
                self.inflight_input = None
                self.processed_count += 1
                vision_tokens = int(message.get("vision_tokens") or 0)
                vision_slices = int(message.get("vision_slices") or 0)
                frame_count = len(item.get("video_frames", []))
                if frame_count and vision_slices >= frame_count and vision_tokens >= frame_count * 64:
                    self.processed_video_frames += len(item["video_frames"])
                    self.last_vision_capture_ms = item.get("video_capture_ms", item.get("capture_start_ms"))
                    self.last_vision_client_input_id = item.get("client_input_id")
                processing_ms = round(((message.get("processing_finished") or 0) - (message.get("processing_started") or 0)) * 1000, 1)
                logging.getLogger("uvicorn.error").info("input_processed session=%s input=%s processing_ms=%s pending=%s skipped_silence=%s vision_slices=%s vision_tokens=%s",
                                                        self.id, message.get("input_id"), processing_ms, self.inputs.qsize(), self.skipped_count,
                                                        vision_slices, vision_tokens)
                await self.send({"type": "input.processed", "input_id": message.get("input_id"),
                                 "client_input_id": item.get("client_input_id"), "processing_ms": processing_ms,
                                 "vision_slices": vision_slices, "vision_tokens": vision_tokens,
                                 "video_frames": len(item.get("video_frames", [])),
                                 "video_frame_times_ms": item.get("video_frame_times_ms"),
                                 **{key: message[key] for key in ("vision_context_tokens", "vision_context_units", "vision_window_mode")
                                    if key in message},
                                 "processing_started": message.get("processing_started"),
                                 "processing_finished": message.get("processing_finished"),
                                 "processing_scope": message.get("processing_scope", "input_and_voice"),
                                 **self.input_status()})
                continue
            if kind == "session.closed":
                raise RuntimeError(f"MiniCPM session closed: {message.get('reason')}")
            if kind != "response.output.delta":
                continue
            source_epoch = str(message.get("input_id", "-1:")).split(":", 1)[0]
            if source_epoch != str(self.epoch):
                continue
            output_kind = message.get("kind")
            response_id = message.get("response_id") or self.current_response
            if output_kind == "audio":
                self.current_response = response_id
                self.generation += 1
                samples = decode_pcm(message["audio"])
                if self.diagnostics:
                    await self.send({"type": "brain.audio", "audio": message["audio"],
                                 "epoch": self.epoch, "response_id": response_id,
                                 "metrics": message.get("metrics", {})})
                if self.output_mode == "audio":
                    self.sequence += 1
                    await self.send({"type": "media", "epoch": self.epoch, "sequence": self.sequence,
                                     "response_id": response_id, "audio": message["audio"], "frames": [],
                                     "sample_rate": 24000, "valid_samples": len(samples),
                                     "pts_samples": self.sample_cursor, "output_mode": "audio"})
                    self.sample_cursor += len(samples)
                else:
                    for block in self.chunks.append(samples):
                        await self.enqueue(block, response_id)
            elif output_kind == "text":
                timestamp = self.response_times.setdefault(response_id, round(time.time() * 1000))
                await self.send({"type": "text", "text": message.get("text", ""),
                                 "response_id": response_id, "epoch": self.epoch, "generated_at_ms": timestamp})
            elif output_kind == "listen":
                if self.current_response and self.output_mode == "avatar":
                    # Append actual zero driving samples before flushing the last partial block.
                    # Transport carries original speech separately from this visual postroll.
                    remaining = len(self.chunks.pending)
                    silence = np.zeros(round(CONFIG["speech_postroll_ms"] * 24), dtype=np.float32)
                    blocks = self.chunks.append(silence)
                    tail = self.chunks.flush()
                    if tail is not None:
                        blocks.append(np.pad(tail, (0, self.chunks.size - len(tail))))
                    for block in blocks:
                        real = min(remaining, len(block))
                        await self.enqueue(block, response_id, speech_samples=real)
                        remaining -= real
                if self.current_response:
                    if self.output_mode == "audio":
                        await self.send({"type": "response.end", "epoch": self.epoch, "response_id": response_id})
                    else:
                        self.outputs.put_nowait({"kind": "end", "epoch": self.epoch, "response_id": response_id})
                self.current_response = None
                await self.send({"type": "listening", "epoch": self.epoch,
                                 "metrics": message.get("metrics", {})})

    async def write_avatar(self):
        """Speech queue has priority; idle work never accumulates in that queue."""
        while True:
            await self.avatar_slots.acquire()
            item = None
            while item is None:
                try:
                    item = await asyncio.wait_for(self.outputs.get(), timeout=0.05)
                    self.outputs.task_done()
                except asyncio.TimeoutError:
                    now = time.perf_counter()
                    media_box = self.media_outbox if self.separate_media else self.outbox
                    # A2V motion is stateful. Idle advances its pose/history even
                    # when the browser later discards those frames. Never insert
                    # it between blocks of an unfinished reply, including a
                    # partial TTS block. Prefetch only in the listening phase.
                    deadline = max(self.playback_until, self.idle_playback_until) - self.chunks.size / 24000
                    if (CONFIG["idle_animation"] and not self.current_response and not self.chunks.pending.size
                            and not self.avatar_pending and not media_box.media and media_box.idle is None
                            and now >= max(self.next_idle, deadline)):
                        item = {"kind": "idle", "epoch": self.epoch, "generation": self.generation,
                                "samples": np.zeros(self.chunks.size, dtype=np.float32), "queued": now}
                        self.next_idle = now + .05
            if item["epoch"] != self.epoch:
                self.avatar_slots.release()
                continue
            self.sequence += 1
            sequence = self.sequence
            item.update(sequence=sequence, started=time.perf_counter())
            self.avatar_pending[sequence] = item
            request = {"type": "end" if item["kind"] == "end" else "render",
                       "session_id": self.id, "epoch": item["epoch"], "sequence": sequence}
            if "samples" in item:
                item["audio"] = encode_pcm(item["samples"])
                request["audio"] = item["audio"]
            await self.avatar.send(json.dumps(request))

    async def read_avatar(self):
        async for raw in self.avatar:
            rendered = json.loads(raw)
            if rendered["type"] == "error":
                raise RuntimeError(rendered["message"])
            item = self.avatar_pending.pop(rendered["sequence"])
            self.avatar_slots.release()
            if item["epoch"] != self.epoch or rendered["type"] == "dropped":
                continue
            if item["kind"] == "end":
                await self.send({"type": "response.end", "response_id": item["response_id"], "epoch": self.epoch})
                continue
            elapsed = (time.perf_counter() - item["started"]) * 1000
            rendered.pop("type")
            media = {**rendered, "epoch": item["epoch"], "sample_rate": 24000,
                     "queue_ms": round((item["started"] - item["queued"]) * 1000, 2),
                     "render_ms": round(elapsed, 2),
                     "rtf": round(elapsed / (len(item["samples"]) / 24), 3)}
            logging.getLogger("uvicorn.error").info(
                "avatar_done session=%s sequence=%s kind=%s inference_ms=%s encode_ms=%s worker_queue_ms=%s total_ms=%.2f",
                self.id, item["sequence"], item["kind"], media.get("inference_ms"), media.get("encode_ms"),
                media.get("worker_queue_ms"), elapsed)
            if item["kind"] in ("idle", "transition"):
                if item["generation"] == self.generation and not self.current_response and self.outputs.empty():
                    self.last_idle_sequence = item["sequence"]
                    self.idle_playback_until = max(self.idle_playback_until, self.playback_until, time.perf_counter()) + len(item["samples"]) / 24000
                    await self.send({**media, "type": "idle", "generation": item["generation"],
                                     "transition": item["kind"] == "transition"})
            else:
                self.idle_playback_until = 0
                self.last_speech_sequence = item["sequence"]
                self.playback_until = max(self.playback_until, time.perf_counter()) + len(item["samples"]) / 24000
                await self.send({**media, "type": "media",
                                 "audio": encode_pcm(item["samples"][:item["speech_samples"]]),
                                 "valid_samples": item["speech_samples"], "duration_samples": len(item["samples"]),
                                 "postroll_samples": len(item["samples"]) - item["speech_samples"],
                                 "response_id": item["response_id"], "pts_samples": item["pts_samples"],
                                 "output_mode": "avatar"})
        raise RuntimeError("Avatar worker disconnected")


@app.websocket("/ws")
async def websocket(browser: WebSocket):
    await browser.accept()
    if lease.locked():
        await browser.send_json({"type": "error", "message": "已有会话正在运行，请先结束该会话。"})
        await browser.close(code=1013)
        return
    async with lease:
        session = Session(browser)
        try:
            first = await asyncio.wait_for(browser.receive_json(), 20)
            if first.get("type") != "session.start":
                raise ValueError("First message must be session.start")
            await session.run(first)
        except (WebSocketDisconnect, websockets.exceptions.ConnectionClosedOK):
            pass
        except Exception as exc:
            import logging
            logging.exception("Avatar session failed")
            with contextlib.suppress(Exception):
                await browser.send_json({"type": "error", "message": str(exc) or type(exc).__name__})
        finally:
            sessions.pop(session.id, None)
            session.done.set()
            with contextlib.suppress(Exception):
                await browser.close()


@app.websocket("/media")
async def media_socket(browser: WebSocket):
    await browser.accept()
    session = None
    tasks = []
    try:
        hello = await asyncio.wait_for(browser.receive_json(), 15)
        candidate = sessions.get(hello.get("session_id"))
        if (candidate is None or candidate.done.is_set() or candidate.media_browser is not None
                or hello.get("token") != candidate.media_token):
            await browser.close(code=1008)
            return
        session = candidate
        session.media_browser = browser
        await browser.send_json({"type": "media.ready"})
        session.media_ready.set()
        tasks = [asyncio.create_task(browser.receive()), asyncio.create_task(session.done.wait())]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except (WebSocketDisconnect, asyncio.TimeoutError, ValueError):
        pass
    finally:
        if session is not None and not session.done.is_set():
            session.media_lost.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        with contextlib.suppress(Exception):
            await browser.close()


app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")

if __name__ == "__main__":
    cert, key = tls_paths()
    uvicorn.run(app, host="127.0.0.1", port=CONFIG["gateway_port"],
                ssl_certfile=str(cert), ssl_keyfile=str(key),
                ws_max_size=4 * 1024 * 1024, ws_per_message_deflate=False)
