"""Resident FlashHead worker: 24 kHz source PCM in, JPEG frames out."""
import asyncio
import base64
import io
import json
import math
import os
import sys
import time
from collections import deque
from contextlib import asynccontextmanager, suppress

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from PIL import Image
from scipy.signal import resample_poly

from duplex_avatar.media import decode_pcm
from duplex_avatar.motion import MotionCache
from duplex_avatar.video_transport import encode_h264, encoder_backend
from duplex_avatar.settings import CONFIG, WORKSPACE, image_path

FLASHHEAD = WORKSPACE / "SoulX-FlashHead"
sys.path.insert(0, str(FLASHHEAD))
os.chdir(FLASHHEAD)


class StartRequest(BaseModel):
    session_id: str
    epoch: int = 0


class RenderRequest(StartRequest):
    audio: str = Field(max_length=400000)
    sequence: int


class Engine:
    def load(self, world_size=1):
        import torch
        from flash_head.src.pipeline import flash_head_pipeline
        flash_head_pipeline.COMPILE_MODEL = CONFIG["compile_avatar"]
        flash_head_pipeline.COMPILE_VAE = CONFIG["compile_avatar"]
        from flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline

        if world_size > 1:
            from duplex_avatar.parallel_avatar import configure_attention
            configure_attention()

        self.torch = torch
        self.get_base_data = get_base_data
        self.get_audio_embedding = get_audio_embedding
        self.run_pipeline = run_pipeline
        self.pipeline = get_pipeline(
            world_size=world_size, ckpt_dir=str(FLASHHEAD / "models/SoulX-FlashHead-1_3B"),
            wav2vec_dir=str(FLASHHEAD / "models/wav2vec2-base-960h"), model_type=CONFIG["model_type"],
        )
        self.params = get_infer_params()
        self.fps = self.params["tgt_fps"]
        self.frames = self.params["frame_num"] - self.params["motion_frames_num"]
        self.chunk_samples = self.frames * 24000 // self.fps
        self.session_id = None
        self.epoch = -1
        self.render_count = 0
        self.motion_cache = MotionCache()
        self.reset("warmup", 0, prepare=True)
        for sequence in range(2):
            self.render(np.zeros(self.chunk_samples, dtype=np.float32), sequence)
        self.resume_motion("warmup", 1, {"sequence": 0, "frame_index": 7})
        self.render(np.zeros(self.chunk_samples, dtype=np.float32))
        self.initial_motion = self.pipeline.latent_motion_frames.clone()
        self.reset(None, 0, prepare=True)
        self.initial_motion = self.pipeline.latent_motion_frames.clone()

    def reset(self, session_id, epoch, prepare=False):
        if prepare:
            self.get_base_data(self.pipeline, cond_image_path_or_dir=str(image_path()),
                               base_seed=42, use_face_crop=False)
        elif hasattr(self, "initial_motion"):
            self.pipeline.latent_motion_frames = self.initial_motion.clone()
        self.session_id, self.epoch = session_id, epoch
        self.last_sequence = -1
        self.motion_cache.clear()
        self.history = deque([0.0] * (self.params["sample_rate"] * self.params["cached_audio_duration"]),
                             maxlen=self.params["sample_rate"] * self.params["cached_audio_duration"])

    def resume_motion(self, session_id, epoch, anchor=None):
        if session_id != self.session_id:
            raise ValueError("Cannot resume motion from a different session")
        started = time.perf_counter()
        checkpoint = self.motion_cache.resume(anchor)
        info = {"anchor_matched": checkpoint is not None, "motion_preserved": True}
        if checkpoint is not None:
            frames, audio, used = checkpoint
            with self.torch.no_grad():
                video = self.torch.from_numpy(frames.copy()).to(device=self.pipeline.device, dtype=self.pipeline.param_dtype)
                video = (video.permute(3, 0, 1, 2).unsqueeze(0).contiguous() / 255 - .5) * 2
                self.pipeline.latent_motion_frames = self.pipeline.vae.encode(video)
            self.history = deque(audio.tolist(), maxlen=self.history.maxlen)
            info.update(anchor=used, motion_frames=len(frames))
        else:
            # Older clients or expired checkpoints retain the current pose. Never
            # jump back to initial_motion just because the reply epoch changed.
            self.history = deque([0.0] * self.history.maxlen, maxlen=self.history.maxlen)
        self.epoch, self.last_sequence = epoch, -1
        info["resume_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return info

    def render(self, original, sequence=None):
        started = time.perf_counter()
        valid_samples = len(original)
        padded = np.pad(original, (0, self.chunk_samples - valid_samples))
        driving = resample_poly(padded, 2, 3).astype(np.float32)
        history_before = np.asarray(self.history, dtype=np.float32) if sequence is not None else None
        self.history.extend(driving.tolist())
        end = self.params["cached_audio_duration"] * self.fps
        embedding = self.get_audio_embedding(self.pipeline, np.asarray(self.history, dtype=np.float32),
                                             end - self.params["frame_num"], end)
        video = self.run_pipeline(self.pipeline, embedding)
        prefix = self.params["motion_frames_num"]
        valid_frames = min(len(video) - prefix, math.ceil(valid_samples * self.fps / 24000))
        full_frames = video[:prefix + valid_frames].clamp(0, 255).to(self.torch.uint8).cpu().numpy()
        frames = full_frames[prefix:]
        if sequence is not None:
            self.motion_cache.remember(sequence, full_frames, history_before, driving, prefix,
                                       self.fps, self.params["sample_rate"])
        self.render_count += 1
        return frames, (time.perf_counter() - started) * 1000


engine = Engine()
operation = asyncio.Lock()
stream_lease = asyncio.Lock()


async def gpu_call(function, *args):
    # Cancellation cannot stop CUDA work in a Python thread. Keep its lock until
    # the thread finishes, so a reconnect cannot reset motion state underneath it.
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(engine.load)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ready", "backend": "flashhead", "variant": CONFIG["model_type"],
            "gpu": CONFIG["avatar_gpu"], "fps": engine.fps, "frames_per_chunk": engine.frames,
            "chunk_samples": engine.chunk_samples, "compile": CONFIG["compile_avatar"],
            "motion_anchor": True, "motion_cache_chunks": len(engine.motion_cache.clips),
            "video_codecs": ["h264", "jpeg"],
            "video_encoder": encoder_backend(),
            "render_count": engine.render_count, "stream_busy": stream_lease.locked()}


@app.websocket("/stream")
async def stream(browser: WebSocket):
    """Ordered GPU execution overlaps CPU JPEG encoding and socket I/O."""
    await browser.accept()
    if stream_lease.locked():
        await browser.send_json({"type": "error", "message": "Avatar worker is occupied"})
        await browser.close()
        return
    async with stream_lease:
        tasks = []
        try:
            first = await asyncio.wait_for(browser.receive_json(), 20)
            if first.get("type") != "session.start":
                raise ValueError("Expected session.start")
            session_id, epoch = str(first["session_id"]), int(first["epoch"])
            codec = first.get("video_codec", "jpeg")
            if codec not in ("h264", "jpeg"):
                raise ValueError("Unsupported video codec")
            async with operation:
                await gpu_call(engine.reset, session_id, epoch)
            await browser.send_json({"type": "session.ready", "video_codec": codec, **await health()})
            pending = asyncio.Queue(maxsize=2)
            encoding = asyncio.Queue(maxsize=2)
            anchors = {}

            async def receive():
                nonlocal epoch
                while True:
                    request = await browser.receive_json()
                    if request.get("type") == "interrupt":
                        epoch = max(epoch, int(request["epoch"]))
                        anchors[epoch] = request.get("anchor")
                        for old in list(anchors):
                            if old < epoch:
                                anchors.pop(old)
                    elif request.get("type") in ("render", "end"):
                        if request.get("session_id") != session_id:
                            raise ValueError("Mismatched avatar session")
                        epoch = max(epoch, int(request["epoch"]))
                        request["received_at"] = time.perf_counter()
                        pending.put_nowait(request)
                    else:
                        raise ValueError("Invalid avatar stream message")

            async def infer():
                while True:
                    request = await pending.get()
                    result = {"sequence": request["sequence"], "epoch": request["epoch"]}
                    if request["epoch"] != epoch:
                        result["type"] = "dropped"
                    elif request["type"] == "end":
                        result["type"] = "end"
                    else:
                        samples = decode_pcm(request["audio"], max_samples=engine.chunk_samples)
                        if not len(samples):
                            raise ValueError("Empty driving audio")
                        async with operation:
                            resume_info = None
                            if engine.epoch != request["epoch"]:
                                resume_info = await gpu_call(engine.resume_motion, session_id, request["epoch"],
                                                             anchors.pop(request["epoch"], None))
                            if request["sequence"] <= engine.last_sequence:
                                raise ValueError("Out-of-order avatar sequence")
                            began = time.perf_counter()
                            frames, elapsed = await gpu_call(engine.render, samples, request["sequence"])
                            engine.last_sequence = request["sequence"]
                        result.update(type="frames", raw_frames=frames, fps=engine.fps,
                                      width=int(frames.shape[2]), height=int(frames.shape[1]),
                                      valid_samples=len(samples), inference_ms=round(elapsed, 2),
                                      worker_queue_ms=round((began - request["received_at"]) * 1000, 2),
                                      inference_started=began, inference_finished=time.perf_counter())
                        if resume_info is not None:
                            result["motion_resume"] = resume_info
                    await encoding.put(result)
                    pending.task_done()

            async def encode_and_send():
                while True:
                    result = await encoding.get()
                    frames = result.pop("raw_frames", None)
                    if result["epoch"] != epoch:
                        result = {"type": "dropped", "epoch": result["epoch"], "sequence": result["sequence"]}
                    elif frames is not None:
                        started = time.perf_counter()
                        if codec == "h264":
                            result["video"] = await asyncio.to_thread(encode_h264, frames, engine.fps)
                        else:
                            result["frames"] = await asyncio.to_thread(encode_frames, frames)
                        result["encode_ms"] = round((time.perf_counter() - started) * 1000, 2)
                    if result["epoch"] != epoch:
                        result = {"type": "dropped", "epoch": result["epoch"], "sequence": result["sequence"]}
                    raw = await asyncio.to_thread(json.dumps, result)
                    await browser.send_text(raw)
                    encoding.task_done()

            tasks = [asyncio.create_task(coro) for coro in (receive(), infer(), encode_and_send())]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            import logging
            logging.exception("Avatar stream failed")
            with suppress(Exception):
                await browser.send_json({"type": "error", "message": str(exc)})
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            with suppress(Exception):
                await browser.close()


@app.post("/session")
async def session(request: StartRequest):
    async with operation:
        await gpu_call(engine.reset, request.session_id, request.epoch)
    return {"status": "ready", "epoch": request.epoch}


def encode_frames(frames):
    result = []
    for frame in frames:
        output = io.BytesIO()
        Image.fromarray(frame).save(output, format="JPEG", quality=CONFIG["jpeg_quality"])
        result.append(base64.b64encode(output.getvalue()).decode("ascii"))
    return result


@app.post("/render")
async def render(request: RenderRequest):
    try:
        samples = decode_pcm(request.audio, max_samples=engine.chunk_samples)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    if not len(samples):
        raise HTTPException(400, "Audio is empty")
    async with operation:
        if request.session_id != engine.session_id or request.epoch != engine.epoch:
            raise HTTPException(409, "Stale avatar session")
        if request.sequence <= engine.last_sequence:
            raise HTTPException(409, "Duplicate or out-of-order chunk")
        frames, inference_ms = await gpu_call(engine.render, samples)
        engine.last_sequence = request.sequence
    # Encoding can overlap GPU work from another queued request.
    started = time.perf_counter()
    encoded = await asyncio.to_thread(encode_frames, frames)
    return {"frames": encoded, "fps": engine.fps, "width": int(frames.shape[2]),
            "height": int(frames.shape[1]), "valid_samples": len(samples),
            "inference_ms": round(inference_ms, 2),
            "encode_ms": round((time.perf_counter() - started) * 1000, 2)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=CONFIG["avatar_port"])
