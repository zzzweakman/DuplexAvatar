"""Real MiniCPM audiovisual inference -> FlashHead -> playable MP4, with PCM identity checks."""
import argparse
import asyncio
import base64
import io
import json
from pathlib import Path
import ssl
import statistics
import subprocess
import sys
import time
import wave

import httpx
import numpy as np
from PIL import Image
from scipy.signal import resample_poly
import websockets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from duplex_avatar.media import decode_pcm, encode_pcm
from duplex_avatar.settings import CONFIG, BRAIN_URL, BRAIN_WS, AVATAR_URL

TLS = ssl._create_unverified_context()
URL = f"https://127.0.0.1:{CONFIG['gateway_port']}"


def save_wav(path, samples, rate):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


async def prepare_question():
    folder = ROOT / "runtime/fixtures"
    folder.mkdir(parents=True, exist_ok=True)
    question = folder / "question.wav"
    if question.exists():
        return question
    bundled = ROOT / "assets/fixtures/question.wav"
    if bundled.exists():
        with question.open("xb") as output:
            output.write(bundled.read_bytes())
        return question
    async with websockets.connect(BRAIN_WS, max_size=16 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "session.init", "payload": {"mode": "turn_based"}}))
        created = json.loads(await ws.recv())
        assert created["type"] == "session.created", created
        await ws.send(json.dumps({"type": "input.append", "input": {
            "messages": [{"role": "user", "content": "请原样朗读这句话，不要添加其他内容：请问你看到的图片主要是什么颜色？请用一句简短的中文回答。"}],
            "streaming": True, "generation": {"max_new_tokens": 96, "do_sample": False},
            "tts": {"enabled": True}, "use_tts_template": True, "enable_thinking": False,
        }}, ensure_ascii=False))
        parts = []
        while True:
            message = json.loads(await asyncio.wait_for(ws.recv(), 180))
            if message.get("kind") == "audio" and message.get("audio"):
                parts.append(decode_pcm(message["audio"]))
            if message["type"] == "response.done":
                if message.get("audio"):
                    parts.append(decode_pcm(message["audio"]))
                break
            if message["type"] == "session.closed":
                raise RuntimeError(message)
        async with httpx.AsyncClient(trust_env=False, timeout=60) as client:
            await client.post(BRAIN_URL + f"/sessions/{created['session_id']}/close", json={"reason": "fixture_ready"})
    samples = np.concatenate(parts)
    assert len(samples) > 24000 and np.max(np.abs(samples)) > 0.01
    save_wav(question, samples, 24000)
    print(f"Question ready: {question}, {len(samples) / 24000:.2f}s", flush=True)
    return question


async def run_test(input_mode="audiovisual", output_mode="avatar"):
    question = await prepare_question()
    async with httpx.AsyncClient(verify=False, timeout=5, trust_env=False) as client:
        for _ in range(900):
            status = (await client.get(URL + "/health")).json()
            if status["status"] == "ready" and not status["busy"]:
                break
            await asyncio.sleep(2)
        else:
            raise RuntimeError("Services did not become ready")
    with wave.open(str(question)) as wav:
        input_audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
        input_audio = resample_poly(input_audio, 16000, wav.getframerate()).astype(np.float32)
    encoded_image = io.BytesIO()
    Image.new("RGB", (224, 224), (237, 41, 41)).save(encoded_image, format="JPEG")
    frame = base64.b64encode(encoded_image.getvalue()).decode()
    avatar_renders_before = status["avatar"]["render_count"]
    output = ROOT / "runtime" / time.strftime(f"smoke-{input_mode}-{output_mode}-%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    (output / "input.jpg").write_bytes(encoded_image.getvalue())
    save_wav(output / "input.wav", input_audio, 16000)
    text, audio, media = {}, {}, {}
    ack = asyncio.Queue()
    completed = asyncio.Event()
    first_media = None
    vision_slices = 0
    chosen = None
    idle = []
    brain_intervals = []
    avatar_intervals = []
    idle_ready = asyncio.Event()
    started = time.perf_counter()

    async with websockets.connect(URL.replace("https:", "wss:") + "/ws", ssl=TLS,
                                  max_size=32 * 1024 * 1024, ping_timeout=180) as ws:
        await ws.send(json.dumps({"type": "session.start", "diagnostics": True,
                                  "input_mode": input_mode, "output_mode": output_mode}))
        ready = json.loads(await asyncio.wait_for(ws.recv(), 180))
        assert ready["type"] == "session.ready", ready
        print("Session ready", ready, flush=True)
        unit_samples = ready["timing"]["input_samples"]
        unit_seconds = ready["timing"]["decision_ms"] / 1000

        async def reader():
            nonlocal first_media, vision_slices, chosen
            async for raw in ws:
                message = json.loads(raw)
                kind = message["type"]
                response_id = message.get("response_id")
                vision_slices += message.get("metrics", {}).get("vision_slices", 0)
                if kind == "error":
                    raise RuntimeError(message)
                if kind == "input.processed":
                    ack.put_nowait(message)
                    brain_intervals.append([message["processing_started"], message["processing_finished"]])
                elif kind == "text":
                    text[response_id] = text.get(response_id, "") + message["text"]
                    print("Text:", message["text"], flush=True)
                elif kind == "brain.audio":
                    audio.setdefault(response_id, []).append(decode_pcm(message["audio"]))
                elif kind == "media":
                    first_media = first_media or time.perf_counter()
                    media.setdefault(response_id, []).append(message)
                    if message.get("inference_started"):
                        avatar_intervals.append([message["inference_started"], message["inference_finished"]])
                    print(f"Media: {len(message['frames'])} frames, RTF={message.get('rtf')}, queue={message.get('queue_ms')} ms", flush=True)
                elif kind == "idle":
                    assert "audio" not in message, "Idle driving silence must not be played as speech"
                    if len(idle) < 4:
                        idle.append(message)
                    avatar_intervals.append([message["inference_started"], message["inference_finished"]])
                    if len(idle) >= 3:
                        idle_ready.set()
                elif kind == "response.end" and media.get(response_id):
                    chosen = response_id
                    completed.set()
                    return

        async def sender():
            if output_mode == "avatar":
                await asyncio.wait_for(idle_ready.wait(), 30)
                # Start near the next idle inference to exercise independent GPUs.
                await asyncio.sleep(0.78)
            count = int(np.ceil(len(input_audio) / unit_samples)) + int(np.ceil(30 / unit_seconds))
            for index in range(count):
                if completed.is_set():
                    return
                t0 = time.perf_counter()
                chunk = np.zeros(unit_samples, dtype=np.float32)
                part = input_audio[index * unit_samples:(index + 1) * unit_samples]
                chunk[:len(part)] = part
                await ws.send(json.dumps({"type": "input", "audio": encode_pcm(chunk),
                                          "video_frames": [frame] if input_mode == "audiovisual" else []}))
                await asyncio.wait_for(ack.get(), 180)
                await asyncio.sleep(max(0, unit_seconds - (time.perf_counter() - t0)))
            await asyncio.wait_for(completed.wait(), 60)

        reading = asyncio.create_task(reader())
        sending = asyncio.create_task(sender())
        try:
            await asyncio.wait_for(asyncio.gather(reading, sending), 300)
        finally:
            for task in (reading, sending):
                task.cancel()
            await asyncio.gather(reading, sending, return_exceptions=True)
            await ws.send(json.dumps({"type": "session.stop"}))
    assert chosen, "No completed audiovisual reply"
    if input_mode == "audiovisual":
        assert "红" in text.get(chosen, ""), text
        assert vision_slices > 0, "No visual tokens were processed"
    else:
        assert vision_slices == 0, "Audio-only mode processed an image"
        assert text.get(chosen), "No text reply"
        assert any(word in text[chosen] for word in ("看不到", "无法看到", "无法看", "不能看到", "没有看到", "没有收到", "摄像头", "没看到", "未收到")), text[chosen]
    original = np.concatenate(audio[chosen])
    delivered = np.concatenate([decode_pcm(item["audio"]) for item in media[chosen]])
    assert np.array_equal(original, delivered), "Delivered audio differs from MiniCPM output"
    assert np.max(np.abs(original)) > 0.01, "Silent response"
    save_wav(output / "minicpm_reply.wav", original, 24000)
    (output / "minicpm_reply.f32").write_bytes(original.astype("<f4").tobytes())
    if output_mode == "audio":
        assert all(not item["frames"] for items in media.values() for item in items)
        assert not idle, "Audio output unexpectedly started avatar animation"
        async with httpx.AsyncClient(trust_env=False) as client:
            after = (await client.get(AVATAR_URL + "/health")).json()["render_count"]
        assert after == avatar_renders_before, "Audio-only output used the avatar GPU"
        result = {"passed": True, "input_mode": input_mode, "output_mode": output_mode,
                  "text": text[chosen], "audio_seconds": len(original) / 24000,
                  "audio_bit_exact_before_encoding": True, "vision_slices": vision_slices,
                  "acknowledged_missing_visual_input": True,
                  "avatar_renders": after - avatar_renders_before, "frames": 0}
        with (output / "results.json").open("x") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return
    idle_images = [Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
                   for item in idle[:3] for value in item["frames"]]
    idle_motion = float(np.mean(np.abs(np.asarray(idle_images[0], dtype=float) - np.asarray(idle_images[-1], dtype=float))))
    assert idle_motion > 0.1, f"Silence-driven video is static: {idle_motion}"
    idle_images[0].save(output / "idle-first.jpg")
    idle_images[-1].save(output / "idle-last.jpg")
    overlaps = [max(0, min(brain[1], avatar[1]) - max(brain[0], avatar[0]))
                for brain in brain_intervals for avatar in avatar_intervals]
    assert any(value > 0 for value in overlaps), "No observed brain-processing/avatar-inference overlap"
    frames = [Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
              for item in media[chosen] for value in item["frames"]]
    assert len(frames) >= 20, "Too few video frames"
    motion = float(np.mean(np.abs(np.asarray(frames[0], dtype=float) - np.asarray(frames[-1], dtype=float))))
    assert motion > 0.1, "Video frames are static"
    frames[0].save(output / "first-frame.jpg")
    frames[-1].save(output / "last-frame.jpg")
    playback_audio = [np.pad(decode_pcm(item["audio"]),
                            (0, item["duration_samples"] - len(decode_pcm(item["audio"]))))
                      for item in media[chosen]]
    save_wav(output / "playback-with-postroll.wav", np.concatenate(playback_audio), 24000)
    width, height = frames[0].size
    with (output / "ffmpeg.log").open("x") as log:
        encoder = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", "25", "-i", "pipe:0", "-i", str(output / "playback-with-postroll.wav"),
            "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(output / "reply.mp4"),
        ], stdin=subprocess.PIPE, stderr=log)
        for frame_image in frames:
            encoder.stdin.write(frame_image.tobytes())
        encoder.stdin.close()
        assert encoder.wait() == 0, "Video encoding failed"
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output / "reply.mp4")]))
    assert {s["codec_name"] for s in probe["streams"]} >= {"h264", "aac"}
    result = {"passed": True, "text": text[chosen], "audio_seconds": len(original) / 24000,
              "input_mode": input_mode, "output_mode": output_mode,
              "idle_chunks": len(idle), "idle_motion_mean_absolute_difference": idle_motion,
              "overlapping_processing_pairs": sum(value > 0 for value in overlaps),
              "processing_overlap_ms": round(sum(overlaps) * 1000, 2),
              "frames": len(frames), "resolution": [width, height], "vision_slices": vision_slices,
              "audio_bit_exact_before_encoding": True, "frame_motion_mean_absolute_difference": motion,
              "first_media_seconds": round(first_media - started, 3),
              "median_avatar_rtf": statistics.median([m["rtf"] for m in media[chosen]]),
              "chunks": [{key: m[key] for key in ("sequence", "valid_samples", "inference_ms", "encode_ms", "queue_ms", "render_ms", "rtf")} for m in media[chosen]],
              "output": str(output / "reply.mp4")}
    (output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--audio-input", action="store_true")
    parser.add_argument("--audio-output", action="store_true")
    args = parser.parse_args()
    asyncio.run(prepare_question() if args.prepare_only else run_test(
        "audio" if args.audio_input else "audiovisual", "audio" if args.audio_output else "avatar"))
