"""Fixed-rate input (never ACK-paced): ASR + MiniCPM + A2V under real concurrency."""
import asyncio
import base64
import io
import json
import math
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
from duplex_avatar.media import encode_pcm, decode_pcm
from duplex_avatar.settings import CONFIG


async def main():
    output = ROOT / "runtime" / time.strftime("realtime-%Y%m%d-%H%M%S")
    output.mkdir(exist_ok=False)
    with wave.open(str(ROOT / "runtime/fixtures/question.wav")) as wav:
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
        audio = resample_poly(audio, 16000, wav.getframerate()).astype(np.float32)
    image = io.BytesIO()
    Image.new("RGB", (224, 224), (237, 41, 41)).save(image, format="JPEG")
    frame = base64.b64encode(image.getvalue()).decode()
    events, transcripts, processing, rtf, overlaps = [], [], [], [], []
    original, delivered, videos, texts, tails, ended = {}, {}, {}, {}, {}, set()
    brain_times, avatar_times = [], []
    pending_max = skipped_max = 0
    received = processed = 0
    seconds = 70
    url = f"https://127.0.0.1:{CONFIG['gateway_port']}"
    async with httpx.AsyncClient(verify=False, trust_env=False) as client:
        health = (await client.get(url + "/health")).json()
        assert health["status"] == "ready" and not health["busy"], health
        assert health["asr"]["status"] == "ok", health
    async with websockets.connect(url.replace("https:", "wss:") + "/ws", ssl=ssl._create_unverified_context(),
                                  compression=None, max_size=32 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "session.start", "diagnostics": True, "media_channel": "separate"}))
        ready = json.loads(await ws.recv())
        assert ready["type"] == "session.ready", ready
        media_ws = None
        if ready.get("media_channel"):
            media_ws = await websockets.connect(url.replace("https:", "wss:") + ready["media_channel"]["path"],
                                                ssl=ssl._create_unverified_context(), compression=None,
                                                max_size=32 * 1024 * 1024)
            await media_ws.send(json.dumps({"session_id": ready["session_id"], "token": ready["media_channel"]["token"]}))
            assert json.loads(await media_ws.recv())["type"] == "media.ready"
        timing = ready["timing"]
        unit_ms, unit_samples = timing["decision_ms"], timing["input_samples"]
        driving_samples = timing["driving_samples"]
        assert unit_ms == 960 and unit_samples == 15360 and driving_samples == 23040, timing
        units = math.ceil(seconds * 1000 / unit_ms)
        second_start = math.ceil(35000 / unit_ms)
        started = time.perf_counter()

        async def read(source):
            nonlocal pending_max, skipped_max, received, processed
            async for raw in source:
                m = json.loads(raw)
                kind, response = m["type"], m.get("response_id")
                events.append({k: v for k, v in m.items() if k not in ("audio", "frames")})
                if kind == "error" or kind == "asr.status" and m["state"] == "error":
                    raise RuntimeError(m)
                if kind in ("input.received", "input.processed"):
                    pending_max = max(pending_max, m["pending_inputs"])
                    skipped_max = max(skipped_max, m["skipped_silence_inputs"])
                    received, processed = m["received_inputs"], m["processed_inputs"]
                if kind == "input.processed":
                    processing.append(m["processing_ms"])
                    brain_times.append((m["processing_started"], m["processing_finished"]))
                elif kind == "transcript":
                    transcripts.append(m)
                    print("ASR:", m["text"], "span", m["start_ms"], m["end_ms"], "latency_ms", m["asr_ms"], flush=True)
                elif kind == "text":
                    assert m["generated_at_ms"] >= ready["session_started_ms"]
                    texts[response] = texts.get(response, "") + m["text"]
                elif kind == "brain.audio":
                    original.setdefault(response, []).append(decode_pcm(m["audio"]))
                elif kind == "media":
                    assert m["duration_samples"] == driving_samples, m["duration_samples"]
                    assert len(m["frames"]) == timing["frames_per_block"], len(m["frames"])
                    delivered.setdefault(response, []).append(decode_pcm(m["audio"]))
                    videos.setdefault(response, []).append(m)
                    tails[response] = tails.get(response, 0) + m.get("postroll_samples", 0)
                    rtf.append(m["rtf"])
                    avatar_times.append((m["inference_started"], m["inference_finished"]))
                elif kind == "idle":
                    avatar_times.append((m["inference_started"], m["inference_finished"]))
                elif kind == "response.end":
                    ended.add(response)
                    assert 480 * 24 <= tails.get(response, 0) < 480 * 24 + driving_samples, (response, tails)
                    print("Completed:", texts.get(response), "postroll_ms", tails[response] / 24, flush=True)

        readers = [asyncio.create_task(read(source)) for source in (ws, media_ws) if source is not None]
        try:
            for index in range(units):
                for reader in readers:
                    if reader.done():
                        reader.result()
                        raise RuntimeError("Reader disconnected")
                await asyncio.sleep(max(0, started + index * unit_ms / 1000 - time.perf_counter()))
                chunk = np.zeros(unit_samples, dtype=np.float32)
                # Two complete utterances with continuous capture between them.
                offset = index if index < second_start else index - second_start
                part = audio[offset * unit_samples:(offset + 1) * unit_samples]
                chunk[:len(part)] = part
                await ws.send(json.dumps({"type": "input", "audio": encode_pcm(chunk), "video_frames": [frame],
                                          "client_input_id": index + 1, "capture_start_ms": index * unit_ms}))
            for _ in range(100):
                if received == units and processed + skipped_max == units and len(ended) >= 2:
                    break
                for reader in readers:
                    if reader.done():
                        reader.result()
                await asyncio.sleep(.1)
            assert received == units and processed + skipped_max == units, (received, processed, skipped_max)
            assert len(ended) >= 2 and len(transcripts) >= 2, (texts, transcripts)
            assert any("颜色" in m["text"] for m in transcripts), transcripts
            assert any(m["start_ms"] >= 35000 for m in transcripts), transcripts
            for response in ended:
                np.testing.assert_array_equal(np.concatenate(original[response]), np.concatenate(delivered[response]))
            await ws.send(json.dumps({"type": "session.stop"}))
        finally:
            for reader in readers:
                reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
            if media_ws is not None:
                await media_ws.close()

    for bs, be in brain_times:
        for avs, ave in avatar_times:
            if min(be, ave) > max(bs, avs):
                overlaps.append((min(be, ave) - max(bs, avs)) * 1000)
    assert overlaps, "No overlapping brain/avatar service intervals"
    first = next(iter(videos))
    clips = videos[first]
    rendered_audio, frame_index = [], 0
    for clip in clips:
        pcm = decode_pcm(clip["audio"])
        rendered_audio.append(np.pad(pcm, (0, clip["duration_samples"] - len(pcm))))
        for frame in clip["frames"]:
            with (output / f"frame-{frame_index:05d}.jpg").open("xb") as f:
                f.write(base64.b64decode(frame))
            frame_index += 1
    with wave.open(str(output / "reply.wav"), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000)
        wav.writeframes((np.concatenate(rendered_audio) * 32767).astype("<i2").tobytes())
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-framerate", "25", "-i", str(output / "frame-%05d.jpg"),
                    "-i", str(output / "reply.wav"), "-c:v", "libx264", "-threads", "4", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(output / "reply-with-postroll.mp4")], check=True)
    result = {"passed": True, "capture_seconds": units * unit_ms / 1000,
              "gpus": health["gpus"], "parallel_voice": health["brain"].get("acceleration", {}).get("parallel_voice", False),
              "separate_media_channel": media_ws is not None,
              "input_pacing": f"{unit_ms} ms, independent of processing ACK", "timing": timing,
              "received": received, "processed": processed, "coalesced_silence": skipped_max,
              "max_pending_seconds": pending_max * unit_ms / 1000, "processing_median_ms": statistics.median(processing),
              "processing_max_ms": max(processing), "avatar_median_rtf": statistics.median(rtf),
              "overlapping_intervals": len(overlaps), "original_pcm_exact": True,
              "responses": texts, "postroll_samples": tails, "transcripts": transcripts, "output": str(output)}
    voice_metrics = [e["metrics"] for e in events if e["type"] == "brain.audio"]
    if voice_metrics:
        result["speech_stages_median_ms"] = {key: statistics.median(m[key] for m in voice_metrics if key in m)
                                             for key in ("wall_clock_ms", "cost_llm_ms", "cost_tts_ms", "cost_token2wav_ms", "tts_total_ms", "tts_queue_ms")
                                             if any(key in m for m in voice_metrics)}
        voice_times = [(m["tts_started"], m["tts_finished"]) for m in voice_metrics if "tts_started" in m]
        result["voice_avatar_overlapping_intervals"] = sum(min(ve, ae) > max(vs, ass)
                                                           for vs, ve in voice_times for ass, ae in avatar_times)
    for name, data in (("results.json", result), ("events.json", events)):
        with (output / name).open("x") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
