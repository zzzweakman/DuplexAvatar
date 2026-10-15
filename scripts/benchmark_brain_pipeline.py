"""Replay identical input as fast as ACKs allow, measuring catch-up throughput.

Model sampling remains enabled: token counts can differ. This measures service
intervals, not CUDA kernels, and is not a real-time first-response benchmark.
"""
import argparse
import asyncio
import base64
import io
import json
from pathlib import Path
import statistics
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
from duplex_avatar.media import encode_pcm
from duplex_avatar.settings import CONFIG


async def main(port, units):
    url = f"http://127.0.0.1:{port}"
    async with httpx.AsyncClient(trust_env=False) as client:
        health = (await client.get(url + "/health")).json()
        acceleration = (await client.get(url + "/acceleration")).json()
    assert health["status"] == "ready" and not health.get("active_session_id"), health
    assert {acceleration["gpu"], acceleration.get("tts_gpu", acceleration["gpu"])} <= set(CONFIG["allowed_gpus"])
    output = ROOT / "runtime" / time.strftime(f"catchup-{port}-%Y%m%d-%H%M%S")
    output.mkdir(exist_ok=False)
    with wave.open(str(ROOT / "runtime/fixtures/question.wav")) as wav:
        samples = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").astype(np.float32) / 32768
        samples = resample_poly(samples, 16000, wav.getframerate()).astype(np.float32)
    image = io.BytesIO()
    Image.new("RGB", (224, 224), (237, 41, 41)).save(image, format="JPEG")
    frame = base64.b64encode(image.getvalue()).decode()
    events, sent = [], {}
    ack, drained = asyncio.Queue(), asyncio.Event()
    async with websockets.connect(url.replace("http:", "ws:") + "/backend", max_size=16 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "session.init", "payload": {"mode": "full_duplex",
            "system_prompt": "你是一个中文音视频助手。认真听用户说话并观察画面，用一句简短的中文回答。",
            "config": {"chunk_ms": 960, "force_listen_count": 1, "length_penalty": 1.0}}}))
        created = json.loads(await ws.recv())
        assert created["type"] == "session.created", created

        async def read():
            async for raw in ws:
                event = json.loads(raw)
                event["received"] = time.perf_counter()
                event["audio_samples"] = len(base64.b64decode(event.pop("audio", ""))) // 4
                events.append(event)
                if event["type"] == "session.closed":
                    raise RuntimeError(event)
                if event["type"] == "input.processed":
                    ack.put_nowait(event)
                if event.get("input_id") == f"0:{units}" and event.get("kind") == "listen":
                    drained.set()

        reader = asyncio.create_task(read())
        started = time.perf_counter()
        try:
            for index in range(units):
                if reader.done():
                    reader.result()
                chunk = np.zeros(15360, dtype=np.float32)
                part = samples[(index % 16) * 15360:((index % 16) + 1) * 15360]
                chunk[:len(part)] = part
                key = f"0:{index + 1}"
                sent[key] = time.perf_counter()
                await ws.send(json.dumps({"type": "input.append", "input": {
                    "input_id": key, "audio": encode_pcm(chunk), "video_frames": [frame],
                    "force_listen": index == units - 1}}))
                await asyncio.wait_for(ack.get(), 180)
            await asyncio.wait_for(drained.wait(), 30)
            elapsed = time.perf_counter() - started
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            async with httpx.AsyncClient(trust_env=False, timeout=60) as client:
                await client.post(url + f"/sessions/{created['session_id']}/close", json={"reason": "benchmark_finished"})
    voices = [event for event in events if event.get("kind") == "audio"]
    spoken = {event["input_id"] for event in voices}
    steps = [event for event in events if event["type"] == "input.processed"]
    assert len(steps) == units and voices
    measure = lambda values: {"median": round(statistics.median(values), 2), "max": round(max(values), 2)}
    voice_intervals = [(e["metrics"]["tts_started"], e["metrics"]["tts_finished"]) for e in voices if "tts_started" in e["metrics"]]
    overlaps = [(min(ve, step["processing_finished"]) - max(vs, step["processing_started"])) * 1000
                for vs, ve in voice_intervals for step in steps
                if min(ve, step["processing_finished"]) > max(vs, step["processing_started"])]
    result = {"passed": True, "acceleration": acceleration, "input_units": units, "input_seconds": units * .960,
              "elapsed_seconds": round(elapsed, 3), "speaking_steps": len(spoken),
              "scope": "ACK-paced catch-up with two recorded utterances; stochastic replies can differ",
              "speech_decision_ack_ms": measure([(e["processing_finished"] - e["processing_started"]) * 1000 for e in steps if e["input_id"] in spoken]),
              "speech_audio_ready_ms": measure([(e["received"] - sent[e["input_id"]]) * 1000 for e in voices]),
              "tts_token_count": sum(e["metrics"].get("n_tts_tokens", 0) for e in voices),
              "voice_input_overlapping_intervals": len(overlaps), "voice_input_overlap_ms": round(sum(overlaps), 2),
              "text": "".join(e.get("text", "") for e in events),
              "voice_stages_median_ms": {key: round(statistics.median(e["metrics"][key] for e in voices if key in e["metrics"]), 2)
                                        for key in ("cost_llm_ms", "cost_tts_ms", "cost_token2wav_ms", "tts_queue_ms")
                                        if any(key in e["metrics"] for e in voices)}}
    for name, data in (("results.json", result), ("events.json", events)):
        with (output / name).open("x") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--units", type=int, default=32)
    args = parser.parse_args()
    asyncio.run(main(args.port, args.units))
