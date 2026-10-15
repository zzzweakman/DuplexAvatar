"""Real streaming vision probe; save inputs, model evidence and original speech."""
import argparse
import asyncio
import base64
import io
import json
from pathlib import Path
import re
import ssl
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


async def main(args):
    output = ROOT / "runtime" / time.strftime("vision-probe-%Y%m%d-%H%M%S")
    output.mkdir(exist_ok=False)
    with wave.open(str(args.question)) as wav:
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").astype(np.float32) / 32768
        question = resample_poly(pcm, 16000, wav.getframerate()).astype(np.float32)
    image = Image.open(args.frame).convert("RGB") if args.frame else Image.new("RGB", (448, 336), "red")
    image.thumbnail((640, 640))
    data = io.BytesIO()
    image.save(data, format="JPEG", quality=85)
    with (output / "input.jpg").open("xb") as handle:
        handle.write(data.getvalue())
    frame = base64.b64encode(data.getvalue()).decode()
    events, texts, speech, metrics = [], {}, {}, []
    done = asyncio.Event()
    url = "https://127.0.0.1:8443"
    async with httpx.AsyncClient(verify=False, trust_env=False, timeout=30) as http:
        health = (await http.get(url + "/health")).json()
        assert health["status"] == "ready" and not health["busy"], health
        async with websockets.connect(url.replace("https:", "wss:") + "/ws",
                                      ssl=ssl._create_unverified_context(), max_size=32 * 1024 * 1024) as ws:
            options = {"type": "session.start", "input_mode": "audiovisual", "output_mode": "audio", "diagnostics": True}
            if args.prompt:
                options["system_prompt"] = args.prompt
            await ws.send(json.dumps(options))
            ready = json.loads(await ws.recv())
            assert ready["type"] == "session.ready", ready
            unit, duration = ready["timing"]["input_samples"], ready["timing"]["decision_ms"] / 1000

            async def receive():
                async for raw in ws:
                    event = json.loads(raw)
                    events.append({k: v for k, v in event.items() if k not in ("audio", "frames", "video")})
                    kind, response = event["type"], event.get("response_id")
                    if kind == "error":
                        raise RuntimeError(event)
                    if event.get("metrics"):
                        metrics.append(event["metrics"])
                    if kind == "text":
                        texts[response] = texts.get(response, "") + event["text"]
                        print(event["text"], end="", flush=True)
                    elif kind == "media":
                        speech.setdefault(response, []).append(decode_pcm(event["audio"]))
                    elif kind == "response.end" and texts.get(response) and speech.get(response):
                        done.set()

            receiver = asyncio.create_task(receive())
            started = time.perf_counter()
            try:
                for index in range(45):
                    await asyncio.sleep(max(0, started + index * duration - time.perf_counter()))
                    if receiver.done():
                        receiver.result()
                        raise RuntimeError("Model disconnected")
                    if done.is_set():
                        break
                    chunk = np.zeros(unit, dtype=np.float32)
                    offset = index * unit
                    part = question[offset:offset + unit]
                    chunk[:len(part)] = part
                    await ws.send(json.dumps({"type": "input", "audio": encode_pcm(chunk), "video_frames": [frame],
                                              "client_input_id": index + 1, "capture_start_ms": index * duration * 1000}))
                await ws.send(json.dumps({"type": "session.stop"}))
            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
        for _ in range(100):
            health = (await http.get(url + "/health")).json()
            if not health["busy"] and not health["brain"].get("active_session_id"):
                break
            await asyncio.sleep(.1)
    text = "".join(texts.values())
    result = {"session_id": ready["session_id"], "text": text, "question": str(args.question),
              "input_size": image.size, "vision_slices": sorted({m.get("vision_slices", 0) for m in metrics}),
              "vision_tokens": sorted({m.get("vision_tokens", 0) for m in metrics}), "events": events,
              "passed": done.is_set() and (not args.expect or re.search(args.expect, text) is not None)}
    with (output / "results.json").open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    if speech:
        samples = np.concatenate([part for parts in speech.values() for part in parts])
        with (output / "reply.wav").open("xb") as handle, wave.open(handle, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    print("\n" + str(output), json.dumps({k: v for k, v in result.items() if k != "events"}, ensure_ascii=False), flush=True)
    assert result["passed"], result["text"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=Path, default=ROOT / "runtime/fixtures/question.wav")
    parser.add_argument("--frame", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--expect")
    asyncio.run(main(parser.parse_args()))
