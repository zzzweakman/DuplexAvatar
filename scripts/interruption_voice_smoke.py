"""Interrupt live speech twice, then verify fresh TTS state and exact delivered PCM."""
import argparse
import asyncio
import base64
import io
import json
from pathlib import Path
import re
import ssl
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
from duplex_avatar.settings import ASR_URL, CONFIG


async def main(audio_output=False):
    mode = "audio" if audio_output else "avatar"
    output = ROOT / "runtime" / time.strftime(f"interrupt-voice-{mode}-%Y%m%d-%H%M%S")
    output.mkdir(exist_ok=False)
    manifest = sorted((ROOT / "runtime").glob("run-*/processes.json"))[-1]
    brain_log = Path(next(r["log"] for r in json.loads(manifest.read_text()) if r["name"] == "brain"))
    log_offset = brain_log.stat().st_size
    with wave.open(str(ROOT / "runtime/fixtures/question.wav")) as wav:
        question = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").astype(np.float32) / 32768
        question = resample_poly(question, 16000, wav.getframerate()).astype(np.float32)
    frames = []
    for color in ((237, 41, 41), (25, 55, 230), (20, 185, 55)):
        image = io.BytesIO()
        Image.new("RGB", (224, 224), color).save(image, format="JPEG")
        frames.append(base64.b64encode(image.getvalue()).decode())
    events, texts, original, delivered, clips, first_voice = [], {}, {}, {}, {}, {}
    audible = {epoch: asyncio.Event() for epoch in (0, 1, 3)}
    completed = asyncio.Event()
    acknowledgements = asyncio.Queue()
    phase = epoch = sent = phase_start = 0
    latest_anchor = None
    stopping = False
    errors, finished_responses = [], set()
    url = f"https://127.0.0.1:{CONFIG['gateway_port']}"
    started = time.perf_counter()

    async with httpx.AsyncClient(verify=False, trust_env=False, timeout=45) as http:
        health = (await http.get(url + "/health")).json()
        assert health["status"] == "ready" and not health["busy"], health
        async with websockets.connect(url.replace("https:", "wss:") + "/ws",
                                      ssl=ssl._create_unverified_context(), compression=None,
                                      max_size=32 * 1024 * 1024) as ws:
            await ws.send(json.dumps({"type": "session.start", "diagnostics": True,
                                     "input_mode": "audiovisual", "output_mode": mode,
                                     "media_channel": "separate", "video_codec": "h264",
                                     "system_prompt": "你是中文助手。每次回答先说当前图片的颜色，然后用四到五句话描述这种颜色。只描述当前画面，不提之前的画面。用户插话时立即停止，听完新问题再回答。"}))
            ready = json.loads(await ws.recv())
            assert ready["type"] == "session.ready", ready
            unit = ready["timing"]["input_samples"]
            unit_ms = ready["timing"]["decision_ms"]
            media = None
            if ready.get("media_channel"):
                media = await websockets.connect(url.replace("https:", "wss:") + "/media",
                                                 ssl=ssl._create_unverified_context(), compression=None,
                                                 max_size=32 * 1024 * 1024)
                await media.send(json.dumps({"session_id": ready["session_id"], "token": ready["media_channel"]["token"]}))
                assert json.loads(await media.recv())["type"] == "media.ready"

            async def receive(source):
                nonlocal latest_anchor
                async for raw in source:
                    message = json.loads(raw)
                    kind = message["type"]
                    events.append({k: v for k, v in message.items() if k not in ("audio", "frames", "video")})
                    if kind == "error" and not stopping:
                        errors.append(message)
                        raise RuntimeError(message)
                    if kind == "interrupted":
                        acknowledgements.put_nowait(message["epoch"])
                    if message.get("epoch", epoch) != epoch:
                        continue
                    response = message.get("response_id")
                    if kind == "text":
                        texts[response] = texts.get(response, "") + message["text"]
                    elif kind == "brain.audio":
                        original.setdefault(response, []).append(decode_pcm(message["audio"]))
                        metrics = message["metrics"]
                        if epoch not in first_voice:
                            first_voice[epoch] = metrics
                            if epoch:
                                assert metrics["tts_fresh_cache"] is True, metrics
                                assert metrics["tts_text_position_before"] == 0, metrics
                                assert metrics["token2wav_buffer_before"] == 3, metrics
                                assert metrics["tts_generation"] > first_voice[0]["tts_generation"], metrics
                            print("First voice:", epoch, json.dumps({k: metrics[k] for k in (
                                "tts_generation", "tts_fresh_cache", "tts_text_position_before", "token2wav_buffer_before")}), flush=True)
                        if epoch in audible and np.max(np.abs(decode_pcm(message["audio"]))) > .002:
                            audible[epoch].set()
                    elif kind in ("media", "idle"):
                        if not audio_output:
                            assert message["video"]["frame_count"] == 24, message["video"]["frame_count"]
                            latest_anchor = {"sequence": message["sequence"], "frame_index": 7}
                        if kind == "media":
                            delivered.setdefault(response, []).append(decode_pcm(message["audio"]))
                            clips.setdefault(response, []).append(message)
                    elif kind == "response.end":
                        finished_responses.add(response)
                        if epoch == 3 and original.get(response):
                            completed.set()

            async def capture():
                nonlocal sent
                capture_started = time.perf_counter()
                while not stopping:
                    await asyncio.sleep(max(0, capture_started + sent * unit_ms / 1000 - time.perf_counter()))
                    block = np.zeros(unit, dtype=np.float32)
                    offset = (sent - phase_start) * unit
                    part = question[offset:offset + unit]
                    block[:len(part)] = part
                    index = sent
                    sent += 1
                    await ws.send(json.dumps({"type": "input", "client_input_id": sent,
                                              "capture_start_ms": index * unit_ms,
                                              "audio": encode_pcm(block), "video_frames": [frames[phase]]}))

            tasks = [asyncio.create_task(receive(source)) for source in (ws, media) if source is not None]
            tasks.append(asyncio.create_task(capture()))

            async def wait_event(event, timeout=70):
                waiter = asyncio.create_task(event.wait())
                try:
                    done, _ = await asyncio.wait([waiter, *tasks], timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                    for task in tasks:
                        if task in done:
                            task.result()
                            raise RuntimeError("Pipeline stopped before the check completed")
                    assert waiter in done, f"Timed out at epoch {epoch}: {texts}"
                finally:
                    waiter.cancel()
                    await asyncio.gather(waiter, return_exceptions=True)

            async def interrupt():
                nonlocal epoch
                epoch += 1
                await ws.send(json.dumps({"type": "interrupt", "anchor": latest_anchor}))
                acknowledged = await asyncio.wait_for(acknowledgements.get(), 15)
                assert acknowledged == epoch, (acknowledged, epoch)
                print("Interrupted:", epoch, "at", round(time.perf_counter() - started, 2), flush=True)

            try:
                await wait_event(audible[0])
                await interrupt()
                phase, phase_start = 1, sent
                await wait_event(audible[1])
                await interrupt()
                await asyncio.sleep(.06)
                await interrupt()
                phase, phase_start = 2, sent
                await wait_event(audible[3])
                await wait_event(completed)
                stopping = True
                tasks[-1].cancel()
                await asyncio.gather(tasks[-1], return_exceptions=True)
                await ws.send(json.dumps({"type": "session.stop"}))
                await asyncio.wait_for(ws.wait_closed(), 20)
            finally:
                stopping = True
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if media is not None:
                    await media.close()

        final_response = next(response for response in finished_responses
                              if clips.get(response) and clips[response][0]["epoch"] == 3)
        speech = np.concatenate(original[final_response])
        np.testing.assert_array_equal(speech, np.concatenate(delivered[final_response]))
        final_clips = clips[final_response]
        first_generation = [first_voice[e]["tts_generation"] for e in (0, 1, 3)]
        assert first_generation[0] < first_generation[1] < first_generation[2], first_generation
        new_log = brain_log.read_bytes()[log_offset:].decode(errors="replace")
        resets = [{"generation": int(m[1]), "discarded_tts_position": int(m[2]),
                   "discarded_buffer_tokens": int(m[3]), "remaining_prefix_tokens": int(m[4])}
                  for m in re.finditer(r"voice_reset generation=(\d+) discarded_tts_position=(\d+) discarded_buffer_tokens=(\d+) remaining_prefix_tokens=(\d+)", new_log)]
        assert len(resets) >= 2 and sum(r["discarded_tts_position"] > 0 for r in resets) >= 2, resets
        assert all(r["remaining_prefix_tokens"] == 3 for r in resets), resets

        def save_wave(name, samples):
            with wave.open(str(output / name), "wb") as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000)
                wav.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())

        save_wave("new-reply.wav", speech)
        for index, (response, chunks) in enumerate(original.items()):
            save_wave(f"response-{index}.wav", np.concatenate(chunks))
        if not audio_output:
            with (output / "new-reply.h264").open("xb") as video:
                for clip in final_clips:
                    for chunk in clip["video"]["chunks"]:
                        video.write(base64.b64decode(chunk))
            with_postroll = np.concatenate([np.pad(decode_pcm(clip["audio"]),
                (0, clip["duration_samples"] - clip["valid_samples"])) for clip in final_clips])
            save_wave("new-reply-with-postroll.wav", with_postroll)
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-r", "25",
                            "-i", str(output / "new-reply.h264"), "-i", str(output / "new-reply-with-postroll.wav"),
                            "-c:v", "copy", "-c:a", "aac", "-shortest", str(output / "new-reply.mp4")], check=True)
        pcm16 = (np.clip(resample_poly(speech, 2, 3), -1, 1) * 32767).astype("<i2").tobytes()
        transcript = await http.post(ASR_URL + "/transcribe", content=pcm16,
                                     params={"start_ms": 0, "language": "zh"},
                                     headers={"Content-Type": "application/octet-stream"})
        transcript.raise_for_status()
        result = {"passed": True, "output_mode": mode, "input_blocks": sent,
                  "interrupt_epochs": [1, 2, 3], "voice_generations": first_generation,
                  "voice_resets": resets, "first_voice_metrics": first_voice,
                  "final_reply_text": texts.get(final_response), "final_reply_asr": transcript.json().get("text"),
                  "original_pcm_exact": True, "video_frames_per_block": None if audio_output else 24,
                  "final_reply_speech_seconds": len(speech) / 24000,
                  "seconds": round(time.perf_counter() - started, 2), "errors": errors, "output": str(output)}
        for name, data in (("results.json", result), ("events.json", events)):
            with (output / name).open("x") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-output", action="store_true")
    asyncio.run(main(parser.parse_args().audio_output))
