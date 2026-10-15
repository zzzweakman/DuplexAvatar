"""GPU 5: rewind a queued reply to the displayed frame and continue with silence."""
import asyncio
import base64
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
import wave

import numpy as np
from PIL import Image, ImageDraw
import websockets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from duplex_avatar.media import encode_pcm
from duplex_avatar.settings import AVATAR_URL


def images(clip):
    return [Image.open(io.BytesIO(base64.b64decode(frame))).convert("RGB") for frame in clip["frames"]]


async def main():
    output = ROOT / "runtime" / time.strftime("motion-%Y%m%d-%H%M%S")
    output.mkdir(exist_ok=False)
    with wave.open(str(ROOT / "runtime/smoke-audiovisual-avatar-20261015-082000/minicpm_reply.wav")) as wav:
        assert wav.getframerate() == 24000
        speech = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    url = AVATAR_URL.replace("http:", "ws:") + "/stream"
    session = uuid.uuid4().hex
    clips = []
    async with websockets.connect(url, compression=None, max_size=32 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "session.start", "session_id": session, "epoch": 0}))
        ready = json.loads(await ws.recv())
        assert ready["type"] == "session.ready", ready
        size, fps = ready["chunk_samples"], ready["fps"]

        async def render(sequence, epoch, samples):
            await ws.send(json.dumps({"type": "render", "session_id": session, "epoch": epoch,
                                      "sequence": sequence, "audio": encode_pcm(samples)}))
            clip = json.loads(await asyncio.wait_for(ws.recv(), 30))
            assert clip["type"] == "frames" and clip["epoch"] == epoch, clip.get("type")
            assert len(clip["frames"]) == ready["frames_per_chunk"]
            return clip

        for index in range(4):
            samples = speech[index * size:(index + 1) * size]
            clips.append(await render(index + 1, 0, np.pad(samples, (0, size - len(samples)))))
        anchor = {"sequence": 2, "frame_index": 7}
        started = time.perf_counter()
        await ws.send(json.dumps({"type": "interrupt", "epoch": 1, "anchor": anchor}))
        resumed = await render(5, 1, np.zeros(size, dtype=np.float32))
        resume_total_ms = (time.perf_counter() - started) * 1000
        assert resumed["motion_resume"]["anchor_matched"] and resumed["motion_resume"]["anchor"] == anchor
        assert resumed["motion_resume"]["motion_frames"] == 9
        continuation = await render(6, 1, np.zeros(size, dtype=np.float32))
        # Repeated interrupt can anchor to an earlier epoch still on the screen.
        await ws.send(json.dumps({"type": "interrupt", "epoch": 2, "anchor": anchor}))
        repeated = await render(7, 2, np.zeros(size, dtype=np.float32))
        assert repeated["motion_resume"]["anchor_matched"]
    await asyncio.sleep(.2)
    session = uuid.uuid4().hex
    async with websockets.connect(url, compression=None, max_size=32 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "session.start", "session_id": session, "epoch": 0}))
        assert json.loads(await ws.recv())["type"] == "session.ready"
        baseline = await render(1, 0, np.zeros(size, dtype=np.float32))

    before = images(clips[0]) + images(clips[1])[:8]
    after = images(resumed) + images(continuation)
    hard_reset = images(baseline)[0]
    delta = lambda a, b: float(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)).mean())
    result = {"passed": True, "gpu": 5, "anchor": anchor, "generated_ahead_blocks": 2,
              "motion_resume": resumed["motion_resume"], "resume_total_ms": round(resume_total_ms, 2),
              "repeated_interrupt_anchor_matched": True, "full_block_frames": len(after) // 2,
              "pixel_mae_previous_frame": delta(before[-2], before[-1]),
              "pixel_mae_resumed_boundary": delta(before[-1], after[0]),
              "pixel_mae_initial_pose_boundary": delta(before[-1], hard_reset),
              "output": str(output)}
    width, height = before[0].size
    grid = Image.new("RGB", (width * 3, (height + 30) * 2), "#202328")
    draw = ImageDraw.Draw(grid)
    for index, (label, frame) in enumerate((("Before interruption", before[-2]), ("Exact anchor", before[-1]),
                                          ("Resumed frame 0", after[0]), ("Resumed frame 3", after[3]),
                                          ("Resumed frame 12", after[12]), ("Initial-pose baseline", hard_reset))):
        x, y = index % 3 * width, index // 3 * (height + 30)
        grid.paste(frame, (x, y)); draw.text((x + 8, y + height + 8), label, fill="white")
    grid.save(output / "transition-contact-sheet.jpg")
    with wave.open(str(output / "interruption.wav"), "wb") as wav:
        cutoff = len(before) * 24000 // fps
        audio = np.pad(speech[:cutoff], (0, len(after) * 24000 // fps))
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000)
        wav.writeframes((audio * 32767).astype("<i2").tobytes())
    with (output / "ffmpeg.log").open("x") as log:
        encoder = subprocess.Popen(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning", "-n",
                                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                                    "-r", str(fps), "-i", "pipe:0", "-i", str(output / "interruption.wav"),
                                    "-c:v", "libx264", "-threads", "4", "-preset", "veryfast",
                                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
                                    str(output / "interruption.mp4")], stdin=subprocess.PIPE, stderr=log)
        for frame in before + after:
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        assert encoder.wait() == 0
    with (output / "results.json").open("x") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
