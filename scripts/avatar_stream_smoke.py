"""Exercise the GPU 5 streaming worker: silence, pipelining, stale cancellation."""
import asyncio
import base64
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np
from PIL import Image
import websockets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from duplex_avatar.settings import AVATAR_URL
from duplex_avatar.media import encode_pcm


async def main():
    output = ROOT / "runtime" / time.strftime("avatar-stream-%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    session = uuid.uuid4().hex
    clips = []
    async with websockets.connect(AVATAR_URL.replace("http:", "ws:") + "/stream", max_size=32 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "session.start", "session_id": session, "epoch": 0}))
        ready = json.loads(await ws.recv())
        assert ready["type"] == "session.ready", ready
        silence = encode_pcm(np.zeros(ready["chunk_samples"], dtype=np.float32))

        async def render(sequence, epoch=0):
            await ws.send(json.dumps({"type": "render", "session_id": session, "epoch": epoch,
                                      "sequence": sequence, "audio": silence}))

        for sequence in (1, 3):
            await render(sequence)
            await render(sequence + 1)
            for expected in (sequence, sequence + 1):
                clip = json.loads(await asyncio.wait_for(ws.recv(), 20))
                assert clip["type"] == "frames" and clip["sequence"] == expected, clip.get("type")
                clips.append(clip)
        await ws.send(json.dumps({"type": "interrupt", "epoch": 1}))
        await render(5, epoch=0)
        await render(6, epoch=1)
        stale = json.loads(await asyncio.wait_for(ws.recv(), 20))
        fresh = json.loads(await asyncio.wait_for(ws.recv(), 20))
        assert stale["type"] == "dropped" and stale["sequence"] == 5, stale
        assert fresh["type"] == "frames" and fresh["epoch"] == 1, fresh.get("type")

    frames = [Image.open(io.BytesIO(base64.b64decode(frame))).convert("RGB") for clip in clips for frame in clip["frames"]]
    difference = float(np.abs(np.asarray(frames[0], dtype=float) - np.asarray(frames[-1], dtype=float)).mean())
    assert difference > 0.1, "Silence did not produce visible frame changes"
    frames[0].save(output / "first.jpg")
    frames[-1].save(output / "last.jpg")
    with (output / "ffmpeg.log").open("x") as log:
        encoder = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-n", "-f", "rawvideo",
                                    "-pix_fmt", "rgb24", "-s", "512x512", "-r", "25", "-i", "pipe:0",
                                    "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                                    "-movflags", "+faststart", str(output / "listening.mp4")], stdin=subprocess.PIPE, stderr=log)
        for frame in frames:
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        assert encoder.wait() == 0
    result = {"passed": True, "gpu": 5, "frames": len(frames), "seconds": len(frames) / 25,
              "silent_input": True, "motion_mean_absolute_difference": difference,
              "two_inflight_requests": True, "stale_epoch_discarded": True,
              "new_epoch_rendered": True, "output": str(output / "listening.mp4")}
    with (output / "results.json").open("x") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))


asyncio.run(main())
