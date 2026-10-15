"""Decode a real browser continuity capture and measure motion across A2V blocks."""
import argparse
import base64
import json
from pathlib import Path
import statistics
import subprocess
import wave

import av
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    targets = [args.capture / name for name in ("block-boundaries.jpg", "speech.wav", "speech.mp4", "pixel-analysis.json")]
    if any(target.exists() for target in targets):
        raise FileExistsError("Analysis outputs already exist; use a new capture directory")
    result = json.loads((args.capture / "results.json").read_text())
    blocks = [e for e in result["events"] if e["type"] == "media"]
    clips, audio, within, boundaries = [], [], [], []
    for block in blocks:
        decoder = av.CodecContext.create("h264", "r")
        decoded = []
        for chunk in block["video"]["chunks"]:
            decoded.extend(decoder.decode(av.Packet(base64.b64decode(chunk))))
        decoded.extend(decoder.decode(None))
        frames = np.stack([frame.to_ndarray(format="rgb24") for frame in decoded])
        assert len(frames) == 24
        delta = np.abs(np.diff(frames.astype(np.float32), axis=0)).mean(axis=(1, 2, 3))
        within.extend(delta.tolist())
        if clips and block["response_id"] == blocks[len(clips) - 1]["response_id"]:
            boundaries.append(float(np.abs(frames[0].astype(np.float32) - clips[-1][-1]).mean()))
        clips.append(frames)
        pcm = np.frombuffer(base64.b64decode(block["audio"]), dtype="<f4")
        audio.append(np.pad(pcm, (0, block["duration_samples"] - len(pcm))))
    report = {"blocks": len(blocks), "within_frame_mae_median": statistics.median(within),
              "boundary_frame_mae": boundaries, "boundary_frame_mae_median": statistics.median(boundaries),
              "boundary_to_within_ratio": statistics.median(boundaries) / statistics.median(within)}
    width, height = clips[0].shape[2], clips[0].shape[1]
    rows = min(len(clips) - 1, 5)
    grid = Image.new("RGB", (width * 2, (height + 32) * rows), "#202328")
    draw = ImageDraw.Draw(grid)
    for i in range(rows):
        y = i * (height + 32)
        grid.paste(Image.fromarray(clips[i][-1]), (0, y)); grid.paste(Image.fromarray(clips[i + 1][0]), (width, y))
        draw.text((8, y + height + 8), f'Block {blocks[i]["sequence"]} frame 23 -> block {blocks[i+1]["sequence"]} frame 0; MAE {boundaries[i]:.3f}', fill="white")
    with (args.capture / "block-boundaries.jpg").open("xb") as file:
        grid.save(file, format="JPEG")
    with (args.capture / "speech.wav").open("xb") as file:
        with wave.open(file, "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000)
            wav.writeframes((np.clip(np.concatenate(audio), -1, 1) * 32767).astype("<i2").tobytes())
    encoder = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", "25", "-i", "pipe:0", "-i", str(args.capture / "speech.wav"),
        "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac",
        str(args.capture / "speech.mp4")], stdin=subprocess.PIPE)
    for clip in clips: encoder.stdin.write(clip.tobytes())
    encoder.stdin.close(); assert encoder.wait() == 0
    with (args.capture / "pixel-analysis.json").open("x") as file: json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
