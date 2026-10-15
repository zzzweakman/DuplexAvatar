"""Compare both CPU encoders on the same real 24-frame block; no GPU is used."""
import argparse
import base64
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from duplex_avatar.video_transport import av, encode_h264


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path, help="Directory containing transition-*.jpg browser frames")
    args = parser.parse_args()
    paths = sorted(args.frames.glob("transition-*.jpg"))[:24]
    assert len(paths) == 24, "Expected at least 24 real frames"
    frames = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])
    output = ROOT / "runtime" / time.strftime("video-encoder-%Y%m%d-%H%M%S")
    output.mkdir(exist_ok=False)
    results = {"frames": 24, "fps": 25, "width": frames.shape[2], "height": frames.shape[1],
               "source": str(args.frames.resolve()), "pyav_version": av.__version__, "gpu_used": False}
    for backend in ("ffmpeg", "pyav"):
        timings, sizes = [], []
        for _ in range(5):
            started = time.perf_counter()
            block = encode_h264(frames, 25, backend=backend)
            timings.append(round((time.perf_counter() - started) * 1000, 2))
            data = [base64.b64decode(chunk) for chunk in block["chunks"]]
            sizes.append(sum(map(len, data)))
            decoder = av.CodecContext.create("h264", "r")
            decoded = []
            for chunk in data:
                decoded.extend(decoder.decode(av.Packet(chunk)))
            decoded.extend(decoder.decode(None))
            assert len(decoded) == 24 and decoded[0].key_frame
        results[backend] = {"ms": timings, "median_ms": statistics.median(timings),
                            "bytes": sizes, "independent_decode_frames": 24}
    results["median_speedup"] = round(results["ffmpeg"]["median_ms"] / results["pyav"]["median_ms"], 2)
    with (output / "results.json").open("x") as file:
        json.dump(results, file, indent=2)
    print(json.dumps({**results, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
