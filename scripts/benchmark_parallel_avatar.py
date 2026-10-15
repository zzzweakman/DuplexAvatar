"""Compare native FlashHead sequence parallelism with identical warmed inputs."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--chunks", type=int, default=18)
args = parser.parse_args()
devices = [int(item) for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]
from duplex_avatar.settings import CONFIG
assert devices and set(devices) <= set(CONFIG["allowed_gpus"]), devices
os.environ.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
                  TORCHINDUCTOR_COMPILE_THREADS="4", NCCL_MIN_NCHANNELS="4",
                  TORCHINDUCTOR_CACHE_DIR=str(ROOT / "runtime/cache/inductor"),
                  TRITON_CACHE_DIR=str(ROOT / "runtime/cache/triton"))
import numpy as np
import torch
from duplex_avatar.avatar_server import Engine

world = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))
engine = Engine()
engine.load(world_size=world)
engine.reset("benchmark-parallel", 0)
values = []
for i in range(args.chunks):
    chunk = (.05 * np.sin(np.arange(engine.chunk_samples) * 2 * np.pi * 180 / 24000)).astype(np.float32)
    if world > 1:
        torch.distributed.barrier()
    torch.cuda.synchronize()
    started = time.perf_counter()
    frames, _ = engine.render(chunk, i)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) * 1000
    assert len(frames) == 24 and frames.shape[1:] == (512, 512, 3)
    if i >= 4:
        values.append(round(elapsed, 3))
if rank == 0:
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {"passed": True, "gpus": devices, "world_size": world, "compile": True,
              "median_ms": round(statistics.median(values), 3),
              "p95_ms": round(float(np.percentile(values, 95)), 3), "samples_ms": values,
              "scope": "960 ms native Lite blocks with CPU motion checkpoint; no JPEG or network"}
    with path.open("x") as file:
        json.dump(result, file, indent=2)
    print("RESULT", json.dumps(result), flush=True)
if world > 1:
    torch.distributed.destroy_process_group()
