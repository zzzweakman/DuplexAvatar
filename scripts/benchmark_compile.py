"""Warm steady-state eager/compiled comparisons on the configured physical GPU."""
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
parser.add_argument("model", choices=["brain", "avatar"])
args = parser.parse_args()
from duplex_avatar.settings import CONFIG
from scripts.manage import role_devices
role_devices(args.model)
gpu = CONFIG["brain_gpu" if args.model == "brain" else "avatar_gpu"]
os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                  TORCHINDUCTOR_COMPILE_THREADS="4", TOKENIZERS_PARALLELISM="false",
                  TORCHINDUCTOR_CACHE_DIR=str(ROOT / "runtime/cache/inductor"),
                  TRITON_CACHE_DIR=str(ROOT / "runtime/cache/triton"))
mini = ROOT.parent / "MiniCPM-o-Demo"
os.environ.update(HF_HOME=str(mini / ".cache/huggingface"),
                  HF_MODULES_CACHE=str(mini / ".cache/huggingface/modules"),
                  NUMBA_CACHE_DIR=str(mini / ".cache/numba"))
output = ROOT / "runtime" / time.strftime(f"compile-{args.model}-%Y%m%d-%H%M%S")
output.mkdir(parents=True, exist_ok=False)
import numpy as np
import torch


def summarize(values):
    return {"median_ms": round(statistics.median(values), 3),
            "p95_ms": round(float(np.percentile(values, 95)), 3), "samples_ms": values}


if args.model == "brain":
    import logging
    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, str(mini))
    os.chdir(mini)
    from core.processors.unified import UnifiedProcessor
    import librosa
    ref = str(mini / "assets/ref_audio/ref_minicpm_signature.wav")
    processor = UnifiedProcessor(model_path=str(mini / "models/MiniCPM-o-4_5"),
                                 ref_audio_path=ref, compile=True,
                                 chat_vocoder="token2wav", attn_implementation="sdpa")
    model = processor.model
    audio, frames = model._extract_mp4_chunks(str(mini / "assets/samples/compile.mp4"), max_chunks=8)
    ref_audio, _ = librosa.load(ref, sr=16000, mono=True)
    measurements = {}
    # Force the same listening decisions and input lengths in both modes.
    for visual in (True, False):
        for compiled in (False, True):
            model.set_compile_enabled(compiled)
            values = []
            for repeat in range(3):
                torch.manual_seed(42)
                model.duplex.prepare(prefix_system_prompt="<|im_start|>system\nStreaming Omni Conversation.\n<|audio_start|>",
                                     suffix_system_prompt="<|audio_end|><|im_end|>",
                                     ref_audio=ref_audio, prompt_wav_path=ref)
                for i, chunk in enumerate(audio):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    model.duplex.streaming_prefill(audio_waveform=chunk,
                                                    frame_list=[frames[i]] if visual else None, max_slice_nums=1)
                    result = model.duplex.streaming_generate(force_listen_override=True)
                    model.duplex.finalize_unit()
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - started) * 1000
                    assert result.get("is_listen"), result
                    if repeat:
                        values.append(round(elapsed, 3))
                model.reset_session(reset_token2wav_cache=True)
            key = f"{'audiovisual' if visual else 'audio'}_{'compiled' if compiled else 'eager'}"
            measurements[key] = summarize(values)
            print(key, measurements[key], flush=True)
    targets = ["llm.model", "tts.model"]
    scope = "Identical forced-listening steps; prefill + generate + finalize; first round excluded. Does not measure speech generation speed."
else:
    from duplex_avatar.avatar_server import Engine
    engine = Engine()
    engine.load()
    pipeline = engine.pipeline
    bindings = [(pipeline, "model", pipeline.model, pipeline.model._orig_mod)]
    vae = pipeline.vae.model if pipeline.use_ltx else pipeline.vae
    for name in ("encode", "decode"):
        function = getattr(vae, name)
        bindings.append((vae, name, function, function._torchdynamo_orig_callable))
    measurements = {}
    for compiled in (False, True):
        for owner, name, optimized, eager in bindings:
            setattr(owner, name, optimized if compiled else eager)
        engine.reset("benchmark", 0)
        values = []
        for index in range(14):
            # Silence also exercises the requested listening animation path.
            frames, elapsed = engine.render(np.zeros(engine.chunk_samples, dtype=np.float32))
            assert np.isfinite(frames).all() and len(frames) == engine.frames
            if index >= 2:
                values.append(round(elapsed, 3))
        measurements["compiled" if compiled else "eager"] = summarize(values)
    targets = ["diffusion model", "VAE encode", "VAE decode"]
    scope = "0.96s silent driving chunks including resampling, audio encoder, diffusion, VAE, CPU transfer; excludes JPEG/network; first 2 chunks excluded."

result = {"model": args.model, "gpu": gpu, "torch": torch.__version__, "targets": targets,
          "scope": scope, "measurements": measurements}
with (output / "results.json").open("x") as file:
    json.dump(result, file, indent=2, ensure_ascii=False)
print("RESULT", output / "results.json", flush=True)
