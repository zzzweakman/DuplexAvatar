"""Reproduce the tested Linux x86_64 / Python 3.10 deployment without sudo."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
MANIFEST = json.loads((ROOT / "deploy/sources.json").read_text())
ENVIRONMENTS = {
    "brain": WORKSPACE / "MiniCPM-o-Demo/.venv/base",
    "avatar": WORKSPACE / "SoulX-FlashHead/.venv",
    "asr": WORKSPACE / "Omni-Interaction-Agent/.venv",
}


def run(*args, **kwargs):
    print("+", " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def sources():
    for source in MANIFEST["repositories"]:
        target = WORKSPACE / source["directory"]
        if target.exists():
            head = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
            if head != source["revision"]:
                raise RuntimeError(f"{target}: expected {source['revision']}, found {head}; existing checkout was not changed")
            print("Verified", source["directory"], head)
            continue
        run("git", "clone", "--no-checkout", source["url"], target)
        run("git", "-C", target, "checkout", "--detach", source["revision"])


def envs(python):
    if not shutil.which("uv"):
        raise RuntimeError("Install uv first: python3 -m pip install --user uv")
    for role, path in ENVIRONMENTS.items():
        executable = path / "bin/python"
        if not executable.exists():
            run("uv", "venv", "--python", python, path)
        version = subprocess.check_output([str(executable), "-c", "import sys;print('.'.join(map(str,sys.version_info[:2])))"], text=True).strip()
        if version != "3.10":
            raise RuntimeError(f"{path} needs Python 3.10 for the pinned FlashAttention wheel")
        args = ["uv", "pip", "install", "--python", executable,
                "--index", "https://download.pytorch.org/whl/cu128",
                "--index-strategy", "unsafe-best-match", "-r", ROOT / f"deploy/requirements-{role}.txt"]
        override = ROOT / f"deploy/overrides-{role}.txt"
        if override.exists():
            args += ["--override", override]
        run(*args)
        if role == "avatar":
            entry = MANIFEST["flash_attention"]
            cache = ROOT / "runtime/downloads"
            cache.mkdir(parents=True, exist_ok=True)
            wheel = cache / entry["url"].rsplit("/", 1)[1]
            if not wheel.exists():
                with urllib.request.urlopen(entry["url"], timeout=120) as response, wheel.open("xb") as out:
                    shutil.copyfileobj(response, out)
            if hashlib.sha256(wheel.read_bytes()).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Checksum mismatch: {wheel}; existing file retained for inspection")
            run("uv", "pip", "install", "--python", executable, "--no-deps", wheel)


def models():
    python = ENVIRONMENTS["brain"] / "bin/python"
    code = """import json,sys
from huggingface_hub import snapshot_download
entry=json.loads(sys.argv[1])
snapshot_download(repo_id=entry['repo_id'], revision=entry['revision'],
                  local_dir=sys.argv[2], ignore_patterns=entry['ignore_patterns'], max_workers=4)
"""
    for entry in MANIFEST["models"]:
        run(python, "-c", code, json.dumps(entry), WORKSPACE / entry["directory"])


def tls():
    sys.path.insert(0, str(ROOT))
    from duplex_avatar.settings import tls_paths
    cert, key = tls_paths()
    if cert.exists() and key.exists():
        print("Existing TLS files retained:", cert, key)
        return
    if cert.exists() or key.exists():
        raise RuntimeError("Only one TLS file exists; refusing to overwrite it. Configure a fresh pair of paths.")
    cert.parent.mkdir(parents=True, exist_ok=True)
    key.parent.mkdir(parents=True, exist_ok=True)
    # Reserve both names, so an existing certificate/key is never overwritten.
    old = os.umask(0o077)
    try:
        for path in (cert, key):
            with path.open("xb"):
                pass
        run("openssl", "req", "-x509", "-newkey", "rsa:2048", "-sha256", "-nodes",
            "-days", "365", "-keyout", key, "-out", cert,
            "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1")
        key.chmod(0o600)
    finally:
        os.umask(old)


def fixtures():
    source = ROOT / "assets/fixtures/question.wav"
    destination = ROOT / "runtime/fixtures/question.wav"
    if destination.exists():
        print("Existing test question retained")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as inp, destination.open("xb") as out:
        shutil.copyfileobj(inp, out)


def doctor():
    sys.path.insert(0, str(ROOT))
    from duplex_avatar.settings import CONFIG, tls_paths, image_path
    from scripts.manage import role_devices
    issues = []
    for binary in ("git", "ffmpeg", "openssl", "nvidia-smi"):
        if not shutil.which(binary):
            issues.append(f"Missing command: {binary}")
    for role, path in ENVIRONMENTS.items():
        if not (path / "bin/python").exists():
            issues.append(f"Missing {role} environment: {path}")
    for source in MANIFEST["repositories"]:
        path = WORKSPACE / source["directory"]
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
        if result.returncode or result.stdout.strip() != source["revision"]:
            issues.append(f"Missing/mismatched source revision: {source['directory']}")
    for path in [*tls_paths(), image_path(), ROOT / "runtime/fixtures/question.wav"]:
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"Missing file: {path}")
    checkpoints = ["MiniCPM-o-Demo/models/MiniCPM-o-4_5/config.json",
                   "MiniCPM-o-Demo/models/MiniCPM-o-4_5/model.safetensors.index.json",
                   "SoulX-FlashHead/models/SoulX-FlashHead-1_3B/Model_Lite",
                   "SoulX-FlashHead/models/wav2vec2-base-960h/config.json",
                   CONFIG["asr_model"] + "/model.bin"]
    for name in checkpoints:
        if not (WORKSPACE / name).exists():
            issues.append(f"Missing model: {name}")
    for role in ("brain", "avatar", "asr", "gateway"):
        print(f"{role}: physical GPU(s) {role_devices(role)}")
    if shutil.which("nvidia-smi"):
        run("nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv")
    if issues:
        raise RuntimeError("\n".join(issues))
    print("Preflight passed (file/config checks only; GPU inference is checked by smoke tests).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["sources", "envs", "models", "tls", "fixtures", "doctor", "all"])
    parser.add_argument("--python", default="3.10", help="Python 3.10 executable or uv interpreter selector")
    args = parser.parse_args()
    for step in (["sources", "envs", "models", "tls", "fixtures", "doctor"] if args.step == "all" else [args.step]):
        if step == "envs":
            envs(args.python)
        else:
            globals()[step]()


if __name__ == "__main__":
    main()
