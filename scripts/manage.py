"""Start, inspect and stop only processes owned by this project."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import ssl
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from duplex_avatar.settings import CONFIG, WORKSPACE, tls_paths

PYTHON = WORKSPACE / "MiniCPM-o-Demo/.venv/base/bin/python"
AVATAR_PYTHON = WORKSPACE / "SoulX-FlashHead/.venv/bin/python"
ASR_PYTHON = WORKSPACE / "Omni-Interaction-Agent/.venv/bin/python"
TLS = ssl._create_unverified_context()


def identity(pid):
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        return None
    fields = (proc / "stat").read_text().split(") ", 1)[1].split()
    if fields[0] == "Z":
        return None
    return {"start": fields[19], "command": (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()}


def latest():
    records = sorted((ROOT / "runtime").glob("run-*/processes.json"))
    if not records:
        raise SystemExit("No recorded run")
    return records[-1], json.loads(records[-1].read_text())


def fetch(url):
    request = urllib.request.Request(url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=TLS))
    with opener.open(request, timeout=3) as response:
        return json.load(response)


def process_env(name, gpu):
    cache = ROOT / "runtime/cache"
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
               TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
               TRITON_CACHE_DIR=str(cache / "triton"), TORCHINDUCTOR_COMPILE_THREADS="4",
               DUPLEX_AVATAR_OVERRIDES=json.dumps(CONFIG),
               CUDA_VISIBLE_DEVICES=(",".join(map(str, gpu)) if isinstance(gpu, list) else str(gpu)) if gpu is not None else "")
    if name == "brain":
        mini_cache = WORKSPACE / "MiniCPM-o-Demo/.cache"
        env.update(HF_HOME=str(mini_cache / "huggingface"),
                   HF_MODULES_CACHE=str(mini_cache / "huggingface/modules"),
                   NUMBA_CACHE_DIR=str(mini_cache / "numba"))
    if name == "asr":
        # CTranslate2 loads these libraries directly; it does not import torch.
        site = ASR_PYTHON.parents[1] / "lib/python3.10/site-packages/nvidia"
        libraries = [str(site / part / "lib") for part in ("cublas", "cudnn")
                     if (site / part / "lib").is_dir()]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(libraries + [env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    return env


def role_devices(name):
    mapping = {"brain": list(dict.fromkeys([CONFIG["brain_gpu"], CONFIG.get("tts_gpu", CONFIG["brain_gpu"])])),
               "avatar": CONFIG["avatar_gpu"], "asr": CONFIG.get("asr_gpu", CONFIG["avatar_gpu"]), "gateway": None}
    devices = mapping[name]
    values = devices if isinstance(devices, list) else [] if devices is None else [devices]
    allowed = CONFIG.get("allowed_gpus", [0, 1, 2, 3])
    if not isinstance(allowed, list) or not allowed or any(type(gpu) is not int or gpu < 0 for gpu in allowed):
        raise ValueError("allowed_gpus must be a nonempty list of physical GPU indices")
    if any(type(gpu) is not int or gpu not in allowed for gpu in values):
        raise ValueError(f"{name} must use authorized physical GPUs {allowed}: {devices}")
    if name == "brain" and CONFIG["avatar_gpu"] in values:
        raise ValueError("Brain and avatar require separate GPUs")
    return devices


def start():
    for name in ("brain", "avatar", "asr"):
        role_devices(name)
    required = [PYTHON, AVATAR_PYTHON, ASR_PYTHON, *tls_paths()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing deployment files; run scripts/setup.py first:\n" + "\n".join(missing))
    for port in (CONFIG["brain_port"], CONFIG["avatar_port"], CONFIG["asr_port"], CONFIG["gateway_port"]):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    run = ROOT / "runtime" / time.strftime("run-%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    definitions = [
        ("brain", PYTHON, "duplex_avatar.brain_server", role_devices("brain"), CONFIG["brain_port"]),
        ("avatar", AVATAR_PYTHON, "duplex_avatar.avatar_server", role_devices("avatar"), CONFIG["avatar_port"]),
        ("asr", ASR_PYTHON, "duplex_avatar.asr_server", role_devices("asr"), CONFIG["asr_port"]),
        ("gateway", PYTHON, "duplex_avatar.gateway", None, CONFIG["gateway_port"]),
    ]
    records, processes = [], []
    for name, python, module, gpu, port in definitions:
        env = process_env(name, gpu)
        with (run / f"{name}.log").open("x") as log:
            p = subprocess.Popen([str(python), "-m", module], cwd=ROOT, env=env,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        processes.append(p)
        records.append({"name": name, "pid": p.pid, "port": port, "gpu": gpu,
                        "log": str(run / f"{name}.log"), **identity(p.pid)})
        print(f"Started {name}: PID {p.pid}; {run / (name + '.log')}", flush=True)
    with (run / "processes.json").open("x") as file:
        json.dump(records, file, indent=2)
    pending = {r["name"]: r for r in records}
    deadline = time.monotonic() + 1800
    while pending and time.monotonic() < deadline:
        for p, r in zip(processes, records):
            if p.poll() is not None:
                raise SystemExit(f"{r['name']} exited ({p.returncode}); inspect {run}")
        for name, r in list(pending.items()):
            try:
                scheme = "https" if name == "gateway" else "http"
                result = fetch(f"{scheme}://127.0.0.1:{r['port']}/health")
                if result.get("status") in ("ready", "ok"):
                    print(f"Ready {name}: {json.dumps(result)}", flush=True)
                    pending.pop(name)
            except Exception:
                pass
        if pending:
            time.sleep(2)
    if pending:
        raise SystemExit(f"Readiness timed out: {list(pending)}; inspect {run}")
    print(f"READY https://localhost:{CONFIG['gateway_port']}\nRUN {run}", flush=True)


def restart_gateway(name="gateway"):
    """Reload one owned process, retaining all other warmed model processes."""
    path, records = latest()
    record = next(r for r in records if r["name"] == name)
    devices = role_devices(name)
    current = identity(record["pid"])
    if current and any(current[k] != record[k] for k in ("start", "command")):
        raise SystemExit(f"{name} identity mismatch; refusing to signal it")
    if name == "gateway" and current is None:
        # A previous graceful stop may finish just after its wait timed out.
        # Resume that reload without touching the warmed model processes.
        brain = fetch(f"http://127.0.0.1:{CONFIG['brain_port']}/health")
        avatar = fetch(f"http://127.0.0.1:{CONFIG['avatar_port']}/health")
        if brain.get("active_session_id") or avatar.get("stream_busy"):
            raise SystemExit("A model session is still active; refusing gateway recovery")
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", record["port"]))
    else:
        health = fetch(f"https://127.0.0.1:{CONFIG['gateway_port']}/health")
        if health.get("busy") or name == "avatar" and health.get("avatar", {}).get("stream_busy"):
            raise SystemExit("Gateway has an active session; finish it before restarting")
        if not current:
            raise SystemExit(f"{name} is not running; refusing to signal it")
        os.killpg(record["pid"], signal.SIGTERM)
        deadline = time.monotonic() + 20
        while identity(record["pid"]) and time.monotonic() < deadline:
            time.sleep(.1)
        if identity(record["pid"]):
            raise SystemExit(f"{name} did not stop; refusing a duplicate process")
    run = ROOT / "runtime" / time.strftime("run-%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    python = {"avatar": AVATAR_PYTHON, "asr": ASR_PYTHON}.get(name, PYTHON)
    module = f"duplex_avatar.{name}_server" if name != "gateway" else "duplex_avatar.gateway"
    with (run / f"{name}.log").open("x") as log:
        process = subprocess.Popen([str(python), "-m", module], cwd=ROOT,
                                   env=process_env(name, devices), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for item in records:
        item.setdefault("log", str(path.parent / f"{item['name']}.log"))
    record.update(pid=process.pid, gpu=devices, log=str(run / f"{name}.log"), **identity(process.pid))
    with (run / "processes.json").open("x") as file:
        json.dump(records, file, indent=2)
    deadline = time.monotonic() + (30 if name == "gateway" else 1800)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"{name} exited; inspect {run}")
        try:
            scheme = "https" if name == "gateway" else "http"
            result = fetch(f"{scheme}://127.0.0.1:{record['port']}/health")
            if name != "gateway" and result.get("status") not in ("ready", "ok"):
                time.sleep(.5)
                continue
            print(json.dumps(result), flush=True)
            print(f"{name} restarted; other processes retained. RUN {run}", flush=True)
            return
        except Exception:
            time.sleep(.2)
    raise SystemExit(f"{name} readiness timed out; inspect {run}")


def start_asr():
    """Attach subtitles to an existing run without reloading warmed model weights."""
    devices = role_devices("asr")
    path, records = latest()
    if any(r["name"] == "asr" and identity(r["pid"]) for r in records):
        raise SystemExit("Recorded ASR is already running")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", CONFIG["asr_port"]))
    run = ROOT / "runtime" / time.strftime("run-%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    with (run / "asr.log").open("x") as log:
        process = subprocess.Popen([str(ASR_PYTHON), "-m", "duplex_avatar.asr_server"], cwd=ROOT,
                                   env=process_env("asr", devices), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    records = [r for r in records if r["name"] != "asr"]
    for item in records:
        item.setdefault("log", str(path.parent / f"{item['name']}.log"))
    records.append({"name": "asr", "pid": process.pid, "port": CONFIG["asr_port"],
                    "gpu": devices, "log": str(run / "asr.log"), **identity(process.pid)})
    with (run / "processes.json").open("x") as file:
        json.dump(records, file, indent=2)
    print(f"Started ASR: PID {process.pid}; RUN {run}", flush=True)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"ASR exited; inspect {run}")
        try:
            result = fetch(f"http://127.0.0.1:{CONFIG['asr_port']}/health")
            print(json.dumps(result), flush=True)
            return
        except Exception:
            time.sleep(.5)
    raise SystemExit(f"ASR readiness timed out; inspect {run}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "status", "stop", "restart-gateway", "restart-avatar", "restart-brain", "restart-asr", "start-asr"])
    args = parser.parse_args()
    if args.action == "start":
        start()
        return
    if args.action.startswith("restart-"):
        restart_gateway(args.action.removeprefix("restart-"))
        return
    if args.action == "start-asr":
        start_asr()
        return
    path, records = latest()
    print(path)
    for record in reversed(records):
        current = identity(record["pid"])
        matches = current and all(current[k] == record[k] for k in ("start", "command"))
        if args.action == "status":
            print(json.dumps({**record, "alive": bool(matches)}, ensure_ascii=False))
        elif matches:
            os.killpg(record["pid"], signal.SIGTERM)
            print("SIGTERM", record["name"], record["pid"])
        elif current:
            raise SystemExit(f"Identity mismatch; refusing to signal PID {record['pid']}")


if __name__ == "__main__":
    main()
