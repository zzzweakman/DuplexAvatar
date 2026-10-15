"""Sample NVIDIA device and owned-process VRAM; never starts or stops services."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.manage import identity, latest


def query(fields, kind="gpu"):
    result = subprocess.run(["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=10)
    return [[part.strip() for part in row] for row in csv.reader(result.stdout.splitlines())]


def number(value):
    try:
        return int(value)
    except ValueError:
        return None


def process_role(pid, owners):
    if pid in owners:
        return owners[pid]
    try:
        # Model runtimes can spawn GPU workers. They inherit the manager-owned
        # process group even though nvidia-smi reports a different PID.
        return owners.get(os.getpgid(pid))
    except (ProcessLookupError, PermissionError):
        return None


def sample(gpus):
    devices = {uuid: {"index": int(index), "name": name, "total_mib": number(total), "used_mib": number(used)}
               for index, uuid, name, total, used in query("index,uuid,name,memory.total,memory.used")
               if int(index) in gpus}
    owners = {}
    try:
        _, records = latest()
        for record in records:
            current = identity(record["pid"])
            if current and all(current[k] == record[k] for k in ("start", "command")):
                owners[record["pid"]] = record["name"]
    except (SystemExit, FileNotFoundError, json.JSONDecodeError, ProcessLookupError):
        pass
    processes = []
    for uuid, pid, used in query("gpu_uuid,pid,used_gpu_memory", "compute-apps"):
        if uuid in devices:
            role = process_role(int(pid), owners)
            processes.append({"gpu": devices[uuid]["index"], "pid": int(pid), "mib": number(used),
                              "role": role, "owned": role is not None})
    return {"utc": datetime.now(timezone.utc).isoformat(), "monotonic": time.monotonic(),
            "devices": list(devices.values()), "processes": processes}


def summarize(samples):
    result = {"samples": len(samples), "duration_seconds": 0, "devices": {},
              "max_simultaneous_device_mib": 0, "max_simultaneous_owned_process_mib": 0}
    if samples:
        result["duration_seconds"] = round(samples[-1]["monotonic"] - samples[0]["monotonic"], 3)
    intervals = [b["monotonic"] - a["monotonic"] for a, b in zip(samples, samples[1:])]
    result["actual_interval_seconds"] = {"median": round(statistics.median(intervals), 3),
                                         "max": round(max(intervals), 3)} if intervals else None
    result["unattributed_gpu_processes_seen"] = any(not p["owned"] for s in samples for p in s["processes"])
    for point in samples:
        result["max_simultaneous_device_mib"] = max(result["max_simultaneous_device_mib"], sum(d["used_mib"] or 0 for d in point["devices"]))
        result["max_simultaneous_owned_process_mib"] = max(result["max_simultaneous_owned_process_mib"],
            sum(p["mib"] or 0 for p in point["processes"] if p["owned"]))
        for device in point["devices"]:
            key = str(device["index"])
            row = result["devices"].setdefault(key, {"name": device["name"], "total_mib": device["total_mib"],
                "peak_device_mib": 0, "peak_owned_process_mib": 0, "roles": []})
            row["peak_device_mib"] = max(row["peak_device_mib"], device["used_mib"] or 0)
            owned = [p for p in point["processes"] if p["gpu"] == device["index"] and p["owned"]]
            row["peak_owned_process_mib"] = max(row["peak_owned_process_mib"], sum(p["mib"] or 0 for p in owned))
            row["roles"] = sorted(set(row["roles"]) | {p["role"] for p in owned})
    result["sum_individual_device_peaks_mib"] = sum(d["peak_device_mib"] for d in result["devices"].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--seconds", type=float, default=0, help="0: continue until Ctrl-C/SIGTERM")
    parser.add_argument("--output", type=Path, required=True, help="A new output directory")
    args = parser.parse_args()
    if args.interval < 0.05 or args.seconds < 0:
        parser.error("interval must be >=0.05 and seconds >=0")
    args.output.mkdir(parents=True, exist_ok=False)
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    samples = []
    try:
        with (args.output / "samples.jsonl").open("x") as out:
            while not stopping and (not args.seconds or time.monotonic() - started < args.seconds):
                tick = time.monotonic()
                value = sample(set(args.gpus))
                samples.append(value)
                out.write(json.dumps(value) + "\n")
                out.flush()
                time.sleep(max(0, args.interval - (time.monotonic() - tick)))
    finally:
        report = {"requested_interval_seconds": args.interval,
                  "scope": "Sampled nvidia-smi VRAM, MiB; device totals include all processes and driver allocations. Owned process memory includes CUDA/allocator reservation; this is not torch allocated memory or a universal upper bound.",
                  **summarize(samples)}
        with (args.output / "summary.json").open("x") as out:
            json.dump(report, out, indent=2)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
