"""Single-input, serial warm inference latency; NOT HTTP throughput."""
import argparse
import importlib.metadata
import json
import math
import platform
import statistics
import time
from pathlib import Path

from quickstart import MODEL_REPO, MODEL_REVISION, QUESTIONS, TICKET, load_agent


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("benchmark_result.json"))
    args = parser.parse_args()
    if args.runs < 20 or args.warmup < 1:
        parser.error("Use at least 20 runs and one warmup.")
    t0 = time.perf_counter()
    agent = load_agent()
    device = str(agent.device)

    def synchronize():
        if device.startswith("cuda"):
            torch.cuda.synchronize(agent.device)
        elif device.startswith("mps"):
            torch.mps.synchronize()

    synchronize()
    load_seconds = time.perf_counter() - t0
    for _ in range(args.warmup):
        agent.predict(TICKET, QUESTIONS)
    synchronize()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(agent.device)
    durations = []
    for _ in range(args.runs):
        synchronize()
        t0 = time.perf_counter()
        agent.predict(TICKET, QUESTIONS)
        synchronize()
        durations.append((time.perf_counter() - t0) * 1000)
    ordered = sorted(durations)
    report = {
        "scope": "same input, serial predict, batch=1, no HTTP, not throughput",
        "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
        "subfolder": "multilingual", "device": device, "dtype": str(agent.dtype),
        "os": platform.platform(), "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "laya")},
        "input": TICKET, "questions": QUESTIONS,
        "load_seconds_includes_download_if_uncached": load_seconds,
        "warmup_runs": args.warmup, "measured_runs": args.runs,
        "p50_ms": statistics.median(durations),
        "p95_ms_nearest_rank": ordered[math.ceil(.95 * len(ordered)) - 1],
        "durations_ms": durations,
    }
    if device.startswith("cuda"):
        report.update(
            gpu=torch.cuda.get_device_name(agent.device),
            peak_allocated_mib=torch.cuda.max_memory_allocated(agent.device) / 1024**2,
            peak_reserved_mib=torch.cuda.max_memory_reserved(agent.device) / 1024**2,
            memory_note="PyTorch allocator only, not complete nvidia-smi process memory",
        )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output.resolve())
    print(f"P50={report['p50_ms']:.2f}ms; P95={report['p95_ms_nearest_rank']:.2f}ms")


if __name__ == "__main__":
    main()
