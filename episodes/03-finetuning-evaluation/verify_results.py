"""Recompute saved metrics without loading a model or requiring a GPU."""

import argparse
import json
from pathlib import Path

from checks import align, assert_metrics, paired
from common import read, sha
from logic import metrics


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def verify(data, result):
    data, result = Path(data), Path(result)
    protocol = read(data / "data_protocol.json")
    labels = sorted(protocol["labels"])
    rows = read(data / "test.json")
    summary = read(result / "summary.json")
    records = {}
    for name in ("original", "trained"):
        path = result / f"{name}-test.jsonl"
        if sha(path) != summary["prediction_sha256"][name]:
            raise ValueError(f"{name} predictions changed")
        values = read_jsonl(path)
        align(rows, values, labels)
        assert_metrics(metrics(values, labels), summary[name])
        records[name] = values
    if paired(records["original"], records["trained"]) != summary["paired_changes"]:
        raise ValueError("Paired comparison mismatch")
    gain = 100 * (summary["trained"]["accuracy"] - summary["original"]["accuracy"])
    if abs(gain - summary["accuracy_gain_pp"]) > 1e-9:
        raise ValueError("Accuracy gain mismatch")
    return {
        "verified": True,
        "n": len(rows),
        "original_accuracy": summary["original"]["accuracy"],
        "trained_accuracy": summary["trained"]["accuracy"],
        "accuracy_gain_pp": gain,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/banking77"))
    parser.add_argument("--result", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    print(json.dumps(verify(args.data, args.result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
