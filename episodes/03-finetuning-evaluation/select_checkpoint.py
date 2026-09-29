"""Recreate the checkpoint selection record after an interrupted session."""

import argparse
import json
from pathlib import Path

from common import read, save, sha
from logic import best_epoch


def complete_export(run_dir, epoch):
    epoch_dir = Path(run_dir) / f"epoch-{epoch:02d}"
    export = epoch_dir / "export"
    required = [
        epoch_dir / "snapshot.json",
        export / "model.safetensors",
        export / "rl_agent_config.json",
        export / "encoder" / "config.json",
        export / "tokenizer" / "tokenizer_config.json",
    ]
    return export if all(path.is_file() for path in required) else None


def select(run_dir):
    run_dir = Path(run_dir)
    history = read(run_dir / "history.json")
    completed = [row for row in history if complete_export(run_dir, row["epoch"])]
    if not completed:
        raise FileNotFoundError("No complete inference snapshot was found")
    picked = best_epoch(completed)
    export = complete_export(run_dir, picked["epoch"])
    weights = export / "model.safetensors"
    result = {
        "best_epoch": picked["epoch"],
        "validation": picked["validation"],
        "model_directory": str(export.resolve()),
        "model_sha256": sha(weights),
        "criterion": "accuracy then macro_f1; earlier wins exact ties",
        "test_seen_in_this_run": False,
        "completed_epochs": [row["epoch"] for row in completed],
    }
    save(run_dir / "selection.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(select(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
