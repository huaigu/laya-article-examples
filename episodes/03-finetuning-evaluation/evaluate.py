"""Evaluate the pinned base model and one selected snapshot on identical test inputs."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

from checks import align, paired
from common import read, save, sha
from settings import EXPECTED_SPLITS, make_training_protocol


def item_digest(items):
    payload = json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_for_evaluation(model_dir):
    import modeling as m

    model, tokenizer, _config, _groups = m.load_model(Path(model_dir))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/banking77"))
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    data_protocol = read(args.data / "data_protocol.json")
    protocol = make_training_protocol(data_protocol["labels"])
    test = args.data / "test.json"
    if sha(test) != EXPECTED_SPLITS["test"]["sha256"]:
        raise ValueError("Unexpected test split hash")
    rows = read(test)
    if len(rows) != EXPECTED_SPLITS["test"]["rows"]:
        raise ValueError("Unexpected test split size")

    import modeling as m

    m.seed_all(protocol["seed"])
    base = m.get_base(protocol)
    models = {"original": base, "trained": args.trained.resolve()}
    statistics, outputs, audits, timings = {}, {}, {}, {}
    encoded_digest = None
    started = time.monotonic()
    for name, model_dir in models.items():
        phase_start = time.monotonic()
        model, tokenizer = load_for_evaluation(model_dir)
        items, audit = m.encode(rows, tokenizer, protocol)
        current_digest = item_digest(items)
        if encoded_digest is None:
            encoded_digest = current_digest
        elif current_digest != encoded_digest:
            raise RuntimeError("The two models did not receive identical encoded inputs")
        metrics, records = m.score(model, items, tokenizer, protocol["labels"])
        align(rows, records, protocol["labels"])
        write_jsonl(output / f"{name}-test.jsonl", records)
        save(output / f"{name}-metrics.json", metrics)
        statistics[name], outputs[name], audits[name] = metrics, records, audit
        timings[name] = time.monotonic() - phase_start
        del model, tokenizer, items
        gc.collect()
        m.torch.cuda.empty_cache()

    comparison = paired(outputs["original"], outputs["trained"])
    summary = {
        "status": "complete",
        "historical_retest": True,
        "is_pristine_blind_test": False,
        "training_examples": EXPECTED_SPLITS["train"]["rows"],
        "validation_examples": EXPECTED_SPLITS["validation"]["rows"],
        "test_examples": EXPECTED_SPLITS["test"]["rows"],
        "test_used_for_training": False,
        "test_used_for_selection": False,
        "original": statistics["original"],
        "trained": statistics["trained"],
        "accuracy_gain_pp": 100
        * (statistics["trained"]["accuracy"] - statistics["original"]["accuracy"]),
        "paired_changes": comparison,
        "input_sequence_sha256": encoded_digest,
        "input_audits": audits,
        "base_sha256": sha(Path(base) / "model.safetensors"),
        "model_sha256": sha(args.trained / "model.safetensors"),
        "test_sha256": sha(test),
        "prediction_sha256": {
            name: sha(output / f"{name}-test.jsonl") for name in models
        },
        "model_load_encode_and_score_seconds": timings,
        "wall_seconds": time.monotonic() - started,
        "limitations": [
            "Historical test exposure is disclosed; this is not a pristine blind test.",
            "The task is an English 77-intent benchmark, not production traffic.",
            "Probabilities are raw model outputs and are not business-threshold calibration.",
        ],
    }
    save(output / "summary.json", summary)
    print(
        json.dumps(
            {
                "original_accuracy": summary["original"]["accuracy"],
                "trained_accuracy": summary["trained"]["accuracy"],
                "accuracy_gain_pp": summary["accuracy_gain_pp"],
                "paired_changes": comparison,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
