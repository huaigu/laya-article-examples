"""Download and deterministically reconstruct the episode 03 BANKING77 split."""

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import unicodedata
from urllib.request import urlopen

from settings import EXPECTED_SPLITS, RAW_BASE, RAW_FILES, SPLIT_SEED


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalize(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def download_raw(raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in RAW_FILES.items():
        destination = raw_dir / name
        if destination.exists() and digest(destination.read_bytes()) == expected:
            continue
        with urlopen(f"{RAW_BASE}/{name}", timeout=60) as response:
            data = response.read()
        if digest(data) != expected:
            raise ValueError(f"Downloaded {name} does not match the pinned SHA-256")
        destination.write_bytes(data)


def clean_official_file(path, split, labels):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    groups = collections.defaultdict(list)
    for index, row in enumerate(source_rows):
        if row["category"] not in labels:
            raise ValueError(f"Unknown label in {path.name}: {row['category']}")
        normalized_sha = digest(normalize(row["text"]).encode())
        groups[normalized_sha].append(
            {
                "id": f"official-{split}-{index:05d}",
                "text": row["text"],
                "label": row["category"],
                "normalized_sha256": normalized_sha,
            }
        )
    selected, removed = [], []
    for rows in groups.values():
        conflict = len({row["label"] for row in rows}) > 1
        if not conflict:
            selected.append(rows[0])
        dropped = rows if conflict else rows[1:]
        if dropped:
            removed.append(
                {
                    "split": split,
                    "reason": "conflicting_labels" if conflict else "normalized_duplicate",
                    "ids": [row["id"] for row in dropped],
                }
            )
    return selected, removed, len(source_rows)


def construct_splits(raw_dir):
    labels = json.loads((raw_dir / "categories.json").read_text(encoding="utf-8"))
    if len(labels) != len(set(labels)) or len(labels) != 77:
        raise ValueError("Expected 77 unique BANKING77 categories")
    selected, removed, original_counts = {}, [], {}
    for split in ("train", "test"):
        selected[split], dropped, original_counts[split] = clean_official_file(
            raw_dir / f"{split}.csv", split, labels
        )
        removed.extend(dropped)

    test_keys = {row["normalized_sha256"] for row in selected["test"]}
    pool = []
    for row in selected["train"]:
        if row["normalized_sha256"] in test_keys:
            removed.append(
                {"split": "train", "reason": "official_test_overlap", "ids": [row["id"]]}
            )
        else:
            pool.append(row)

    training, validation = [], []
    for label in labels:
        rows = sorted((row for row in pool if row["label"] == label), key=lambda row: row["id"])
        random.Random(f"{SPLIT_SEED}:{label}").shuffle(rows)
        validation_count = max(1, round(len(rows) * 0.2))
        validation.extend(rows[:validation_count])
        training.extend(rows[validation_count:])

    splits = {
        "train": sorted(training, key=lambda row: row["id"]),
        "validation": sorted(validation, key=lambda row: row["id"]),
        "test": sorted(selected["test"], key=lambda row: row["id"]),
    }
    keys = {
        name: {row["normalized_sha256"] for row in rows}
        for name, rows in splits.items()
    }
    overlap = {
        f"{left}__{right}": len(keys[left] & keys[right])
        for left in splits
        for right in splits
        if left < right
    }
    if any(overlap.values()):
        raise ValueError(f"Normalized text overlap remains: {overlap}")
    return labels, splits, removed, original_counts, overlap


def prepare(output, force=False):
    output = Path(output)
    if output.exists():
        if not force:
            raise FileExistsError(f"{output} already exists; pass --force to rebuild it")
        shutil.rmtree(output)
    raw_dir = output / "raw"
    download_raw(raw_dir)
    labels, splits, removed, original_counts, overlap = construct_splits(raw_dir)

    files = {}
    for name, rows in splits.items():
        if {row["label"] for row in rows} != set(labels):
            raise ValueError(f"{name} does not contain all labels")
        path = output / f"{name}.json"
        write_json(path, rows)
        actual = {"rows": len(rows), "sha256": digest(path.read_bytes())}
        if actual != EXPECTED_SPLITS[name]:
            raise ValueError(f"{name} reconstruction mismatch: {actual}")
        files[name] = actual

    protocol = {
        "task": "full 77-class BANKING77 English intent classification",
        "source_revision": RAW_BASE,
        "raw_sha256": RAW_FILES,
        "original_counts": original_counts,
        "split_seed": SPLIT_SEED,
        "labels": labels,
        "files": files,
        "removed": removed,
        "normalized_overlap_counts": overlap,
        "selection_policy": "training and validation only; test after checkpoint selection",
    }
    write_json(output / "data_protocol.json", protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/banking77"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protocol = prepare(args.output, args.force)
    print(json.dumps({"files": protocol["files"], "overlap": protocol["normalized_overlap_counts"]}, indent=2))


if __name__ == "__main__":
    main()
