"""Build the uploadable training-only ZIP. The official test split stays outside."""

import argparse
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

from common import read, save, sha
from settings import EXPECTED_SPLITS, make_training_protocol

RUNTIME_FILES = [
    "common.py",
    "logic.py",
    "modeling.py",
    "training_core.py",
    "train.py",
    "select_checkpoint.py",
    "requirements.txt",
]


def build(data, output):
    here = Path(__file__).resolve().parent
    data, output = Path(data), Path(output)
    data_protocol = read(data / "data_protocol.json")
    labels = data_protocol["labels"]
    for name in ("train", "validation"):
        path = data / f"{name}.json"
        if sha(path) != EXPECTED_SPLITS[name]["sha256"]:
            raise ValueError(f"Unexpected {name} data hash")
    if (data / "test.json").exists() is False:
        raise FileNotFoundError("Prepare the complete public split before building the isolated bundle")
    if sha(data / "test.json") != EXPECTED_SPLITS["test"]["sha256"]:
        raise ValueError("Unexpected test data hash")

    for name in RUNTIME_FILES:
        source = here / name
        if name.endswith(".py"):
            ast.parse(source.read_text(encoding="utf-8"), filename=name)

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temporary:
        staging = Path(temporary)
        for name in RUNTIME_FILES:
            (staging / name).write_bytes((here / name).read_bytes())
        for name in ("train.json", "validation.json"):
            (staging / name).write_bytes((data / name).read_bytes())
        save(staging / "protocol.json", make_training_protocol(labels))
        files = RUNTIME_FILES + ["train.json", "validation.json", "protocol.json"]
        save(
            staging / "manifest.json",
            {
                "files": {name: sha(staging / name) for name in files},
                "test_included": False,
                "credentials_included": False,
                "private_project_included": False,
            },
        )
        if output.exists():
            output.unlink()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in files + ["manifest.json"]:
                archive.write(staging / name, arcname=name)
    result = {"archive": str(output), "sha256": sha(output), "bytes": output.stat().st_size}
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/banking77"))
    parser.add_argument("--output", type=Path, default=Path("outputs/episode03-training.zip"))
    args = parser.parse_args()
    build(args.data, args.output)


if __name__ == "__main__":
    main()
