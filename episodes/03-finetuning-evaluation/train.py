"""Run only the validation-selection training phase; the test split is never loaded."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

from common import read, save, sha, verify_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("outputs/run"))
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = verify_bundle(root)
    if (root / "test.json").exists() or "test.json" in manifest["files"]:
        raise RuntimeError("The training workspace must not contain test.json")

    protocol = read(root / "protocol.json")
    training, validation = read(root / "train.json"), read(root / "validation.json")
    if sha(root / "train.json") != protocol["data_sha256"]["train"]:
        raise ValueError("Training data hash mismatch")
    if sha(root / "validation.json") != protocol["data_sha256"]["validation"]:
        raise ValueError("Validation data hash mismatch")
    if len(training) != 7957 or len(validation) != 1989:
        raise ValueError("Unexpected split size")
    if {row["id"] for row in training} & {row["id"] for row in validation}:
        raise ValueError("Training and validation IDs overlap")

    import modeling as m
    from training_core import train_selection

    if not m.torch.cuda.is_available():
        raise RuntimeError("This registered recipe requires a CUDA GPU")
    runtime = {
        "python": platform.python_version(),
        "torch": str(m.torch.__version__),
        "cuda": m.torch.version.cuda,
        "gpu": m.torch.cuda.get_device_name(0),
        "precision": "FP16 autocast / FP32 parameters / GradScaler",
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "laya",
                "transformers",
                "tokenizers",
                "numpy",
                "safetensors",
                "huggingface-hub",
            )
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    base = m.get_base(protocol)
    selection = train_selection(base, training, validation, protocol, output)
    save(output / "runtime.json", runtime)
    print(json.dumps({"selection": selection, "runtime": runtime}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
