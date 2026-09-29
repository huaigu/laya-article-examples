# Episode 03: Fine-tuning and evaluation

[简体中文](README.md)

This directory accompanies episode 03. It reconstructs the public BANKING77 data, builds an isolated training package, selects a checkpoint using validation data, and compares the selected snapshot with the pinned base model.

Datasets, model weights, and run artifacts are not committed. `RESULTS.md` records the completed article experiment; it is not a promise that a new run on different hardware will produce bit-identical scores. Publishing this package did not start another full GPU training run.

## 1. Environment

Use Python 3.10+. Install a CUDA-compatible PyTorch build from the official PyTorch instructions, then install the remaining pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

The completed reference run used an NVIDIA T4 with 16 GB VRAM. An earlier RTX 3060 Ti 8 GB experiment used a different trainable scope and recipe, so the score difference should not be attributed to VRAM alone.

## 2. Reconstruct the data

```bash
python prepare.py --output data/banking77
```

The script downloads three files from a pinned upstream commit, verifies their SHA-256 digests, removes normalized duplicates and label conflicts, excludes normalized train/test overlap, and creates a class-stratified validation split. Expected counts are 7,957 training, 1,989 validation, and 3,075 test examples, with zero normalized-text overlap between the three splits.

## 3. Build the training-only ZIP

```bash
python build_training_bundle.py \
  --data data/banking77 \
  --output outputs/episode03-training.zip
```

The archive contains runtime code, training data, validation data, a protocol, and a manifest. It deliberately excludes `test.json`. Upload and extract it in Colab, install the dependencies, and run:

```bash
python train.py --root . --output outputs/run
```

The recipe updates the encoder, native decision head, and scoring path while keeping the action head frozen. Parameters remain FP32 with FP16 autocast and GradScaler. A complete inference snapshot is saved after each validation epoch. Checkpoints are ranked by validation accuracy, then Macro-F1, with the earlier epoch retained on a full tie.

A free Colab session can be interrupted. To request a clean stop, create a `STOP` file in the output directory. If the session ends after complete epochs were saved, rebuild the selection record with:

```bash
python select_checkpoint.py outputs/run
```

The package does not save optimizer state and does not automatically merge training and validation data for a second refit.

## 4. Evaluate identical test inputs

Download the selected `export` directory, then run the comparison outside the isolated training bundle:

```bash
python evaluate.py \
  --data data/banking77 \
  --trained outputs/run/epoch-07/export \
  --output outputs/evaluation

python verify_results.py \
  --data data/banking77 \
  --result outputs/evaluation
```

Both models receive the same label order, question, test examples, and encoded token sequences. The verifier does not load either model; it recomputes accuracy, Macro-F1, NLL, and paired prediction changes from the saved probabilities.

Part of the official test split had already been used during exploratory work, so later scoring is a disclosed historical retest rather than a pristine blind evaluation. A production study should retain a separate untouched acceptance set.

## 5. CPU-only tests

```bash
python -m unittest discover -p "test_*.py" -v
```

The tests cover cleaning rules, checkpoint selection, metric recomputation, training/test isolation, and file chunking. They do not start a GPU training run.

## Files

- `prepare.py`: download, clean, and reconstruct the three splits.
- `build_training_bundle.py`: create the ZIP that excludes the test split.
- `modeling.py`: load Laya, encode all 77 options, and save inference weights.
- `training_core.py` / `train.py`: run validation-based checkpoint selection.
- `select_checkpoint.py`: recover the selection record from completed epochs.
- `evaluate.py`: compare the base and selected models on identical inputs.
- `verify_results.py`: independently recompute saved results on CPU.
- `RESULTS.md`: historical article results and limitations.
