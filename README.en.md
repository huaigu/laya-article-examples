# Laya Article Examples

[简体中文](README.md)

Runnable examples for the Laya article series. This repository is separate from the development and training workspace and does not inherit its Git history. It currently includes episode 02 local inference and serving, plus episode 03 public-data preparation, fine-tuning, checkpoint selection, and comparative evaluation.

## Episodes

- `episodes/02-local-service`: Chinese ticket inference, routing policy, a FastAPI service, latency recording, and generic dataset auditing.
- `episodes/03-finetuning-evaluation`: pinned BANKING77 reconstruction, a training-only archive, native Laya fine-tuning, and independent result recomputation.

## Scope

Included: example code, public-data download and processing rules, demonstration thresholds, mocked HTTP tests, explicitly synthetic fixtures, and documentation. Excluded: customer tickets, committed dataset files, weights, fine-tuning artifacts, internal logs, credentials, local configuration, article drafts, image-generation records, and the original Git history.

## Episode 02 setup and execution

Use Python 3.10+ and Git, preferably in a virtual environment. Install a suitable PyTorch build using its official installation instructions. From the repository root:

```bash
python -m pip install -r episodes/02-local-service/requirements.txt
cd episodes/02-local-service
python minimal.py
python quickstart.py
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1
```

The first load fetches required model artifacts; weights are not bundled. For offline use, prepare and test a complete cache for the pinned revision. Open `http://127.0.0.1:8000/docs`, expand `POST /triage`, select `Try it out`, enter a Chinese ticket with a `text` field, and select `Execute`.

The service loads and warms up during startup and returns 503 when busy. `/healthz` checks liveness; `/readyz` checks readiness. This is a teaching example, not a complete production deployment. Policy thresholds are illustrative and need validation on business data.

## Runtime and deployment notes

- **External access:** Add authentication, request size limits, rate limits, timeout and cancellation handling, redacted logs, monitoring, gradual rollout, and rollback. The example's busy response only manages inference occupancy within one process; it does not replace these controls.
- **Input length:** Check token counts with the model's tokenizer in addition to the API's character limit, so oversized text is not silently truncated.
- **Processes and devices:** The example uses one worker. Additional workers typically load their own model copies; budget RAM or VRAM accordingly. On MPS, verify compatibility of the operations used by the model.
- **Offline deployment:** Prepare the pinned model, configuration, and tokenizer cache, then test startup with networking disabled to verify that all required files are present.
- **Bilingual preloading:** Use `router.preload(["english", "multilingual"])` when only those two checkpoints are needed. The pinned version registers three checkpoints by default; `preload=True` loads all registered checkpoints, not just two when `max_loaded=2`. Warm up English and Chinese inputs and questions separately, explicitly selecting `model="english"` and `model="multilingual"`.

`threatens_cancellation.noul` concerns cancellation intent expressed in the input, not a forecast of future customer churn. The policy first sends low-confidence department classifications to human review; only classifications that meet the threshold proceed to the cancellation-intent check for priority follow-up.

## Episode 02 files and tests

In the episode directory, `minimal.py` demonstrates classification; `quickstart.py` defines three questions and routing policy; `server.py` serves the API; `benchmark.py` records warm inference; and `audit_dataset.py` inspects supplied data. The two `test_*.py` files cover policy/mocked HTTP and dataset auditing.

From the root, run tests without downloading a model:

```bash
python -m pip install "fastapi>=0.115,<1" "httpx>=0.27,<1"
python -m unittest discover -s episodes/02-local-service -p "test_*.py" -v
```

HTTP tests use FakeAgent, and audit tests use synthetic records. Neither validates real-model accuracy or performance.

## Episode 03 fine-tuning and evaluation

The episode 03 directory downloads BANKING77 from a pinned commit and verifies both upstream and reconstructed split hashes. Its uploadable training ZIP contains only training and validation data; the test split remains outside until checkpoint selection is complete.

```bash
cd episodes/03-finetuning-evaluation
python prepare.py --output data/banking77
python build_training_bundle.py --data data/banking77 --output outputs/episode03-training.zip
python -m unittest discover -p "test_*.py" -v
```

Full training requires a CUDA GPU and is never started by the ordinary tests. See the episode `README.md` and `RESULTS.md` for training, selection, evaluation, and the historical article results.
From the episode directory, measure real-model inference:

```bash
python benchmark.py --warmup 5 --runs 100 --output benchmark_result.json
```

This records fixed-input, batch-size-1, serial warm inference: P50, P95, individual timings, and environment details. It is not an HTTP concurrency benchmark. No hardware latency, memory, or accuracy results are pre-filled. Generated records are excluded from commits by default.

Record initial downloads, model loading, and warm inference separately. Model revision, language, input length, question count, dtype, and device affect timing; keep the model and input conditions consistent when comparing devices. The result includes device, model revision, dtype, dependency versions, and CUDA peak memory reported by the PyTorch allocator, with device synchronization around GPU timing. This memory metric differs from whole-process usage shown by `nvidia-smi`. Measure HTTP capacity separately, including queueing, networking, serialization, varying input lengths, and busy-response rates.

## Dataset auditing

Supply JSON arrays whose records contain `state` and `gold.triage_category.probabilities`. The auditor counts the argmax gold label, without loading a model or printing raw tickets.

```bash
python audit_dataset.py --train data/train.json --test data/test.json --output outputs/audit.json
```

The article's original data files are not distributed. The auditor reports class counts, a majority baseline, missing training classes, and exact state overlap. These examples alone do not reproduce the article's specific counts or calculate model accuracy. Zero exact overlap does not rule out semantic leakage.

## Versions and publication checks

- Laya source commit: `4066d5d5fbf08b66c6757ddeedbd797bd7655bc0`.
- Model: the `multilingual` subfolder of `convaiinnovations/laya`.
- Model revision: `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`.
- Other dependencies use compatibility ranges; record the actual environment for reproducibility.
- The maintainer has not yet chosen a publication license. Review dependency and model license declarations separately.

Run `python tools/check_public_content.py` from the root to check the file allowlist and common credential patterns. It does not replace human review. Explicitly add approved new files to `PUBLIC_FILES.txt`; do not copy the entire development directory or its history.
