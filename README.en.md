# Laya Article Examples

[简体中文](README.md)

Runnable examples for the Laya article series. This repository is separate from the development and training workspace and does not inherit its Git history. The current scope is episode 02: Chinese ticket inference, routing policy, a FastAPI service, latency recording, and generic dataset auditing.

## Scope

Included: example code, demonstration thresholds, mocked HTTP tests, explicitly synthetic fixtures, and documentation. Excluded: customer tickets, business or training data, weights, fine-tuning artifacts, internal logs, credentials, local configuration, article drafts, image-generation records, and the original Git history.

## Setup and execution

Use Python 3.10+ and Git, preferably in a virtual environment. Install a suitable PyTorch build using its official installation instructions. From the repository root:

```bash
python -m pip install -r episodes/02-local-service/requirements.txt
cd episodes/02-local-service
python minimal.py
python quickstart.py
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1
```

The first load fetches required model artifacts; weights are not bundled. For offline use, prepare and test a complete cache for the pinned revision. Open the local service on port 8000 at `/docs` and send a Chinese ticket with a `text` field to `/triage`.

The service loads and warms up during startup and returns 503 when busy. `/healthz` checks liveness; `/readyz` checks readiness. This is a teaching example, not a complete production deployment. Policy thresholds are illustrative and need validation on business data.

## Files and tests

In the episode directory, `minimal.py` demonstrates classification; `quickstart.py` defines three questions and routing policy; `server.py` serves the API; `benchmark.py` records warm inference; and `audit_dataset.py` inspects supplied data. The two `test_*.py` files cover policy/mocked HTTP and dataset auditing.

From the root, run tests without downloading a model:

```bash
python -m pip install "fastapi>=0.115,<1" "httpx>=0.27,<1"
python -m unittest discover -s episodes/02-local-service -p "test_*.py" -v
```

HTTP tests use FakeAgent, and audit tests use synthetic records. Neither validates real-model accuracy or performance.
From the episode directory, measure real-model inference:

```bash
python benchmark.py --warmup 5 --runs 100 --output benchmark_result.json
```

This records fixed-input, batch-size-1, serial warm inference: P50, P95, individual timings, and environment details. It is not an HTTP concurrency benchmark. No hardware latency, memory, or accuracy results are pre-filled. Generated records are excluded from commits by default.

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