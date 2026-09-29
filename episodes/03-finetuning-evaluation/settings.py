"""Pinned public inputs and the training recipe used by episode 03."""

DATA_REPOSITORY = "PolyAI-LDN/task-specific-datasets"
DATA_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
RAW_BASE = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    f"{DATA_REVISION}/banking_data"
)
RAW_FILES = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    "categories.json": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
}

SPLIT_SEED = 20260927
EXPECTED_SPLITS = {
    "train": {
        "rows": 7957,
        "sha256": "a35018d3d9211bbbf3d3010c3b72258424657ef511e1a83bcbcf3c337c13ca82",
    },
    "validation": {
        "rows": 1989,
        "sha256": "2450633c92ed5a08c2de42c4cfb690b255ced78d13a777973e0e7b73920a0745",
    },
    "test": {
        "rows": 3075,
        "sha256": "01061f2766dee5a35696d5470df9252969dcc00fef4a9736a15d724a770fa99d",
    },
}

MODEL_ID = "convaiinnovations/laya"
MODEL_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
BASE_MODEL_SHA256 = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
REFERENCE_REPOSITORY = "deepopen-com/deepopen"
REFERENCE_COMMIT = "6bd28ae5375959d72afc7d6682a05e9973179e1d"

TRAINING = {
    "seed": 123,
    "lr": 5e-5,
    "batch": 32,
    "micro": 4,
    "epochs": 8,
    "patience": 3,
    "warmup": 0.06,
}

QUESTION_INSTRUCTION = "Which banking intent does `message` express?"


def make_training_protocol(labels):
    labels = sorted(labels)
    assert len(labels) == len(set(labels)) == 77
    rendered = [label.replace("_", " ") for label in labels]
    assert len(rendered) == len(set(rendered))
    return {
        "run_id": "episode-03-public-reproduction",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "base_sha256": BASE_MODEL_SHA256,
        "labels": labels,
        "question": {
            "t": "choice",
            "ins": QUESTION_INSTRUCTION,
            "crit": dict.fromkeys(rendered),
        },
        **TRAINING,
        "test_sha256": EXPECTED_SPLITS["test"]["sha256"],
        "test_count": EXPECTED_SPLITS["test"]["rows"],
        "data_sha256": {
            name: EXPECTED_SPLITS[name]["sha256"]
            for name in ("train", "validation")
        },
        "selection": "validation accuracy, then macro_f1; earlier wins full ties",
        "target_accuracy": 0.9,
        "historical_retest": True,
        "test_included": False,
        "prior_test_exposure": (
            "320 official test examples from eight classes were evaluated in an "
            "earlier exploratory study; do not describe a later score as a pristine blind test"
        ),
        "reference_repo": REFERENCE_REPOSITORY,
        "reference_commit": REFERENCE_COMMIT,
        "checkpoint_policy": "complete inference snapshot after every validation epoch",
        "exact_optimizer_resume_supported": False,
    }
