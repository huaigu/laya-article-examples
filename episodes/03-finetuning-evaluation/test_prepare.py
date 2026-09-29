import csv
import json
from pathlib import Path
import tempfile
import unittest

from prepare import clean_official_file, normalize
from settings import make_training_protocol


class PrepareTests(unittest.TestCase):
    def test_normalization_is_case_and_punctuation_insensitive(self):
        self.assertEqual(normalize("  Card—ARRIVAL?! "), "card arrival")
        self.assertEqual(normalize("ＡＢＣ 123"), "abc 123")

    def test_duplicate_and_conflict_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["text", "category"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"text": "same text", "category": "a"},
                        {"text": "Same, text!", "category": "a"},
                        {"text": "conflict", "category": "a"},
                        {"text": "CONFLICT", "category": "b"},
                    ]
                )
            selected, removed, count = clean_official_file(path, "train", ["a", "b"])
            self.assertEqual(count, 4)
            self.assertEqual([row["label"] for row in selected], ["a"])
            self.assertEqual(
                sorted(item["reason"] for item in removed),
                ["conflicting_labels", "normalized_duplicate"],
            )

    def test_training_protocol_has_all_options_and_no_test_payload(self):
        labels = [f"label_{index:02d}" for index in range(77)]
        protocol = make_training_protocol(labels)
        self.assertEqual(len(protocol["labels"]), 77)
        self.assertEqual(len(protocol["question"]["crit"]), 77)
        self.assertFalse(protocol["test_included"])
        self.assertNotIn("test", protocol["data_sha256"])
        json.dumps(protocol)


if __name__ == "__main__":
    unittest.main()
