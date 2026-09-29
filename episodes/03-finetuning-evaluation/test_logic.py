import hashlib
from pathlib import Path
import tempfile
import unittest

from logic import best_epoch, groups, metrics, split_file


class LogicTests(unittest.TestCase):
    def test_accuracy_is_primary(self):
        history = [
            {"epoch": 1, "validation": {"accuracy": 0.8, "macro_f1": 0.9}},
            {"epoch": 2, "validation": {"accuracy": 0.81, "macro_f1": 0.7}},
        ]
        self.assertEqual(best_epoch(history)["epoch"], 2)

    def test_macro_f1_breaks_accuracy_tie_and_keeps_earlier_full_tie(self):
        history = [
            {"epoch": epoch, "validation": {"accuracy": 0.8, "macro_f1": score}}
            for epoch, score in ((1, 0.7), (2, 0.8), (3, 0.8))
        ]
        self.assertEqual(best_epoch(history)["epoch"], 2)

    def test_partial_accumulation_group_is_preserved(self):
        self.assertEqual(list(groups(range(10), 4)), [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])

    def test_metrics_include_zero_support_classes(self):
        result = metrics(
            [
                {"gold": 0, "pred": 0, "probs": [0.8, 0.1, 0.1]},
                {"gold": 1, "pred": 0, "probs": [0.6, 0.3, 0.1]},
            ],
            ["a", "b", "c"],
        )
        self.assertEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], 2 / 9)

    def test_bad_probabilities_are_rejected(self):
        with self.assertRaises(AssertionError):
            metrics([{"gold": 0, "pred": 0, "probs": [0.8, 0.8]}], ["a", "b"])

    def test_chunk_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = bytes(range(256)) * 11
            source = root / "source"
            source.write_bytes(data)
            result = split_file(source, root, "part", 113)
            recovered = b"".join((root / part["name"]).read_bytes() for part in result["parts"])
            self.assertEqual(recovered, data)
            self.assertEqual(result["sha256"], hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    unittest.main()
