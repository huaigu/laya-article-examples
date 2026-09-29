import unittest

from checks import align, paired


class CheckTests(unittest.TestCase):
    def test_alignment_requires_matching_ids(self):
        with self.assertRaises(AssertionError):
            align([{"id": "a", "label": "x"}], [{"id": "b", "gold": 0}], ["x"])

    def test_paired_counts(self):
        original = [
            {"id": str(index), "gold": 0, "pred": prediction}
            for index, prediction in enumerate([0, 0, 1, 1])
        ]
        trained = [
            {"id": str(index), "gold": 0, "pred": prediction}
            for index, prediction in enumerate([0, 1, 0, 1])
        ]
        self.assertTrue(all(value == 1 for value in paired(original, trained).values()))


if __name__ == "__main__":
    unittest.main()
