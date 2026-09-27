"""Synthetic schema fixtures only; these are not article or customer records."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from audit_dataset import audit, inspect_dataset


def record(state, category):
    return {'state': state, 'gold': {'triage_category': {'probabilities': {category: 1.0}}}}


class AuditTests(unittest.TestCase):
    def test_counts_and_missing_class(self):
        with TemporaryDirectory() as directory:
            train, test = Path(directory) / 'train.json', Path(directory) / 'test.json'
            train.write_text(json.dumps([record('fixture-a', 'a')]), encoding='utf-8')
            test.write_text(json.dumps([record('fixture-b', 'b'), record('fixture-c', 'b'),
                                       record('fixture-a', 'a')]), encoding='utf-8')
            result = audit(train, test)
            self.assertEqual(result['missing_train_classes'], ['b'])
            self.assertAlmostEqual(result['majority_baseline'], 2 / 3)
            self.assertEqual(result['exact_state_overlap'], 1)
            self.assertEqual(result['train']['file'], 'train.json')
            self.assertNotIn('fixture-a', json.dumps(result))

    def test_empty_data_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'empty.json'
            path.write_text('[]', encoding='utf-8')
            with self.assertRaises(ValueError):
                inspect_dataset(path, 'triage_category')

    def test_missing_labels_are_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            path.write_text('[{}]', encoding='utf-8')
            with self.assertRaises(ValueError):
                inspect_dataset(path, 'triage_category')

    def test_non_finite_probabilities_are_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            row = record('synthetic', 'a')
            row['gold']['triage_category']['probabilities']['a'] = float('nan')
            path.write_text(json.dumps([row]), encoding='utf-8')
            with self.assertRaises(ValueError):
                inspect_dataset(path, 'triage_category')


if __name__ == '__main__':
    unittest.main()
