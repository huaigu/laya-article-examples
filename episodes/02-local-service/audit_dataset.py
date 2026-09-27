"""Audit user-supplied label files. No dataset or internal path is bundled."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def inspect_dataset(path, question):
    raw = Path(path).read_bytes()
    rows = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a non-empty JSON array.')
    counts = Counter()
    states = set()
    for index, row in enumerate(rows):
        try:
            probabilities = row['gold'][question]['probabilities']
            state = row['state']
        except (KeyError, TypeError) as exc:
            raise ValueError(f'Row {index}: required label or state is missing.') from exc
        if not isinstance(probabilities, dict) or not probabilities:
            raise ValueError(f'Row {index}: expected a non-empty probabilities object.')
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or not 0 <= v <= 1
               for v in probabilities.values()):
            raise ValueError(f'Row {index}: probabilities must be finite numbers in [0, 1].')
        counts[max(probabilities, key=probabilities.get)] += 1
        states.add(json.dumps(state, ensure_ascii=False, sort_keys=True))
    return {
        'file': Path(path).name, 'rows': len(rows),
        'sha256': hashlib.sha256(raw).hexdigest(),
        'counts': dict(sorted(counts.items())),
    }, states


def audit(train, test, question='triage_category'):
    train_info, train_states = inspect_dataset(train, question)
    test_info, test_states = inspect_dataset(test, question)
    majority = max(test_info['counts'], key=test_info['counts'].get)
    return {
        'question': question, 'train': train_info, 'test': test_info,
        'missing_train_classes': sorted(set(test_info['counts']) - set(train_info['counts'])),
        'majority_label': majority,
        'majority_baseline': test_info['counts'][majority] / test_info['rows'],
        'exact_state_overlap': len(train_states & test_states),
        'scope': 'Label distribution only, not model accuracy. Exact non-overlap does not rule out semantic leakage.',
        'label_rule': 'Argmax of gold probabilities; ties retain the first label in input order.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--question', default='triage_category')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.train, args.test, args.question)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    else:
        print(text)


if __name__ == '__main__':
    main()
