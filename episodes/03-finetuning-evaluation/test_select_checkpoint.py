from pathlib import Path
import tempfile
import unittest

from common import save
from select_checkpoint import select


def make_complete_epoch(root, epoch):
    epoch_dir = root / f"epoch-{epoch:02d}"
    export = epoch_dir / "export"
    (export / "encoder").mkdir(parents=True)
    (export / "tokenizer").mkdir()
    for path in (
        epoch_dir / "snapshot.json",
        export / "model.safetensors",
        export / "rl_agent_config.json",
        export / "encoder" / "config.json",
        export / "tokenizer" / "tokenizer_config.json",
    ):
        path.write_text("{}\n", encoding="utf-8")


class SelectionTests(unittest.TestCase):
    def test_incomplete_latest_epoch_is_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history = [
                {"epoch": 1, "validation": {"accuracy": 0.8, "macro_f1": 0.8}},
                {"epoch": 2, "validation": {"accuracy": 0.9, "macro_f1": 0.9}},
            ]
            save(root / "history.json", history)
            make_complete_epoch(root, 1)
            result = select(root)
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(result["completed_epochs"], [1])


if __name__ == "__main__":
    unittest.main()
