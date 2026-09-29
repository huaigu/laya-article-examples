from pathlib import Path
import tempfile
import unittest

from common import save, sha, verify_bundle


class BundleTests(unittest.TestCase):
    def test_training_bundle_rejects_test_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train.json").write_text("[]\n", encoding="utf-8")
            save(
                root / "manifest.json",
                {
                    "files": {"train.json": sha(root / "train.json")},
                    "test_included": False,
                },
            )
            self.assertFalse(verify_bundle(root)["test_included"])
            manifest = {
                "files": {
                    "train.json": sha(root / "train.json"),
                    "test.json": "not-used",
                },
                "test_included": False,
            }
            save(root / "manifest.json", manifest)
            with self.assertRaises(AssertionError):
                verify_bundle(root)

    def test_nested_manifest_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save(root / "manifest.json", {"files": {"nested/file": "x"}, "test_included": False})
            with self.assertRaises(AssertionError):
                verify_bundle(root)


if __name__ == "__main__":
    unittest.main()
