"""Checks for staging and input protection, independent of scientific packages."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('runner', Path(__file__).resolve().parents[1] / 'reproduce.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_rejects_archive_paths_outside_workspace(self):
        with tempfile.TemporaryDirectory() as folder:
            for path in ['../escape', '/absolute', 'C:/absolute', 'code\\escape', 'code/../../escape']:
                with self.subTest(path=path), self.assertRaises(ValueError):
                    runner.contained_path(Path(folder), path)

    def test_existing_workspace_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            sentinel = p / 'keep.txt'
            sentinel.write_text('keep', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                runner.stage(p, {'frozen_files': {}}, None)
            self.assertEqual(sentinel.read_text(encoding='utf-8'), 'keep')

    def test_frozen_manifest_matches_repository(self):
        runner.verify_frozen(runner.read_manifest())


if __name__ == '__main__':
    unittest.main()
