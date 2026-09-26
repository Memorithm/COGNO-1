import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from offline_bundle import export_commit, inventory, package, verify_inventory


class BundleTests(unittest.TestCase):
    def test_integrity_rejects_changed_added_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            file = source / 'example'
            file.write_text('original')
            manifest = dict(schema='cogno-offline-bundle/v1', source_sha256=inventory(source))
            (root / 'manifest.json').write_text(json.dumps(manifest))
            verify_inventory(root)
            file.write_text('modified')
            with self.assertRaises(ValueError):
                verify_inventory(root)
            file.write_text('original')
            extra = source / 'extra'
            extra.write_text('extra')
            with self.assertRaises(ValueError):
                verify_inventory(root)
            extra.unlink()
            file.unlink()
            with self.assertRaises(ValueError):
                verify_inventory(root)

    def test_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'link').symlink_to('/nonexistent')
            with self.assertRaises(ValueError):
                inventory(root)

    def test_existing_destination_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'keep').write_text('preserve')
            with self.assertRaises(FileExistsError):
                package(root, root)
            self.assertEqual((root / 'keep').read_text(), 'preserve')

    def test_export_ignores_uncommitted_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / 'repo'
            repo.mkdir()
            def git(*args):
                return subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True)
            git('init')
            (repo / 'file').write_text('committed')
            git('add', 'file')
            git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                'commit', '-m', 'fixture')
            (repo / 'file').write_text('dirty')
            (repo / 'private').write_text('not packaged')
            out = root / 'out'
            out.mkdir()
            export_commit(repo, out)
            self.assertEqual((out / 'file').read_text(), 'committed')
            self.assertFalse((out / 'private').exists())


if __name__ == '__main__':
    unittest.main()
