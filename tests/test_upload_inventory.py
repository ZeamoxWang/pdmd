"""Verify backup checksum comparison for ordinary Git and large-file objects."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from scripts.upload_cache import hashes, matches, destination_info


class UploadInventoryTest(unittest.TestCase):
    def test_git_and_lfs_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'sample'
            path.write_bytes(b'hello\n')
            expected = hashes(path)
            self.assertEqual(expected['blob_id'], 'ce013625030ba8dba906f756967f9e9ca394464a')
            remote = SimpleNamespace(size=6, blob_id=expected['blob_id'], lfs=None)
            self.assertTrue(matches(remote, expected))
            remote.blob_id = 'wrong'
            self.assertFalse(matches(remote, expected))
            remote.lfs = SimpleNamespace(sha256=expected['sha256'])
            self.assertTrue(matches(remote, expected))
            remote.size = 7
            self.assertFalse(matches(remote, expected))

    def test_requires_authorized_visibility(self):
        api = SimpleNamespace(repo_info=lambda *args, **kwargs: SimpleNamespace(private=True))
        with self.assertRaisesRegex(RuntimeError, 'visibility differs'):
            destination_info(api)


if __name__ == '__main__':
    unittest.main()
