import hashlib
import json
import unittest
import tempfile
from pathlib import Path
import admit_rust_corpus as corpus
import prepare_rust_diagnostic as diagnostic


def fixture():
    rows = []
    for split in corpus.SPLITS:
        for label in (0, 1):
            source = f'fn {split}_{label}() {{}}\n'
            rows.append(dict(project=split, revision='a'*40, license='MIT', classification='Internal',
                             compiler='rustc test-fixture', provenance='unit fixture; labels not compiler verified',
                             source=source, sha256=hashlib.sha256(source.encode()).hexdigest(), split=split, label=label))
    return rows


def encoded(rows):
    return ''.join(json.dumps(r)+'\n' for r in rows).encode()


class AdmissionTests(unittest.TestCase):
    def test_reused_fixture_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'diagnostic'
            manifest = diagnostic.prepare(output)
            self.assertEqual(manifest['counts'], {'train': 16, 'validation': 4, 'test': 4})
            before = (output / 'corpus.crust').read_bytes()
            with self.assertRaises(FileExistsError):
                diagnostic.prepare(output)
            self.assertEqual(before, (output / 'corpus.crust').read_bytes())
    def test_lossless_export_and_determinism(self):
        rows, wire = corpus.admit(encoded(fixture()), {'MIT'})
        self.assertEqual(len(rows), 6)
        self.assertEqual(bytes.fromhex(wire.decode().splitlines()[1].split('\t')[4]), rows[0]['source'].encode())
        self.assertEqual(corpus.admit(encoded(fixture()), {'MIT'})[1], wire)

    def test_metadata_hash_labels_and_leakage(self):
        for field, value in [('license', 'unknown'), ('classification', 'Secret'), ('revision', 'main'),
                             ('sha256', '0'*64), ('label', True), ('project', 'test'), ('source', 'x'*16385)]:
            rows = fixture()
            rows[0][field] = value
            with self.assertRaises(ValueError, msg=field):
                corpus.admit(encoded(rows), {'MIT'})
        rows = fixture()
        for bad in [rows[:-1], rows + [rows[0]]]:
            with self.assertRaises(ValueError):
                corpus.admit(encoded(bad), {'MIT'})

    def test_duplicate_json_keys_and_bounds(self):
        for data in [b'', b'{}\n', b'{"label":0,"label":1}\n', b'x'*(corpus.LIMIT+1)]:
            with self.assertRaises(ValueError):
                corpus.admit(data, {'MIT'})


if __name__ == '__main__':
    unittest.main()
