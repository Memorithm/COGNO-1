import csv
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
import check_bpe_evidence as evidence


class EvidenceTests(unittest.TestCase):
    def test_reference_counts(self):
        summary = evidence.check_predictions(evidence.REFERENCE)
        self.assertEqual(summary['bpe/42/test'], {'correct': 2, 'total': 4, 'tokens': 151})
        self.assertEqual(len(summary), 18)

    def test_missing_duplicate_nonfinite_and_changed_rows(self):
        lines = evidence.REFERENCE.read_text().splitlines()
        variants = ['\n'.join(lines[:-1]), '\n'.join(lines + [lines[-1]])]
        for column, value in [('p_compile', 'nan'), ('p_compile', '1.5'), ('prediction', '9'), ('parameters', '0')]:
            rows = list(csv.DictReader(io.StringIO('\n'.join(lines))))
            rows[0][column] = value
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=evidence.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            variants.append(stream.getvalue())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'predictions.csv'
            for variant in variants:
                path.write_text(variant)
                with self.assertRaises(ValueError):
                    evidence.check_predictions(path)

    def test_inventory_rejects_missing_seeds_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'checkpoints.tsv'
            for body in ['', '1\t../outside\t0\t' + '0' * 64 + '\n']:
                path.write_text('seed\tfile\tbytes\tsha256\n' + body)
                with self.assertRaises(ValueError):
                    evidence.check_checkpoints(temp)

    def test_inventory_detects_corruption_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            data = b'CBPC0001inventory-test-only'
            rows = ['seed\tfile\tbytes\tsha256']
            for seed in (1, 7, 42):
                name = f'bpe-seed-{seed}.cbpc'
                (directory / name).write_bytes(data)
                rows.append(f'{seed}\t{name}\t{len(data)}\t{hashlib.sha256(data).hexdigest()}')
            (directory / 'checkpoints.tsv').write_text('\n'.join(rows) + '\n')
            evidence.check_checkpoints(directory)
            target = directory / 'bpe-seed-1.cbpc'
            target.write_bytes(data[:-1] + b'x')
            with self.assertRaises(ValueError):
                evidence.check_checkpoints(directory)
            target.unlink()
            target.symlink_to(directory / 'bpe-seed-7.cbpc')
            with self.assertRaises(ValueError):
                evidence.check_checkpoints(directory)


if __name__ == '__main__':
    unittest.main()
