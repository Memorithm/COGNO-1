import copy
import unittest
from select_curriculum_probe import ARMS, SEEDS, select


def fixture():
    return [dict(arm=a, seed=str(s), split='validation', source_sha256=f'{i:064x}',
                 target=str(i % 2), p_compile='0.5') for a in ARMS for s in SEEDS for i in range(16)]


class SelectionTests(unittest.TestCase):
    def test_tie_and_no_test_dependency(self):
        rows = fixture()
        self.assertEqual(select(rows)['selection']['arm'], 'full')
        self.assertEqual(select(rows), select(rows + [dict(split='test', p_compile='nan')]))

    def test_missing_duplicate_nonfinite(self):
        rows = fixture()
        for bad in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                select(bad)
        for p in ('nan', '-0.1', '1.1'):
            bad = copy.deepcopy(rows)
            bad[0]['p_compile'] = p
            with self.assertRaises(ValueError):
                select(bad)

    def test_all_seeds_contribute(self):
        rows = fixture()
        for row in rows:
            if row['arm'] == 'byte_mix':
                row['p_compile'] = '0.9' if row['target'] == '1' else '0.1'
        self.assertEqual(select(rows)['selection']['arm'], 'byte_mix')
