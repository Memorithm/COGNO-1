import copy
import unittest
from select_rust_ablation import CONFIGS, SEEDS, select


def fixture():
    return [dict(context=str(c), order=o, seed=str(s), split='validation',
                 source_sha256=f'{i:064x}', target=str(i % 2), p_compile='0.5')
            for c, o in CONFIGS for s in SEEDS for i in range(4)]


class SelectionTests(unittest.TestCase):
    def test_tie_and_no_test_dependency(self):
        rows = fixture()
        result = select(rows)
        self.assertEqual(result['selection']['context'], 128)
        self.assertEqual(result['selection']['order'], 'ordered')
        self.assertEqual(select(rows + [dict(split='test', p_compile='nan')]), result)

    def test_missing_duplicate_and_invalid(self):
        rows = fixture()
        for bad in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                select(bad)
        for value in ('nan', 'inf', '-0.1', '1.1'):
            bad = copy.deepcopy(rows)
            bad[0]['p_compile'] = value
            with self.assertRaises(ValueError):
                select(bad)

    def test_all_seeds_aggregated(self):
        rows = fixture()
        for row in rows:
            if row['order'] == 'shuffled' and row['context'] == '256':
                row['p_compile'] = '0.9' if row['target'] == '1' else '0.1'
        self.assertEqual(select(rows)['selection']['context'], 256)
        self.assertEqual(select(rows)['selection']['order'], 'shuffled')


if __name__ == '__main__':
    unittest.main()
