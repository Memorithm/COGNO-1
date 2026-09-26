import copy
import unittest
from verify_rust_ablation import compare_diagnostic, read_csv, REFERENCE


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.rows = read_csv(REFERENCE / 'predictions.csv')

    def test_frozen_evidence(self):
        self.assertEqual(len(self.rows), 288)
        compare_diagnostic(self.rows, self.rows)

    def test_missing_duplicate_and_category(self):
        altered = copy.deepcopy(self.rows)
        altered[0]['target'] = str(1-int(altered[0]['target']))
        for bad in (self.rows[:-1], self.rows + [self.rows[0]], altered):
            with self.assertRaises(ValueError):
                compare_diagnostic(bad, self.rows)

    def test_invalid_and_drifting_probability(self):
        for value in ('nan', 'inf', '-1', '2', '0.5'):
            bad = copy.deepcopy(self.rows)
            bad[0]['p_compile'] = value
            with self.assertRaises(ValueError):
                compare_diagnostic(bad, self.rows)


if __name__ == '__main__':
    unittest.main()
