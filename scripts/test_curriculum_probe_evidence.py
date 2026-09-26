import copy
import unittest
from verify_curriculum_probe import compare, rows, REFERENCE


class EvidenceTests(unittest.TestCase):
    def test_identity_and_probability_tampering(self):
        data = rows(REFERENCE / 'predictions-full.csv')
        self.assertEqual(len(data), 288)
        compare(data, data)
        for bad in (data[:-1], data + [data[0]]):
            with self.assertRaises(ValueError):
                compare(bad, data)
        for key, value in [('target', '9'), ('p_compile', 'nan'), ('p_compile', '2'), ('p_compile', '0.123456')]:
            bad = copy.deepcopy(data)
            bad[0][key] = value
            with self.assertRaises(ValueError):
                compare(bad, data)
