import copy
import unittest
from prepare_domain_holdout import DOMAIN_SPLITS, group_mapping
from select_curriculum_probe import ARMS, SEEDS, select


class DomainHoldoutTests(unittest.TestCase):
    def test_whole_domains_and_fixed_inventory(self):
        rows = [dict(project=f'synthetic/curriculum/{d}/family{i//2}')
                for d in DOMAIN_SPLITS for i in range(24)]
        mapping = group_mapping(rows).decode().splitlines()[1:]
        self.assertEqual(len(mapping), 144)
        for line in mapping:
            project, domain, split = line.split('\t')
            self.assertEqual(project.split('/')[2], domain)
            self.assertEqual(split, DOMAIN_SPLITS[domain])
        with self.assertRaises(ValueError):
            group_mapping(rows[:-1])
        bad = copy.deepcopy(rows)
        bad[0]['project'] = 'upstream/real/project'
        with self.assertRaises(ValueError):
            group_mapping(bad)

    def test_48_row_selection_keeps_test_irrelevant(self):
        rows = [dict(arm=a, seed=str(s), split='validation', source_sha256=f'{i:064x}',
                     target=str(i % 2), p_compile='0.5')
                for a in ARMS for s in SEEDS for i in range(48)]
        self.assertEqual(select(rows, 48)['selection']['arm'], 'full')
        self.assertEqual(select(rows, 48), select(rows + [dict(split='test')], 48))
        with self.assertRaises(ValueError):
            select(rows)
        with self.assertRaises(ValueError):
            select(rows[:-1], 48)
        with self.assertRaises(ValueError):
            select(rows, 47)
