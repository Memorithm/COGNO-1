import copy
import json
from pathlib import Path
import unittest
from prepare_domain_holdout import DOMAIN_SPLITS
from select_curriculum_probe import ARMS, SEEDS
from summarize_domain_holdout import summarize

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    lines, rows = ['CRUST001'], []
    for domain, split in DOMAIN_SPLITS.items():
        path = ROOT / 'experiments/rust-curriculum' / domain / 'corpus.jsonl'
        for text in path.read_text().splitlines():
            record = json.loads(text)
            digest, label = record['sha256'], record['label']
            lines.append(f"{split}\t{record['project']}\t{label}\t{digest}\t{record['source'].encode().hex()}")
            for arm in ARMS:
                for seed in SEEDS:
                    rows.append(dict(arm=arm, seed=str(seed), split=split, source_sha256=digest,
                                     target=str(label), prediction='0', p_compile='0.5', tokens='4'))
    return rows, ('\n'.join(lines) + '\n').encode('ascii')


class EvidenceTests(unittest.TestCase):
    def test_complete_uniform_and_test_independent_selection(self):
        rows, corpus = fixture()
        report = summarize(rows, corpus)
        self.assertEqual(report['predictions'], 3456)
        self.assertEqual(report['selection']['selection']['arm'], 'full')
        self.assertTrue(all(r['test_accuracy'] == 0.5 for r in report['arm_means']))
        self.assertFalse(report['model_promoted'])
        for row in rows:
            if row['split'] == 'test':
                row['prediction'] = row['target']
                row['p_compile'] = '0.9' if row['target'] == '1' else '0.1'
        changed = summarize(rows, corpus)
        self.assertEqual(report['selection'], changed['selection'])
        self.assertTrue(all(r['test_accuracy'] == 1 for r in changed['arm_means']))

    def test_missing_duplicate_corrupt_probability_and_identity(self):
        rows, corpus = fixture()
        for bad in (rows[:-1], rows[1:] + [rows[1]]):
            with self.assertRaises(ValueError):
                summarize(bad, corpus)
        for field, value in [('p_compile', 'nan'), ('p_compile', '1.1'), ('p_compile', '0.9'), ('split', 'test'),
                             ('arm', 'unknown'), ('seed', '2'), ('tokens', '513'),
                             ('source_sha256', 'f'*64), ('target', '9')]:
            bad = copy.deepcopy(rows)
            bad[0][field] = value
            with self.assertRaises(ValueError):
                summarize(bad, corpus)
        with self.assertRaises(ValueError):
            summarize(rows, corpus + b'\n')
