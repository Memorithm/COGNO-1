import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import unittest
from prepare_domain_holdout import DOMAIN_SPLITS
from select_curriculum_probe import ARMS, SEEDS
from summarize_domain_holdout import summarize, read_csv
from verify_domain_holdout import compare_report

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
    def test_thor_summary_matches_retained_x86_evidence(self):
        reference = ROOT / 'experiments/domain-holdout-v1'
        thor = json.loads((reference / 'thor-summary.json').read_text())
        report = json.loads((reference / 'report.json').read_text())
        self.assertEqual(thor['architecture'], 'aarch64')
        for key in ('checkpoints', 'selection', 'arm_means', 'corpus_sha256', 'predictions'):
            compare_report(thor[key], report[key])
        rows = [r for arm in ARMS for r in read_csv(reference / f'predictions-{arm}.csv')]
        data = io.StringIO(newline='')
        writer = csv.DictWriter(data, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
        self.assertEqual(hashlib.sha256(data.getvalue().encode()).hexdigest(), thor['predictions_sha256'])

    def test_report_rejects_tampered_metrics_and_inventory(self):
        value = dict(count=48, nll=0.6, models=['seed-1'], promoted=False)
        compare_report(value, dict(value, nll=0.6+1e-14))
        for altered in (dict(value, count=47), dict(value, nll=0.7),
                        dict(value, nll=float('nan')), dict(value, models=[]),
                        dict(value, promoted=True), dict(value, extra=0)):
            with self.assertRaises(ValueError):
                compare_report(value, altered)

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
