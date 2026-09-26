#!/usr/bin/env python3
"""Fresh-process classification regression on reused diagnostic data, not fresh quality evidence."""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
from prepare_rust_diagnostic import prepare
from check_bpe_evidence import CHECKPOINT_REFERENCE, checkpoint_rows, check_checkpoints

ROOT = Path(__file__).resolve().parents[1]
CORPUS_HASH = '766f1df55b2b3befb8c55cf4bd7882f61ee900972a8ffd81b0f3679abb0495ee'


def check_result(rows, expected):
    seen = set()
    for row in rows:
        identity = row['source_sha256']
        if identity in seen or identity not in expected:
            raise ValueError('duplicate or unexpected evaluated source')
        seen.add(identity)
        ref = expected[identity]
        if any(row[field] != ref[field] for field in ('split', 'project', 'target', 'prediction')):
            raise ValueError('evaluation categorical mismatch')
        p = float(row['p_compile'])
        if not math.isfinite(p) or abs(p - float(ref['p_compile'])) > 1e-6:
            raise ValueError('evaluation probability drift')
    if seen != expected.keys():
        raise ValueError('missing evaluation rows')


def verify(examples, checkpoints, output):
    check_checkpoints(checkpoints)
    output.mkdir()
    corpus_dir = output / 'corpus'
    manifest = prepare(corpus_dir)
    if manifest['corpus_sha256'] != CORPUS_HASH:
        raise ValueError('diagnostic corpus identity drift')
    coverage = subprocess.run([str(examples / 'rust_corpus_coverage'), str(corpus_dir / 'corpus.crust'),
                               CORPUS_HASH], capture_output=True, text=True, check=True)
    if coverage.stdout != (ROOT / 'experiments/rust-corpus-pipeline/coverage.csv').read_text():
        raise ValueError('coverage drift')
    (output / 'coverage.csv').write_text(coverage.stdout)
    source = [json.loads(line) for line in (corpus_dir / 'provenance.jsonl').read_text().splitlines()]
    identities = {(r['split'], r['family'], str(r['label'])): r for r in source}
    with (ROOT / 'experiments/bpe-rust-pilot/predictions.csv').open() as stream:
        predictions = list(csv.DictReader(stream))
    checked = 0
    rejected = 0
    for seed, item in checkpoint_rows(CHECKPOINT_REFERENCE).items():
        for split in ('train', 'validation', 'test'):
            expected = {}
            for ref in predictions:
                if ref['arm'] == 'bpe' and ref['stage'] == 'trained' and ref['seed'] == seed and ref['split'] == split:
                    record = identities[(split, ref['family'], ref['target'])]
                    expected[record['sha256']] = dict(ref, project=record['project'])
            command = [str(examples / 'bpe_corpus_eval'), str(checkpoints / item['file']), item['sha256'],
                       str(corpus_dir / 'corpus.crust'), CORPUS_HASH, split]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            rows = list(csv.DictReader(result.stdout.splitlines()))
            check_result(rows, expected)
            (output / f'seed-{seed}-{split}.csv').write_text(result.stdout)
            checked += len(rows)
            if seed == '1' and split == 'test':
                for index in (2, 4):
                    bad = command.copy()
                    bad[index] = '0' * 64
                    refused = subprocess.run(bad, capture_output=True, text=True)
                    if refused.returncode == 0 or refused.stdout:
                        raise ValueError('wrong digest did not fail before output')
                    rejected += 1
    report = {'corpus_sha256': CORPUS_HASH, 'processes': 9, 'predictions_checked': checked,
              'wrong_digest_requests_rejected': rejected,
              'fresh_holdout': False, 'training_performed_by_evaluator': False}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples-dir', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.examples_dir.resolve(), args.checkpoints.resolve(), args.output)
