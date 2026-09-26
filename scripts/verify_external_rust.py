#!/usr/bin/env python3
"""Reproduce frozen external-panel evidence offline; no quality promotion."""
import argparse
import csv
import json
import math
import hashlib
from pathlib import Path
import subprocess
from evaluate_external_rust import evaluate
from prepare_external_rust import PANEL, checked_panel
from prepare_rust_diagnostic import SOURCE, SOURCE_HASH
from check_bpe_evidence import CHECKPOINT_REFERENCE, checkpoint_rows


def compare_predictions(actual, expected):
    def index(rows):
        result = {}
        for row in rows:
            key = row['source_sha256']
            if key in result:
                raise ValueError('duplicate source prediction')
            result[key] = row
        return result
    actual, expected = index(actual), index(expected)
    if actual.keys() != expected.keys():
        raise ValueError('external prediction set differs')
    for key, row in actual.items():
        ref = expected[key]
        if row.keys() != ref.keys() or any(row[k] != ref[k] for k in ref if k != 'p_compile'):
            raise ValueError('external categorical/coverage drift')
        if ref['status'] == 'capacity':
            if row['p_compile']:
                raise ValueError('refusal must not contain probability')
        else:
            probability = float(row['p_compile'])
            if not math.isfinite(probability) or abs(probability-float(ref['p_compile'])) > 1e-6:
                raise ValueError('external probability drift')


def verify(executable, checkpoints, output, rustc):
    diagnostic = SOURCE.read_bytes()
    if hashlib.sha256(diagnostic).hexdigest() != SOURCE_HASH:
        raise ValueError('original model diagnostic corpus changed')
    prior_hashes = {json.loads(line)['sha256'] for line in diagnostic.split(b'\n') if line}
    if any(c[k] in prior_hashes for c in checked_panel()['cases'] for k in ('raw_sha256', 'source_sha256')):
        raise ValueError('exact source overlap with original model corpus')
    report = evaluate(executable, checkpoints, output, rustc)
    if report != json.loads((PANEL / 'results.json').read_text()):
        raise ValueError('external aggregate evidence drift')
    if json.loads((output / 'corpus/compiler-results.json').read_text()) != json.loads((PANEL / 'compiler-results.json').read_text()):
        raise ValueError('external compiler evidence drift')
    for seed in ('1', '7', '42'):
        with (output / f'seed-{seed}.csv').open() as actual, (PANEL / f'seed-{seed}.csv').open() as expected:
            compare_predictions(list(csv.DictReader(actual)), list(csv.DictReader(expected)))
    first = checkpoint_rows(CHECKPOINT_REFERENCE)['1']
    command = [str(executable), str(checkpoints / first['file']), first['sha256'],
               str(output / 'corpus/corpus.crust'), report['corpus_sha256']]
    for position in (2, 4):
        bad = command.copy()
        bad[position] = '0'*64
        result = subprocess.run(bad, capture_output=True, text=True)
        if result.returncode == 0 or result.stdout:
            raise ValueError('wrong digest accepted or emitted predictions')
    verification = dict(rows_verified=24, compiler_pairs_verified=8, wrong_digest_requests_rejected=2,
                        exact_source_overlap=0, program_execution=False, training=False, model_quality_improved=False)
    (output / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps(verification, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rustc', type=Path, required=True)
    args = parser.parse_args()
    verify(args.executable.resolve(), args.checkpoints.resolve(), args.output, args.rustc)
