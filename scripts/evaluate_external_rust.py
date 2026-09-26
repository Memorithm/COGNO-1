#!/usr/bin/env python3
"""Evaluate the fixed external panel with frozen checkpoints, preserving refusals and every seed."""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
from check_bpe_evidence import check_checkpoints, checkpoint_rows, CHECKPOINT_REFERENCE
from prepare_external_rust import prepare, checked_panel


def summarize(rows, expected):
    seen = set()
    correct = accepted = 0
    confusion = {k: 0 for k in ('tn', 'fp', 'fn', 'tp')}
    accepted_labels = []
    for row in rows:
        identity = row['source_sha256']
        if identity in seen or identity not in expected or row['target'] != str(expected[identity]):
            raise ValueError('missing, duplicate, unexpected or mislabelled row')
        seen.add(identity)
        if row['status'] == 'capacity':
            if any(row[k] for k in ('prediction', 'p_compile', 'tokens')):
                raise ValueError('refused example has a prediction')
            continue
        if row['status'] != 'accepted' or row['prediction'] not in ('0', '1'):
            raise ValueError('invalid prediction status')
        p = float(row['p_compile'])
        if not math.isfinite(p) or not 0 <= p <= 1 or not 2 <= int(row['tokens']) <= 128:
            raise ValueError('invalid probability or context')
        target, prediction = int(row['target']), int(row['prediction'])
        accepted += 1
        correct += target == prediction
        accepted_labels.append(target)
        confusion[('tn', 'fp', 'fn', 'tp')[target*2 + prediction]] += 1
    if seen != expected.keys():
        raise ValueError('incomplete evaluation')
    majority = max(sum(v == label for v in expected.values()) for label in (0, 1))
    return dict(total=len(expected), accepted=accepted, refused=len(expected)-accepted, correct=correct,
                accuracy_on_accepted=correct/accepted if accepted else None, confusion=confusion,
                majority_correct_on_full_panel=majority,
                majority_correct_on_accepted=max((accepted_labels.count(i) for i in (0, 1)), default=0))


def evaluate(executable, checkpoints, output, rustc):
    check_checkpoints(checkpoints)
    output.mkdir()
    manifest = prepare(output / 'corpus', rustc)
    expected = {c['source_sha256']: c['expected_label'] for c in checked_panel()['cases']}
    reports = {}
    for seed, item in checkpoint_rows(CHECKPOINT_REFERENCE).items():
        result = subprocess.run([str(executable), str(checkpoints / item['file']), item['sha256'],
                                 str(output / 'corpus/corpus.crust'), manifest['corpus_sha256']],
                                capture_output=True, text=True, check=True)
        reader = csv.DictReader(result.stdout.splitlines())
        if reader.fieldnames != ['source_sha256', 'target', 'status', 'prediction', 'p_compile', 'tokens']:
            raise ValueError('invalid evaluator CSV schema')
        reports[seed] = dict(summarize(list(reader), expected), checkpoint_sha256=item['sha256'])
        (output / f'seed-{seed}.csv').write_text(result.stdout)
    report = dict(corpus_sha256=manifest['corpus_sha256'], project='rust-lang/rust', models=reports,
                  training_performed=False, representative_benchmark=False)
    (output / 'results.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rustc', type=Path)
    args = parser.parse_args()
    rustc = args.rustc or Path(subprocess.run(['rustup', 'which', '--toolchain', '1.97.1', 'rustc'], capture_output=True, text=True, check=True).stdout.strip())
    print(json.dumps(evaluate(args.executable.resolve(), args.checkpoints.resolve(), args.output, rustc), sort_keys=True))
