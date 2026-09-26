#!/usr/bin/env python3
"""Reproduce every fixed ablation arm and selected external regression, offline."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from prepare_rust_diagnostic import prepare as prepare_diagnostic
from prepare_external_rust import prepare as prepare_external
from select_rust_ablation import select, SEEDS
from verify_external_rust import compare_predictions

REFERENCE = Path(__file__).resolve().parents[1] / 'experiments/rust-training-ablation'


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def compare_diagnostic(actual, expected):
    def index(rows):
        indexed = {}
        for row in rows:
            key = tuple(row[k] for k in ('context', 'order', 'seed', 'split', 'source_sha256'))
            if key in indexed:
                raise ValueError('duplicate diagnostic row')
            indexed[key] = row
        return indexed
    actual, expected = index(actual), index(expected)
    if actual.keys() != expected.keys():
        raise ValueError('diagnostic inventory changed')
    for key, row in actual.items():
        ref = expected[key]
        if row.keys() != ref.keys() or any(row[k] != ref[k] for k in ref if k != 'p_compile'):
            raise ValueError('diagnostic categorical drift')
        p = float(row['p_compile'])
        if not math.isfinite(p) or not 0 <= p <= 1 or abs(p-float(ref['p_compile'])) > 1e-6:
            raise ValueError('diagnostic probability drift')


def verify(examples, output, rustc):
    output.mkdir()
    corpus = prepare_diagnostic(output / 'diagnostic')
    models = output / 'models'
    subprocess.run([str(examples / 'bpe_training_ablation'),
                    str(output / 'diagnostic/corpus.crust'), corpus['corpus_sha256'], str(models)], check=True)
    inventory = read_csv(models / 'checkpoints.csv')
    if inventory != read_csv(REFERENCE / 'checkpoints.csv'):
        raise ValueError('checkpoint inventory differs from frozen evidence')
    for row in inventory:
        if hashlib.sha256((models / row['file']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('checkpoint file identity differs')
    predictions = read_csv(models / 'predictions.csv')
    compare_diagnostic(predictions, read_csv(REFERENCE / 'predictions.csv'))
    selection = select(predictions)
    reference = json.loads((REFERENCE / 'selection.json').read_text())
    chosen = selection['selection']
    if (chosen['context'], chosen['order']) != (reference['selection']['context'], reference['selection']['order']):
        raise ValueError('validation-selected configuration changed')
    for a, b in zip(selection['scores'], reference['scores'], strict=True):
        if (a['context'], a['order']) != (b['context'], b['order']) or abs(a['validation_nll']-b['validation_nll']) > 1e-6:
            raise ValueError('validation score drift')
    # Freeze selection before opening the external regression corpus.
    (output / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    panel = prepare_external(output / 'external', rustc)
    selected = [r for r in inventory if int(r['context']) == chosen['context'] and r['order'] == chosen['order']]
    if sorted(int(r['seed']) for r in selected) != sorted(SEEDS):
        raise ValueError('selected seed inventory incomplete')
    for row in selected:
        result = subprocess.run([str(examples / 'bpe_external_eval'), str(models / row['file']), row['sha256'],
                                 str(output / 'external/corpus.crust'), panel['corpus_sha256']],
                                capture_output=True, text=True, check=True)
        name = f"external-seed-{row['seed']}.csv"
        compare_predictions(list(csv.DictReader(result.stdout.splitlines())), read_csv(REFERENCE / name))
        (output / name).write_text(result.stdout)
    report = dict(checkpoints_verified=len(inventory), diagnostic_predictions_verified=len(predictions),
                  selected_seeds_verified=len(selected), external_predictions_verified=8*len(selected),
                  model_promoted=False, expert_capability_demonstrated=False)
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--examples-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rustc', type=Path, required=True)
    a = p.parse_args()
    verify(a.examples_dir.resolve(), a.output.resolve(), a.rustc.resolve())
