#!/usr/bin/env python3
"""Reproduce the fixed prefix campaign; no promotion from successful reproduction."""
import argparse
import csv
import collections
import hashlib
import json
import math
from pathlib import Path
import subprocess
from prepare_rust_curriculum import prepare
from select_curriculum_probe import select, ARMS

REFERENCE = Path(__file__).resolve().parents[1] / 'experiments/curriculum-prefix-campaign'
CORPUS_SHA = '9d7ed2ef18a30f760696dff926c1c2a2684446d12905c9a205018b6a95f50376'


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def compare(actual, expected):
    def indexed(data):
        result = {}
        for r in data:
            key = tuple(r[k] for k in ('arm', 'seed', 'split', 'source_sha256'))
            if key in result:
                raise ValueError('duplicate prediction')
            result[key] = r
        return result
    actual, expected = indexed(actual), indexed(expected)
    if actual.keys() != expected.keys():
        raise ValueError('prediction inventory drift')
    for key, r in actual.items():
        e = expected[key]
        if r.keys() != e.keys() or any(r[k] != e[k] for k in e if k != 'p_compile'):
            raise ValueError('categorical prediction drift')
        p = float(r['p_compile'])
        if not math.isfinite(p) or not 0 <= p <= 1 or abs(p-float(e['p_compile'])) > 1e-6:
            raise ValueError('probability drift')


def verify(executable, output, rustc):
    output.mkdir()
    manifest = prepare(output / 'corpus', rustc)
    if manifest['corpus_sha256'] != CORPUS_SHA:
        raise ValueError('fixed campaign corpus drift')
    models = output / 'models'
    subprocess.run([str(executable), str(output / 'corpus/corpus.crust'), CORPUS_SHA, str(models)], check=True)
    inventory = rows(models / 'checkpoints.csv')
    if inventory != rows(REFERENCE / 'checkpoints.csv') or len(inventory) != 12:
        raise ValueError('checkpoint inventory drift')
    for r in inventory:
        if hashlib.sha256((models / r['file']).read_bytes()).hexdigest() != r['sha256']:
            raise ValueError('checkpoint bytes drift')
    predicted = rows(models / 'predictions.csv')
    expected = [r for arm in ARMS for r in rows(REFERENCE / f'predictions-{arm}.csv')]
    compare(predicted, expected)
    groups = collections.defaultdict(list)
    for row in predicted:
        groups[row['arm'], row['seed'], row['split']].append(row)
    measured = {}
    for (arm, seed, split), data in groups.items():
        counts = [0]*4
        for row in data:
            counts[2*int(row['target'])+int(row['prediction'])] += 1
        nll = sum(-math.log(min(1-1e-7, max(1e-7, float(row['p_compile']) if row['target'] == '1'
                                              else 1-float(row['p_compile'])))) for row in data)/len(data)
        measured.setdefault(arm, {}).setdefault(seed, {})[split] = dict(
            count=len(data), correct=counts[0]+counts[3],
            confusion=dict(zip(('tn', 'fp', 'fn', 'tp'), counts)), nll=nll)
    if measured != json.loads((REFERENCE / 'metrics.json').read_text()):
        raise ValueError('aggregate metrics drift')
    selected = select(predicted)
    golden = json.loads((REFERENCE / 'selection.json').read_text())
    if selected['selection']['arm'] != golden['selection']['arm']:
        raise ValueError('selection drift')
    for a, b in zip(selected['scores'], golden['scores'], strict=True):
        if a['arm'] != b['arm'] or abs(a['validation_nll']-b['validation_nll']) > 1e-6:
            raise ValueError('validation NLL drift')
    if rows(models / 'exposure.csv') != rows(REFERENCE / 'exposure.csv'):
        raise ValueError('training exposure drift')
    report = dict(checkpoints=12, predictions=len(predicted), compiled_sources=96,
                  selected_arm=selected['selection']['arm'], promoted=False)
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rustc', type=Path, required=True)
    a = p.parse_args()
    verify(a.executable.resolve(), a.output.resolve(), a.rustc.resolve())
