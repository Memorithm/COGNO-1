#!/usr/bin/env python3
"""Rebuild every model and compare the entire frozen domain campaign."""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
from prepare_domain_holdout import prepare, CORPUS_SHA256
from select_curriculum_probe import ARMS
from summarize_domain_holdout import inspect, read_csv

REFERENCE = Path(__file__).resolve().parents[1] / 'experiments/domain-holdout-v1'


def compare_report(actual, expected):
    if type(actual) is not type(expected):
        raise ValueError('report type mismatch')
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise ValueError('report fields mismatch')
        for key in actual:
            compare_report(actual[key], expected[key])
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            raise ValueError('report length mismatch')
        for a, b in zip(actual, expected):
            compare_report(a, b)
    elif isinstance(actual, float):
        if not math.isfinite(actual) or not math.isfinite(expected) or abs(actual-expected) > 1e-12:
            raise ValueError('report metric mismatch')
    elif actual != expected:
        raise ValueError('report value mismatch')


def verify(output, splitter, runner, rustc):
    output.mkdir()
    prepare(output / 'corpus', splitter, rustc)
    models = output / 'models'
    corpus = output / 'corpus/grouped/corpus.crust'
    subprocess.run([str(runner), str(corpus), CORPUS_SHA256, str(models), '--domain-holdout-v1'], check=True)
    report = inspect(models, corpus.read_bytes())
    compare_report(report, json.loads((REFERENCE / 'report.json').read_text()))
    if read_csv(models / 'checkpoints.csv') != read_csv(REFERENCE / 'checkpoints.csv'):
        raise ValueError('checkpoint inventory drift')
    expected = [row for arm in ARMS for row in read_csv(REFERENCE / f'predictions-{arm}.csv')]
    if read_csv(models / 'predictions.csv') != expected:
        raise ValueError('prediction evidence drift')
    if read_csv(models / 'exposure.csv') != read_csv(REFERENCE / 'exposure.csv'):
        raise ValueError('exposure evidence drift')
    print(json.dumps(dict(checkpoints=12, predictions=3456, compiled_sources=288,
                         selected_arm=report['selection']['selection']['arm'], promoted=False)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--split-executable', type=Path, required=True)
    p.add_argument('--runner', type=Path, required=True)
    p.add_argument('--rustc', type=Path, required=True)
    a = p.parse_args()
    verify(a.output.resolve(), a.split_executable.resolve(), a.runner.resolve(), a.rustc.resolve())
