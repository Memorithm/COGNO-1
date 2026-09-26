#!/usr/bin/env python3
"""Check frozen pilot reproduction and checkpoint inventory, not model quality."""
import argparse
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'experiments/bpe-rust-pilot/predictions.csv'
CHECKPOINT_REFERENCE = ROOT / 'experiments/bpe-rust-pilot/checkpoints.tsv'
FIELDS = 'arm,seed,stage,split,family,target,prediction,p_compile,tokens,parameters'.split(',')
KEY = ('arm', 'seed', 'stage', 'split', 'family', 'target')


def bounded_read(path, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'not a regular non-symlink file: {path}')
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('evidence exceeds size bound')
    return data


def prediction_rows(path):
    reader = csv.DictReader(io.StringIO(bounded_read(path, 1_048_576).decode('utf-8')))
    if reader.fieldnames != FIELDS:
        raise ValueError('prediction columns differ')
    rows = {}
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError('malformed row')
        key = tuple(row[k] for k in KEY)
        if key in rows:
            raise ValueError('duplicate prediction')
        probability = float(row['p_compile'])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('invalid probability')
        rows[key] = row
    return rows


def check_predictions(path):
    expected = prediction_rows(REFERENCE)
    actual = prediction_rows(path)
    if actual.keys() != expected.keys():
        raise ValueError('missing or unexpected examples/seeds/arms/stages')
    summary = {}
    for key, row in actual.items():
        ref = expected[key]
        for field in ('prediction', 'tokens', 'parameters'):
            if row[field] != ref[field]:
                raise ValueError(f'{field} mismatch for {key}')
        if abs(float(row['p_compile']) - float(ref['p_compile'])) > 1e-6:
            raise ValueError(f'probability drift for {key}')
        if row['stage'] == 'trained':
            group = '/'.join(row[k] for k in ('arm', 'seed', 'split'))
            counts = summary.setdefault(group, {'correct': 0, 'total': 0, 'tokens': 0})
            counts['correct'] += row['prediction'] == row['target']
            counts['total'] += 1
            counts['tokens'] += int(row['tokens'])
    return summary


def checkpoint_rows(path):
    reader = csv.DictReader(
        io.StringIO(bounded_read(path, 16384).decode('utf-8')),
        delimiter='\t',
    )
    if reader.fieldnames != ['seed', 'file', 'bytes', 'sha256']:
        raise ValueError('invalid checkpoint inventory columns')
    rows = {}
    for row in reader:
        seed = row['seed']
        if seed not in {'1', '7', '42'} or seed in rows:
            raise ValueError('invalid or duplicate checkpoint seed')
        if row['file'] != f'bpe-seed-{seed}.cbpc':
            raise ValueError('unexpected checkpoint filename')
        if not row['bytes'].isdigit() or int(row['bytes']) > 2_097_152:
            raise ValueError('invalid checkpoint size')
        if not re.fullmatch('[0-9a-f]{64}', row['sha256']):
            raise ValueError('invalid checkpoint digest')
        rows[seed] = row
    if rows.keys() != {'1', '7', '42'}:
        raise ValueError('incomplete checkpoint inventory')
    return rows


def check_checkpoints(directory, reference=CHECKPOINT_REFERENCE):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('invalid checkpoint directory')
    expected = checkpoint_rows(reference)
    actual = checkpoint_rows(directory / 'checkpoints.tsv')
    if actual != expected:
        raise ValueError('checkpoint inventory differs from frozen evidence')
    for seed, row in actual.items():
        data = bounded_read(directory / row['file'], 2_097_152)
        if int(row['bytes']) != len(data) or data[:8] != b'CBPC0001':
            raise ValueError('checkpoint size/schema mismatch')
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('checkpoint digest mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--checkpoints', type=Path)
    args = parser.parse_args()
    summary = check_predictions(args.predictions)
    if args.checkpoints:
        check_checkpoints(args.checkpoints)
    print(json.dumps({'reproduced': True, 'trained_counts': summary}, sort_keys=True))


if __name__ == '__main__':
    main()
