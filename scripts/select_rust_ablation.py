#!/usr/bin/env python3
"""Select a configuration using validation NLL only; never select a seed."""
import csv
import itertools
import json
import math
import sys

CONFIGS = tuple(itertools.product((128, 256), ('ordered', 'shuffled')))
SEEDS = (1, 7, 42)


def select(rows):
    groups = {(c, o, s): {} for c, o in CONFIGS for s in SEEDS}
    for row in rows:
        if row['split'] != 'validation':
            continue
        key = int(row['context']), row['order'], int(row['seed'])
        if key not in groups:
            raise ValueError('unexpected configuration')
        digest, target = row['source_sha256'], int(row['target'])
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid source identity')
        p = float(row['p_compile'])
        if target not in (0, 1) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('invalid probability or label')
        if digest in groups[key]:
            raise ValueError('duplicate validation source')
        groups[key][digest] = (target, p)
    reference = {h: t for h, (t, _) in groups[(128, 'ordered', 1)].items()}
    if len(reference) != 4 or set(reference.values()) != {0, 1}:
        raise ValueError('fixed diagnostic requires four validation rows and both labels')
    scores = []
    for context, order in CONFIGS:
        losses = []
        for seed in SEEDS:
            group = groups[context, order, seed]
            if {h: t for h, (t, _) in group.items()} != reference:
                raise ValueError('missing or inconsistent validation rows')
            for target, p in group.values():
                p = min(1 - 1e-7, max(1e-7, p))
                losses.append(-math.log(p if target else 1 - p))
        scores.append(dict(context=context, order=order, validation_nll=sum(losses)/len(losses)))
    best = min(scores, key=lambda x: (x['validation_nll'], x['context'], x['order']))
    return dict(selection=best, seeds=list(SEEDS), scores=scores,
                limitation='Exploratory previously observed diagnostic; no expert or generalization claim')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: select_rust_ablation.py PREDICTIONS_CSV')
    with open(sys.argv[1], newline='') as f:
        print(json.dumps(select(csv.DictReader(f)), indent=2, allow_nan=False))
