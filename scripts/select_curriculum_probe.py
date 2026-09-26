#!/usr/bin/env python3
"""Fixed validation-only curriculum arm selection; retain every seed."""
import csv
import json
import math
import sys

ARMS = ('full', 'byte_mix', 'half_mix', 'cycle_mix')
SEEDS = (1, 7, 42)


def select(rows):
    groups = {(a, s): {} for a in ARMS for s in SEEDS}
    for row in rows:
        if row['split'] != 'validation':
            continue
        key = row['arm'], int(row['seed'])
        if key not in groups:
            raise ValueError('unexpected arm or seed')
        digest = row['source_sha256']
        p, y = float(row['p_compile']), int(row['target'])
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid source identity')
        if digest in groups[key] or y not in (0, 1) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('duplicate source or invalid observation')
        groups[key][digest] = y, p
    reference = {h: y for h, (y, _) in groups['full', 1].items()}
    if len(reference) != 16 or set(reference.values()) != {0, 1}:
        raise ValueError('sixteen validation rows and both labels required')
    scores = []
    for arm in ARMS:
        losses = []
        for seed in SEEDS:
            group = groups[arm, seed]
            if {h: y for h, (y, _) in group.items()} != reference:
                raise ValueError('incomplete or inconsistent validation set')
            for y, p in group.values():
                p = min(1-1e-7, max(1e-7, p))
                losses.append(-math.log(p if y else 1-p))
        scores.append(dict(arm=arm, validation_nll=sum(losses)/len(losses)))
    best = min(scores, key=lambda x: (x['validation_nll'], ARMS.index(x['arm'])))
    return dict(selection=best, scores=scores, seeds=list(SEEDS), expert_claim=False)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: select_curriculum_probe.py PREDICTIONS_CSV')
    with open(sys.argv[1], newline='') as f:
        print(json.dumps(select(csv.DictReader(f)), indent=2, allow_nan=False))
