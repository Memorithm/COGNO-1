#!/usr/bin/env python3
"""Validate complete 12-model evidence before summarizing the fixed domain experiment."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
from prepare_domain_holdout import CORPUS_SHA256, DOMAIN_SPLITS
from select_curriculum_probe import ARMS, SEEDS, select


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def summarize(rows, corpus):
    if hashlib.sha256(corpus).hexdigest() != CORPUS_SHA256:
        raise ValueError('fixed corpus hash mismatch')
    expected = {}
    for line in corpus.decode('ascii').splitlines()[1:]:
        split, project, label, digest, _ = line.split('\t')
        domain = project.split('/')[2]
        if DOMAIN_SPLITS[domain] != split:
            raise ValueError('domain leakage')
        expected[digest] = (split, int(label), domain)
    if len(rows) != 12 * 288:
        raise ValueError('complete twelve-model prediction inventory required')
    seen, groups, domains = set(), collections.defaultdict(list), collections.defaultdict(list)
    for row in rows:
        arm, seed, digest = row['arm'], int(row['seed']), row['source_sha256']
        key = (arm, seed, digest)
        if arm not in ARMS or seed not in SEEDS or digest not in expected or key in seen:
            raise ValueError('unexpected or duplicate observation')
        seen.add(key)
        split, label, domain = expected[digest]
        probability, prediction = float(row['p_compile']), int(row['prediction'])
        if (row['split'] != split or int(row['target']) != label or prediction not in (0, 1)
                or not math.isfinite(probability) or not 0 <= probability <= 1
                or not 2 <= int(row['tokens']) <= 512):
            raise ValueError('prediction metadata or probability mismatch')
        probability = min(1-1e-7, max(1e-7, probability))
        item = (int(prediction == label), -math.log(probability if label else 1-probability))
        groups[arm, seed, split].append(item)
        domains[arm, seed, domain].append(item)
    def metric(items):
        return dict(count=len(items), correct=sum(x[0] for x in items),
                    accuracy=sum(x[0] for x in items)/len(items), nll=sum(x[1] for x in items)/len(items))
    details = [dict(arm=a, seed=s, split=p, **metric(v)) for (a, s, p), v in sorted(groups.items())]
    domain_metrics = [dict(arm=a, seed=s, domain=d, split=DOMAIN_SPLITS[d], **metric(v))
                      for (a, s, d), v in sorted(domains.items())]
    means = []
    for arm in ARMS:
        item = dict(arm=arm)
        for split in ('train', 'validation', 'test'):
            parts = [r for r in details if r['arm'] == arm and r['split'] == split]
            item[split + '_accuracy'] = sum(r['accuracy'] for r in parts)/3
            item[split + '_nll'] = sum(r['nll'] for r in parts)/3
        means.append(item)
    return dict(schema=1, corpus_sha256=CORPUS_SHA256, predictions=len(rows),
                selection=select(rows, 48), arm_means=means, per_seed=details,
                per_domain=domain_metrics, uniform_nll=math.log(2),
                unique_test_sources=48, synthetic_domains=True, model_promoted=False)


def inspect(directory, corpus):
    inventory = read_csv(directory / 'checkpoints.csv')
    if len(inventory) != 12:
        raise ValueError('twelve checkpoints required')
    keys = set()
    for row in inventory:
        arm, seed = row['arm'], int(row['seed'])
        filename = f'{arm}-seed-{seed}.cbpc'
        if arm not in ARMS or seed not in SEEDS or (arm, seed) in keys or row['file'] != filename:
            raise ValueError('checkpoint inventory mismatch')
        keys.add((arm, seed))
        if row['corpus_sha256'] != CORPUS_SHA256:
            raise ValueError('checkpoint corpus mismatch')
        if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('checkpoint bytes mismatch')
    report = summarize(read_csv(directory / 'predictions.csv'), corpus)
    report['checkpoints'] = inventory
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('corpus', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.directory, args.corpus.read_bytes()), indent=2, sort_keys=True))
