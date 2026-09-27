#!/usr/bin/env python3
"""Recertify 288 known synthetic cases, keeping whole domains in one split."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from admit_rust_corpus import write_admitted

ROOT = Path(__file__).resolve().parents[1]
CORPUS_SHA256 = 'fa327c59f97f46b5ddbf8d120e0283ce7fbb80085a31fcca0d7b9a004cc5fe1f'
GROUPS_SHA256 = 'ea343f65801e0da63b6de455bf9731fca8586c3d1c39b63ce1aab7675fd8c606'
DOMAIN_SPLITS = dict.fromkeys(('ownership', 'borrowing', 'traits', 'lifetimes',
    'pattern_matching', 'closures', 'iterators', 'error_handling'), 'train')
DOMAIN_SPLITS.update(smart_pointers='validation', async_futures='validation',
                     macros='test', concurrency='test')


def group_mapping(records):
    projects = {}
    counts = {domain: 0 for domain in DOMAIN_SPLITS}
    for row in records:
        parts = row['project'].split('/')
        if len(parts) != 4 or parts[:2] != ['synthetic', 'curriculum'] or parts[2] not in DOMAIN_SPLITS:
            raise ValueError('unexpected synthetic project identity')
        domain = parts[2]
        counts[domain] += 1
        projects[row['project']] = domain
    if set(counts.values()) != {24}:
        raise ValueError('exactly 24 rows per reviewed domain required')
    return ('project\tgroup\tsplit\n' + ''.join(
        f'{project}\t{domain}\t{DOMAIN_SPLITS[domain]}\n'
        for project, domain in sorted(projects.items()))).encode('ascii')


def prepare(output, executable, rustc):
    output.mkdir()
    records, evidence = [], {}
    with tempfile.TemporaryDirectory(prefix='cogno-domain-holdout-') as temp:
        for domain in DOMAIN_SPLITS:
            spec = importlib.util.spec_from_file_location('domain_' + domain,
                ROOT / 'experiments/rust-curriculum' / domain / 'verify.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            directory = Path(temp) / domain
            evidence[domain] = module.prepare(directory, rustc)
            records.extend(json.loads(line) for line in (directory / 'corpus.jsonl').read_text().splitlines())
    mapping = group_mapping(records)
    data = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in records).encode()
    original = write_admitted(data, output / 'original', ['MIT'])
    (output / 'groups.tsv').write_bytes(mapping)
    group_hash = hashlib.sha256(mapping).hexdigest()
    if group_hash != GROUPS_SHA256:
        raise ValueError('fixed grouping inventory changed')
    subprocess.run([str(executable), str(output / 'original/corpus.crust'), original['corpus_sha256'],
                    str(output / 'groups.tsv'), group_hash, str(output / 'grouped')], check=True)
    wire = (output / 'grouped/corpus.crust').read_bytes()
    if hashlib.sha256(wire).hexdigest() != CORPUS_SHA256:
        raise ValueError('fixed grouped corpus changed')
    counts = {split: sum(line.startswith((split + '\t').encode()) for line in wire.splitlines())
              for split in ('train', 'validation', 'test')}
    if counts != dict(train=192, validation=48, test=48):
        raise ValueError('fixed domain split counts differ')
    manifest = dict(schema=1, campaign='domain-holdout-v1', records=288, counts=counts,
                    domains=DOMAIN_SPLITS, original=original, compiler_admissions=evidence,
                    groups_sha256=group_hash, corpus_sha256=hashlib.sha256(wire).hexdigest(),
                    independent_upstream_projects=False, previously_published_sources=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--split-executable', type=Path, required=True)
    p.add_argument('--rustc', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.output.resolve(), a.split_executable.resolve(), a.rustc.resolve()), sort_keys=True))
