#!/usr/bin/env python3
"""Bounded, offline admission of declared Rust classification records; never runs source."""
import argparse
import hashlib
import json
import re
from pathlib import Path

LIMIT = 4_194_304
SPLITS = ('train', 'validation', 'test')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def admit(data, approved_licenses):
    if not data or len(data) > LIMIT:
        raise ValueError('input size')
    records, seen, projects = [], set(), {}
    labels = {split: set() for split in SPLITS}
    total = 0
    lines = data.split(b'\n')
    if lines[-1:] == [b'']:
        lines.pop()
    if not lines:
        raise ValueError('record count or empty line')
    for raw_line in lines:
        if len(records) >= 4096 or not raw_line:
            raise ValueError('record count or empty line')
        r = json.loads(raw_line.decode('utf-8'), object_pairs_hook=unique_object)
        if not isinstance(r, dict):
            raise ValueError('record object required')
        for field in ('project', 'revision', 'license', 'classification', 'compiler', 'provenance', 'split', 'sha256'):
            if not isinstance(r.get(field), str) or not 1 <= len(r[field]) <= 512:
                raise ValueError(f'missing/bounded metadata: {field}')
        if not re.fullmatch(r'[A-Za-z0-9_./-]{1,128}', r['project']):
            raise ValueError('project identity')
        if not re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', r['revision']):
            raise ValueError('immutable revision required')
        if r['license'] not in approved_licenses or r['classification'] not in ('Public', 'Internal'):
            raise ValueError('license or data classification not admitted')
        if not r['compiler'].startswith('rustc ') or type(r.get('label')) is not int or r['label'] not in (0, 1):
            raise ValueError('compiler declaration or binary label')
        if not isinstance(r.get('source'), str):
            raise ValueError('source string required')
        source = r['source'].encode('utf-8')
        total += len(source)
        if not 1 <= len(source) <= 16384 or total > 1_048_576:
            raise ValueError('source capacity')
        digest = hashlib.sha256(source).hexdigest()
        if digest != r['sha256'] or digest in seen:
            raise ValueError('content digest mismatch or duplicate source')
        seen.add(digest)
        split = r['split']
        if split not in SPLITS or projects.setdefault(r['project'], split) != split:
            raise ValueError('invalid split or project leakage')
        labels[split].add(r['label'])
        records.append(r)
    if any(value != {0, 1} for value in labels.values()):
        raise ValueError('each split must contain both labels')
    wire = 'CRUST001\n' + ''.join(
        f"{r['split']}\t{r['project']}\t{r['label']}\t{r['sha256']}\t{r['source'].encode().hex()}\n"
        for r in records)
    return records, wire.encode('ascii')


def provenance_bytes(records):
    return ''.join(
        json.dumps(record, sort_keys=True) + '\n' for record in records
    ).encode('utf-8')


def build_manifest(data, wire, provenance, records, approved_licenses):
    return {
        'schema': 1,
        'input_sha256': hashlib.sha256(data).hexdigest(),
        'corpus_sha256': hashlib.sha256(wire).hexdigest(),
        'provenance_sha256': hashlib.sha256(provenance).hexdigest(),
        'records': len(records),
        'approved_licenses': sorted(set(approved_licenses)),
        'counts': {
            split: sum(record['split'] == split for record in records)
            for split in SPLITS
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path, help='new directory')
    parser.add_argument('--license', action='append', required=True, dest='licenses')
    args = parser.parse_args()
    with args.input.open('rb') as stream:
        data = stream.read(LIMIT + 1)
    records, wire = admit(data, set(args.licenses))
    args.output.mkdir()
    (args.output / 'corpus.crust').write_bytes(wire)
    provenance = provenance_bytes(records)
    (args.output / 'provenance.jsonl').write_bytes(provenance)
    manifest = build_manifest(data, wire, provenance, records, args.licenses)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, sort_keys=True))


if __name__ == '__main__':
    main()
