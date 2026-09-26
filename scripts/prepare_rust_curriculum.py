#!/usr/bin/env python3
"""Compile and join the four fixed original curriculum domains; no new holdout claim."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
from admit_rust_corpus import write_admitted

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ('ownership', 'borrowing', 'traits', 'lifetimes')


def prepare(output, rustc=None):
    if output.exists():
        raise ValueError('output must be new')
    records = []
    evidence = {}
    with tempfile.TemporaryDirectory(prefix='cogno-curriculum-') as temp:
        for domain in DOMAINS:
            spec = importlib.util.spec_from_file_location('curriculum_' + domain,
                ROOT / 'experiments/rust-curriculum' / domain / 'verify.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            directory = Path(temp) / domain
            evidence[domain] = module.prepare(directory, rustc)
            records.extend(json.loads(line) for line in (directory / 'corpus.jsonl').read_text().splitlines())
    data = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in records).encode()
    manifest = write_admitted(data, output, ['MIT'])
    (output / 'domains.json').write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--rustc', type=Path)
    a = p.parse_args()
    print(json.dumps(prepare(a.output, a.rustc), sort_keys=True))
