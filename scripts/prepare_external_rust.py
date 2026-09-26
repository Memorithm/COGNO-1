#!/usr/bin/env python3
"""Compile only the pinned reviewed panel, never run its programs or download data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from admit_rust_corpus import write_admitted

PANEL = Path(__file__).resolve().parents[1] / 'experiments/external-rust-panel'
PANEL_HASH = '0a20b5c8fe9eb8e383901dd7d7cf7393d66926a195cc48f38e46193396e2cf81'


def checked_panel():
    data = (PANEL / 'panel.json').read_bytes()
    if hashlib.sha256(data).hexdigest() != PANEL_HASH:
        raise ValueError('reviewed panel inventory changed')
    panel = json.loads(data)
    for case in panel['cases']:
        for directory, key in [('upstream', 'raw_sha256'), ('source', 'source_sha256')]:
            path = PANEL / directory / (case['id'] + '.rs')
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != case[key]:
                raise ValueError('reviewed source changed')
    return panel


def compile_source(rustc, path, output):
    result = subprocess.run([str(rustc), str(path), '--edition=2021', '--crate-type=bin',
                             '--crate-name=external_panel', '--emit=metadata', '--error-format=json',
                             '-o', str(output)], capture_output=True, text=True, timeout=20)
    if result.returncode not in (0, 1) or 'internal compiler error' in result.stderr:
        raise ValueError('compiler infrastructure failure')
    diagnostics = [json.loads(line) for line in result.stderr.splitlines() if line]
    errors = [d for d in diagnostics if d.get('level') == 'error']
    if (result.returncode == 0) == bool(errors):
        raise ValueError('compiler outcome/diagnostic disagreement')
    return int(result.returncode == 0), sorted({d['code']['code'] for d in errors if d.get('code')})


def prepare(output, rustc):
    if output.exists():
        raise ValueError('output must be new')
    panel = checked_panel()
    version = subprocess.run([str(rustc), '--version'], capture_output=True, text=True, check=True).stdout.strip()
    if not version.startswith('rustc 1.97.1 '):
        raise ValueError('pinned rustc 1.97.1 required')
    records, results = [], []
    with tempfile.TemporaryDirectory(prefix='cogno-external-compile-') as temporary:
        for case in panel['cases']:
            raw = compile_source(rustc, PANEL / 'upstream' / (case['id'] + '.rs'), Path(temporary) / 'raw.rmeta')
            clean = compile_source(rustc, PANEL / 'source' / (case['id'] + '.rs'), Path(temporary) / 'clean.rmeta')
            if raw != clean or clean[0] != case['expected_label']:
                raise ValueError(f'compiler label/code mismatch: {case["id"]}')
            source = (PANEL / 'source' / (case['id'] + '.rs')).read_text()
            records.append(dict(source=source, sha256=case['source_sha256'], label=clean[0], split='test',
                                project=case['project'], revision=case['revision'], license='MIT', classification='Public',
                                compiler=version, provenance=f"{case['path']}; line comments removed; original and derived compile outcomes matched",
                                upstream_sha256=case['raw_sha256'], diagnostics=clean[1], case_id=case['id']))
            results.append(dict(case=case['id'], label=clean[0], diagnostics=clean[1]))
    data = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in records).encode()
    manifest = write_admitted(data, output, ['MIT'], evaluation_only=True)
    report = dict(compiler=version, panel_sha256=PANEL_HASH, program_execution=False, results=results)
    (output / 'compiler-results.json').write_text(json.dumps(report, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--rustc', type=Path)
    args = parser.parse_args()
    rustc = args.rustc or Path(subprocess.run(['rustup', 'which', '--toolchain', '1.97.1', 'rustc'], capture_output=True, text=True, check=True).stdout.strip())
    print(json.dumps(prepare(args.output, rustc), sort_keys=True))
