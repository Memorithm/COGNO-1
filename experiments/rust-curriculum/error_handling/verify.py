#!/usr/bin/env python3
"""Certify the pinned original error_handling curriculum using metadata-only compilation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]
sys.path.insert(0, str(REPOSITORY / 'scripts'))
from admit_rust_corpus import write_admitted  # noqa: E402

FIXTURES_SHA256 = '1871d84c720d0b4f23ff7fee1c2e1545203fe8edf35a59570ac8b5e388611845'
COMPILER = 'rustc 1.97.1 (8bab26f4f 2026-07-14)'
COMPILER_COMMIT = '8bab26f4f68e0e26f0bb7960be334d5b520ea452'


def checked_fixtures():
    data = (ROOT / 'fixtures.json').read_bytes()
    if hashlib.sha256(data).hexdigest() != FIXTURES_SHA256:
        raise ValueError('reviewed fixture inventory changed')
    manifest = json.loads(data)
    seen = set()
    family_splits = {}
    for case in manifest['cases']:
        source_file = ROOT / 'source' / (case['id'] + '.rs')
        if source_file.is_symlink():
            raise ValueError('symlink fixture refused')
        source = source_file.read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        if digest != case['source_sha256'] or digest in seen:
            raise ValueError('source digest mismatch or duplicate')
        seen.add(digest)
        if family_splits.setdefault(case['family'], case['split']) != case['split']:
            raise ValueError('synthetic family split leakage')
    return manifest


def classify(result, expected_label, expected_diagnostics):
    """Infrastructure failures and unexpected diagnostics must never become label 0."""
    if result.returncode not in (0, 1):
        raise ValueError('compiler terminated or infrastructure failure')
    diagnostics = []
    for line in result.stderr.splitlines():
        item = json.loads(line)
        if item.get('level') != 'error':
            continue
        message = item.get('message', '')
        if message.startswith('aborting due to '):
            continue
        code = item.get('code')
        if isinstance(code, dict) and isinstance(code.get('code'), str):
            diagnostics.append(code['code'])
        elif message == 'lifetime may not live long enough':
            diagnostics.append('LIFETIME')
        else:
            raise ValueError('unrecognized compiler failure')
    label = int(result.returncode == 0)
    diagnostics = sorted(set(diagnostics))
    if label != expected_label or diagnostics != expected_diagnostics:
        raise ValueError('compiler outcome differs from reviewed semantic expectation')
    if not label and not diagnostics:
        raise ValueError('failure without a recognized diagnostic')
    return label, diagnostics


def prepare(output, rustc=None):
    if output.exists():
        raise ValueError('output directory must be new')
    fixtures = checked_fixtures()
    command = [str(rustc)] if rustc else ['rustc', '+1.97.1']
    version = subprocess.run(command + ['--version'], check=True, capture_output=True,
                             text=True, timeout=20).stdout.strip()
    verbose = subprocess.run(command + ['-vV'], check=True, capture_output=True,
                             text=True, timeout=20).stdout.splitlines()
    if version != COMPILER or 'commit-hash: ' + COMPILER_COMMIT not in verbose:
        raise ValueError('pinned compiler identity required')
    records, observations = [], []
    with tempfile.TemporaryDirectory(prefix='cogno-error_handling-') as directory:
        for case in fixtures['cases']:
            source_file = ROOT / 'source' / (case['id'] + '.rs')
            source = source_file.read_bytes()
            # Compile the verified snapshot, preventing a changed file after admission.
            if hashlib.sha256(source).hexdigest() != case['source_sha256']:
                raise ValueError('source changed before compilation')
            snapshot = Path(directory) / 'input.rs'
            snapshot.write_bytes(source)
            result = subprocess.run(command + [str(snapshot), '--edition=2021',
                '--emit=metadata', '--error-format=json', '--crate-name=curriculum',
                '-o', str(Path(directory) / 'case.rmeta')], capture_output=True,
                text=True, timeout=20)
            label, diagnostics = classify(result, case['expected_label'], case['expected_diagnostics'])
            observations.append(dict(id=case['id'], source_sha256=case['source_sha256'],
                                     label=label, exit_code=result.returncode, diagnostics=diagnostics))
            records.append(dict(project='synthetic/curriculum/error_handling/' + case['family'],
                revision=FIXTURES_SHA256, license='MIT', classification='Public',
                compiler=version, provenance='Original COGNO-1 synthetic error_handling contrastive fixture; '
                    'revision is SHA256 of fixtures.json (which pins all source hashes), not an upstream commit; '
                    'project is a synthetic family, not an independent upstream project; '
                    'compiler metadata only, no execution; edition 2021; family=' + case['family'],
                split=case['split'], sha256=case['source_sha256'], label=label,
                source=source.decode('utf-8')))
    data = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in records).encode()
    report = dict(schema=1, domain='error_handling', compiler=version, compiler_commit=COMPILER_COMMIT,
                  edition='2021', fixture_manifest_sha256=FIXTURES_SHA256,
                  source_execution=False, observations=observations)
    manifest = write_admitted(data, output, ['MIT'])
    (output / 'corpus.jsonl').write_bytes(data)
    (output / 'compiler-results.json').write_text(json.dumps(report, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='new directory')
    parser.add_argument('--rustc', type=Path, help='pinned rustc binary; defaults to rustc +1.97.1')
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.rustc), sort_keys=True))
