"""Verify saved Rust50 evidence and optionally a complete fresh qualification run."""
import argparse
import hashlib
import json
from pathlib import Path
import re

REFERENCE = Path(__file__).resolve().parents[1] / 'experiments/rust50-v2'
BUNDLE = '1a39007a9430133c85f6c101b4409df86d08a077c811940e9612ae549f0be4f6'
TEST_BUNDLE = '924ba1379df78d03872be67110879f2da4ac1e698c0302ea71b868d2b87b78d4'
VARIABLE = {'architecture', 'source_commit', 'source_tree', 'benchmark_logs'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable(summary):
    return {key: value for key, value in summary.items() if key not in VARIABLE}


def manifest(directory, expected, checkpoints):
    if digest(directory / 'COMPLETE') != expected:
        raise ValueError('completion inventory changed')
    lines = (directory / 'COMPLETE').read_text().splitlines()
    start = lines.index('file\tsha256') + 1
    seen = set()
    for line in lines[start:]:
        name, sha = line.split('\t')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in seen:
            raise ValueError('invalid inventory filename')
        seen.add(name)
        actual = checkpoints.get(name) if name.endswith('.cbpc') else digest(directory / name)
        if actual != sha:
            raise ValueError('saved experiment file changed: ' + name)
    return seen


def verify(reference=REFERENCE, actual=None):
    local = json.loads((reference / 'local-summary.json').read_text())
    thor = json.loads((reference / 'thor-summary.json').read_text())
    execution = json.loads((reference / 'thor-execution.json').read_text())
    if stable(local) != stable(thor):
        raise ValueError('cross-architecture deterministic evidence differs')
    if (local['architecture'], thor['architecture']) != ('x86_64', 'aarch64'):
        raise ValueError('architecture evidence differs')
    if (thor['source_commit'] != '3624917ae1f98843adeb8920799b1c3487b5248e'
            or execution['source_commit'] != thor['source_commit']
            or execution['conclusion'] != 'success'
            or (execution['run_id'], execution['job_id']) != (36296752085, 108556928364)):
        raise ValueError('successful pinned Thor execution required')
    if (thor['frozen_checkpoints'], thor['frozen_predictions'], thor['external_sources']) != (12, 3456, 18):
        raise ValueError('qualification counts differ')
    if thor['blind_test'] or thor['model_promoted'] or thor['gpu_training']:
        raise ValueError('scope or promotion claim differs')
    if thor['protocol_sha256'] != digest(reference / 'protocol.json'):
        raise ValueError('preregistered protocol binding differs')
    if thor['configurable_bundle_sha256'] != BUNDLE or thor['selected_test_sha256'] != TEST_BUNDLE:
        raise ValueError('frozen training evidence differs')
    for prefix, summary in [('local', local), ('thor', thor)]:
        for name, sha in summary['benchmark_logs'].items():
            if digest(reference / f'{prefix}-{name}') != sha:
                raise ValueError('benchmark log changed')
        if digest(reference / f'{prefix}-selection.tsv') != summary['selection_sha256']:
            raise ValueError('selection evidence changed')
    checkpoints = thor['configurable_checkpoints']
    if len(checkpoints) != 6:
        raise ValueError('six model identities required')
    manifest(reference / 'trainer', BUNDLE, checkpoints)
    manifest(reference / 'selected-test', TEST_BUNDLE, {})
    if actual is not None:
        candidate = json.loads((actual / 'summary.json').read_text())
        if stable(candidate) != stable(thor):
            raise ValueError('fresh qualification differs from saved evidence')
        for name, sha in checkpoints.items():
            if digest(actual / 'trainer' / name) != sha:
                raise ValueError('fresh model checkpoint differs')
        manifest(actual / 'trainer', BUNDLE, checkpoints)
        manifest(actual / 'selected-test', TEST_BUNDLE, {})
    return dict(frozen_checkpoints=12, configurable_checkpoints=6, predictions=3456,
                external_sources=18, cross_architecture_equal=True, promoted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('actual', type=Path, nargs='?')
    args = parser.parse_args()
    print(json.dumps(verify(actual=args.actual), sort_keys=True))
