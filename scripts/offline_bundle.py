#!/usr/bin/env python3
"""Package committed source and locked dependencies; never installs a toolchain."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib


def run(argv, cwd, **kwargs):
    return subprocess.run(argv, cwd=cwd, check=True, **kwargs)


def inventory(root):
    if root.is_symlink():
        raise ValueError(f'symlink forbidden as inventory root: {root}')
    if not root.is_dir():
        raise ValueError(f'inventory root is not a directory: {root}')
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'symlink forbidden: {path}')
        if path.is_file():
            result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def export_commit(repo, destination):
    revision = run(['git', 'rev-parse', 'HEAD'], repo, capture_output=True, text=True).stdout.strip()
    archive = run(['git', 'archive', '--format=tar', revision], repo, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for item in tar.getmembers():
            path = Path(item.name)
            if path.is_absolute() or '..' in path.parts or not (item.isfile() or item.isdir()):
                raise ValueError(f'unsupported source archive member: {item.name}')
        tar.extractall(destination, filter='data')
    return revision


def verify_inventory(bundle):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest.get('schema') != 'cogno-offline-bundle/v1':
        raise ValueError('unsupported bundle schema')
    source = bundle / 'source'
    if inventory(source) != manifest['source_sha256']:
        raise ValueError('source inventory mismatch')
    toolchain_file = source / 'rust-toolchain.toml'
    try:
        declared_toolchain = tomllib.loads(toolchain_file.read_text())['toolchain']['channel']
    except (FileNotFoundError, KeyError, tomllib.TOMLDecodeError) as error:
        raise ValueError('invalid source toolchain declaration') from error
    if manifest.get('toolchain') != declared_toolchain:
        raise ValueError('manifest toolchain does not match inventoried source')
    return manifest


def package(repo, output):
    # Refuse any existing destination, including dangling symlinks.
    output.mkdir(exist_ok=False)
    source = output / 'source'
    source.mkdir()
    revision = export_commit(repo, source)
    toolchain = tomllib.loads((source / 'rust-toolchain.toml').read_text())['toolchain']['channel']
    cargo = ['cargo', '+' + toolchain]
    if (source / 'vendor').exists() or (source / '.cargo' / 'config.toml').exists():
        raise ValueError('existing vendor/config needs explicit packaging review')
    # Preparation may fetch dependencies. The delivered config uses local sources.
    config = run(cargo + ['vendor', '--locked', '--versioned-dirs', 'vendor'], source,
                 capture_output=True, text=True).stdout
    (source / '.cargo').mkdir(exist_ok=True)
    (source / '.cargo' / 'config.toml').write_text(config + '\n[net]\noffline = true\n')
    lock = tomllib.loads((source / 'Cargo.lock').read_text())
    dependencies = [p for p in lock['package'] if 'source' in p]
    manifest = dict(schema='cogno-offline-bundle/v1', revision=revision,
                    toolchain=toolchain, dependencies=dependencies,
                    source_sha256=inventory(source))
    # Written last: a preparation failure leaves an explicitly incomplete bundle.
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    verify_inventory(output)


def verify(bundle):
    manifest = verify_inventory(bundle)
    source = bundle / 'source'
    toolchain = manifest['toolchain']
    # Resolve installed binaries first. No rustup proxy is invoked in isolation.
    binaries = {}
    for name in ('cargo', 'rustc'):
        binaries[name] = run(['rustup', 'which', '--toolchain', toolchain, name], source,
                             capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory(prefix='cogno-offline-check-') as temporary:
        temp = Path(temporary)
        work = temp / 'source'
        shutil.copytree(source, work)
        cargo_home = temp / 'cargo-home'
        cargo_home.mkdir()
        # Do not inherit Cargo source overrides or build wrappers from the host.
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('CARGO_', 'RUSTUP_', 'RUSTC_', 'RUSTFLAGS'))
               and k not in ('RUSTC', 'RUSTDOC')}
        env.update(CARGO_HOME=str(cargo_home), CARGO_TARGET_DIR=str(temp / 'target'),
                   CARGO_NET_OFFLINE='true', RUSTC=binaries['rustc'])
        cargo = [binaries['cargo']]
        run(cargo + ['build', '--workspace', '--all-targets', '--release', '--frozen'], work, env=env)
        run(cargo + ['test', '--workspace', '--all-targets', '--frozen'], work, env=env)
        run(cargo + ['run', '--release', '--frozen', '-p', 'cogno-model', '--example',
                     'rust_expert_pilot', '--', 'experiments/rust-expert-pilot/corpus.tsv',
                     str(temp / 'checkpoints')], work, env=env)
        predictions = temp / 'bpe-predictions.csv'
        with predictions.open('w') as output:
            run(cargo + ['run', '--release', '--frozen', '-p', 'cogno-model', '--example',
                         'bpe_rust_probe', '--', str(temp / 'bpe-checkpoints')], work, env=env, stdout=output)
        run([sys.executable, 'scripts/check_bpe_evidence.py', str(predictions),
             '--checkpoints', str(temp / 'bpe-checkpoints')], work, env=env)
        run([sys.executable, 'scripts/verify_rust_pipeline.py',
             '--examples-dir', str(temp / 'target' / 'release' / 'examples'),
             '--checkpoints', str(temp / 'bpe-checkpoints'),
             '--output', str(temp / 'rust-evaluation')], work, env=env)
        run([sys.executable, 'scripts/verify_external_rust.py',
             '--executable', str(temp / 'target/release/examples/bpe_external_eval'),
             '--checkpoints', str(temp / 'bpe-checkpoints'),
             '--output', str(temp / 'external-rust'), '--rustc', binaries['rustc']], work, env=env)
    # Verification never modifies the delivered source or checkpoints.
    verify_inventory(bundle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('pack', 'verify'))
    parser.add_argument('bundle', type=Path)
    args = parser.parse_args()
    bundle = args.bundle.absolute()
    if args.action == 'pack':
        package(Path(__file__).resolve().parents[1], bundle)
    else:
        verify(bundle)
    print(f'{args.action}: OK: {bundle}')


if __name__ == '__main__':
    main()
