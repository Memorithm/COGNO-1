#!/usr/bin/env python3
"""Fixed engineering qualification; observed synthetic test, no model promotion."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'experiments/rust50-v2/protocol.json'
CORPUS_SHA = 'fa327c59f97f46b5ddbf8d120e0283ce7fbb80085a31fcca0d7b9a004cc5fe1f'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def regroup_provenance(corpus, original):
    """Bind every provenance source to its actual newly grouped corpus row."""
    if sha(corpus) != CORPUS_SHA:
        raise ValueError('fixed corpus changed')
    rows = {}
    for line in corpus.decode().splitlines()[1:]:
        split, project, label, digest, source_hex = line.split('\t')
        source = bytes.fromhex(source_hex)
        if sha(source) != digest or digest in rows:
            raise ValueError('invalid corpus source binding')
        rows[digest] = (split, project, int(label), source)
    output, seen = [], set()
    for line in original.decode().splitlines():
        record = json.loads(line)
        digest = record['sha256']
        if digest in seen or digest not in rows:
            raise ValueError('duplicate or unknown provenance source')
        seen.add(digest)
        split, project, label, source = rows[digest]
        if (record['project'], record['label'], record['source'].encode()) != (project, label, source):
            raise ValueError('provenance content mismatch')
        record['split'] = split
        output.append(json.dumps(record, sort_keys=True) + '\n')
    if seen != set(rows):
        raise ValueError('incomplete provenance')
    return ''.join(output).encode()


def trainer_protocol(provenance):
    fields = [
        ('version', 'rust-train-v2'), ('corpus_sha256', CORPUS_SHA),
        ('provenance_sha256', sha(provenance)), ('seeds', '1,7,42'),
        ('arms', 'full,cycle_mix'), ('epochs', '24'), ('vocab', '384'),
        ('context', '512'), ('embedding', '8'), ('hidden', '16'),
        ('batch', '1'), ('learning_rate', '0.003'), ('max_updates', '100000'),
    ]
    return ''.join(f'{key}\t{value}\n' for key, value in fields).encode()


def run(args, log):
    with log.open('w') as output:
        subprocess.run([str(x) for x in args], cwd=ROOT, stdout=output,
                       stderr=subprocess.STDOUT, check=True)


def qualify(out, binaries, rustc):
    if sha(PROTOCOL.read_bytes()) != 'e9f991f4332706d2b69f60762991f3a019b6488bcf34ccb62bc49f24d915f94d':
        raise ValueError('preregistered qualification protocol changed')
    out.mkdir()
    frozen = out / 'frozen'
    run([sys.executable, ROOT / 'scripts/verify_domain_holdout.py', frozen,
         '--split-executable', binaries / 'rust_project_split',
         '--runner', binaries / 'bpe_curriculum_probe', '--rustc', rustc],
        out / 'frozen.log')
    corpus = frozen / 'corpus/grouped/corpus.crust'
    provenance = regroup_provenance(corpus.read_bytes(),
                                   (frozen / 'corpus/original/provenance.jsonl').read_bytes())
    provenance_path = out / 'provenance.jsonl'
    provenance_path.write_bytes(provenance)
    protocol = trainer_protocol(provenance)
    protocol_path = out / 'trainer.tsv'
    protocol_path.write_bytes(protocol)
    admission = [protocol_path, sha(protocol), corpus, provenance_path]
    runner = binaries / 'rust_train_v2'
    models = out / 'trainer'
    run([runner, 'plan', *admission], out / 'plan.log')
    run([runner, 'train', *admission, models], out / 'training.log')
    complete_sha = sha((models / 'COMPLETE').read_bytes())
    run([runner, 'verify', *admission, models, complete_sha], out / 'verify.log')
    selection = out / 'selection.tsv'
    run([runner, 'select', *admission, models, complete_sha, selection], out / 'select.log')
    run([runner, 'test', *admission, models, complete_sha, selection, out / 'selected-test'],
        out / 'selected-test.log')
    run([runner, 'resume', *admission, models, out / 'resumed'], out / 'resume.log')
    if (out / 'resumed/COMPLETE').read_bytes() != (models / 'COMPLETE').read_bytes():
        raise ValueError('completed-run replay changed the bundle')
    frozen_inventory = list(csv.DictReader((frozen / 'models/checkpoints.csv').open()))
    matched = {}
    for row in frozen_inventory:
        if row['arm'] not in ('full', 'cycle_mix'):
            continue
        actual = sha((models / row['file']).read_bytes())
        if actual != row['sha256']:
            raise ValueError('configurable trainer checkpoint differs from frozen control')
        matched[row['file']] = actual
    if len(matched) != 6:
        raise ValueError('six configurable checkpoints required')
    run([sys.executable, ROOT / 'scripts/external_corpus_qualify.py', out / 'external',
         '--rustc', rustc], out / 'external.log')
    external = json.loads((out / 'external/COMPLETE.json').read_text())
    if external['sources'] != 18 or len(external['upstream_repositories']) != 3:
        raise ValueError('external panel count differs')
    for name in ['sequence_paths_probe', 'optimizer_training_probe']:
        run([binaries / name], out / f'{name}.txt')
    run([binaries / 'bpe_tokenizer_bench', '384', '5',
         ROOT / 'crates/cogno-model/src/tokenizer.rs',
         ROOT / 'crates/cogno-model/src/rust_corpus.rs',
         ROOT / 'crates/cogno-core/src/lib.rs'], out / 'bpe_tokenizer_bench.txt')
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    source_tree = subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip()
    summary = dict(schema=1, architecture=platform.machine(),
                   source_commit=source_commit, source_tree=source_tree,
                   protocol_sha256=sha(PROTOCOL.read_bytes()),
                   corpus_sha256=CORPUS_SHA, frozen_checkpoints=12,
                   frozen_predictions=3456, configurable_checkpoints=matched,
                   configurable_bundle_sha256=complete_sha,
                   selected_test_sha256=sha((out / 'selected-test/COMPLETE').read_bytes()),
                   selection_sha256=sha(selection.read_bytes()),
                   external_sources=external['sources'],
                   external_repositories=external['upstream_repositories'],
                   external_corpus_sha256=sha((out / 'external/admitted/corpus.crust').read_bytes()),
                   benchmark_logs={name: sha((out / name).read_bytes()) for name in
                                   ['sequence_paths_probe.txt', 'optimizer_training_probe.txt',
                                    'bpe_tokenizer_bench.txt']},
                   predictions_sha256=sha((frozen / 'models/predictions.csv').read_bytes()),
                   gpu_training=False, blind_test=False, model_promoted=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print('COGNO_RUST50_SUMMARY=' + json.dumps(summary, sort_keys=True))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--binaries', type=Path, required=True)
    parser.add_argument('--rustc', type=Path, required=True)
    args = parser.parse_args()
    qualify(args.output.resolve(), args.binaries.resolve(), args.rustc.resolve())
