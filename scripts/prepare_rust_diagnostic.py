#!/usr/bin/env python3
"""Adapt the already examined synthetic pilot; this is NOT a fresh project holdout."""
import argparse
import hashlib
import json
from pathlib import Path
from admit_rust_corpus import write_admitted

SOURCE = Path(__file__).resolve().parents[1] / 'experiments/rust-expert-pilot/corpus.jsonl'
SOURCE_HASH = '13841a24f58c0bc8f1c9636b783baef71e0bf62692cde0234d882ce94aec44eb'
REVISION = 'e656dcf47d05f3a127e1fa34d76a957c08374de7'


def prepare(output):
    data = SOURCE.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_HASH:
        raise ValueError('frozen diagnostic source changed')
    records = []
    for line in data.splitlines():
        row = json.loads(line)
        row.update(project='synthetic/' + row['family'], revision=REVISION, license='MIT')
        row['provenance'] += '; reused COGNO-1 diagnostic; synthetic family grouping, not independent projects'
        records.append(row)
    return write_admitted(''.join(json.dumps(r, sort_keys=True) + '\n' for r in records).encode(), output, ['MIT'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    print(json.dumps(prepare(parser.parse_args().output), sort_keys=True))
