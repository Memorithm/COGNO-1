"""Reproduce the fixed observed external panel entirely offline, including compiler labels."""
import argparse
import json
from pathlib import Path
import tempfile
from external_corpus_inventory import digest,unique,path
from external_corpus_pipeline import prepare,bounded

ROOT=Path(__file__).resolve().parents[1]
PANEL=ROOT/'experiments/external-corpus-v2'
INVENTORY_SHA='be9c98861e89a52fb0d8b45554cfc8646964bc042bde9b5dc3223df413878b2c'
SPLITS_SHA='4026657839431e8e588be89d6377faa739f66176e07593e6118d430ac262e9c3'
KNOWN_SHA='f746b82354e55ce2f29e3eda18cd416ed65b8df774558af69e2b856cc5b5e07d'


def prior_snapshot():
    data=bounded(PANEL/'known-registry.json',65536)
    if digest(data)!=KNOWN_SHA:raise ValueError('reviewed known registry changed')
    registry=json.loads(data,object_pairs_hook=unique);records=[]
    for item in registry['files']:
        filename=ROOT/path(item['path']);source=bounded(filename,4_194_304)
        if digest(source)!=item['sha256']:raise ValueError('known repository artifact changed')
        if item['kind']=='jsonl':records.extend(json.loads(line,object_pairs_hook=unique) for line in source.splitlines())
        elif item['kind']=='source':records.append(dict(source=source.decode(),sha256=digest(source)))
        else:raise ValueError('unknown registry kind')
    return ''.join(json.dumps(r,sort_keys=True)+'\n' for r in records).encode()


def qualify(output,rustc):
    with tempfile.TemporaryDirectory(prefix='cogno-known-inventory-') as directory:
        previous=prior_snapshot();known=Path(directory)/'known.jsonl';known.write_bytes(previous)
        manifest=prepare(bounded(PANEL/'inventory.json',1_048_576),INVENTORY_SHA,
            bounded(PANEL/'splits.json',65536),SPLITS_SHA,PANEL/'cache',output,rustc,[(known,digest(previous))],finalize=False)
    expected=PANEL/'compiler-results.json'
    if (output/'compiler-results.json').read_bytes()!=expected.read_bytes():raise ValueError('compiler evidence regression')
    if (output/'admitted/corpus.crust').read_bytes()!=(PANEL/'corpus.crust').read_bytes():raise ValueError('corpus reproduction differs')
    if (output/'admitted/provenance.jsonl').read_bytes()!=(PANEL/'provenance.jsonl').read_bytes():raise ValueError('provenance reproduction differs')
    (output/'COMPLETE.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--rustc',type=Path,required=True);a=p.parse_args()
    print(json.dumps(qualify(a.output,a.rustc.resolve()),sort_keys=True))
