"""Offline audited upstream acquisition -> recertification -> grouped CRUST001 admission."""
import argparse
import json
from pathlib import Path
from admit_rust_corpus import write_admitted
from external_corpus_inventory import validate,digest,MAX_BYTES
from external_corpus_fetch import obtain
from external_corpus_audit import derive
from external_corpus_compile import identify,compile_source,COMPILER
from external_corpus_splits import assignments,project
from external_corpus_contamination import known_sources,exclude


def bounded(path,limit):
    with path.open('rb') as stream:data=stream.read(limit+1)
    if len(data)>limit:raise ValueError('file bound')
    return data


def prepare(inventory_bytes,inventory_sha,split_bytes,split_sha,cache,output,rustc,known):
    if output.exists():raise ValueError('output must be new')
    inventory=validate(inventory_bytes,inventory_sha)
    mapping=assignments(split_bytes,split_sha,inventory['sources'])
    previous,known_evidence=known_sources(known)
    identify(rustc)
    records=[];reports=[];payloads=[];licenses={}
    for item in inventory['sources']:
        raw=obtain(item,cache,offline=True)
        licenses[item['license_sha256']]=obtain(item,cache,license_file=True,offline=True)
        source,audit=derive(raw)
        raw_result=compile_source(raw,rustc,item['expected_label'],item['expected_diagnostics'])
        derived_result=compile_source(source,rustc,item['expected_label'],item['expected_diagnostics'])
        assignment=mapping[item['repository']]
        records.append(dict(project=project(item),revision=item['revision'],license=item['license'],
            classification='Public',compiler=COMPILER,split=assignment['split'],sha256=digest(source),
            label=derived_result['label'],source=source.decode('utf-8'),
            provenance=f"https://github.com/{item['repository']}/blob/{item['revision']}/{item['path']}; raw_sha256={digest(raw)}; line-comments-v1; group={assignment['lineage_group']}"))
        reports.append(dict(id=item['id'],audit=audit,raw=raw_result,derived=derived_result))
        payloads.append((item['id'],raw,source))
    collisions=exclude([(name,source) for name,_,source in payloads],previous)
    if collisions:raise ValueError('contamination: '+json.dumps(collisions,sort_keys=True))
    data=''.join(json.dumps(r,sort_keys=True)+'\n' for r in records).encode()
    # Validate before creating the destination; write_admitted revalidates during emission.
    from admit_rust_corpus import admit
    admit(data,{'MIT','Apache-2.0'})
    output.mkdir();(output/'raw').mkdir();(output/'source').mkdir();(output/'licenses').mkdir()
    admitted=write_admitted(data,output/'admitted',['MIT','Apache-2.0'])
    for name,raw,source in payloads:
        (output/'raw'/f'{name}.rs').write_bytes(raw);(output/'source'/f'{name}.rs').write_bytes(source)
    for sha,license_bytes in licenses.items():(output/'licenses'/sha).write_bytes(license_bytes)
    (output/'inventory.json').write_bytes(inventory_bytes);(output/'splits.json').write_bytes(split_bytes)
    report_bytes=(json.dumps(reports,indent=2,sort_keys=True)+'\n').encode();(output/'compiler-results.json').write_bytes(report_bytes)
    manifest=dict(schema=1,inventory_sha256=inventory_sha,splits_sha256=split_sha,admitted=admitted,
        compiler_report_sha256=digest(report_bytes),known_corpora=known_evidence,
        upstream_repositories=sorted(mapping),repository_groups=mapping,
        sources=len(records),raw_and_derived_compilations=len(records)*2,
        contamination=dict(exact_or_whitespace_matches=0,semantic_independence_proven=False),
        source_execution=False,model_trained=False,curated_observed_panel=True)
    (output/'COMPLETE.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inventory',type=Path);p.add_argument('inventory_sha');p.add_argument('splits',type=Path);p.add_argument('splits_sha');p.add_argument('cache',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--rustc',type=Path,required=True);p.add_argument('--known',nargs=2,action='append',required=True,metavar=('JSONL','SHA256'));a=p.parse_args()
    print(json.dumps(prepare(bounded(a.inventory,MAX_BYTES),a.inventory_sha,bounded(a.splits,65536),a.splits_sha,a.cache,a.output,a.rustc.resolve(),[(Path(p),h) for p,h in a.known]),sort_keys=True))
