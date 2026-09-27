"""Pinned metadata-only compiler certification; no source program or build script runs."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import tempfile
from external_corpus_inventory import digest

COMPILER='rustc 1.97.1 (8bab26f4f 2026-07-14)'
COMMIT='8bab26f4f68e0e26f0bb7960be334d5b520ea452'
MAX_DIAGNOSTICS=2_097_152


def identify(rustc):
    version=subprocess.run([str(rustc),'--version'],check=True,capture_output=True,text=True,timeout=20).stdout.strip()
    verbose=subprocess.run([str(rustc),'-vV'],check=True,capture_output=True,text=True,timeout=20).stdout.splitlines()
    if version!=COMPILER or 'commit-hash: '+COMMIT not in verbose:raise ValueError('pinned compiler required')


def classify(returncode,stderr,expected_label,expected_codes):
    if returncode not in (0,1) or len(stderr)>MAX_DIAGNOSTICS:raise ValueError('compiler infrastructure failure')
    codes=[]
    for line in stderr.decode('utf-8').splitlines():
        item=json.loads(line)
        if item.get('level')!='error':continue
        message=item.get('message','')
        if message.startswith('aborting due to '):continue
        code=item.get('code')
        if not isinstance(code,dict) or not isinstance(code.get('code'),str):raise ValueError('uncoded failure refused')
        codes.append(code['code'])
    codes=sorted(set(codes));label=int(returncode==0)
    if any(c in {'E0432','E0433','E0463','E0583'} for c in codes):raise ValueError('unresolved dependency is not a semantic label')
    if label!=expected_label or codes!=expected_codes or (not label and not codes):raise ValueError('unexpected compiler outcome')
    return dict(label=label,exit_code=returncode,diagnostics=codes)


def limits():
    resource.setrlimit(resource.RLIMIT_CPU,(15,15))
    resource.setrlimit(resource.RLIMIT_AS,(2_147_483_648,2_147_483_648))
    resource.setrlimit(resource.RLIMIT_FSIZE,(MAX_DIAGNOSTICS,MAX_DIAGNOSTICS))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))


def compile_source(source,rustc,label,codes):
    with tempfile.TemporaryDirectory(prefix='cogno-external-compile-') as temp:
        p=Path(temp);(p/'input.rs').write_bytes(source)
        with (p/'stdout').open('wb') as out,(p/'stderr').open('wb') as err:
            result=subprocess.run([str(rustc),str(p/'input.rs'),'--edition=2021','--emit=metadata',
                '--error-format=json','--crate-name=external_fixture','-o',str(p/'out.rmeta')],
                stdout=out,stderr=err,timeout=20,preexec_fn=limits)
        stderr=(p/'stderr').read_bytes()
        observation=classify(result.returncode,stderr,label,codes)
        if label and not (p/'out.rmeta').is_file():raise ValueError('success without metadata')
        observation.update(source_sha256=digest(source),compiler=COMPILER,compiler_commit=COMMIT,
                           edition='2021',source_execution=False)
        return observation


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--rustc',type=Path,required=True);p.add_argument('--label',type=int,choices=[0,1],required=True);p.add_argument('--code',action='append',default=[]);a=p.parse_args()
    identify(a.rustc);source=a.source.read_bytes()
    if not 1<=len(source)<=16384:raise ValueError('source bound')
    print(json.dumps(compile_source(source,a.rustc,a.label,sorted(set(a.code))),sort_keys=True))
