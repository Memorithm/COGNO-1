"""Exact and whitespace-normalized exclusion against explicit known-source inventories."""
import json
from external_corpus_inventory import digest,unique


def normalized(source):
    # Diagnostic identity only: whitespace in string literals is also removed.
    return digest(bytes(b for b in source if b not in b' \t\r\n\v\f'))


def known_sources(inputs):
    sources=[];evidence=[];total=0
    if not inputs:raise ValueError('at least one explicit known corpus required')
    for path,expected in inputs:
        with path.open('rb') as stream:data=stream.read(4_194_305)
        if len(data)>4_194_304 or digest(data)!=expected:raise ValueError('known corpus size/hash')
        count=0
        for line in data.splitlines():
            row=json.loads(line,object_pairs_hook=unique);source=row['source'].encode('utf-8')
            if not 1<=len(source)<=16384 or digest(source)!=row['sha256']:raise ValueError('known source hash/size')
            total+=len(source);count+=1
            if total>4_194_304 or len(sources)>=8192:raise ValueError('known corpus total bound')
            sources.append(source)
        if not count:raise ValueError('empty known corpus')
        evidence.append(dict(file=str(path),sha256=expected,records=count))
    return sources,evidence


def exclude(candidates,known):
    exact={digest(s):'known' for s in known};normal={normalized(s):'known' for s in known}
    findings=[]
    for name,source in candidates:
        sha,norm=digest(source),normalized(source)
        if sha in exact:findings.append(dict(source_id=name,match=exact[sha],kind='exact'))
        elif norm in normal:findings.append(dict(source_id=name,match=normal[norm],kind='whitespace_normalized'))
        exact[sha]=name;normal[norm]=name
    return findings
