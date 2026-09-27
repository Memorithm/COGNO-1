"""Explicit repository lineage groups: no generated family aliases or random heldout claims."""
import json
import re
from external_corpus_inventory import digest,unique


def assignments(data,expected,sources):
    if len(data)>65536 or digest(data)!=expected:raise ValueError('split inventory size/hash')
    obj=json.loads(data,object_pairs_hook=unique)
    if set(obj)!={'schema','repositories'} or type(obj['schema']) is not int or obj['schema']!=1 or not isinstance(obj['repositories'],list):raise ValueError('split schema')
    wanted={s['repository'] for s in sources};seen={};groups={}
    for r in obj['repositories']:
        if set(r)!={'repository','lineage_group','split'} or r['repository'] not in wanted or r['repository'] in seen:raise ValueError('unknown or duplicate repository')
        if not isinstance(r['lineage_group'],str) or not re.fullmatch(r'[a-z0-9][a-z0-9_./-]{0,127}',r['lineage_group']):raise ValueError('lineage group')
        if r['split'] not in ('train','validation','test'):raise ValueError('split')
        if groups.setdefault(r['lineage_group'],r['split'])!=r['split']:raise ValueError('lineage crosses splits')
        seen[r['repository']]=r
    if set(seen)!=wanted:raise ValueError('all repositories must be assigned exactly once')
    if set(groups.values())!={'train','validation','test'}:raise ValueError('all three splits required')
    return seen


def project(source):
    # All paths, versions, exercises and solutions from a repository share one identity.
    return 'github/'+source['repository']
