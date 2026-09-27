import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from external_corpus_inventory import digest
from external_corpus_pipeline import prepare
from test_external_corpus_inventory import row
class PipelineTests(unittest.TestCase):
    def fixture(self,base):
        cache=base/'cache';cache.mkdir();sources=[];repos=[]
        for i,split in enumerate(['train','validation','test']):
            repos.append(dict(repository=f'o/r{i}',lineage_group=f'g{i}',split=split))
            for label in [0,1]:
                r=row();data=f'fn item_{i}_{label}(){{}}\n'.encode();lic=b'MIT license fixture'
                r.update(id=f'r{i}-{label}',path=f'tests/{label}.rs',repository=f'o/r{i}',sha256=digest(data),bytes=len(data),license_sha256=digest(lic),license_bytes=len(lic),expected_label=label,expected_diagnostics=[] if label else ['E0308'])
                (cache/r['sha256']).write_bytes(data);(cache/r['license_sha256']).write_bytes(lic);sources.append(r)
        inv=json.dumps(dict(schema=1,sources=sources)).encode();splits=json.dumps(dict(schema=1,repositories=repos)).encode();known=base/'known.jsonl';data=(json.dumps(dict(source='fn prior(){}',sha256=digest(b'fn prior(){}')))+'\n').encode();known.write_bytes(data)
        return inv,digest(inv),splits,digest(splits),cache,base/'output',Path('/mock/rustc'),[(known,digest(data))]
    def test_full_pipeline_preserves_repo_projects_and_completion(self):
        with tempfile.TemporaryDirectory() as t:
            args=self.fixture(Path(t))
            with patch('external_corpus_pipeline.identify'),patch('external_corpus_pipeline.compile_source',side_effect=lambda s,r,l,c:dict(label=l,diagnostics=c)):
                result=prepare(*args)
            self.assertEqual(result['admitted']['counts'],dict(train=2,validation=2,test=2));self.assertEqual(result['raw_and_derived_compilations'],12)
            self.assertTrue((args[5]/'COMPLETE.json').exists());self.assertFalse(result['model_trained'])
    def test_infrastructure_failure_leaves_no_admitted_output(self):
        with tempfile.TemporaryDirectory() as t:
            args=self.fixture(Path(t))
            with patch('external_corpus_pipeline.identify'),patch('external_corpus_pipeline.compile_source',side_effect=ValueError('infra')):
                with self.assertRaises(ValueError):prepare(*args)
            self.assertFalse(args[5].exists())
