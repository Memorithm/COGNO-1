import json
import unittest
from external_corpus_inventory import digest
from external_corpus_splits import assignments,project
class SplitTests(unittest.TestCase):
    def fixture(self):
        sources=[{'repository':f'o/{x}'} for x in 'abc']
        rows=[dict(repository=f'o/{x}',lineage_group=x,split=s) for x,s in zip('abc',['train','validation','test'])]
        return sources,rows
    def check(self,s,r):
        b=json.dumps(dict(schema=1,repositories=r)).encode();return assignments(b,digest(b),s)
    def test_real_repository_identity_and_complete_assignment(self):
        s,r=self.fixture();self.assertEqual(len(self.check(s,r)),3)
        self.assertEqual(project(dict(repository='o/a',path='a',revision='1')),project(dict(repository='o/a',path='b',revision='2')))
    def test_related_repositories_cannot_cross_splits(self):
        s,r=self.fixture();r[1]['lineage_group']='a'
        with self.assertRaises(ValueError):self.check(s,r)
    def test_unknown_missing_duplicate_and_incomplete_splits(self):
        s,r=self.fixture()
        for bad in [r[:-1],r+[r[0]],[dict(x,split='train') for x in r]]:
            with self.assertRaises(ValueError):self.check(s,bad)
