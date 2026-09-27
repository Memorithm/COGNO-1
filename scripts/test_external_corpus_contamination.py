import json
from pathlib import Path
import tempfile
import unittest
from external_corpus_contamination import exclude,known_sources
from external_corpus_inventory import digest
class ContaminationTests(unittest.TestCase):
    def test_known_and_within_candidate_duplicates(self):
        findings=exclude([('a',b'abc'),('b',b'a b c'),('c',b'xyz'),('d',b'xyz')],[b'abc'])
        self.assertEqual([x['kind'] for x in findings],['exact','whitespace_normalized','exact'])
    def test_explicit_known_inventory_hash(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'known.jsonl';data=(json.dumps(dict(source='fn old(){}',sha256=digest(b'fn old(){}')))+'\n').encode();p.write_bytes(data)
            sources,evidence=known_sources([(p,digest(data))]);self.assertEqual(sources,[b'fn old(){}']);self.assertEqual(evidence[0]['records'],1)
            with self.assertRaises(ValueError):known_sources([(p,'0'*64)])
        with self.assertRaises(ValueError):known_sources([])
    def test_whitespace_diagnostic_is_not_semantic_equivalence(self):
        self.assertEqual(exclude([('x',b'let x="a b";')],[b'let x="ab";'])[0]['kind'],'whitespace_normalized')
