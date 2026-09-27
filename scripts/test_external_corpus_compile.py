import json
import unittest
from external_corpus_compile import classify
class CompileTests(unittest.TestCase):
    def test_reviewed_error_and_success(self):
        err=json.dumps(dict(level='error',code={'code':'E0382'},message='moved')).encode()
        self.assertEqual(classify(1,err,0,['E0382'])['label'],0)
        self.assertEqual(classify(0,b'',1,[])['label'],1)
        with self.assertRaises(ValueError):classify(1,err,0,['E0308'])
    def test_infrastructure_is_never_negative_training(self):
        for code,err in [(-9,b''),(2,b''),(1,b''),(1,b'not json'),(1,b'{"level":"error","message":"disk full"}')]:
            with self.assertRaises(ValueError):classify(code,err,0,[])
    def test_missing_dependency_rejected_even_if_expected(self):
        err=b'{"level":"error","code":{"code":"E0433"},"message":"unresolved"}'
        with self.assertRaises(ValueError):classify(1,err,0,['E0433'])
