from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from external_corpus_qualify import qualify,prior_snapshot,PANEL
from external_corpus_inventory import digest

class QualificationTests(unittest.TestCase):
    def test_complete_offline_three_repository_compiler_pipeline(self):
        rustc=Path(subprocess.check_output(['rustup','which','--toolchain','1.97.1','rustc'],text=True).strip())
        with tempfile.TemporaryDirectory() as t:
            output=Path(t)/'qualified';result=qualify(output,rustc)
            self.assertEqual(result['sources'],18)
            self.assertEqual(result['raw_and_derived_compilations'],36)
            self.assertEqual(result['admitted']['counts'],dict(train=7,validation=7,test=4))
            self.assertEqual(len(result['upstream_repositories']),3)
            self.assertEqual(result['known_corpora'][0]['records'],328)
            saved=output/result['known_corpora'][0]['artifact']
            self.assertEqual(digest(saved.read_bytes()),result['known_corpora'][0]['sha256'])
            self.assertFalse(result['source_execution']);self.assertFalse(result['model_trained'])
            with self.assertRaises(ValueError):qualify(output,rustc)

    def test_frozen_evidence_failure_has_no_completion_marker(self):
        with tempfile.TemporaryDirectory() as t:
            output=Path(t)/'rejected'
            def mocked(*args,**kwargs):
                self.assertFalse(kwargs['finalize'])
                output.mkdir();(output/'compiler-results.json').write_bytes(b'changed')
                return {}
            with patch('external_corpus_qualify.prepare',side_effect=mocked):
                with self.assertRaisesRegex(ValueError,'compiler evidence regression'):
                    qualify(output,Path('/mock/rustc'))
            self.assertFalse((output/'COMPLETE.json').exists())
