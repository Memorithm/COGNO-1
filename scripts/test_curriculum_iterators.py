"""Fail-closed provenance and compiler evidence tests for the iterators curriculum."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

FOLDER = Path(__file__).resolve().parents[1] / 'experiments/rust-curriculum/iterators'
SPEC = importlib.util.spec_from_file_location('curriculum_iterators', FOLDER / 'verify.py')
curriculum = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curriculum)


class CurriculumTests(unittest.TestCase):
    def test_pinned_sources_and_balanced_pairs(self):
        cases = curriculum.checked_fixtures()['cases']
        self.assertEqual(len(cases), 24)
        families = {}
        for case in cases:
            families.setdefault(case['family'], []).append(case)
        self.assertEqual(len(families), 12)
        for pair in families.values():
            self.assertEqual({case['expected_label'] for case in pair}, {0, 1})
            self.assertEqual(len({case['split'] for case in pair}), 1)
        with patch.object(curriculum, 'FIXTURES_SHA256', '0' * 64):
            with self.assertRaises(ValueError):
                curriculum.checked_fixtures()

    def test_infrastructure_failure_never_becomes_negative_label(self):
        for returncode, stderr in [(-9, ''), (2, ''), (1, ''), (1, 'not json'),
                (1, json.dumps({'level': 'error', 'message': 'could not write output'}))]:
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                curriculum.classify(subprocess.CompletedProcess([], returncode, '', stderr), 0, [])
        diagnostic = json.dumps({'level': 'error', 'code': {'code': 'E0382'}, 'message': 'moved'})
        failed = subprocess.CompletedProcess([], 1, '', diagnostic)
        self.assertEqual(curriculum.classify(failed, 0, ['E0382']), (0, ['E0382']))
        with self.assertRaises(ValueError):
            curriculum.classify(failed, 0, ['E0499'])
        with self.assertRaises(ValueError):
            curriculum.classify(failed, 1, [])

    def test_all_compiler_labels_reproduce_frozen_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'verified'
            manifest = curriculum.prepare(output)
            self.assertEqual(manifest['counts'], {'train': 16, 'validation': 4, 'test': 4})
            for filename in ['corpus.jsonl', 'compiler-results.json']:
                self.assertEqual((output / filename).read_bytes(), (FOLDER / filename).read_bytes())
            with self.assertRaises(ValueError):
                curriculum.prepare(output)


if __name__ == '__main__':
    unittest.main()
