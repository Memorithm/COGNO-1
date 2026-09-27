import json
from pathlib import Path
import shutil
import tempfile
import unittest
from verify_rust50_v2_evidence import REFERENCE, verify


class EvidenceTests(unittest.TestCase):
    def test_all_saved_evidence_matches(self):
        self.assertEqual(verify()['configurable_checkpoints'], 6)

    def test_changed_model_marker_timing_and_prediction_are_rejected(self):
        for name in ['trainer/COMPLETE', 'thor-sequence_paths_probe.txt',
                     'selected-test/cycle_mix-seed-1.test.tsv']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / 'evidence'
                shutil.copytree(REFERENCE, target)
                with (target / name).open('ab') as output:
                    output.write(b'changed\n')
                with self.assertRaises(ValueError):
                    verify(target)

    def test_failed_or_wrong_source_remote_job_is_not_completion(self):
        for change in [dict(conclusion='failure'), dict(source_commit='0' * 40)]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / 'evidence'
                shutil.copytree(REFERENCE, target)
                path = target / 'thor-execution.json'
                record = json.loads(path.read_text())
                record.update(change)
                path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    verify(target)
