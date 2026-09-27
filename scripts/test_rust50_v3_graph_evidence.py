import json
from pathlib import Path
import shutil
import tempfile
import unittest

from verify_rust50_v3_graph_evidence import REFERENCE, verify


class GraphEvidenceTests(unittest.TestCase):
    def copied_reference(self):
        directory = tempfile.TemporaryDirectory()
        target = Path(directory.name) / "evidence"
        shutil.copytree(REFERENCE, target)
        self.addCleanup(directory.cleanup)
        return target

    def test_saved_thor_record_is_complete(self):
        result = verify()
        self.assertEqual(result["campaign"], "rust50-v3-graph-comparison")
        self.assertEqual(result["updates"], 55296)
        self.assertEqual(result["matched_files_per_round"], 18)
        self.assertFalse(result["gpu_training"])
        self.assertFalse(result["model_promoted"])

    def test_changed_identity_measurement_or_scope_is_rejected(self):
        for field, value in [
            ("source_commit", "0" * 40),
            ("protocol_sha256", "0" * 64),
            ("dense_ns", 0),
            ("remoteops_run_id", 1),
            ("model_promoted", True),
        ]:
            with self.subTest(field=field):
                target = self.copied_reference()
                path = target / "thor-v3-graph-summary.json"
                summary = json.loads(path.read_text())
                summary[field] = value
                path.write_text(json.dumps(summary))
                with self.assertRaises(ValueError):
                    verify(target)

    def test_missing_or_extra_fields_are_rejected(self):
        for operation in ["remove", "add"]:
            with self.subTest(operation=operation):
                target = self.copied_reference()
                path = target / "thor-v3-graph-summary.json"
                summary = json.loads(path.read_text())
                if operation == "remove":
                    summary.pop("gather_ns")
                else:
                    summary["unreviewed_claim"] = "speedup"
                path.write_text(json.dumps(summary))
                with self.assertRaises(ValueError):
                    verify(target)
