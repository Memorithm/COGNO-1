import json
from pathlib import Path
import shutil
import tempfile
import unittest

from verify_rust50_v3_graph_evidence import EXPECTED, REFERENCE, verify, verify_actual


class GraphEvidenceTests(unittest.TestCase):
    def copied_reference(self):
        directory = tempfile.TemporaryDirectory()
        target = Path(directory.name) / "evidence"
        shutil.copytree(REFERENCE, target)
        self.addCleanup(directory.cleanup)
        return target

    def actual_summary(self):
        return {
            "schema": 1,
            "architecture": "aarch64",
            "source_commit": "a" * 40,
            "source_tree": "b" * 40,
            "source_protocol_sha256": EXPECTED["protocol_sha256"],
            "corpus_sha256": EXPECTED["corpus_sha256"],
            "provenance_sha256": EXPECTED["provenance_sha256"],
            "rounds": 1,
            "updates": 55296,
            "matched_files_per_round": 18,
            "dense_ns": 123,
            "gather_ns": 456,
            "dense_bundle_sha256": EXPECTED["dense_bundle_sha256"],
            "gather_bundle_sha256": EXPECTED["gather_bundle_sha256"],
            "comparison_complete_sha256": "c" * 64,
            "gpu_training": False,
            "model_promoted": False,
        }

    def test_saved_thor_records_are_complete(self):
        result = verify()
        self.assertEqual(result["campaign"], "rust50-v3-graph-comparison")
        self.assertEqual(result["updates"], 55296)
        self.assertEqual(result["matched_files_per_round"], 18)
        self.assertEqual(result["gated_source_commit"],
                         "baabdc226f42221c1cb64a0ccccd75afa67c8d2b")
        self.assertEqual(result["gated_remoteops_run_id"], 36303993254)
        self.assertEqual(result["gated_remoteops_job_id"], 108576919803)
        self.assertEqual(result["runtime_source_commit"],
                         "9989cf5edd2f9d34a037877d2af746762715a84e")
        self.assertEqual(result["runtime_remoteops_run_id"], 36327304698)
        self.assertEqual(result["runtime_remoteops_job_id"], 108642451598)
        self.assertFalse(result["gpu_training"])
        self.assertFalse(result["model_promoted"])

    def test_fresh_summary_is_checked_against_the_declared_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(self.actual_summary()))
            result = verify_actual(path, "a" * 40, "b" * 40)
            self.assertEqual(result["updates"], 55296)
            self.assertEqual(result["source_tree"], "b" * 40)

    def test_changed_fresh_summary_is_rejected(self):
        for field, value in [
            ("source_commit", "0" * 40),
            ("source_protocol_sha256", "0" * 64),
            ("dense_ns", 0),
            ("comparison_complete_sha256", "not-a-digest"),
            ("model_promoted", True),
        ]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "summary.json"
                summary = self.actual_summary()
                summary[field] = value
                path.write_text(json.dumps(summary))
                with self.assertRaises(ValueError):
                    verify_actual(path, "a" * 40, "b" * 40)

    def test_changed_identity_measurement_or_scope_is_rejected(self):
        for filename, field, value in [
            ("thor-v3-graph-summary.json", "source_commit", "0" * 40),
            ("thor-v3-graph-summary.json", "protocol_sha256", "0" * 64),
            ("thor-v3-graph-summary.json", "dense_ns", 0),
            ("thor-v3-graph-summary.json", "remoteops_run_id", 1),
            ("thor-v3-graph-summary.json", "model_promoted", True),
            ("thor-v3-gated-graph-summary.json", "source_commit", "0" * 40),
            ("thor-v3-gated-graph-summary.json", "comparison_complete_sha256", "0" * 64),
            ("thor-v3-gated-graph-summary.json", "gather_ns", 0),
            ("thor-v3-runtime-verified-summary.json", "source_commit", "0" * 40),
            ("thor-v3-runtime-verified-summary.json", "comparison_complete_sha256", "0" * 64),
            ("thor-v3-runtime-verified-summary.json", "dense_ns", 0),
        ]:
            with self.subTest(filename=filename, field=field):
                target = self.copied_reference()
                path = target / filename
                summary = json.loads(path.read_text())
                summary[field] = value
                path.write_text(json.dumps(summary))
                with self.assertRaises(ValueError):
                    verify(target)

    def test_missing_or_extra_fields_are_rejected(self):
        for filename in ["thor-v3-graph-summary.json",
                         "thor-v3-gated-graph-summary.json"]:
            for operation in ["remove", "add"]:
                with self.subTest(filename=filename, operation=operation):
                    target = self.copied_reference()
                    path = target / filename
                    summary = json.loads(path.read_text())
                    if operation == "remove":
                        summary.pop("gather_ns")
                    else:
                        summary["unreviewed_claim"] = "speedup"
                    path.write_text(json.dumps(summary))
                    with self.assertRaises(ValueError):
                        verify(target)
