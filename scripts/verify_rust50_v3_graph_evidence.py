"""Verify the committed Rust50 v3 Thor graph comparison record."""
import argparse
import json
import re
from pathlib import Path

REFERENCE = Path(__file__).resolve().parents[1] / "experiments/rust50-v2"

EXPECTED = {
    "schema": 1,
    "campaign": "rust50-v3-graph-comparison",
    "source_commit": "80cf940b1165b16a721185d27b63a55e8e6b4a7c",
    "source_tree": "ffdd8918dacd80f0412099ef5bd3d764f5c939ee",
    "toolchain": "rustc 1.97.1",
    "architecture": "aarch64",
    "remoteops_commit": "4ca2f5d3e8ca38b46da05f3d4820cc0290cdc1dc",
    "remoteops_run_id": 36303253691,
    "remoteops_job_id": 108574837971,
    "protocol_sha256": "d22e03d490425e0f2b0938e6a577912b47e7054d4415f2aa32c723ad888c4df5",
    "corpus_sha256": "fa327c59f97f46b5ddbf8d120e0283ce7fbb80085a31fcca0d7b9a004cc5fe1f",
    "provenance_sha256": "e57d9fc4c498d47b442ab8179e7ff489780937aca8ea97559f68fbc6ab39d926",
    "rounds": 1,
    "updates": 55296,
    "matched_files_per_round": 18,
    "dense_ns": 28715786569,
    "gather_ns": 4851739391,
    "dense_bundle_sha256": "1721b15c57ff9114850c19e6efa4cafce1a82ce38fae3b611314f0b9f518188d",
    "gather_bundle_sha256": "2ef04b2209c28f8430de536de93e1492d6bfb928c1c71afa0cfe69bbad2a5dfa",
    "comparison_complete_sha256": "506563683be54dc513a7132c0a86c1766cb39ea0face431b253dc62783da97ff",
    "gpu_training": False,
    "model_promoted": False,
}

HEX40_FIELDS = {"source_commit", "source_tree", "remoteops_commit"}
HEX64_FIELDS = {
    "protocol_sha256",
    "corpus_sha256",
    "provenance_sha256",
    "dense_bundle_sha256",
    "gather_bundle_sha256",
    "comparison_complete_sha256",
}
POSITIVE_FIELDS = {
    "remoteops_run_id",
    "remoteops_job_id",
    "rounds",
    "updates",
    "matched_files_per_round",
    "dense_ns",
    "gather_ns",
}


def load_summary(reference=REFERENCE):
    path = reference / "thor-v3-graph-summary.json"
    try:
        summary = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read v3 graph evidence: {path}") from exc
    if not isinstance(summary, dict):
        raise ValueError("v3 graph evidence must be a JSON object")
    if set(summary) != set(EXPECTED):
        raise ValueError("v3 graph evidence fields changed")
    if summary != EXPECTED:
        raise ValueError("v3 graph evidence values changed")
    for field in HEX40_FIELDS:
        if not re.fullmatch(r"[0-9a-f]{40}", summary[field]):
            raise ValueError(f"{field} is not a lowercase commit identity")
    for field in HEX64_FIELDS:
        if not re.fullmatch(r"[0-9a-f]{64}", summary[field]):
            raise ValueError(f"{field} is not a lowercase SHA-256 digest")
    for field in POSITIVE_FIELDS:
        if type(summary[field]) is not int or summary[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if summary["gpu_training"] or summary["model_promoted"]:
        raise ValueError("v3 evidence cannot claim GPU training or promotion")
    return summary


def verify(reference=REFERENCE):
    summary = load_summary(reference)
    return {
        "campaign": summary["campaign"],
        "updates": summary["updates"],
        "matched_files_per_round": summary["matched_files_per_round"],
        "dense_ns": summary["dense_ns"],
        "gather_ns": summary["gather_ns"],
        "gpu_training": summary["gpu_training"],
        "model_promoted": summary["model_promoted"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, nargs="?", default=REFERENCE)
    args = parser.parse_args()
    print(json.dumps(verify(args.reference.resolve()), sort_keys=True))
