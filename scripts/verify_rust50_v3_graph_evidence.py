"""Verify committed Rust50 v3 records and fresh graph comparison summaries."""
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

GATED_FILENAME = "thor-v3-gated-graph-summary.json"
GATED_EXPECTED = {
    "schema": 1,
    "campaign": "rust50-v3-graph-comparison-gated",
    "source_commit": "baabdc226f42221c1cb64a0ccccd75afa67c8d2b",
    "source_tree": "8b75bb781a1869b6177722755fda5e953cce81a6",
    "toolchain": "rustc 1.97.1",
    "architecture": "aarch64",
    "remoteops_commit": "11549073787d47ae7f7e3a1e5fd1cb124cb2600b",
    "remoteops_run_id": 36303993254,
    "remoteops_job_id": 108576919803,
    "protocol_sha256": "d22e03d490425e0f2b0938e6a577912b47e7054d4415f2aa32c723ad888c4df5",
    "corpus_sha256": "fa327c59f97f46b5ddbf8d120e0283ce7fbb80085a31fcca0d7b9a004cc5fe1f",
    "provenance_sha256": "e57d9fc4c498d47b442ab8179e7ff489780937aca8ea97559f68fbc6ab39d926",
    "rounds": 1,
    "updates": 55296,
    "matched_files_per_round": 18,
    "dense_ns": 28898623619,
    "gather_ns": 4956827068,
    "dense_bundle_sha256": "1721b15c57ff9114850c19e6efa4cafce1a82ce38fae3b611314f0b9f518188d",
    "gather_bundle_sha256": "2ef04b2209c28f8430de536de93e1492d6bfb928c1c71afa0cfe69bbad2a5dfa",
    "comparison_complete_sha256": "9ee9552194f98a5515ef120fa87cd763002c5fe791d11a33806716bbf8edc711",
    "gpu_training": False,
    "model_promoted": False,
}


RUNTIME_FILENAME = "thor-v3-runtime-verified-summary.json"
RUNTIME_EXPECTED = {
    "schema": 1,
    "campaign": "rust50-v3-graph-comparison-runtime",
    "source_commit": "9989cf5edd2f9d34a037877d2af746762715a84e",
    "source_tree": "702e5e90135820027b66d93caef63a1f71cbb5ef",
    "toolchain": "rustc 1.97.1",
    "architecture": "aarch64",
    "remoteops_commit": "8a50d46723b4566cdaf04c08d32704659f349549",
    "remoteops_run_id": 36327304698,
    "remoteops_job_id": 108642451598,
    "protocol_sha256": "d22e03d490425e0f2b0938e6a577912b47e7054d4415f2aa32c723ad888c4df5",
    "corpus_sha256": "fa327c59f97f46b5ddbf8d120e0283ce7fbb80085a31fcca0d7b9a004cc5fe1f",
    "provenance_sha256": "e57d9fc4c498d47b442ab8179e7ff489780937aca8ea97559f68fbc6ab39d926",
    "rounds": 1,
    "updates": 55296,
    "matched_files_per_round": 18,
    "dense_ns": 28714368040,
    "gather_ns": 4841435851,
    "dense_bundle_sha256": "1721b15c57ff9114850c19e6efa4cafce1a82ce38fae3b611314f0b9f518188d",
    "gather_bundle_sha256": "2ef04b2209c28f8430de536de93e1492d6bfb928c1c71afa0cfe69bbad2a5dfa",
    "comparison_complete_sha256": "b45236aef1f5918c29f52d326569caa349a2765142403d4426aaaad476c02a51",
    "gpu_training": False,
    "model_promoted": False,
}


LATEST_RUNTIME_FILENAME = "thor-v3-runtime-verified-summary-latest.json"
LATEST_RUNTIME_EXPECTED = {
    "schema": 1,
    "campaign": "rust50-v3-graph-comparison-runtime-latest",
    "source_commit": "8d71d7cfbe2d4bd4a8f7710e62370d9317bb00b5",
    "source_tree": "ac5c2e139dc61b39721170d53e44fcb284497f2a",
    "toolchain": "rustc 1.97.1",
    "architecture": "aarch64",
    "remoteops_commit": "59ae30c783c4eacd0bef8aa63ef92b0f38fdc3f0",
    "remoteops_run_id": 36348846012,
    "remoteops_job_id": 108703433769,
    "protocol_sha256": "d22e03d490425e0f2b0938e6a577912b47e7054d4415f2aa32c723ad888c4df5",
    "corpus_sha256": "fa327c59f97f46b5ddbf8d120e0283ce7fbb80085a31fcca0d7b9a004cc5fe1f",
    "provenance_sha256": "e57d9fc4c498d47b442ab8179e7ff489780937aca8ea97559f68fbc6ab39d926",
    "rounds": 1,
    "updates": 55296,
    "matched_files_per_round": 18,
    "dense_ns": 28464820093,
    "gather_ns": 4802861883,
    "dense_bundle_sha256": "1721b15c57ff9114850c19e6efa4cafce1a82ce38fae3b611314f0b9f518188d",
    "gather_bundle_sha256": "2ef04b2209c28f8430de536de93e1492d6bfb928c1c71afa0cfe69bbad2a5dfa",
    "comparison_complete_sha256": "6dfef49dd6a8e2d3352f874d2bad36dafadddc779dcdabf5751c4fec4f202cac",
    "gpu_training": false,
    "model_promoted": false
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


def load_record(reference, filename, expected):
    path = reference / filename
    try:
        summary = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read v3 graph evidence: {path}") from exc
    if not isinstance(summary, dict):
        raise ValueError(f"{filename} must be a JSON object")
    if set(summary) != set(expected):
        raise ValueError(f"{filename} fields changed")
    if summary != expected:
        raise ValueError(f"{filename} values changed")
    for field in HEX40_FIELDS:
        if not re.fullmatch(r"[0-9a-f]{40}", summary[field]):
            raise ValueError(f"{filename}: {field} is not a lowercase commit identity")
    for field in HEX64_FIELDS:
        if not re.fullmatch(r"[0-9a-f]{64}", summary[field]):
            raise ValueError(f"{filename}: {field} is not a lowercase SHA-256 digest")
    for field in POSITIVE_FIELDS:
        if type(summary[field]) is not int or summary[field] <= 0:
            raise ValueError(f"{filename}: {field} must be a positive integer")
    if summary["gpu_training"] or summary["model_promoted"]:
        raise ValueError(f"{filename} cannot claim GPU training or promotion")
    return summary


def load_summary(reference=REFERENCE):
    return load_record(reference, "thor-v3-graph-summary.json", EXPECTED)


def load_gated_summary(reference=REFERENCE):
    return load_record(reference, GATED_FILENAME, GATED_EXPECTED)


def load_runtime_summary(reference=REFERENCE):
    return load_record(reference, RUNTIME_FILENAME, RUNTIME_EXPECTED)


def load_latest_runtime_summary(reference=REFERENCE):
    return load_record(reference, LATEST_RUNTIME_FILENAME, LATEST_RUNTIME_EXPECTED)


ACTUAL_FIELDS = {
    "schema",
    "architecture",
    "source_commit",
    "source_tree",
    "source_protocol_sha256",
    "corpus_sha256",
    "provenance_sha256",
    "rounds",
    "updates",
    "matched_files_per_round",
    "dense_ns",
    "gather_ns",
    "dense_bundle_sha256",
    "gather_bundle_sha256",
    "comparison_complete_sha256",
    "gpu_training",
    "model_promoted",
}
ACTUAL_HASH_FIELDS = {
    "source_protocol_sha256",
    "corpus_sha256",
    "provenance_sha256",
    "dense_bundle_sha256",
    "gather_bundle_sha256",
    "comparison_complete_sha256",
}
ACTUAL_POSITIVE_FIELDS = {
    "rounds",
    "updates",
    "matched_files_per_round",
    "dense_ns",
    "gather_ns",
}


def load_actual(path, source_commit, source_tree):
    try:
        summary = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read fresh v3 graph summary: {path}") from exc
    if not isinstance(summary, dict):
        raise ValueError("fresh v3 graph summary must be a JSON object")
    if set(summary) != ACTUAL_FIELDS:
        raise ValueError("fresh v3 graph summary fields changed")
    expected = {
        "schema": 1,
        "architecture": "aarch64",
        "source_commit": source_commit,
        "source_tree": source_tree,
        "source_protocol_sha256": EXPECTED["protocol_sha256"],
        "corpus_sha256": EXPECTED["corpus_sha256"],
        "provenance_sha256": EXPECTED["provenance_sha256"],
        "rounds": 1,
        "updates": 55296,
        "matched_files_per_round": 18,
        "dense_bundle_sha256": EXPECTED["dense_bundle_sha256"],
        "gather_bundle_sha256": EXPECTED["gather_bundle_sha256"],
        "gpu_training": False,
        "model_promoted": False,
    }
    for field, value in expected.items():
        if summary[field] != value:
            raise ValueError(f"fresh v3 graph summary value changed: {field}")
    if not re.fullmatch(r"[0-9a-f]{40}", summary["source_commit"]):
        raise ValueError("fresh source commit is not a lowercase commit identity")
    if not re.fullmatch(r"[0-9a-f]{40}", summary["source_tree"]):
        raise ValueError("fresh source tree is not a lowercase tree identity")
    for field in ACTUAL_HASH_FIELDS:
        if not re.fullmatch(r"[0-9a-f]{64}", summary[field]):
            raise ValueError(f"fresh {field} is not a lowercase SHA-256 digest")
    for field in ACTUAL_POSITIVE_FIELDS:
        if type(summary[field]) is not int or summary[field] <= 0:
            raise ValueError(f"fresh {field} must be a positive integer")
    return summary


def verify_actual(path, source_commit, source_tree):
    summary = load_actual(path, source_commit, source_tree)
    return {
        "source_commit": summary["source_commit"],
        "source_tree": summary["source_tree"],
        "updates": summary["updates"],
        "matched_files_per_round": summary["matched_files_per_round"],
        "dense_ns": summary["dense_ns"],
        "gather_ns": summary["gather_ns"],
        "gpu_training": summary["gpu_training"],
        "model_promoted": summary["model_promoted"],
}


def verify(reference=REFERENCE):
    summary = load_summary(reference)
    gated = load_gated_summary(reference)
    runtime = load_runtime_summary(reference)
    latest = load_latest_runtime_summary(reference)
    return {
        "campaign": summary["campaign"],
        "updates": summary["updates"],
        "matched_files_per_round": summary["matched_files_per_round"],
        "dense_ns": summary["dense_ns"],
        "gather_ns": summary["gather_ns"],
        "gpu_training": summary["gpu_training"],
        "model_promoted": summary["model_promoted"],
        "gated_source_commit": gated["source_commit"],
        "gated_remoteops_run_id": gated["remoteops_run_id"],
        "gated_remoteops_job_id": gated["remoteops_job_id"],
        "runtime_source_commit": runtime["source_commit"],
        "runtime_remoteops_run_id": runtime["remoteops_run_id"],
        "runtime_remoteops_job_id": runtime["remoteops_job_id"],
        "latest_runtime_source_commit": latest["source_commit"],
        "latest_runtime_remoteops_run_id": latest["remoteops_run_id"],
        "latest_runtime_remoteops_job_id": latest["remoteops_job_id"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, nargs="?", default=REFERENCE)
    parser.add_argument("--actual", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--source-tree")
    args = parser.parse_args()
    if args.actual is not None:
        if not args.source_commit or not args.source_tree:
            parser.error("--actual requires --source-commit and --source-tree")
        result = verify_actual(args.actual.resolve(), args.source_commit, args.source_tree)
    else:
        if args.source_commit or args.source_tree:
            parser.error("--source-commit/--source-tree require --actual")
        result = verify(args.reference.resolve())
    print(json.dumps(result, sort_keys=True))
