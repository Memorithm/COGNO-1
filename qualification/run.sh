#!/usr/bin/env bash
set -euo pipefail

# Run only in the ephemeral GitHub-hosted qualification job (or an equivalently
# reviewed isolated development environment), never on an untrusted host input.
cd "$(dirname "$0")/.."
test "$(git -C qualification/canonical rev-parse HEAD)" = "68d544679df465d3b85539c80926ed2644fbbba1"
test "$(git hash-object qualification/canonical/scirust-core/src/matrix/backend.rs)" = "e9c550fcd10ae54c936899cf31ef80e7cc5c7b11"
test "$(git hash-object qualification/canonical/scirust-core/src/matrix/view.rs)" = "067b71e6fa59a77588ac81c5f16e4bdf6a76c223"
test "$(rustc --version | awk '{print $2}')" = "1.97.1"

mkdir -p qualification/results
{
  echo 'schema=cogno.canonical-scalar-qualification@1'
  echo "cogno_sha=$(git rev-parse HEAD)"
  echo "cogno_tree=$(git rev-parse HEAD^{tree})"
  echo 'canonical_sha=68d544679df465d3b85539c80926ed2644fbbba1'
  echo 'canonical_backend_blob=e9c550fcd10ae54c936899cf31ef80e7cc5c7b11'
  echo 'canonical_view_blob=067b71e6fa59a77588ac81c5f16e4bdf6a76c223'
  echo 'canonical_license=https://github.com/Memorithm/scirust/blob/68d544679df465d3b85539c80926ed2644fbbba1/LICENSE.md'
  echo 'Required Notice: Copyright 2026 Tarek Zekriti (https://github.com/Memorithm/)'
  echo 'scope=finite_f32_scalar_forward_dot_relu_matmul_only'
  echo 'cases=288'
  rustc -vV
  sha256sum qualification/common.rs qualification/canonical.rs qualification/matrix/mod.rs crates/cogno-scirust/examples/canonical_scalar_fixture.rs
} > qualification/results/identity.txt

# Existing locked graph only; canonical modules are a separate std-only binary.
timeout --kill-after=5s 180s cargo run --locked --quiet -p cogno-scirust --example canonical_scalar_fixture > qualification/results/local.txt
timeout --kill-after=5s 120s rustc --edition=2024 -C opt-level=0 -C debuginfo=0 qualification/canonical.rs -o qualification/canonical-fixture
timeout --kill-after=5s 30s qualification/canonical-fixture > qualification/results/canonical.txt
test "$(wc -l < qualification/results/local.txt)" -eq 288
test "$(wc -l < qualification/results/canonical.txt)" -eq 288
diff -u qualification/results/canonical.txt qualification/results/local.txt > qualification/results/diff.txt
echo 'result=288_of_288_bit_exact_forward_cases;96_dot_and_96_relu_local_analytical_gradient_checks_passed' >> qualification/results/identity.txt
sha256sum qualification/results/local.txt qualification/results/canonical.txt >> qualification/results/identity.txt
