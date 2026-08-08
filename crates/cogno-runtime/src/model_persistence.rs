//! Transactional persistence and restart-time replay of reviewed neural models.
//!
//! Model generations live in a namespace separate from scientific taste:
//! `model-generation-N/` plus a strict `MODEL_CURRENT` pointer. A commit is
//! accepted only from an [`EligibleMetaModelReview`] plus an explicit host
//! persistence attestation. The model artifact is never hot-swapped here.

use crate::model_generation::{
    ModelGenerationChain, ModelGenerationError, ModelGenerationManifest, MODEL_GENERATION_MANIFEST_BYTES,
    MODEL_GENESIS_DIGEST,
};
use cogno_core::{
    ArchitectureId, MANIFEST_SCHEMA_VERSION, ModelFamily, ModelManifest,
};
use cogno_model::{
    load_neural_artifact, EligibleMetaModelReview, EncodedNeuralArtifact, NeuralArtifactError,
    NeuralModel, MAX_NEURAL_ARTIFACT_BYTES,
};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

const MODEL_CURRENT_FILE: &str = "MODEL_CURRENT";
const MODEL_CURRENT_TMP_FILE: &str = ".MODEL_CURRENT.tmp";
const MODEL_ARTIFACT_FILE: &str = "model.bin";
const MODEL_MANIFEST_FILE: &str = "model.manifest";
const GENERATION_MANIFEST_FILE: &str = "generation.manifest";
const MODEL_MANIFEST_BYTES: usize = 95;
const MAX_PERSISTED_MODEL_GENERATIONS: u64 = 4_096;

/// Explicit host authorization to persist one already-reviewed candidate.
///
/// This token grants only persistence authority; it does not activate Meta,
/// install weights into a live runtime, enable tools, or mutate policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostModelPromotionAttestation {
    _private: (),
}

impl HostModelPromotionAttestation {
    #[must_use]
    pub const fn approve_reviewed_candidate_for_controlled_persistence() -> Self {
        Self { _private: () }
    }
}

/// Result of an atomic model-generation commit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelGenerationCommit {
    pub generation: u64,
    pub generation_path: PathBuf,
    pub generation_manifest_sha256: [u8; 32],
    pub artifact_sha256: [u8; 32],
}

/// Fully replayed and verified persisted model selection.
#[derive(Clone, Debug)]
pub struct PersistedModelGenerationSelection {
    pub chain: ModelGenerationChain,
    pub selected_generation: u64,
    pub selected_path: PathBuf,
    pub selected_artifact: EncodedNeuralArtifact,
    pub selected_model: NeuralModel,
}

/// Fail-closed persistence/replay errors.
#[derive(Debug)]
pub enum ModelPersistenceError {
    Io(io::Error),
    InvalidGeneration,
    GenerationLimitExceeded,
    CurrentAlreadyExistsForGenesis,
    MissingCurrentForSuccessor,
    CurrentGenerationMismatch { expected: u64, actual: u64 },
    GenerationAlreadyExists(u64),
    StagingAlreadyExists(u64),
    MalformedCurrent,
    MissingGeneration(u64),
    InvalidGenerationManifestLength,
    InvalidModelManifestLength,
    GenerationChain(ModelGenerationError),
    ModelArtifact(NeuralArtifactError),
    ArtifactDigestMismatch,
    ModelManifestDigestMismatch,
    ManifestArtifactDigestMismatch,
    HeldOutMetricMismatch,
    ArtifactTooLarge { actual: u64, maximum: u64 },
    UnsupportedModelManifest,
    ArithmeticOverflow,
}

impl From<io::Error> for ModelPersistenceError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<ModelGenerationError> for ModelPersistenceError {
    fn from(error: ModelGenerationError) -> Self {
        Self::GenerationChain(error)
    }
}

impl From<NeuralArtifactError> for ModelPersistenceError {
    fn from(error: NeuralArtifactError) -> Self {
        Self::ModelArtifact(error)
    }
}

/// Persist one reviewed neural candidate as the exact next immutable model
/// generation and atomically advance `MODEL_CURRENT`.
pub fn commit_reviewed_model_generation(
    root: impl AsRef<Path>,
    generation: u64,
    review: &EligibleMetaModelReview,
    _host: HostModelPromotionAttestation,
) -> Result<ModelGenerationCommit, ModelPersistenceError> {
    if generation == 0 {
        return Err(ModelPersistenceError::InvalidGeneration);
    }
    if generation > MAX_PERSISTED_MODEL_GENERATIONS {
        return Err(ModelPersistenceError::GenerationLimitExceeded);
    }
    let root = root.as_ref();
    fs::create_dir_all(root)?;

    let final_path = generation_path(root, generation);
    if final_path.exists() {
        return Err(ModelPersistenceError::GenerationAlreadyExists(generation));
    }
    let stage_path = staging_path(root, generation);
    if stage_path.exists() {
        return Err(ModelPersistenceError::StagingAlreadyExists(generation));
    }

    let previous_manifest_sha256 = predecessor_digest_for_commit(root, generation)?;
    let artifact = review.artifact();
    let verified_model = load_neural_artifact(&artifact.manifest, &artifact.bytes)?;
    let model_manifest_bytes = encode_model_manifest(&artifact.manifest)?;
    let model_manifest_sha256 = sha256(&model_manifest_bytes);
    let generation_manifest = ModelGenerationManifest {
        generation,
        previous_manifest_sha256,
        artifact_sha256: artifact.manifest.weights_hash,
        model_manifest_sha256,
        validation_accuracy_bps: review.validation_accuracy_bps(),
        test_accuracy_bps: review.test_accuracy_bps(),
    };
    // Validate the manifest through the same chain primitive used at replay.
    let mut validation_chain = if generation == 1 {
        ModelGenerationChain::default()
    } else {
        load_persisted_model_generation_selection(root)?.chain
    };
    validation_chain.append(generation_manifest)?;

    fs::create_dir(&stage_path)?;
    let staged = (|| -> Result<(), ModelPersistenceError> {
        write_synced_file(&stage_path.join(MODEL_ARTIFACT_FILE), &artifact.bytes)?;
        write_synced_file(&stage_path.join(MODEL_MANIFEST_FILE), &model_manifest_bytes)?;
        write_synced_file(
            &stage_path.join(GENERATION_MANIFEST_FILE),
            &generation_manifest.canonical_bytes(),
        )?;
        sync_directory(&stage_path)?;
        Ok(())
    })();
    if let Err(error) = staged {
        let _ = fs::remove_dir_all(&stage_path);
        return Err(error);
    }

    fs::rename(&stage_path, &final_path)?;
    sync_directory(root)?;
    write_current_atomically(root, generation)?;
    let _ = verified_model;
    Ok(ModelGenerationCommit {
        generation,
        generation_path: final_path,
        generation_manifest_sha256: generation_manifest.digest(),
        artifact_sha256: generation_manifest.artifact_sha256,
    })
}

/// Replay every persisted generation from genesis through `MODEL_CURRENT`,
/// verifying chain links, both manifest/artifact digests and hostile model
/// decoding before returning the selected candidate.
pub fn load_persisted_model_generation_selection(
    root: impl AsRef<Path>,
) -> Result<PersistedModelGenerationSelection, ModelPersistenceError> {
    let root = root.as_ref();
    let selected_generation = read_current_generation(root)?;
    if selected_generation == 0 || selected_generation > MAX_PERSISTED_MODEL_GENERATIONS {
        return Err(ModelPersistenceError::GenerationLimitExceeded);
    }

    let mut chain = ModelGenerationChain::default();
    let mut selected_artifact = None;
    let mut selected_model = None;
    for generation in 1..=selected_generation {
        let path = generation_path(root, generation);
        if !path.is_dir() {
            return Err(ModelPersistenceError::MissingGeneration(generation));
        }
        let generation_manifest = read_generation_manifest(&path)?;
        if generation_manifest.generation != generation {
            return Err(ModelPersistenceError::CurrentGenerationMismatch {
                expected: generation,
                actual: generation_manifest.generation,
            });
        }
        let model_manifest_bytes = read_exact_file(
            &path.join(MODEL_MANIFEST_FILE),
            MODEL_MANIFEST_BYTES,
            ModelPersistenceError::InvalidModelManifestLength,
        )?;
        if sha256(&model_manifest_bytes) != generation_manifest.model_manifest_sha256 {
            return Err(ModelPersistenceError::ModelManifestDigestMismatch);
        }
        let model_manifest = decode_model_manifest(&model_manifest_bytes)?;
        if model_manifest.weights_hash != generation_manifest.artifact_sha256 {
            return Err(ModelPersistenceError::ManifestArtifactDigestMismatch);
        }
        let artifact_bytes = read_bounded_artifact(&path.join(MODEL_ARTIFACT_FILE))?;
        if sha256(&artifact_bytes) != generation_manifest.artifact_sha256 {
            return Err(ModelPersistenceError::ArtifactDigestMismatch);
        }
        let model = load_neural_artifact(&model_manifest, &artifact_bytes)?;
        chain.append(generation_manifest)?;
        if generation == selected_generation {
            selected_artifact = Some(EncodedNeuralArtifact {
                bytes: artifact_bytes,
                manifest: model_manifest,
            });
            selected_model = Some(model);
        }
    }

    let selected_path = generation_path(root, selected_generation);
    Ok(PersistedModelGenerationSelection {
        chain,
        selected_generation,
        selected_path,
        selected_artifact: selected_artifact.ok_or(ModelPersistenceError::MalformedCurrent)?,
        selected_model: selected_model.ok_or(ModelPersistenceError::MalformedCurrent)?,
    })
}

fn predecessor_digest_for_commit(
    root: &Path,
    generation: u64,
) -> Result<[u8; 32], ModelPersistenceError> {
    if generation == 1 {
        if root.join(MODEL_CURRENT_FILE).exists() {
            return Err(ModelPersistenceError::CurrentAlreadyExistsForGenesis);
        }
        return Ok(MODEL_GENESIS_DIGEST);
    }
    if !root.join(MODEL_CURRENT_FILE).exists() {
        return Err(ModelPersistenceError::MissingCurrentForSuccessor);
    }
    let selection = load_persisted_model_generation_selection(root)?;
    let expected = generation
        .checked_sub(1)
        .ok_or(ModelPersistenceError::InvalidGeneration)?;
    if selection.selected_generation != expected {
        return Err(ModelPersistenceError::CurrentGenerationMismatch {
            expected,
            actual: selection.selected_generation,
        });
    }
    selection
        .chain
        .selected()
        .map(ModelGenerationManifest::digest)
        .ok_or(ModelPersistenceError::MalformedCurrent)
}

fn write_current_atomically(root: &Path, generation: u64) -> Result<(), ModelPersistenceError> {
    let temp = root.join(MODEL_CURRENT_TMP_FILE);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    writeln!(file, "{generation}")?;
    file.sync_all()?;
    fs::rename(&temp, root.join(MODEL_CURRENT_FILE))?;
    sync_directory(root)?;
    Ok(())
}

fn read_current_generation(root: &Path) -> Result<u64, ModelPersistenceError> {
    let text = fs::read_to_string(root.join(MODEL_CURRENT_FILE))?;
    if text.is_empty()
        || !text.ends_with('\n')
        || text[..text.len() - 1].contains('\n')
        || text[..text.len() - 1].starts_with('0')
        || !text[..text.len() - 1].bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(ModelPersistenceError::MalformedCurrent);
    }
    text[..text.len() - 1]
        .parse::<u64>()
        .map_err(|_| ModelPersistenceError::MalformedCurrent)
}

fn read_generation_manifest(path: &Path) -> Result<ModelGenerationManifest, ModelPersistenceError> {
    let bytes = read_exact_file(
        &path.join(GENERATION_MANIFEST_FILE),
        MODEL_GENERATION_MANIFEST_BYTES,
        ModelPersistenceError::InvalidGenerationManifestLength,
    )?;
    let mut offset = 0usize;
    let generation = take_u64(&bytes, &mut offset)?;
    let previous_manifest_sha256 = take_digest(&bytes, &mut offset)?;
    let artifact_sha256 = take_digest(&bytes, &mut offset)?;
    let model_manifest_sha256 = take_digest(&bytes, &mut offset)?;
    let validation_accuracy_bps = take_u16(&bytes, &mut offset)?;
    let test_accuracy_bps = take_u16(&bytes, &mut offset)?;
    if offset != MODEL_GENERATION_MANIFEST_BYTES {
        return Err(ModelPersistenceError::InvalidGenerationManifestLength);
    }
    Ok(ModelGenerationManifest {
        generation,
        previous_manifest_sha256,
        artifact_sha256,
        model_manifest_sha256,
        validation_accuracy_bps,
        test_accuracy_bps,
    })
}

fn encode_model_manifest(manifest: &ModelManifest) -> Result<[u8; MODEL_MANIFEST_BYTES], ModelPersistenceError> {
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION || manifest.model_family != ModelFamily::Generic {
        return Err(ModelPersistenceError::UnsupportedModelManifest);
    }
    let mut bytes = [0u8; MODEL_MANIFEST_BYTES];
    let mut offset = 0usize;
    put(&mut bytes, &mut offset, &manifest.schema_version.to_le_bytes());
    put(&mut bytes, &mut offset, &[0]);
    put(&mut bytes, &mut offset, &manifest.architecture_id.0.to_le_bytes());
    put(&mut bytes, &mut offset, &manifest.tensor_count.to_le_bytes());
    put(&mut bytes, &mut offset, &manifest.parameter_count.to_le_bytes());
    put(&mut bytes, &mut offset, &manifest.max_context_tokens.to_le_bytes());
    put(&mut bytes, &mut offset, &manifest.tokenizer_hash);
    put(&mut bytes, &mut offset, &manifest.weights_hash);
    put(&mut bytes, &mut offset, &manifest.expected_file_bytes.to_le_bytes());
    debug_assert_eq!(offset, MODEL_MANIFEST_BYTES);
    Ok(bytes)
}

fn decode_model_manifest(bytes: &[u8]) -> Result<ModelManifest, ModelPersistenceError> {
    if bytes.len() != MODEL_MANIFEST_BYTES {
        return Err(ModelPersistenceError::InvalidModelManifestLength);
    }
    let mut offset = 0usize;
    let schema_version = take_u16(bytes, &mut offset)?;
    let family = *bytes
        .get(offset)
        .ok_or(ModelPersistenceError::InvalidModelManifestLength)?;
    offset = offset
        .checked_add(1)
        .ok_or(ModelPersistenceError::ArithmeticOverflow)?;
    if family != 0 {
        return Err(ModelPersistenceError::UnsupportedModelManifest);
    }
    let architecture_id = ArchitectureId(take_u32(bytes, &mut offset)?);
    let tensor_count = take_u32(bytes, &mut offset)?;
    let parameter_count = take_u64(bytes, &mut offset)?;
    let max_context_tokens = take_u32(bytes, &mut offset)?;
    let tokenizer_hash = take_digest(bytes, &mut offset)?;
    let weights_hash = take_digest(bytes, &mut offset)?;
    let expected_file_bytes = take_u64(bytes, &mut offset)?;
    if offset != MODEL_MANIFEST_BYTES {
        return Err(ModelPersistenceError::InvalidModelManifestLength);
    }
    Ok(ModelManifest {
        schema_version,
        model_family: ModelFamily::Generic,
        architecture_id,
        tensor_count,
        parameter_count,
        max_context_tokens,
        tokenizer_hash,
        weights_hash,
        expected_file_bytes,
    })
}

fn read_bounded_artifact(path: &Path) -> Result<Vec<u8>, ModelPersistenceError> {
    let metadata = fs::metadata(path)?;
    let maximum = u64::try_from(MAX_NEURAL_ARTIFACT_BYTES)
        .map_err(|_| ModelPersistenceError::ArithmeticOverflow)?;
    if metadata.len() > maximum {
        return Err(ModelPersistenceError::ArtifactTooLarge {
            actual: metadata.len(),
            maximum,
        });
    }
    let capacity = usize::try_from(metadata.len()).map_err(|_| ModelPersistenceError::ArithmeticOverflow)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| ModelPersistenceError::ArithmeticOverflow)?;
    File::open(path)?.take(maximum.saturating_add(1)).read_to_end(&mut bytes)?;
    if bytes.len() > MAX_NEURAL_ARTIFACT_BYTES {
        return Err(ModelPersistenceError::ArtifactTooLarge {
            actual: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            maximum,
        });
    }
    Ok(bytes)
}

fn read_exact_file(
    path: &Path,
    expected: usize,
    error: ModelPersistenceError,
) -> Result<Vec<u8>, ModelPersistenceError> {
    let bytes = fs::read(path)?;
    if bytes.len() != expected {
        return Err(error);
    }
    Ok(bytes)
}

fn write_synced_file(path: &Path, bytes: &[u8]) -> Result<(), ModelPersistenceError> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<(), ModelPersistenceError> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn generation_path(root: &Path, generation: u64) -> PathBuf {
    root.join(format!("model-generation-{generation}"))
}

fn staging_path(root: &Path, generation: u64) -> PathBuf {
    root.join(format!(".model-stage-{generation}"))
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn put<const N: usize>(target: &mut [u8; N], offset: &mut usize, value: &[u8]) {
    let end = *offset + value.len();
    target[*offset..end].copy_from_slice(value);
    *offset = end;
}

fn take_u16(bytes: &[u8], offset: &mut usize) -> Result<u16, ModelPersistenceError> {
    let slice = take(bytes, offset, 2)?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn take_u32(bytes: &[u8], offset: &mut usize) -> Result<u32, ModelPersistenceError> {
    let slice = take(bytes, offset, 4)?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn take_u64(bytes: &[u8], offset: &mut usize) -> Result<u64, ModelPersistenceError> {
    let slice = take(bytes, offset, 8)?;
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn take_digest(bytes: &[u8], offset: &mut usize) -> Result<[u8; 32], ModelPersistenceError> {
    let slice = take(bytes, offset, 32)?;
    let mut digest = [0u8; 32];
    digest.copy_from_slice(slice);
    Ok(digest)
}

fn take<'a>(
    bytes: &'a [u8],
    offset: &mut usize,
    len: usize,
) -> Result<&'a [u8], ModelPersistenceError> {
    let end = offset
        .checked_add(len)
        .ok_or(ModelPersistenceError::ArithmeticOverflow)?;
    let slice = bytes
        .get(*offset..end)
        .ok_or(ModelPersistenceError::ArithmeticOverflow)?;
    *offset = end;
    Ok(slice)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogno_core::{EvidenceOrigin, InputOrigin};
    use cogno_model::{
        review_neural_model_for_meta, Corpus, CorpusSplit, Label, LabeledExample,
        MetaNeuralReviewPolicy, NeuralConfig, SplitKind,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    fn root() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("cogno-model-persistence-{}-{nonce}", std::process::id()))
    }

    fn eligible_review() -> EligibleMetaModelReview {
        let mut corpus = Corpus::with_seed(7);
        for (label, payload) in [
            (Label(0), b"alpha train one".as_slice()),
            (Label(1), b"omega train one"),
            (Label(0), b"alpha train two"),
            (Label(1), b"omega train two"),
            (Label(0), b"alpha validation"),
            (Label(1), b"omega validation"),
            (Label(0), b"alpha test"),
            (Label(1), b"omega test"),
        ] {
            assert!(corpus.add(LabeledExample::new(
                label,
                payload.to_vec(),
                InputOrigin::ExplicitUserInstruction,
                EvidenceOrigin::ExplicitUserApproval,
            )));
        }
        let train = CorpusSplit {
            kind: SplitKind::Train,
            indices: vec![0, 1, 2, 3],
        };
        let validation = CorpusSplit {
            kind: SplitKind::Validation,
            indices: vec![4, 5],
        };
        let test = CorpusSplit {
            kind: SplitKind::Test,
            indices: vec![6, 7],
        };
        review_neural_model_for_meta(
            &corpus,
            &train,
            &validation,
            &test,
            NeuralConfig {
                input_dim: 64,
                epochs: 64,
                learning_rate: 0.02,
                max_payload_bytes: 128,
            },
            MetaNeuralReviewPolicy {
                minimum_validation_accuracy_bps: 5_000,
                minimum_test_accuracy_bps: 5_000,
                maximum_regression_bps: 5_000,
                artifact_max_context_tokens: 2_048,
            },
        )
        .expect("review")
        .into_eligible()
        .expect("eligible")
    }

    fn host() -> HostModelPromotionAttestation {
        HostModelPromotionAttestation::approve_reviewed_candidate_for_controlled_persistence()
    }

    #[test]
    fn commits_and_replays_two_hash_linked_generations() {
        let root = root();
        let review = eligible_review();
        let first = commit_reviewed_model_generation(&root, 1, &review, host()).expect("first");
        let second = commit_reviewed_model_generation(&root, 2, &review, host()).expect("second");
        assert_ne!(first.generation_manifest_sha256, second.generation_manifest_sha256);
        let selection = load_persisted_model_generation_selection(&root).expect("replay");
        assert_eq!(selection.selected_generation, 2);
        assert_eq!(selection.chain.len(), 2);
        assert_eq!(
            selection.selected_artifact.manifest.weights_hash,
            review.artifact().manifest.weights_hash
        );
        assert_eq!(
            selection.selected_model.weights(),
            load_neural_artifact(&review.artifact().manifest, &review.artifact().bytes)
                .expect("review model")
                .weights()
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn skipped_generation_and_duplicate_fail_before_current_changes() {
        let root = root();
        let review = eligible_review();
        assert!(matches!(
            commit_reviewed_model_generation(&root, 2, &review, host()),
            Err(ModelPersistenceError::MissingCurrentForSuccessor)
        ));
        commit_reviewed_model_generation(&root, 1, &review, host()).expect("first");
        assert!(matches!(
            commit_reviewed_model_generation(&root, 1, &review, host()),
            Err(ModelPersistenceError::GenerationAlreadyExists(1))
        ));
        assert_eq!(read_current_generation(&root).expect("current"), 1);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn artifact_tamper_is_detected_during_restart_replay() {
        let root = root();
        let review = eligible_review();
        commit_reviewed_model_generation(&root, 1, &review, host()).expect("first");
        let artifact_path = generation_path(&root, 1).join(MODEL_ARTIFACT_FILE);
        let mut bytes = fs::read(&artifact_path).expect("artifact");
        bytes[0] ^= 1;
        fs::write(&artifact_path, bytes).expect("tamper");
        assert!(matches!(
            load_persisted_model_generation_selection(&root),
            Err(ModelPersistenceError::ArtifactDigestMismatch)
        ));
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn malformed_current_fails_closed() {
        let root = root();
        fs::create_dir_all(&root).expect("root");
        fs::write(root.join(MODEL_CURRENT_FILE), b"01\n").expect("current");
        assert!(matches!(
            load_persisted_model_generation_selection(&root),
            Err(ModelPersistenceError::MalformedCurrent)
        ));
        fs::remove_dir_all(root).expect("cleanup");
    }
}
