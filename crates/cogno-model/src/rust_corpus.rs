//! Bounded CRUST001 reader. Consistency checking is not provenance authentication.
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;

pub const MAX_CORPUS_BYTES: usize = 4_194_304;
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CorpusSplit {
    Train,
    Validation,
    Test,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorpusError {
    Io,
    Capacity,
    Hash,
    Format,
    Duplicate,
    Leakage,
    Labels,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RustRecord {
    pub split: CorpusSplit,
    pub project: String,
    pub label: usize,
    pub source: Vec<u8>,
    pub source_hash: [u8; 32],
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RustCorpus {
    records: Vec<RustRecord>,
    hash: [u8; 32],
}
/// Narrow borrowed view for code that fits tokenizers or model parameters.
/// Construction is restricted to admitted train rows; evaluation rows cannot be
/// obtained from this view. This is an API boundary, not a process sandbox.
#[derive(Clone, Debug)]
pub struct RustTrainingView<'a> {
    records: Vec<&'a RustRecord>,
}
impl<'a> RustTrainingView<'a> {
    pub fn records(&self) -> &[&'a RustRecord] {
        &self.records
    }
    pub fn sources(&self) -> Vec<&'a [u8]> {
        self.records
            .iter()
            .map(|record| record.source.as_slice())
            .collect()
    }
}

impl RustCorpus {
    pub fn records(&self) -> &[RustRecord] {
        &self.records
    }
    pub fn hash(&self) -> [u8; 32] {
        self.hash
    }
    /// Obtain a training-only view. Evaluation-only corpora are refused, so a
    /// caller cannot accidentally fit an empty tokenizer from a holdout panel.
    pub fn training_view(&self) -> Result<RustTrainingView<'_>, CorpusError> {
        let records: Vec<_> = self
            .records
            .iter()
            .filter(|r| r.split == CorpusSplit::Train)
            .collect();
        if records.is_empty() {
            return Err(CorpusError::Labels);
        }
        Ok(RustTrainingView { records })
    }

    /// Expected identity must come from an independent inventory.
    pub fn read(reader: impl Read, expected: [u8; 32]) -> Result<Self, CorpusError> {
        Self::read_mode(reader, expected, false)
    }
    /// Require exclusively test records, containing both classes; never invent train rows.
    pub fn read_test_only(reader: impl Read, expected: [u8; 32]) -> Result<Self, CorpusError> {
        Self::read_mode(reader, expected, true)
    }
    fn read_mode(
        reader: impl Read,
        expected: [u8; 32],
        test_only: bool,
    ) -> Result<Self, CorpusError> {
        let mut bytes = Vec::new();
        reader
            .take((MAX_CORPUS_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|_| CorpusError::Io)?;
        Self::parse_mode(&bytes, expected, test_only)
    }
    pub fn parse(bytes: &[u8], expected: [u8; 32]) -> Result<Self, CorpusError> {
        Self::parse_mode(bytes, expected, false)
    }
    fn parse_mode(bytes: &[u8], expected: [u8; 32], test_only: bool) -> Result<Self, CorpusError> {
        if bytes.len() > MAX_CORPUS_BYTES {
            return Err(CorpusError::Capacity);
        }
        let hash: [u8; 32] = Sha256::digest(bytes).into();
        if hash != expected {
            return Err(CorpusError::Hash);
        }
        let text = std::str::from_utf8(bytes).map_err(|_| CorpusError::Format)?;
        if !text.ends_with('\n') || text.contains('\r') {
            return Err(CorpusError::Format);
        }
        let mut lines = text.lines();
        if lines.next() != Some("CRUST001") {
            return Err(CorpusError::Format);
        }
        let mut records = Vec::new();
        let mut seen = BTreeSet::new();
        let mut projects = BTreeMap::new();
        let mut labels = BTreeSet::new();
        let mut total = 0;
        for line in lines {
            if records.len() >= 4096 {
                return Err(CorpusError::Capacity);
            }
            let f: Vec<_> = line.splitn(6, '\t').collect();
            if f.len() != 5 {
                return Err(CorpusError::Format);
            }
            let split = match f[0] {
                "train" => CorpusSplit::Train,
                "validation" => CorpusSplit::Validation,
                "test" => CorpusSplit::Test,
                _ => return Err(CorpusError::Format),
            };
            if f[1].is_empty()
                || f[1].len() > 128
                || !f[1]
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_./-".contains(&b))
            {
                return Err(CorpusError::Format);
            }
            let label = match f[2] {
                "0" => 0,
                "1" => 1,
                _ => return Err(CorpusError::Labels),
            };
            let source_hash = parse_hash(f[3])?;
            if f[4].is_empty() || f[4].len() > 32768 {
                return Err(CorpusError::Capacity);
            }
            let source = decode_hex(f[4])?;
            std::str::from_utf8(&source).map_err(|_| CorpusError::Format)?;
            total += source.len();
            if total > 1_048_576 {
                return Err(CorpusError::Capacity);
            }
            if <[u8; 32]>::from(Sha256::digest(&source)) != source_hash {
                return Err(CorpusError::Hash);
            }
            if !seen.insert(source_hash) {
                return Err(CorpusError::Duplicate);
            }
            if projects.insert(f[1], split).is_some_and(|old| old != split) {
                return Err(CorpusError::Leakage);
            }
            labels.insert((split, label));
            records.push(RustRecord {
                split,
                project: f[1].into(),
                label,
                source,
                source_hash,
            });
        }
        if (test_only && labels != BTreeSet::from([(CorpusSplit::Test, 0), (CorpusSplit::Test, 1)]))
            || (!test_only && labels.len() != 6)
        {
            return Err(CorpusError::Labels);
        }
        Ok(Self { records, hash })
    }
}

/// Strict lowercase SHA-256 identity used by corpus/checkpoint CLI tools.
pub fn parse_hash(text: &str) -> Result<[u8; 32], CorpusError> {
    if text.len() != 64 {
        return Err(CorpusError::Format);
    }
    let bytes = decode_hex(text)?;
    let mut hash = [0; 32];
    hash.copy_from_slice(&bytes);
    Ok(hash)
}
fn decode_hex(text: &str) -> Result<Vec<u8>, CorpusError> {
    if !text.len().is_multiple_of(2) {
        return Err(CorpusError::Format);
    }
    fn digit(b: u8) -> Result<u8, CorpusError> {
        match b {
            b'0'..=b'9' => Ok(b - b'0'),
            b'a'..=b'f' => Ok(b - b'a' + 10),
            _ => Err(CorpusError::Format),
        }
    }
    text.as_bytes()
        .chunks_exact(2)
        .map(|p| Ok(digit(p[0])? * 16 + digit(p[1])?))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn training_view_never_exposes_validation_or_test_sources() {
        let corpus = parse(&wire()).unwrap();
        let view = corpus.training_view().unwrap();
        assert_eq!(view.records().len(), 2);
        assert!(view.records().iter().all(|r| r.split == CorpusSplit::Train));
        assert_eq!(
            view.sources(),
            vec![
                b"fn train_0() {}\n\t".as_slice(),
                b"fn train_1() {}\n\t".as_slice()
            ]
        );
        let test_only = RustCorpus {
            records: corpus
                .records
                .iter()
                .filter(|r| r.split == CorpusSplit::Test)
                .cloned()
                .collect(),
            hash: corpus.hash,
        };
        assert!(matches!(
            test_only.training_view(),
            Err(CorpusError::Labels)
        ));
    }

    #[test]
    fn test_only_never_accepts_training_rows() {
        let w = wire();
        let test = String::from("CRUST001\n")
            + &w.lines()
                .filter(|l| l.starts_with("test\t"))
                .collect::<Vec<_>>()
                .join("\n")
            + "\n";
        let hash = Sha256::digest(test.as_bytes()).into();
        assert_eq!(
            RustCorpus::read_test_only(test.as_bytes(), hash)
                .unwrap()
                .records()
                .len(),
            2
        );
        assert!(RustCorpus::parse(test.as_bytes(), hash).is_err());
        assert!(
            RustCorpus::read_test_only(w.as_bytes(), Sha256::digest(w.as_bytes()).into()).is_err()
        );
    }
    fn wire() -> String {
        let mut out = String::from("CRUST001\n");
        for split in ["train", "validation", "test"] {
            for label in 0..2 {
                let source = format!("fn {split}_{label}() {{}}\n\t");
                let hash = Sha256::digest(source.as_bytes())
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>();
                let hex = source
                    .bytes()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>();
                out.push_str(&format!("{split}\t{split}\t{label}\t{hash}\t{hex}\n"));
            }
        }
        out
    }
    fn parse(w: &str) -> Result<RustCorpus, CorpusError> {
        RustCorpus::parse(w.as_bytes(), Sha256::digest(w.as_bytes()).into())
    }
    #[test]
    fn exact_bytes_and_stream_roundtrip() {
        let w = wire();
        let c = parse(&w).unwrap();
        assert_eq!(c.records.len(), 6);
        assert_eq!(c.records[0].source, b"fn train_0() {}\n\t");
        assert_eq!(RustCorpus::read(w.as_bytes(), c.hash()).unwrap(), c);
        assert_eq!(
            RustCorpus::parse(w.as_bytes(), [0; 32]),
            Err(CorpusError::Hash)
        );
    }
    #[test]
    fn tampering_duplicates_leakage_and_missing_labels() {
        let w = wire();
        assert_eq!(
            parse(&w.replace("validation\tvalidation", "validation\ttrain")),
            Err(CorpusError::Leakage)
        );
        assert_eq!(
            parse(&(w.clone() + w.lines().nth(1).unwrap() + "\n")),
            Err(CorpusError::Duplicate)
        );
        let partial = w.lines().take(6).collect::<Vec<_>>().join("\n") + "\n";
        assert_eq!(parse(&partial), Err(CorpusError::Labels));
        assert!(parse(&w.replacen("666e", "666f", 1)).is_err());
        assert!(parse(&w.replacen("666e", "zzzz", 1)).is_err());
        assert!(parse(&w.replace("CRUST001", "CRUST002")).is_err());
        assert!(parse(w.trim_end()).is_err());
        assert!(parse_hash(&"A".repeat(64)).is_err());
    }
}
