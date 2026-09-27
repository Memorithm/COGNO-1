//! Explicit curator-owned group assignment; does not establish project independence.
#![forbid(unsafe_code)]
use cogno_model::rust_corpus::{parse_hash, CorpusSplit, RustCorpus};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::Path;

const MAX_INVENTORY_BYTES: usize = 1_048_576;
const MAX_PROJECTS: usize = 4096;
const HEADER: &str = "project\tgroup\tsplit";

#[derive(Clone, Debug, PartialEq, Eq)]
struct Assignment {
    group: String,
    split: CorpusSplit,
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn split_name(split: CorpusSplit) -> &'static str {
    match split {
        CorpusSplit::Train => "train",
        CorpusSplit::Validation => "validation",
        CorpusSplit::Test => "test",
    }
}
fn identifier(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_./-".contains(&b))
}
fn inventory(bytes: &[u8], expected: [u8; 32]) -> Result<BTreeMap<String, Assignment>, String> {
    if bytes.len() > MAX_INVENTORY_BYTES {
        return Err("inventory exceeds 1 MiB".into());
    }
    if <[u8; 32]>::from(Sha256::digest(bytes)) != expected {
        return Err("inventory hash mismatch".into());
    }
    let text = std::str::from_utf8(bytes).map_err(|_| "inventory requires UTF-8")?;
    if !text.ends_with('\n') || text.contains('\r') {
        return Err("inventory requires LF-terminated lines without CR".into());
    }
    let mut lines = text.lines();
    if lines.next() != Some(HEADER) {
        return Err("inventory header must be project<TAB>group<TAB>split".into());
    }
    let mut assignments = BTreeMap::new();
    let mut groups = BTreeMap::new();
    for line in lines {
        if assignments.len() >= MAX_PROJECTS {
            return Err("inventory exceeds 4096 projects".into());
        }
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 3 || !identifier(fields[0]) || !identifier(fields[1]) {
            return Err("invalid inventory columns or identifier".into());
        }
        let split = match fields[2] {
            "train" => CorpusSplit::Train,
            "validation" => CorpusSplit::Validation,
            "test" => CorpusSplit::Test,
            _ => return Err("invalid target split".into()),
        };
        if groups
            .insert(fields[1], split)
            .is_some_and(|old| old != split)
        {
            return Err("declared group crosses target splits".into());
        }
        if assignments
            .insert(
                fields[0].into(),
                Assignment {
                    group: fields[1].into(),
                    split,
                },
            )
            .is_some()
        {
            return Err("duplicate project assignment".into());
        }
    }
    if assignments.is_empty() {
        return Err("empty project inventory".into());
    }
    Ok(assignments)
}

fn repartition(
    corpus: &RustCorpus,
    assignments: &BTreeMap<String, Assignment>,
) -> Result<(Vec<u8>, String), String> {
    let projects: BTreeSet<_> = corpus
        .records()
        .iter()
        .map(|r| r.project.as_str())
        .collect();
    if projects != assignments.keys().map(String::as_str).collect() {
        return Err("inventory must cover exactly the corpus projects".into());
    }
    let mut wire = String::from("CRUST001\n");
    let mut stats: BTreeMap<&str, (CorpusSplit, [usize; 2])> = BTreeMap::new();
    for row in corpus.records() {
        let assignment = &assignments[&row.project];
        wire.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\n",
            split_name(assignment.split),
            row.project,
            row.label,
            hex(&row.source_hash),
            hex(&row.source)
        ));
        let entry = stats.entry(&row.project).or_insert((row.split, [0; 2]));
        entry.1[row.label] += 1;
    }
    let bytes = wire.into_bytes();
    RustCorpus::parse(&bytes, Sha256::digest(&bytes).into())
        .map_err(|e| format!("regrouped corpus rejected: {e:?}"))?;
    let mut report =
        String::from("project\tgroup\toriginal_split\ttarget_split\tlabel_0_rows\tlabel_1_rows\n");
    for (project, (original, counts)) in stats {
        let assignment = &assignments[project];
        report.push_str(&format!(
            "{project}\t{}\t{}\t{}\t{}\t{}\n",
            assignment.group,
            split_name(original),
            split_name(assignment.split),
            counts[0],
            counts[1]
        ));
    }
    Ok((bytes, report))
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    file.write_all(bytes).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())
}
fn save(
    dir: &Path,
    corpus_hash: [u8; 32],
    mapping: &[u8],
    wire: &[u8],
    report: &str,
) -> Result<String, String> {
    std::fs::create_dir(dir).map_err(|e| e.to_string())?;
    write_new(&dir.join("corpus.crust"), wire)?;
    write_new(&dir.join("groups.tsv"), mapping)?;
    write_new(&dir.join("report.tsv"), report.as_bytes())?;
    let manifest = format!("schema=1\ninput_corpus_sha256={}\ngroups_sha256={}\noutput_corpus_sha256={}\nreport_sha256={}\n", hex(&corpus_hash), hex(&Sha256::digest(mapping)), hex(&Sha256::digest(wire)), hex(&Sha256::digest(report.as_bytes())));
    write_new(&dir.join("COMPLETE"), manifest.as_bytes())?;
    Ok(manifest)
}
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err(
            "usage: rust_project_split CORPUS CORPUS_SHA GROUPS_TSV GROUPS_SHA NEW_DIRECTORY"
                .into(),
        );
    }
    let corpus = RustCorpus::read(
        std::fs::File::open(&args[1]).map_err(|e| e.to_string())?,
        parse_hash(&args[2]).map_err(|e| format!("{e:?}"))?,
    )
    .map_err(|e| format!("{e:?}"))?;
    let mut mapping = Vec::new();
    std::fs::File::open(&args[3])
        .map_err(|e| e.to_string())?
        .take((MAX_INVENTORY_BYTES + 1) as u64)
        .read_to_end(&mut mapping)
        .map_err(|e| e.to_string())?;
    let assignments = inventory(
        &mapping,
        parse_hash(&args[4]).map_err(|e| format!("{e:?}"))?,
    )?;
    let (wire, report) = repartition(&corpus, &assignments)?;
    print!(
        "{}",
        save(Path::new(&args[5]), corpus.hash(), &mapping, &wire, &report)?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> RustCorpus {
        let mut wire = String::from("CRUST001\n");
        for (i, split) in ["train", "validation", "test", "train"].iter().enumerate() {
            for label in 0..2 {
                let source = format!("fn source_{i}_{label}() {{}}\n");
                wire.push_str(&format!(
                    "{split}\tp{i}\t{label}\t{}\t{}\n",
                    hex(&Sha256::digest(source.as_bytes())),
                    hex(source.as_bytes())
                ));
            }
        }
        RustCorpus::parse(wire.as_bytes(), Sha256::digest(wire.as_bytes()).into()).unwrap()
    }
    fn parsed(text: &str) -> Result<BTreeMap<String, Assignment>, String> {
        inventory(text.as_bytes(), Sha256::digest(text.as_bytes()).into())
    }
    const MAPPING: &str =
        "project\tgroup\tsplit\np0\ta\ttest\np1\tb\ttrain\np2\tc\tvalidation\np3\ta\ttest\n";
    #[test]
    fn grouping_preserves_every_payload_and_changes_only_split() {
        let original = fixture();
        let mapping = parsed(MAPPING).unwrap();
        let (wire, report) = repartition(&original, &mapping).unwrap();
        let regrouped = RustCorpus::parse(&wire, Sha256::digest(&wire).into()).unwrap();
        assert_eq!(original.records().len(), regrouped.records().len());
        for (before, after) in original.records().iter().zip(regrouped.records()) {
            let mut expected = before.clone();
            expected.split = mapping[&before.project].split;
            assert_eq!(&expected, after);
        }
        assert!(report.contains("p3\ta\ttrain\ttest\t1\t1\n"));
        let reverse =
            "project\tgroup\tsplit\np3\ta\ttest\np2\tc\tvalidation\np1\tb\ttrain\np0\ta\ttest\n";
        assert_eq!(
            (wire, report),
            repartition(&original, &parsed(reverse).unwrap()).unwrap()
        );
    }
    #[test]
    fn refuses_group_leakage_duplicate_unknown_and_missing_projects() {
        assert!(parsed(&MAPPING.replace("p3\ta\ttest", "p3\ta\ttrain")).is_err());
        assert!(parsed(&format!("{MAPPING}p0\ta\ttest\n")).is_err());
        let c = fixture();
        let missing = MAPPING.replace("p3\ta\ttest\n", "");
        assert!(repartition(&c, &parsed(&missing).unwrap()).is_err());
        let unknown = MAPPING.replace("p3\ta\ttest", "unknown\ta\ttest");
        assert!(repartition(&c, &parsed(&unknown).unwrap()).is_err());
    }
    #[test]
    fn requires_all_output_split_label_combinations() {
        let all_train = MAPPING
            .replace("\ttest\n", "\ttrain\n")
            .replace("\tvalidation\n", "\ttrain\n");
        assert!(repartition(&fixture(), &parsed(&all_train).unwrap()).is_err());
    }
    #[test]
    fn strict_format_hash_and_capacity() {
        assert!(inventory(MAPPING.as_bytes(), [0; 32]).is_err());
        for bad in [
            "",
            HEADER,
            "project\tgroup\tsplit\n",
            "project\tgroup\tsplit\np\tg\ttrain\textra\n",
            "project\tgroup\tsplit\np\tg\tTRAIN\n",
            "project\tgroup\tsplit\np\t☃\ttrain\n",
            "project\tgroup\tsplit\np\tg\ttrain\r\n",
        ] {
            assert!(parsed(bad).is_err(), "{bad:?}");
        }
        let mut too_many = format!("{HEADER}\n");
        for i in 0..=MAX_PROJECTS {
            too_many.push_str(&format!("p{i}\tg\ttrain\n"));
        }
        assert!(parsed(&too_many).is_err());
        assert!(inventory(&vec![b'x'; MAX_INVENTORY_BYTES + 1], [0; 32]).is_err());
        assert!(parsed(&format!("{HEADER}\n{}\tg\ttrain\n", "p".repeat(129))).is_err());
    }
    #[test]
    fn completion_manifest_binds_all_artifacts_and_existing_output_is_refused() {
        let dir =
            std::env::temp_dir().join(format!("cogno-project-split-test-{}", std::process::id()));
        let c = fixture();
        let (wire, report) = repartition(&c, &parsed(MAPPING).unwrap()).unwrap();
        let manifest = save(&dir, c.hash(), MAPPING.as_bytes(), &wire, &report).unwrap();
        assert_eq!(std::fs::read(dir.join("corpus.crust")).unwrap(), wire);
        assert_eq!(
            std::fs::read(dir.join("groups.tsv")).unwrap(),
            MAPPING.as_bytes()
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("report.tsv")).unwrap(),
            report
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("COMPLETE")).unwrap(),
            manifest
        );
        assert!(manifest.contains(&format!(
            "output_corpus_sha256={}",
            hex(&Sha256::digest(&wire))
        )));
        assert!(save(&dir, c.hash(), MAPPING.as_bytes(), &wire, &report).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
