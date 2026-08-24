//! Host consent settings for the taste system (étape 4).
//!
//! [`TasteSettings`] lives in `cogno-core` as pure data; this module owns the
//! fail-closed file layer: bounded reads, strict JSON, atomic writes, and
//! privacy-preserving defaults when no settings file exists yet.

use cogno_core::TasteSettings;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

const SETTINGS_FILE: &str = "taste.settings.json";
const SCHEMA_VERSION: u16 = 1;
const MAX_SETTINGS_BYTES: usize = 4 * 1024;

#[derive(Debug)]
pub enum TasteSettingsError {
    Io(std::io::Error),
    /// The settings file exists but exceeds the size bound.
    SettingsTooLarge,
    /// The settings file is not strictly valid JSON for schema v1.
    InvalidSettingsJson,
}

impl core::fmt::Display for TasteSettingsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "settings io error: {error}"),
            Self::SettingsTooLarge => write!(f, "settings file exceeds {MAX_SETTINGS_BYTES} bytes"),
            Self::InvalidSettingsJson => write!(f, "invalid settings json"),
        }
    }
}

impl std::error::Error for TasteSettingsError {}

/// Serializable mirror of [`TasteSettings`] with an explicit schema version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedSettings {
    schema_version: u16,
    learning_enabled: bool,
    export_allowed: bool,
    import_allowed: bool,
}

impl From<TasteSettings> for PersistedSettings {
    fn from(settings: TasteSettings) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            learning_enabled: settings.learning_enabled,
            export_allowed: settings.export_allowed,
            import_allowed: settings.import_allowed,
        }
    }
}

impl From<PersistedSettings> for TasteSettings {
    fn from(settings: PersistedSettings) -> Self {
        Self {
            learning_enabled: settings.learning_enabled,
            export_allowed: settings.export_allowed,
            import_allowed: settings.import_allowed,
        }
    }
}

fn settings_path(store_root: impl AsRef<Path>) -> PathBuf {
    store_root.as_ref().join(SETTINGS_FILE)
}

/// Load host consent. A missing file is not an error: it means "defaults",
/// which keep data local (learning on, export/import off). An existing file
/// that fails to parse is an error — never silently downgrade consent.
pub fn load_settings(store_root: impl AsRef<Path>) -> Result<TasteSettings, TasteSettingsError> {
    let path = settings_path(store_root);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(TasteSettings::default())
        }
        Err(error) => return Err(TasteSettingsError::Io(error)),
    };
    if bytes.len() > MAX_SETTINGS_BYTES {
        return Err(TasteSettingsError::SettingsTooLarge);
    }
    let persisted: PersistedSettings =
        serde_json::from_slice(&bytes).map_err(|_| TasteSettingsError::InvalidSettingsJson)?;
    if persisted.schema_version != SCHEMA_VERSION {
        return Err(TasteSettingsError::InvalidSettingsJson);
    }
    Ok(persisted.into())
}

/// Atomically persist host consent (write temp + rename, like the stores).
pub fn save_settings(
    store_root: impl AsRef<Path>,
    settings: TasteSettings,
) -> Result<(), TasteSettingsError> {
    let root = store_root.as_ref();
    fs::create_dir_all(root).map_err(TasteSettingsError::Io)?;
    let body = serde_json::to_vec(&PersistedSettings::from(settings))
        .map_err(|_| TasteSettingsError::InvalidSettingsJson)?;
    let tmp = root.join(format!("{SETTINGS_FILE}.tmp"));
    fs::write(&tmp, &body).map_err(TasteSettingsError::Io)?;
    fs::rename(&tmp, settings_path(root)).map_err(TasteSettingsError::Io)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root(tag: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cogno-settings-{tag}-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn missing_file_means_privacy_preserving_defaults() {
        let root = temp_root("missing");
        let settings = load_settings(&root).expect("defaults");
        assert_eq!(settings, TasteSettings::default());
        assert!(settings.can_learn());
        assert!(!settings.can_export());
        assert!(!settings.can_import());
    }

    #[test]
    fn save_then_load_round_trips_consent() {
        let root = temp_root("round");
        let granted = TasteSettings {
            learning_enabled: true,
            export_allowed: true,
            import_allowed: true,
        };
        save_settings(&root, granted).expect("save");
        assert_eq!(load_settings(&root).expect("load"), granted);
        // Learning off disables export even when the flag is set.
        let muted = TasteSettings {
            learning_enabled: false,
            ..granted
        };
        assert!(!muted.can_export());
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn invalid_or_hostile_settings_fail_closed() {
        let root = temp_root("hostile");
        fs::create_dir_all(&root).expect("dir");
        let path = settings_path(&root);

        fs::write(&path, b"{\"schema_version\":2,\"learning_enabled\":true}").expect("write");
        assert!(matches!(
            load_settings(&root),
            Err(TasteSettingsError::InvalidSettingsJson)
        ));

        fs::write(&path, b"not json at all").expect("write");
        assert!(matches!(
            load_settings(&root),
            Err(TasteSettingsError::InvalidSettingsJson)
        ));

        let big = vec![b'x'; MAX_SETTINGS_BYTES + 1];
        fs::write(&path, &big).expect("write");
        assert!(matches!(
            load_settings(&root),
            Err(TasteSettingsError::SettingsTooLarge)
        ));

        fs::remove_dir_all(root).expect("cleanup");
    }
}
