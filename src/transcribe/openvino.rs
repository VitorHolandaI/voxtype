//! OpenVINO GenAI backend: Whisper models in OpenVINO IR, on Intel CPU/iGPU/NPU.
//!
//! Binds `openvino-genai` directly — no Python, no subprocess. The crate is
//! built with `runtime-linking`, so the shared libraries are opened at run
//! time rather than linked at build time; that keeps a binary compiled with
//! this feature runnable on machines that don't have OpenVINO installed,
//! failing with a clear message instead of refusing to start.
//!
//! Three libraries have to be reachable, and they are not packaged by any
//! distribution yet — they come from building `openvinotoolkit/openvino.genai`
//! against the system OpenVINO:
//!
//! - `libopenvino_genai_c.so`     the C API this crate dlopens
//! - `libopenvino_genai.so`       the C++ core it wraps
//! - `libopenvino_tokenizers.so`  the tokenizer plugin `WhisperPipeline` needs
//!
//! Missing the third one surfaces as an opaque `UnknownException` from
//! pipeline construction, so it is worth checking first when that happens.

use std::sync::{Mutex, OnceLock};

use openvino_genai::WhisperPipeline;

use super::{TranscribeError, Transcriber};
use crate::config::OpenVinoConfig;

/// Result of the one-and-only attempt to open the GenAI shared libraries.
static LIBRARY: OnceLock<Result<(), String>> = OnceLock::new();

/// Where the packaged install puts the GenAI libraries. Not on the loader's
/// default search path, so it has to be tried explicitly — otherwise every
/// package user would have to set `[openvino] library` by hand.
const PACKAGED_LIBRARY: &str = "/usr/lib/voxtype/openvino/libopenvino_genai_c.so";

/// Open the GenAI shared libraries once per process.
///
/// Order: the config's explicit path, then the finder's own search (which
/// honours `LD_LIBRARY_PATH` and the system directories), then the location
/// the deb/rpm use. The explicit path goes first so that setting it always
/// wins over a stale copy somewhere else on the system.
fn load_library(explicit: Option<&str>) -> Result<(), TranscribeError> {
    let outcome = LIBRARY.get_or_init(|| {
        let mut tried: Vec<String> = Vec::new();

        if let Some(path) = explicit {
            match openvino_genai::load_from(path) {
                Ok(()) => return Ok(()),
                Err(e) => tried.push(format!("[openvino] library = {path}: {e}")),
            }
        }

        match openvino_genai::load() {
            Ok(()) => return Ok(()),
            Err(e) => tried.push(format!("automatic search: {e}")),
        }

        if explicit != Some(PACKAGED_LIBRARY) && std::path::Path::new(PACKAGED_LIBRARY).exists() {
            match openvino_genai::load_from(PACKAGED_LIBRARY) {
                Ok(()) => return Ok(()),
                Err(e) => tried.push(format!("{PACKAGED_LIBRARY}: {e}")),
            }
        }

        Err(format!(
            "{}. Set [openvino] library to the full path of \
             libopenvino_genai_c.so, or put its directory on LD_LIBRARY_PATH.",
            tried.join("; ")
        ))
    });

    outcome.clone().map_err(|e| {
        TranscribeError::InitFailed(format!("OpenVINO GenAI library not loadable: {e}"))
    })
}

/// Sends anything written to fd 1 to fd 2 until dropped.
///
/// OpenVINO's C++ layer prints straight to the file descriptor — "Non default
/// env value for CACHE_DIR = ..." on every pipeline build — which lands in the
/// middle of `voxtype transcribe` output that scripts parse. Rust-level
/// redirection cannot catch a write from C++, so the descriptor itself has to
/// move. Same trick the project already uses for signals in `src/cpu.rs`.
struct StdoutToStderr {
    saved: i32,
}

impl StdoutToStderr {
    fn new() -> Option<Self> {
        use std::io::Write;
        // Flush our own buffered output first, or it lands on stderr too.
        let _ = std::io::stdout().flush();
        // Safety: dup/dup2 on the process's own descriptors, restored on drop.
        unsafe {
            let saved = libc::dup(libc::STDOUT_FILENO);
            if saved < 0 {
                return None;
            }
            if libc::dup2(libc::STDERR_FILENO, libc::STDOUT_FILENO) < 0 {
                libc::close(saved);
                return None;
            }
            Some(Self { saved })
        }
    }
}

impl Drop for StdoutToStderr {
    fn drop(&mut self) {
        // Safety: `saved` came from dup() above and is closed exactly once.
        unsafe {
            libc::dup2(self.saved, libc::STDOUT_FILENO);
            libc::close(self.saved);
        }
    }
}

/// Accept either a path to an IR directory or a bare catalog name, the way
/// the ONNX engines do, so the TUI's model picker resolves to something real.
fn resolve_model(model: &str, models_dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let direct = std::path::Path::new(model);
    if direct.exists() {
        return Some(direct.to_path_buf());
    }
    let in_models = models_dir.join(model);
    in_models.exists().then_some(in_models)
}

pub struct OpenVinoTranscriber {
    config: OpenVinoConfig,
    model_dir: std::path::PathBuf,
    /// `WhisperPipeline` is `Send` but not `Sync`, and `generate` needs
    /// `&mut`, so the mutex is doing double duty: it makes the transcriber
    /// `Sync` for the trait, and serialises access to the pipeline.
    pipeline: Mutex<Option<WhisperPipeline>>,
}

impl OpenVinoTranscriber {
    pub fn new(config: &OpenVinoConfig) -> Result<Self, TranscribeError> {
        if config.model.is_empty() {
            return Err(TranscribeError::ConfigError(
                "[openvino] model is not set".to_string(),
            ));
        }

        let models_dir = crate::config::Config::models_dir();
        let model_dir = resolve_model(&config.model, &models_dir).ok_or_else(|| {
            TranscribeError::ModelNotFound(format!(
                "[openvino] model not found: {} (tried the path itself and {})",
                config.model,
                models_dir.display()
            ))
        })?;

        // The crate has no way to pass CACHE_DIR, so the environment variable
        // is the only lever. Without it every start recompiles the model for
        // the device — measured at ~1s on the iGPU and ~107s on the NPU for
        // large-v3-turbo. Set before the pipeline is built, and early enough
        // that no other thread is reading the environment.
        let cache = config.resolved_cache_dir();
        if let Err(e) = std::fs::create_dir_all(&cache) {
            tracing::warn!(?e, path = %cache.display(), "could not create OpenVINO cache dir");
        } else {
            std::env::set_var("OV_CACHE_DIR", &cache);
        }

        Ok(Self {
            config: config.clone(),
            model_dir,
            pipeline: Mutex::new(None),
        })
    }

    fn build_pipeline(&self) -> Result<WhisperPipeline, TranscribeError> {
        load_library(self.config.library.as_deref())?;

        let model = self.model_dir.to_string_lossy();
        tracing::info!(
            device = %self.config.device,
            model = %model,
            "building OpenVINO pipeline"
        );

        let _quiet = StdoutToStderr::new();
        WhisperPipeline::new(&model, &self.config.device).map_err(|e| {
            TranscribeError::InitFailed(format!(
                "OpenVINO pipeline failed for device '{}': {e:?}. \
                 Check that the device exists (NPU needs the 'render' group) \
                 and that libopenvino_tokenizers.so sits beside libopenvino_genai.so.",
                self.config.device
            ))
        })
    }
}

impl Transcriber for OpenVinoTranscriber {
    fn transcribe(&self, samples: &[f32]) -> Result<String, TranscribeError> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        let mut guard = self
            .pipeline
            .lock()
            .map_err(|_| TranscribeError::InferenceFailed("pipeline lock poisoned".to_string()))?;

        if guard.is_none() {
            *guard = Some(self.build_pipeline()?);
        }

        let pipeline = guard.as_mut().expect("pipeline present");
        let _quiet = StdoutToStderr::new();
        let results = pipeline
            .generate(samples, None)
            .map_err(|e| TranscribeError::InferenceFailed(format!("{e:?}")))?;
        let text = results
            .get_string()
            .map_err(|e| TranscribeError::InferenceFailed(format!("{e:?}")))?;

        Ok(text.trim().to_string())
    }

    fn prepare(&self) {
        let Ok(mut guard) = self.pipeline.lock() else {
            return;
        };
        if guard.is_none() {
            match self.build_pipeline() {
                Ok(p) => *guard = Some(p),
                Err(e) => tracing::warn!(?e, "OpenVINO pipeline prepare failed"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_model_is_a_config_error() {
        let cfg = OpenVinoConfig::default();
        let Err(err) = OpenVinoTranscriber::new(&cfg) else {
            panic!("expected construction to fail without a model");
        };
        assert!(
            matches!(err, TranscribeError::ConfigError(_)),
            "expected ConfigError, got {err:?}"
        );
    }

    #[test]
    fn unknown_model_is_reported_as_missing() {
        let cfg = OpenVinoConfig {
            model: "definitely-not-a-model-anywhere".to_string(),
            ..Default::default()
        };
        let Err(err) = OpenVinoTranscriber::new(&cfg) else {
            panic!("expected construction to fail on a bogus model");
        };
        assert!(
            matches!(err, TranscribeError::ModelNotFound(_)),
            "expected ModelNotFound, got {err:?}"
        );
    }

    #[test]
    fn a_path_resolves_before_the_models_dir() {
        let tmp = std::env::temp_dir();
        let resolved = resolve_model(&tmp.to_string_lossy(), std::path::Path::new("/nonexistent"));
        assert_eq!(resolved, Some(tmp));
    }
}
