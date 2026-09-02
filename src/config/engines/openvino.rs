//! OpenVINO engine configuration.

use serde::{Deserialize, Serialize};

use super::super::default_on_demand_loading;

fn default_device() -> String {
    "GPU".to_string()
}

/// OpenVINO GenAI speech-to-text configuration (Whisper models in OpenVINO IR
/// format, run on Intel CPU / iGPU / NPU).
///
/// Requires: cargo build --features openvino
///
/// Needs `libopenvino_genai_c.so`, `libopenvino_genai.so` and
/// `libopenvino_tokenizers.so` reachable at run time. No distribution ships
/// them yet; they come from building `openvinotoolkit/openvino.genai` against
/// the system OpenVINO.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenVinoConfig {
    /// Path to the OpenVINO IR model directory. Expects the layout published
    /// under the `OpenVINO/` org on Hugging Face: `openvino_encoder_model.xml`
    /// + `.bin`, the matching decoder, and the tokenizer/detokenizer pair.
    pub model: String,

    /// Device to run on: "GPU" (Intel iGPU, default), "NPU", "CPU", "AUTO",
    /// or a heterogeneous spec like "HETERO:NPU,GPU". Passed through to
    /// OpenVINO untouched, so any device string it accepts works here.
    #[serde(default = "default_device")]
    pub device: String,

    /// Full path to `libopenvino_genai_c.so`, for when it is not on the
    /// loader's search path. Leave unset if it lives somewhere the dynamic
    /// linker already looks.
    #[serde(default)]
    pub library: Option<String>,

    /// Where OpenVINO caches its compiled per-device blobs. Defaults to
    /// `$XDG_CACHE_HOME/voxtype/openvino` (falling back to `~/.cache`).
    ///
    /// This matters more than it looks: without a cache the pipeline
    /// recompiles on every start, which measured at 1.1s on the iGPU and
    /// 106.8s on the NPU for large-v3-turbo.
    #[serde(default)]
    pub cache_dir: Option<String>,

    /// Load model on-demand when recording starts (true) or keep loaded (false)
    #[serde(default = "default_on_demand_loading")]
    pub on_demand_loading: bool,
}

impl Default for OpenVinoConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            device: default_device(),
            library: None,
            cache_dir: None,
            on_demand_loading: false,
        }
    }
}

impl OpenVinoConfig {
    /// Cache directory to hand the worker, resolving the default when unset.
    pub fn resolved_cache_dir(&self) -> std::path::PathBuf {
        if let Some(dir) = &self.cache_dir {
            return std::path::PathBuf::from(dir);
        }
        let base = std::env::var_os("XDG_CACHE_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache"))
            })
            .unwrap_or_else(|| std::path::PathBuf::from("/tmp"));
        base.join("voxtype").join("openvino")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_device_is_igpu() {
        assert_eq!(OpenVinoConfig::default().device, "GPU");
    }

    #[test]
    fn cache_dir_falls_back_under_voxtype() {
        let cfg = OpenVinoConfig::default();
        let dir = cfg.resolved_cache_dir();
        assert!(dir.ends_with("voxtype/openvino"), "got {}", dir.display());
    }

    #[test]
    fn explicit_cache_dir_wins() {
        let cfg = OpenVinoConfig {
            cache_dir: Some("/somewhere/else".to_string()),
            ..Default::default()
        };
        assert_eq!(
            cfg.resolved_cache_dir(),
            std::path::PathBuf::from("/somewhere/else")
        );
    }
}
