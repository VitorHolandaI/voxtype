//! `voxtype transcribe <file>...` — one-shot transcription of audio files.
//!
//! `resample` lives here rather than in `src/audio/` because it has exactly
//! one call site (this command). Per the refactoring policy: don't extract
//! an abstraction from a single use site.

use std::path::{Path, PathBuf};
use voxtype::{config, transcribe, vad};

/// Transcribe one or more audio files, reusing a single loaded model.
///
/// The transcriber is built once and shared across every file. The first
/// file therefore pays model load, and the rest run against a warm model,
/// which is the state a running daemon is in.
pub(crate) fn transcribe_files(config: &config::Config, paths: &[PathBuf]) -> anyhow::Result<()> {
    let transcriber = transcribe::create_transcriber(config)?;
    let vad = vad::create_vad(config).ok().flatten();
    let multiple = paths.len() > 1;

    for path in paths {
        if multiple {
            println!("=== {} ===", path.display());
        }

        let samples = load_samples(path)?;

        if let Some(vad) = &vad {
            match vad.detect(&samples) {
                Ok(result) => {
                    println!(
                        "VAD: {:.2}s speech ({:.1}% of audio)",
                        result.speech_duration_secs,
                        result.speech_ratio * 100.0
                    );
                    if !result.has_speech {
                        println!("No speech detected, skipping transcription.");
                        continue;
                    }
                }
                Err(e) => {
                    eprintln!("VAD warning: {}", e);
                    // Continue with transcription if VAD fails
                }
            }
        }

        let start = std::time::Instant::now();
        let text = transcriber.transcribe(&samples)?;
        let elapsed = start.elapsed().as_secs_f64();

        println!("Transcribed in {:.4}s", elapsed);
        println!("\n{}", text);
    }

    Ok(())
}

/// Decode a WAV file to f32 mono at 16kHz.
fn load_samples(path: &Path) -> anyhow::Result<Vec<f32>> {
    use hound::WavReader;

    println!("Loading audio file: {:?}", path);

    let reader = WavReader::open(path)?;
    let spec = reader.spec();

    println!(
        "Audio format: {} Hz, {} channel(s), {:?}",
        spec.sample_rate, spec.channels, spec.sample_format
    );

    // Convert samples to f32 mono at 16kHz
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
    };

    // Mix to mono if stereo
    let mono_samples: Vec<f32> = if spec.channels > 1 {
        samples
            .chunks(spec.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
            .collect()
    } else {
        samples
    };

    // Resample to 16kHz if needed
    let final_samples = if spec.sample_rate != 16000 {
        println!("Resampling from {} Hz to 16000 Hz...", spec.sample_rate);
        voxtype::audio::resampler::resample_buffer(&mono_samples, spec.sample_rate, 16000)
    } else {
        mono_samples
    };

    println!(
        "Processing {} samples ({:.2}s)...",
        final_samples.len(),
        final_samples.len() as f32 / 16000.0
    );

    Ok(final_samples)
}
