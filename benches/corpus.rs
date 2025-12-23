//! Shared corpus utilities for cross-language benchmarks.
//!
//! Goals:
//! - Single source of truth for synthetic inputs (gradient + noisy RGB).
//! - Optional loading of real fixtures under `tests/fixtures`.
//! - Documented corpus definitions so Rust and JS runners can stay in sync.

use std::path::{Path, PathBuf};

use image::GenericImageView;

/// Default square sizes used across benches.
pub const DEFAULT_SQUARE_SIZES: &[u32] = &[256, 512];

/// JPEG qualities used for cross-language comparisons.
pub const DEFAULT_JPEG_QUALITIES: &[u8] = &[75, 85];

/// Synthetic corpus entry describing how to generate a buffer.
#[derive(Clone, Copy, Debug)]
pub struct SyntheticSpec {
    pub name: &'static str,
    pub width: u32,
    pub height: u32,
    pub kind: SyntheticKind,
}

/// The type of synthetic data.
#[derive(Clone, Copy, Debug)]
pub enum SyntheticKind {
    Gradient,
    Noisy,
}

/// Return the default synthetic corpus definitions.
pub fn synthetic_corpus() -> Vec<SyntheticSpec> {
    let mut specs = Vec::new();
    for &size in DEFAULT_SQUARE_SIZES {
        specs.push(SyntheticSpec {
            name: "gradient",
            width: size,
            height: size,
            kind: SyntheticKind::Gradient,
        });
        specs.push(SyntheticSpec {
            name: "noisy",
            width: size,
            height: size,
            kind: SyntheticKind::Noisy,
        });
    }
    specs
}

/// Generate a gradient RGB buffer.
pub fn generate_gradient_rgb(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width) as u8;
            let g = ((y * 255) / height) as u8;
            let b = (((x + y) * 127) / (width + height)) as u8;
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    pixels
}

/// Generate a deterministic noisy RGB buffer (harder to compress).
pub fn generate_noisy_rgb(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    let mut seed = 12345u32;
    for _ in 0..(width * height) {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let r = (seed >> 16) as u8;
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let g = (seed >> 16) as u8;
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let b = (seed >> 16) as u8;
        pixels.extend_from_slice(&[r, g, b]);
    }
    pixels
}

/// Load a fixture from `tests/fixtures` and decode to RGB8.
///
/// Returns (pixels, width, height).
pub fn load_fixture_rgb(name: &str) -> image::ImageResult<(Vec<u8>, u32, u32)> {
    let path = fixtures_dir().join(name);
    let img = image::open(&path)?;
    let (width, height) = img.dimensions();
    let rgb = img.to_rgb8().into_raw();
    Ok((rgb, width, height))
}

/// Path to the shared fixture directory.
pub fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}
