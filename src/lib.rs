//! # comprs
//!
//! A minimal-dependency, high-performance image compression library.
//!
//! This library provides PNG and JPEG encoding with hand-implemented
//! compression algorithms (DEFLATE, DCT, Huffman) for maximum control
//! and performance.
//!
//! ## Features
//!
//! - **Zero runtime dependencies by default**
//! - **PNG encoding** with all filter types and DEFLATE compression
//! - **JPEG encoding** with DCT, quantization, and Huffman coding
//! - Optional SIMD acceleration via `simd` feature
//! - Optional parallel processing via `parallel` feature
//! - Optional WebAssembly bindings via `wasm` feature
//! - Buffer reuse helpers: `png::encode_into` and
//!   `jpeg::encode_with_options_into` let you supply the output buffer to
//!   avoid repeated allocations when encoding multiple images.
//!
//! ## Example
//!
//! ```rust
//! use comprs::{png, jpeg, ColorType};
//! use comprs::png::{FilterStrategy, QuantizationMode, QuantizationOptions};
//!
//! // Encode as PNG (lossless or quantized)
//! let pixels: Vec<u8> = vec![255, 0, 0, 255]; // 1x1 red RGBA pixel
//! let mut opts = png::PngOptions::default();
//! opts.compression_level = 9;
//! opts.filter_strategy = FilterStrategy::Adaptive;
//! // Optional: quantize to palette (lossy, smaller)
//! opts.quantization = QuantizationOptions {
//!     mode: QuantizationMode::Force,
//!     max_colors: 256,
//!     dithering: false,
//! };
//! let png_data = png::encode_with_options(&pixels, 1, 1, ColorType::Rgba, &opts).unwrap();
//!
//! // Encode as JPEG
//! let rgb_pixels: Vec<u8> = vec![255, 0, 0]; // 1x1 red RGB pixel
//! let jpeg_data = jpeg::encode(&rgb_pixels, 1, 1, 85).unwrap();
//! ```

// Allow unsafe code only when SIMD or WASM feature is enabled
#![cfg_attr(not(any(feature = "simd", feature = "wasm")), forbid(unsafe_code))]
#![warn(missing_docs)]

pub mod bits;
pub mod color;
pub mod compress;
pub mod error;
pub mod jpeg;
pub mod png;

#[cfg(feature = "simd")]
pub mod simd;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use color::ColorType;
pub use error::{Error, Result};
