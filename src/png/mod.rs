//! PNG encoder implementation.
//!
//! Implements PNG encoding according to the PNG specification (RFC 2083).

pub mod chunk;
pub mod filter;

use crate::color::ColorType;
use crate::compress::deflate::deflate_zlib_packed;
#[cfg(feature = "timing")]
use crate::compress::deflate::{deflate_zlib_packed_with_stats, DeflateStats};
use crate::error::{Error, Result};

/// PNG file signature (magic bytes).
const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

/// Maximum supported image dimension.
const MAX_DIMENSION: u32 = 1 << 24; // 16 million pixels

/// PNG encoding options.
#[derive(Debug, Clone)]
pub struct PngOptions {
    /// Compression level (1-9, default 6).
    pub compression_level: u8,
    /// Filter selection strategy.
    pub filter_strategy: FilterStrategy,
}

impl Default for PngOptions {
    fn default() -> Self {
        Self {
            // Prefer speed; level 2 favors throughput over ratio.
            compression_level: 2,
            // AdaptiveFast reduces per-row work with minimal compression impact.
            filter_strategy: FilterStrategy::AdaptiveFast,
        }
    }
}

impl PngOptions {
    /// Speed-focused preset (matches current default).
    pub fn fast() -> Self {
        Self {
            compression_level: 2,
            filter_strategy: FilterStrategy::AdaptiveFast,
        }
    }

    /// Balanced preset targeting better compression at moderate speed.
    pub fn balanced() -> Self {
        Self {
            compression_level: 6,
            filter_strategy: FilterStrategy::Adaptive,
        }
    }

    /// Highest compression preset; slowest.
    pub fn max_compression() -> Self {
        Self {
            compression_level: 9,
            filter_strategy: FilterStrategy::AdaptiveSampled { interval: 2 },
        }
    }
}

/// PNG filter selection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterStrategy {
    /// Always use no filter (fastest encoding).
    None,
    /// Always use Sub filter.
    Sub,
    /// Always use Up filter.
    Up,
    /// Always use Average filter.
    Average,
    /// Always use Paeth filter.
    Paeth,
    /// Choose best filter per row (best compression, slower).
    Adaptive,
    /// Adaptive but with early cut and limited trials (faster).
    AdaptiveFast,
    /// Adaptive on sampled rows, reuse chosen filter on intervening rows.
    /// `interval` must be >= 1. Example: interval=4 runs full adaptive
    /// every 4th row and reuses the last chosen filter for others.
    AdaptiveSampled {
        /// Number of rows between full adaptive evaluations (minimum 1).
        interval: u32,
    },
}

/// Encode raw pixel data as PNG.
///
/// # Arguments
/// * `data` - Raw pixel data (row-major order)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `color_type` - Color type of the input data
///
/// # Returns
/// Complete PNG file as bytes.
pub fn encode(data: &[u8], width: u32, height: u32, color_type: ColorType) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    encode_into(
        &mut output,
        data,
        width,
        height,
        color_type,
        &PngOptions::default(),
    )?;
    Ok(output)
}

/// Encode indexed (palette) pixel data as PNG.
///
/// The `data` slice contains palette indices (0..palette.len()). The palette
/// must contain between 1 and 256 entries. Optional `transparency` supplies
/// per-entry alpha values (tRNS); its length must not exceed the palette
/// length. Bit depth is fixed to 8 for indexed output.
pub fn encode_indexed(
    data: &[u8],
    width: u32,
    height: u32,
    palette: &[[u8; 3]],
    transparency: Option<&[u8]>,
) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    encode_indexed_into(
        &mut output,
        data,
        width,
        height,
        palette,
        transparency,
        &PngOptions::default(),
    )?;
    Ok(output)
}

/// Encode indexed (palette) pixel data as PNG with custom options.
pub fn encode_indexed_with_options(
    data: &[u8],
    width: u32,
    height: u32,
    palette: &[[u8; 3]],
    transparency: Option<&[u8]>,
    options: &PngOptions,
) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    encode_indexed_into(
        &mut output,
        data,
        width,
        height,
        palette,
        transparency,
        options,
    )?;
    Ok(output)
}

/// Encode indexed (palette) pixel data into a caller-provided buffer.
pub fn encode_indexed_into(
    output: &mut Vec<u8>,
    data: &[u8],
    width: u32,
    height: u32,
    palette: &[[u8; 3]],
    transparency: Option<&[u8]>,
    options: &PngOptions,
) -> Result<()> {
    if !(1..=9).contains(&options.compression_level) {
        return Err(Error::InvalidCompressionLevel(options.compression_level));
    }

    if width == 0 || height == 0 {
        return Err(Error::InvalidDimensions { width, height });
    }

    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(Error::ImageTooLarge {
            width,
            height,
            max: MAX_DIMENSION,
        });
    }

    let palette_len = palette.len();
    if palette_len == 0 || palette_len > 256 {
        return Err(Error::InvalidPaletteLength { len: palette_len });
    }
    if let Some(alpha) = transparency {
        if alpha.len() > palette_len {
            return Err(Error::InvalidTransparencyLength {
                palette_len,
                alpha_len: alpha.len(),
            });
        }
    }

    let expected_len = width as usize * height as usize; // 1 byte per index
    if data.len() != expected_len {
        return Err(Error::InvalidDataLength {
            expected: expected_len,
            actual: data.len(),
        });
    }

    output.clear();
    output.reserve(expected_len / 2 + 2048);

    // Write signature and IHDR (color type 3, bit depth 8).
    output.extend_from_slice(&PNG_SIGNATURE);
    write_ihdr_indexed(output, width, height);

    write_plte_chunk(output, palette);
    if let Some(alpha) = transparency {
        write_trns_chunk(output, alpha);
    }

    // Palette-aware filtering: avoid adaptive overhead; prefer None/Sub.
    let mut palette_options = options.clone();
    palette_options.filter_strategy = match options.filter_strategy {
        FilterStrategy::Adaptive
        | FilterStrategy::AdaptiveFast
        | FilterStrategy::AdaptiveSampled { .. } => FilterStrategy::None,
        other => other,
    };

    let filtered = filter::apply_filters(data, width, height, 1, &palette_options);
    let compressed = deflate_zlib_packed(&filtered, palette_options.compression_level);

    write_idat_chunks(output, &compressed);
    write_iend(output);
    Ok(())
}

/// Encode raw pixel data as PNG with custom options.
pub fn encode_with_options(
    data: &[u8],
    width: u32,
    height: u32,
    color_type: ColorType,
    options: &PngOptions,
) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    encode_into(&mut output, data, width, height, color_type, options)?;
    Ok(output)
}

/// Encode raw pixel data as PNG into a caller-provided buffer.
///
/// The `output` buffer will be cleared before writing. This API allows callers
/// to reuse an allocation across multiple encodes.
pub fn encode_into(
    output: &mut Vec<u8>,
    data: &[u8],
    width: u32,
    height: u32,
    color_type: ColorType,
    options: &PngOptions,
) -> Result<()> {
    if !(1..=9).contains(&options.compression_level) {
        return Err(Error::InvalidCompressionLevel(options.compression_level));
    }

    // Validate dimensions
    if width == 0 || height == 0 {
        return Err(Error::InvalidDimensions { width, height });
    }

    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(Error::ImageTooLarge {
            width,
            height,
            max: MAX_DIMENSION,
        });
    }

    // Validate data length
    let bytes_per_pixel = color_type.bytes_per_pixel();
    let expected_len = width as usize * height as usize * bytes_per_pixel;
    if data.len() != expected_len {
        return Err(Error::InvalidDataLength {
            expected: expected_len,
            actual: data.len(),
        });
    }

    output.clear();

    // Estimate output size (compressed data + overhead)
    output.reserve(expected_len / 2 + 1024);

    // Write PNG signature
    output.extend_from_slice(&PNG_SIGNATURE);

    // Write IHDR chunk
    write_ihdr(output, width, height, color_type);

    // Apply filtering and compression
    let filtered = filter::apply_filters(data, width, height, bytes_per_pixel, options);
    let compressed = deflate_zlib_packed(&filtered, options.compression_level);

    // Write IDAT chunk(s)
    write_idat_chunks(output, &compressed);

    // Write IEND chunk
    write_iend(output);

    Ok(())
}

/// Encode raw pixel data as PNG into a caller-provided buffer, returning DEFLATE timing stats.
///
/// Only available when the `timing` feature is enabled. This mirrors `encode_into`
/// but surfaces per-stage DEFLATE timings to aid profiling without external tools.
#[cfg(feature = "timing")]
pub fn encode_into_with_stats(
    output: &mut Vec<u8>,
    data: &[u8],
    width: u32,
    height: u32,
    color_type: ColorType,
    options: &PngOptions,
) -> Result<DeflateStats> {
    if !(1..=9).contains(&options.compression_level) {
        return Err(Error::InvalidCompressionLevel(options.compression_level));
    }

    if width == 0 || height == 0 {
        return Err(Error::InvalidDimensions { width, height });
    }

    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(Error::ImageTooLarge {
            width,
            height,
            max: MAX_DIMENSION,
        });
    }

    let bytes_per_pixel = color_type.bytes_per_pixel();
    let expected_len = width as usize * height as usize * bytes_per_pixel;
    if data.len() != expected_len {
        return Err(Error::InvalidDataLength {
            expected: expected_len,
            actual: data.len(),
        });
    }

    output.clear();
    output.reserve(expected_len / 2 + 1024);
    output.extend_from_slice(&PNG_SIGNATURE);
    write_ihdr(output, width, height, color_type);

    let filtered = filter::apply_filters(data, width, height, bytes_per_pixel, options);
    let (compressed, stats) = deflate_zlib_packed_with_stats(&filtered, options.compression_level);
    write_idat_chunks(output, &compressed);
    write_iend(output);

    Ok(stats)
}

/// Write IHDR (image header) chunk.
fn write_ihdr(output: &mut Vec<u8>, width: u32, height: u32, color_type: ColorType) {
    let mut ihdr_data = Vec::with_capacity(13);

    // Width (4 bytes, big-endian)
    ihdr_data.extend_from_slice(&width.to_be_bytes());

    // Height (4 bytes, big-endian)
    ihdr_data.extend_from_slice(&height.to_be_bytes());

    // Bit depth (1 byte)
    ihdr_data.push(color_type.png_bit_depth());

    // Color type (1 byte)
    ihdr_data.push(color_type.png_color_type());

    // Compression method (1 byte) - always 0 (DEFLATE)
    ihdr_data.push(0);

    // Filter method (1 byte) - always 0 (adaptive)
    ihdr_data.push(0);

    // Interlace method (1 byte) - 0 (no interlace)
    ihdr_data.push(0);

    chunk::write_chunk(output, b"IHDR", &ihdr_data);
}

/// Write IHDR for indexed color (bit depth 8, color type 3).
fn write_ihdr_indexed(output: &mut Vec<u8>, width: u32, height: u32) {
    let mut ihdr_data = Vec::with_capacity(13);
    ihdr_data.extend_from_slice(&width.to_be_bytes());
    ihdr_data.extend_from_slice(&height.to_be_bytes());
    ihdr_data.push(8); // bit depth
    ihdr_data.push(3); // color type: indexed
    ihdr_data.push(0); // compression
    ihdr_data.push(0); // filter
    ihdr_data.push(0); // interlace
    chunk::write_chunk(output, b"IHDR", &ihdr_data);
}

/// Write PLTE chunk.
fn write_plte_chunk(output: &mut Vec<u8>, palette: &[[u8; 3]]) {
    let mut data = Vec::with_capacity(palette.len() * 3);
    for entry in palette {
        data.extend_from_slice(entry);
    }
    chunk::write_chunk(output, b"PLTE", &data);
}

/// Write optional tRNS chunk for palette transparency.
fn write_trns_chunk(output: &mut Vec<u8>, alpha: &[u8]) {
    chunk::write_chunk(output, b"tRNS", alpha);
}

/// Write IDAT (image data) chunks.
fn write_idat_chunks(output: &mut Vec<u8>, compressed: &[u8]) {
    // Write in larger chunks to reduce per-chunk overhead (CRC/length)
    const CHUNK_SIZE: usize = 256 * 1024;

    for chunk_data in compressed.chunks(CHUNK_SIZE) {
        chunk::write_chunk(output, b"IDAT", chunk_data);
    }
}

/// Write IEND (image end) chunk.
fn write_iend(output: &mut Vec<u8>) {
    chunk::write_chunk(output, b"IEND", &[]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_1x1_rgb() {
        let pixels = vec![255, 0, 0]; // Red pixel
        let png = encode(&pixels, 1, 1, ColorType::Rgb).unwrap();

        // Check PNG signature
        assert_eq!(&png[0..8], &PNG_SIGNATURE);

        // Should have IHDR, IDAT, IEND
        assert!(png.len() > 8 + 12 + 12 + 12); // signature + 3 chunks minimum
    }

    #[test]
    fn test_encode_1x1_rgba() {
        let pixels = vec![255, 0, 0, 255]; // Red opaque pixel
        let png = encode(&pixels, 1, 1, ColorType::Rgba).unwrap();

        assert_eq!(&png[0..8], &PNG_SIGNATURE);
    }

    #[test]
    fn test_encode_invalid_dimensions() {
        let pixels = vec![255, 0, 0];
        let result = encode(&pixels, 0, 1, ColorType::Rgb);
        assert!(matches!(result, Err(Error::InvalidDimensions { .. })));
    }

    #[test]
    fn test_encode_invalid_data_length() {
        let pixels = vec![255, 0]; // Too short for 1x1 RGB
        let result = encode(&pixels, 1, 1, ColorType::Rgb);
        assert!(matches!(result, Err(Error::InvalidDataLength { .. })));
    }

    #[test]
    fn test_encode_into_reuses_buffer() {
        let mut output = Vec::with_capacity(64);
        let pixels1 = vec![0u8, 0, 0]; // black 1x1 RGB
        encode_into(
            &mut output,
            &pixels1,
            1,
            1,
            ColorType::Rgb,
            &PngOptions::default(),
        )
        .unwrap();
        let first = output.clone();
        let first_cap = output.capacity();
        assert!(!first.is_empty());

        let pixels2 = vec![255u8, 0, 0]; // red 1x1 RGB
        encode_into(
            &mut output,
            &pixels2,
            1,
            1,
            ColorType::Rgb,
            &PngOptions::default(),
        )
        .unwrap();

        assert_ne!(
            first, output,
            "buffer should have been reused and rewritten"
        );
        assert!(output.capacity() >= first_cap);
        assert_eq!(&output[0..8], &PNG_SIGNATURE);
    }

    #[test]
    fn test_encode_2x2_checkerboard() {
        // 2x2 black and white checkerboard
        let pixels = vec![
            0, 0, 0, 255, 255, 255, // Row 1: black, white
            255, 255, 255, 0, 0, 0, // Row 2: white, black
        ];
        let png = encode(&pixels, 2, 2, ColorType::Rgb).unwrap();

        assert_eq!(&png[0..8], &PNG_SIGNATURE);
    }

    #[test]
    fn test_encode_grayscale() {
        let pixels = vec![128, 255, 0, 64]; // 2x2 grayscale
        let png = encode(&pixels, 2, 2, ColorType::Gray).unwrap();

        assert_eq!(&png[0..8], &PNG_SIGNATURE);
    }

    #[test]
    fn test_encode_indexed_basic() {
        // 2x2 image with four palette entries
        let pixels = vec![0u8, 1, 2, 3];
        let palette = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0]];
        let png = encode_indexed(&pixels, 2, 2, &palette, None).unwrap();

        // Signature + IHDR + PLTE + IDAT + IEND (minimal structure)
        assert_eq!(&png[0..8], &PNG_SIGNATURE);
        // IHDR color type byte at offset 25 should be 3 (indexed)
        assert_eq!(png[25], 3);
        // Ensure PLTE exists
        assert!(png.windows(4).any(|w| w == b"PLTE"), "PLTE chunk missing");
    }
}
