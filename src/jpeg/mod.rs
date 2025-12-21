//! JPEG encoder implementation.
//!
//! Implements baseline JPEG encoding (DCT-based lossy compression).

pub mod dct;
pub mod huffman;
pub mod quantize;

use crate::bits::BitWriterMsb;
use crate::color::{rgb_to_ycbcr, ColorType};
use crate::error::{Error, Result};

use dct::dct_2d;
use huffman::{encode_block, HuffmanTables};
use quantize::{quantize_block, QuantizationTables};

/// Maximum supported image dimension for JPEG.
const MAX_DIMENSION: u32 = 65535;

/// JPEG markers.
const SOI: u16 = 0xFFD8; // Start of Image
const EOI: u16 = 0xFFD9; // End of Image
const APP0: u16 = 0xFFE0; // JFIF marker
const DQT: u16 = 0xFFDB; // Define Quantization Table
const SOF0: u16 = 0xFFC0; // Start of Frame (baseline DCT)
const DHT: u16 = 0xFFC4; // Define Huffman Table
const SOS: u16 = 0xFFDA; // Start of Scan

/// Encode raw pixel data as JPEG.
///
/// # Arguments
/// * `data` - Raw pixel data (RGB, row-major order)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `quality` - Quality level 1-100 (higher = better quality, larger file)
///
/// # Returns
/// Complete JPEG file as bytes.
pub fn encode(data: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
    encode_with_color(data, width, height, quality, ColorType::Rgb)
}

/// Encode raw pixel data as JPEG with specified color type.
pub fn encode_with_color(
    data: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    color_type: ColorType,
) -> Result<Vec<u8>> {
    // Validate quality
    if quality == 0 || quality > 100 {
        return Err(Error::InvalidQuality(quality));
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

    // Validate color type (JPEG only supports RGB and Gray)
    let bytes_per_pixel = match color_type {
        ColorType::Rgb => 3,
        ColorType::Gray => 1,
        _ => return Err(Error::UnsupportedColorType),
    };

    // Validate data length
    let expected_len = width as usize * height as usize * bytes_per_pixel;
    if data.len() != expected_len {
        return Err(Error::InvalidDataLength {
            expected: expected_len,
            actual: data.len(),
        });
    }

    let mut output = Vec::with_capacity(expected_len / 4);

    // Create quantization and Huffman tables
    let quant_tables = QuantizationTables::with_quality(quality);
    let huff_tables = HuffmanTables::default();

    // Write JPEG headers
    write_soi(&mut output);
    write_app0(&mut output);
    write_dqt(&mut output, &quant_tables);
    write_sof0(&mut output, width, height, color_type);
    write_dht(&mut output, &huff_tables);

    // Write scan data
    write_sos(&mut output, color_type);
    encode_scan(
        &mut output,
        data,
        width,
        height,
        color_type,
        &quant_tables,
        &huff_tables,
    );

    // Write end marker
    write_eoi(&mut output);

    Ok(output)
}

/// Write SOI (Start of Image) marker.
fn write_soi(output: &mut Vec<u8>) {
    output.extend_from_slice(&SOI.to_be_bytes());
}

/// Write EOI (End of Image) marker.
fn write_eoi(output: &mut Vec<u8>) {
    output.extend_from_slice(&EOI.to_be_bytes());
}

/// Write APP0 (JFIF) marker.
fn write_app0(output: &mut Vec<u8>) {
    output.extend_from_slice(&APP0.to_be_bytes());

    // Length (16 bytes including length field)
    output.extend_from_slice(&16u16.to_be_bytes());

    // JFIF identifier
    output.extend_from_slice(b"JFIF\0");

    // Version 1.01
    output.push(1);
    output.push(1);

    // Units: 0 = no units (aspect ratio only)
    output.push(0);

    // X density
    output.extend_from_slice(&1u16.to_be_bytes());

    // Y density
    output.extend_from_slice(&1u16.to_be_bytes());

    // Thumbnail dimensions (0x0 = no thumbnail)
    output.push(0);
    output.push(0);
}

/// Write DQT (Define Quantization Table) marker.
fn write_dqt(output: &mut Vec<u8>, tables: &QuantizationTables) {
    // Luminance table
    output.extend_from_slice(&DQT.to_be_bytes());
    output.extend_from_slice(&67u16.to_be_bytes()); // Length: 2 + 1 + 64
    output.push(0); // Table 0, 8-bit precision
    output.extend_from_slice(&tables.luminance);

    // Chrominance table
    output.extend_from_slice(&DQT.to_be_bytes());
    output.extend_from_slice(&67u16.to_be_bytes());
    output.push(1); // Table 1, 8-bit precision
    output.extend_from_slice(&tables.chrominance);
}

/// Write SOF0 (Start of Frame) marker.
fn write_sof0(output: &mut Vec<u8>, width: u32, height: u32, color_type: ColorType) {
    output.extend_from_slice(&SOF0.to_be_bytes());

    let num_components = match color_type {
        ColorType::Gray => 1,
        _ => 3,
    };

    // Length: 8 + 3*num_components
    let length = 8 + 3 * num_components;
    output.extend_from_slice(&(length as u16).to_be_bytes());

    // Precision: 8 bits
    output.push(8);

    // Height and width
    output.extend_from_slice(&(height as u16).to_be_bytes());
    output.extend_from_slice(&(width as u16).to_be_bytes());

    // Number of components
    output.push(num_components);

    if num_components == 1 {
        // Grayscale: 1 component
        output.push(1); // Component ID
        output.push(0x11); // Sampling factor (1x1)
        output.push(0); // Quantization table 0
    } else {
        // YCbCr: 3 components
        // Y component
        output.push(1); // Component ID
        output.push(0x11); // Sampling factor (1x1, no subsampling)
        output.push(0); // Quantization table 0 (luminance)

        // Cb component
        output.push(2);
        output.push(0x11);
        output.push(1); // Quantization table 1 (chrominance)

        // Cr component
        output.push(3);
        output.push(0x11);
        output.push(1);
    }
}

/// Write DHT (Define Huffman Table) marker.
fn write_dht(output: &mut Vec<u8>, tables: &HuffmanTables) {
    // DC luminance
    write_huffman_table(output, 0x00, &tables.dc_lum_bits, &tables.dc_lum_vals);

    // DC chrominance
    write_huffman_table(output, 0x01, &tables.dc_chrom_bits, &tables.dc_chrom_vals);

    // AC luminance
    write_huffman_table(output, 0x10, &tables.ac_lum_bits, &tables.ac_lum_vals);

    // AC chrominance
    write_huffman_table(output, 0x11, &tables.ac_chrom_bits, &tables.ac_chrom_vals);
}

/// Write a single Huffman table.
fn write_huffman_table(output: &mut Vec<u8>, table_id: u8, bits: &[u8; 16], vals: &[u8]) {
    output.extend_from_slice(&DHT.to_be_bytes());

    // Length: 2 + 1 + 16 + num_values
    let length = 2 + 1 + 16 + vals.len();
    output.extend_from_slice(&(length as u16).to_be_bytes());

    // Table class and ID
    output.push(table_id);

    // Number of codes of each length
    output.extend_from_slice(bits);

    // Values
    output.extend_from_slice(vals);
}

/// Write SOS (Start of Scan) marker.
fn write_sos(output: &mut Vec<u8>, color_type: ColorType) {
    output.extend_from_slice(&SOS.to_be_bytes());

    let num_components = match color_type {
        ColorType::Gray => 1,
        _ => 3,
    };

    // Length: 6 + 2*num_components
    let length = 6 + 2 * num_components;
    output.extend_from_slice(&(length as u16).to_be_bytes());

    // Number of components
    output.push(num_components);

    if num_components == 1 {
        output.push(1); // Component ID
        output.push(0x00); // DC/AC table selectors
    } else {
        // Y component: DC table 0, AC table 0
        output.push(1);
        output.push(0x00);

        // Cb component: DC table 1, AC table 1
        output.push(2);
        output.push(0x11);

        // Cr component: DC table 1, AC table 1
        output.push(3);
        output.push(0x11);
    }

    // Spectral selection and successive approximation
    output.push(0); // Start of spectral selection
    output.push(63); // End of spectral selection
    output.push(0); // Successive approximation
}

/// Encode the image scan data.
fn encode_scan(
    output: &mut Vec<u8>,
    data: &[u8],
    width: u32,
    height: u32,
    color_type: ColorType,
    quant_tables: &QuantizationTables,
    huff_tables: &HuffmanTables,
) {
    let mut writer = BitWriterMsb::new();

    // Convert to YCbCr and process in 8x8 blocks
    let width = width as usize;
    let height = height as usize;

    // Calculate padded dimensions (multiple of 8)
    let padded_width = (width + 7) & !7;
    let padded_height = (height + 7) & !7;

    // Previous DC values for differential encoding
    let mut prev_dc_y = 0i16;
    let mut prev_dc_cb = 0i16;
    let mut prev_dc_cr = 0i16;

    // Process blocks
    for block_y in (0..padded_height).step_by(8) {
        for block_x in (0..padded_width).step_by(8) {
            // Extract and convert 8x8 block
            let (y_block, cb_block, cr_block) =
                extract_block(data, width, height, block_x, block_y, color_type);

            // Process Y channel
            let y_dct = dct_2d(&y_block);
            let y_quant = quantize_block(&y_dct, &quant_tables.luminance_table);
            prev_dc_y = encode_block(
                &mut writer,
                &y_quant,
                prev_dc_y,
                true,  // is_luminance
                huff_tables,
            );

            if color_type != ColorType::Gray {
                // Process Cb channel
                let cb_dct = dct_2d(&cb_block);
                let cb_quant = quantize_block(&cb_dct, &quant_tables.chrominance_table);
                prev_dc_cb = encode_block(
                    &mut writer,
                    &cb_quant,
                    prev_dc_cb,
                    false, // is_chrominance
                    huff_tables,
                );

                // Process Cr channel
                let cr_dct = dct_2d(&cr_block);
                let cr_quant = quantize_block(&cr_dct, &quant_tables.chrominance_table);
                prev_dc_cr = encode_block(
                    &mut writer,
                    &cr_quant,
                    prev_dc_cr,
                    false,
                    huff_tables,
                );
            }
        }
    }

    // Flush the bit writer and append to output
    output.extend_from_slice(&writer.finish());
}

/// Extract an 8x8 block from the image and convert to YCbCr.
fn extract_block(
    data: &[u8],
    width: usize,
    height: usize,
    block_x: usize,
    block_y: usize,
    color_type: ColorType,
) -> ([f32; 64], [f32; 64], [f32; 64]) {
    let mut y_block = [0.0f32; 64];
    let mut cb_block = [0.0f32; 64];
    let mut cr_block = [0.0f32; 64];

    for dy in 0..8 {
        for dx in 0..8 {
            let x = (block_x + dx).min(width - 1);
            let y = (block_y + dy).min(height - 1);
            let idx = dy * 8 + dx;

            match color_type {
                ColorType::Gray => {
                    let gray = data[y * width + x];
                    y_block[idx] = gray as f32 - 128.0;
                    cb_block[idx] = 0.0;
                    cr_block[idx] = 0.0;
                }
                ColorType::Rgb => {
                    let pixel_idx = (y * width + x) * 3;
                    let r = data[pixel_idx];
                    let g = data[pixel_idx + 1];
                    let b = data[pixel_idx + 2];
                    let (yc, cb, cr) = rgb_to_ycbcr(r, g, b);
                    y_block[idx] = yc as f32 - 128.0;
                    cb_block[idx] = cb as f32 - 128.0;
                    cr_block[idx] = cr as f32 - 128.0;
                }
                _ => unreachable!(),
            }
        }
    }

    (y_block, cb_block, cr_block)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_1x1_rgb() {
        let pixels = vec![255, 0, 0]; // Red pixel
        let jpeg = encode(&pixels, 1, 1, 85).unwrap();

        // Check JPEG markers
        assert_eq!(&jpeg[0..2], &SOI.to_be_bytes());
        assert_eq!(&jpeg[jpeg.len() - 2..], &EOI.to_be_bytes());
    }

    #[test]
    fn test_encode_8x8_rgb() {
        // 8x8 gradient
        let mut pixels = Vec::with_capacity(8 * 8 * 3);
        for y in 0..8 {
            for x in 0..8 {
                let val = ((x + y) * 16) as u8;
                pixels.extend_from_slice(&[val, val, val]);
            }
        }

        let jpeg = encode(&pixels, 8, 8, 85).unwrap();
        assert_eq!(&jpeg[0..2], &SOI.to_be_bytes());
    }

    #[test]
    fn test_encode_invalid_quality() {
        let pixels = vec![255, 0, 0];
        assert!(matches!(
            encode(&pixels, 1, 1, 0),
            Err(Error::InvalidQuality(0))
        ));
        assert!(matches!(
            encode(&pixels, 1, 1, 101),
            Err(Error::InvalidQuality(101))
        ));
    }

    #[test]
    fn test_encode_invalid_dimensions() {
        let pixels = vec![255, 0, 0];
        assert!(matches!(
            encode(&pixels, 0, 1, 85),
            Err(Error::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn test_encode_grayscale() {
        let pixels = vec![128; 64]; // 8x8 gray
        let jpeg = encode_with_color(&pixels, 8, 8, 85, ColorType::Gray).unwrap();
        assert_eq!(&jpeg[0..2], &SOI.to_be_bytes());
    }
}
