//! PNG conformance tests.
//!
//! Tests PNG encoding against expected output and validates
//! that encoded images can be decoded correctly.

use comprs::{png, ColorType};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Test that PNG output has correct header.
#[test]
fn test_png_signature() {
    let pixels = vec![255, 0, 0]; // 1x1 red pixel
    let result = png::encode(&pixels, 1, 1, ColorType::Rgb).unwrap();

    // PNG signature
    assert_eq!(
        &result[0..8],
        &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    );
}

/// Test IHDR chunk format.
#[test]
fn test_ihdr_chunk() {
    let pixels = vec![0u8; 100 * 100 * 3]; // 100x100 RGB
    let result = png::encode(&pixels, 100, 100, ColorType::Rgb).unwrap();

    // IHDR should be right after signature
    // Length (4 bytes) + "IHDR" (4 bytes) + data (13 bytes) + CRC (4 bytes)

    // Length should be 13
    assert_eq!(&result[8..12], &[0, 0, 0, 13]);

    // Chunk type should be IHDR
    assert_eq!(&result[12..16], b"IHDR");

    // Width (100 = 0x64)
    assert_eq!(&result[16..20], &[0, 0, 0, 100]);

    // Height (100 = 0x64)
    assert_eq!(&result[20..24], &[0, 0, 0, 100]);

    // Bit depth (8)
    assert_eq!(result[24], 8);

    // Color type (2 = RGB)
    assert_eq!(result[25], 2);

    // Compression method (0 = DEFLATE)
    assert_eq!(result[26], 0);

    // Filter method (0 = adaptive)
    assert_eq!(result[27], 0);

    // Interlace method (0 = none)
    assert_eq!(result[28], 0);
}

/// Test that IEND chunk is present at end.
#[test]
fn test_iend_chunk() {
    let pixels = vec![128u8; 10 * 10 * 3];
    let result = png::encode(&pixels, 10, 10, ColorType::Rgb).unwrap();

    // IEND chunk should be at the end
    // It's 12 bytes: length (4) + type (4) + CRC (4)
    let iend_start = result.len() - 12;

    // Length should be 0
    assert_eq!(&result[iend_start..iend_start + 4], &[0, 0, 0, 0]);

    // Type should be IEND
    assert_eq!(&result[iend_start + 4..iend_start + 8], b"IEND");

    // CRC of "IEND" should be 0xAE426082
    assert_eq!(
        &result[iend_start + 8..iend_start + 12],
        &[0xAE, 0x42, 0x60, 0x82]
    );
}

/// Test encoding different color types.
#[test]
fn test_color_types() {
    // Grayscale
    let gray = vec![128u8; 4 * 4];
    let result = png::encode(&gray, 4, 4, ColorType::Gray).unwrap();
    assert_eq!(result[25], 0); // Color type 0

    // Grayscale + Alpha
    let gray_alpha = vec![128u8; 4 * 4 * 2];
    let result = png::encode(&gray_alpha, 4, 4, ColorType::GrayAlpha).unwrap();
    assert_eq!(result[25], 4); // Color type 4

    // RGB
    let rgb = vec![128u8; 4 * 4 * 3];
    let result = png::encode(&rgb, 4, 4, ColorType::Rgb).unwrap();
    assert_eq!(result[25], 2); // Color type 2

    // RGBA
    let rgba = vec![128u8; 4 * 4 * 4];
    let result = png::encode(&rgba, 4, 4, ColorType::Rgba).unwrap();
    assert_eq!(result[25], 6); // Color type 6
}

/// Test that different images produce different output.
#[test]
fn test_different_images() {
    let black = vec![0u8; 8 * 8 * 3];
    let white = vec![255u8; 8 * 8 * 3];

    let black_png = png::encode(&black, 8, 8, ColorType::Rgb).unwrap();
    let white_png = png::encode(&white, 8, 8, ColorType::Rgb).unwrap();

    // Should be different
    assert_ne!(black_png, white_png);
}

/// Ensure encoded PNGs decode correctly via the `image` crate (zlib wrapper validity).
#[test]
fn test_png_roundtrip_decode_rgb() {
    let width = 3;
    let height = 2;
    let pixels = vec![
        // row 0
        255, 0, 0, // red
        0, 255, 0, // green
        0, 0, 255, // blue
        // row 1
        255, 255, 0, // yellow
        0, 255, 255, // cyan
        255, 0, 255, // magenta
    ];

    let encoded = png::encode(&pixels, width, height, ColorType::Rgb).unwrap();

    let decoded = image::load_from_memory(&encoded).expect("decode").to_rgb8();
    assert_eq!(decoded.width(), width);
    assert_eq!(decoded.height(), height);
    assert_eq!(decoded.as_raw(), &pixels);
}

/// Randomized small-images roundtrip across color types to ensure decodability.
#[test]
fn test_png_roundtrip_random_small() {
    let mut rng = StdRng::seed_from_u64(42);
    let dims = [(1, 1), (2, 3), (3, 2), (4, 4), (8, 5)];
    let color_types = [
        ColorType::Gray,
        ColorType::GrayAlpha,
        ColorType::Rgb,
        ColorType::Rgba,
    ];

    for &(w, h) in &dims {
        for &ct in &color_types {
            let bpp = ct.bytes_per_pixel();
            let mut pixels = vec![0u8; (w * h) as usize * bpp];
            rng.fill(pixels.as_mut_slice());

            let encoded =
                png::encode(&pixels, w as u32, h as u32, ct).expect("encode random png");
            let decoded = image::load_from_memory(&encoded).expect("decode").to_rgba8();

            assert_eq!(decoded.width(), w as u32);
            assert_eq!(decoded.height(), h as u32);
        }
    }
}

/// Test filter strategies produce valid output.
#[test]
fn test_filter_strategies() {
    use png::{FilterStrategy, PngOptions};

    let pixels = vec![128u8; 16 * 16 * 3];

    let strategies = [
        FilterStrategy::None,
        FilterStrategy::Sub,
        FilterStrategy::Up,
        FilterStrategy::Average,
        FilterStrategy::Paeth,
        FilterStrategy::Adaptive,
    ];

    for strategy in &strategies {
        let options = PngOptions {
            filter_strategy: *strategy,
            compression_level: 6,
        };

        let result = png::encode_with_options(&pixels, 16, 16, ColorType::Rgb, &options).unwrap();

        // All should produce valid PNG files
        assert_eq!(&result[0..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    }
}

/// Test compression levels.
#[test]
fn test_compression_levels() {
    use png::PngOptions;

    let pixels: Vec<u8> = (0..64 * 64 * 3).map(|i| (i % 256) as u8).collect();

    let mut sizes = Vec::new();

    for level in 1..=9 {
        let options = PngOptions {
            compression_level: level,
            ..Default::default()
        };

        let result = png::encode_with_options(&pixels, 64, 64, ColorType::Rgb, &options).unwrap();
        sizes.push((level, result.len()));
    }

    // Higher compression levels should generally produce smaller files
    // (though not strictly monotonic for all images)
    let level1_size = sizes[0].1;
    let level9_size = sizes[8].1;
    assert!(level9_size <= level1_size);
}

/// Test error handling for invalid input.
#[test]
fn test_invalid_input() {
    // Zero dimensions
    assert!(png::encode(&[0, 0, 0], 0, 1, ColorType::Rgb).is_err());
    assert!(png::encode(&[0, 0, 0], 1, 0, ColorType::Rgb).is_err());

    // Wrong data length
    assert!(png::encode(&[0, 0], 1, 1, ColorType::Rgb).is_err()); // Too short
    assert!(png::encode(&[0, 0, 0, 0], 1, 1, ColorType::Rgb).is_err()); // Too long
}

/// Test large image encoding.
#[test]
fn test_large_image() {
    // 1000x1000 RGB image
    let pixels = vec![100u8; 1000 * 1000 * 3];
    let result = png::encode(&pixels, 1000, 1000, ColorType::Rgb).unwrap();

    // Should produce valid PNG
    assert_eq!(&result[0..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

    // IHDR should have correct dimensions
    assert_eq!(&result[16..20], &[0, 0, 0x03, 0xE8]); // 1000 in big-endian
    assert_eq!(&result[20..24], &[0, 0, 0x03, 0xE8]);
}
