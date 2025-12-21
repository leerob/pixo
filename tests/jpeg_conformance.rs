//! JPEG conformance tests.
//!
//! Tests JPEG encoding for correctness and validates
//! that encoded images contain proper markers.

use comprs::{jpeg, ColorType};

/// Test that JPEG output has correct markers.
#[test]
fn test_jpeg_markers() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let result = jpeg::encode(&pixels, 8, 8, 85).unwrap();

    // SOI marker
    assert_eq!(&result[0..2], &[0xFF, 0xD8]);

    // EOI marker at end
    assert_eq!(&result[result.len() - 2..], &[0xFF, 0xD9]);
}

/// Test APP0 (JFIF) marker.
#[test]
fn test_app0_marker() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let result = jpeg::encode(&pixels, 8, 8, 85).unwrap();

    // APP0 should be right after SOI
    assert_eq!(&result[2..4], &[0xFF, 0xE0]);

    // JFIF identifier
    assert_eq!(&result[6..11], b"JFIF\0");
}

/// Test different quality levels.
#[test]
fn test_quality_levels() {
    let pixels: Vec<u8> = (0..64 * 64 * 3).map(|i| (i % 256) as u8).collect();

    let sizes: Vec<(u8, usize)> = [10, 25, 50, 75, 90, 100]
        .iter()
        .map(|&q| {
            let result = jpeg::encode(&pixels, 64, 64, q).unwrap();
            (q, result.len())
        })
        .collect();

    // Higher quality should produce larger files
    for i in 1..sizes.len() {
        assert!(
            sizes[i].1 >= sizes[i - 1].1,
            "Quality {} produced {} bytes, but quality {} produced {} bytes",
            sizes[i].0,
            sizes[i].1,
            sizes[i - 1].0,
            sizes[i - 1].1
        );
    }
}

/// Test different image sizes.
#[test]
fn test_various_sizes() {
    let sizes = [
        (1, 1),
        (7, 7),   // Not multiple of 8
        (8, 8),   // Exact MCU
        (9, 9),   // Just over one MCU
        (16, 16), // Two MCUs
        (100, 50),
        (50, 100),
    ];

    for (width, height) in sizes {
        let pixels = vec![128u8; (width * height * 3) as usize];
        let result = jpeg::encode(&pixels, width, height, 85);

        assert!(result.is_ok(), "Failed for size {}x{}", width, height);

        let data = result.unwrap();
        assert_eq!(&data[0..2], &[0xFF, 0xD8], "Missing SOI for {}x{}", width, height);
        assert_eq!(
            &data[data.len() - 2..],
            &[0xFF, 0xD9],
            "Missing EOI for {}x{}",
            width,
            height
        );
    }
}

/// Test grayscale encoding.
#[test]
fn test_grayscale() {
    let pixels = vec![128u8; 32 * 32];
    let result = jpeg::encode_with_color(&pixels, 32, 32, 85, ColorType::Gray).unwrap();

    // Should have proper markers
    assert_eq!(&result[0..2], &[0xFF, 0xD8]);
    assert_eq!(&result[result.len() - 2..], &[0xFF, 0xD9]);

    // Should be smaller than RGB (1 component vs 3)
    let rgb_pixels = vec![128u8; 32 * 32 * 3];
    let rgb_result = jpeg::encode(&rgb_pixels, 32, 32, 85).unwrap();
    assert!(result.len() < rgb_result.len());
}

/// Test error handling.
#[test]
fn test_error_handling() {
    // Invalid quality
    let pixels = vec![0u8; 8 * 8 * 3];
    assert!(jpeg::encode(&pixels, 8, 8, 0).is_err());
    assert!(jpeg::encode(&pixels, 8, 8, 101).is_err());

    // Invalid dimensions
    assert!(jpeg::encode(&pixels, 0, 8, 85).is_err());
    assert!(jpeg::encode(&pixels, 8, 0, 85).is_err());

    // Wrong data length
    assert!(jpeg::encode(&[0, 0], 8, 8, 85).is_err());
}

/// Test that encoding produces deterministic output.
#[test]
fn test_deterministic() {
    let pixels = vec![100u8; 16 * 16 * 3];

    let result1 = jpeg::encode(&pixels, 16, 16, 85).unwrap();
    let result2 = jpeg::encode(&pixels, 16, 16, 85).unwrap();

    assert_eq!(result1, result2);
}

/// Test different patterns compress differently.
#[test]
fn test_pattern_compression() {
    // Solid color (should compress very well)
    let solid = vec![128u8; 64 * 64 * 3];
    let solid_result = jpeg::encode(&solid, 64, 64, 85).unwrap();

    // Gradient (compresses reasonably)
    let mut gradient = Vec::with_capacity(64 * 64 * 3);
    for y in 0..64 {
        for x in 0..64 {
            gradient.push(((x * 4) % 256) as u8);
            gradient.push(((y * 4) % 256) as u8);
            gradient.push((((x + y) * 2) % 256) as u8);
        }
    }
    let gradient_result = jpeg::encode(&gradient, 64, 64, 85).unwrap();

    // Random-ish (compresses poorly)
    let mut noisy = Vec::with_capacity(64 * 64 * 3);
    let mut seed = 42u32;
    for _ in 0..(64 * 64 * 3) {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        noisy.push((seed >> 16) as u8);
    }
    let noisy_result = jpeg::encode(&noisy, 64, 64, 85).unwrap();

    // Solid should be smallest, noisy should be largest
    assert!(solid_result.len() < gradient_result.len());
    assert!(gradient_result.len() < noisy_result.len());
}

/// Test that DQT tables are present.
#[test]
fn test_dqt_present() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let result = jpeg::encode(&pixels, 8, 8, 85).unwrap();

    // Look for DQT marker (0xFFDB)
    let mut found_dqt = false;
    for i in 0..result.len() - 1 {
        if result[i] == 0xFF && result[i + 1] == 0xDB {
            found_dqt = true;
            break;
        }
    }
    assert!(found_dqt, "DQT marker not found");
}

/// Test that SOF0 marker is present.
#[test]
fn test_sof0_present() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let result = jpeg::encode(&pixels, 8, 8, 85).unwrap();

    // Look for SOF0 marker (0xFFC0)
    let mut found_sof0 = false;
    for i in 0..result.len() - 1 {
        if result[i] == 0xFF && result[i + 1] == 0xC0 {
            found_sof0 = true;
            break;
        }
    }
    assert!(found_sof0, "SOF0 marker not found");
}

/// Test that DHT markers are present.
#[test]
fn test_dht_present() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let result = jpeg::encode(&pixels, 8, 8, 85).unwrap();

    // Count DHT markers (0xFFC4) - should have 4 (DC lum, DC chrom, AC lum, AC chrom)
    let mut dht_count = 0;
    for i in 0..result.len() - 1 {
        if result[i] == 0xFF && result[i + 1] == 0xC4 {
            dht_count += 1;
        }
    }
    assert_eq!(dht_count, 4, "Expected 4 DHT markers, found {}", dht_count);
}

/// Test that SOS marker is present.
#[test]
fn test_sos_present() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let result = jpeg::encode(&pixels, 8, 8, 85).unwrap();

    // Look for SOS marker (0xFFDA)
    let mut found_sos = false;
    for i in 0..result.len() - 1 {
        if result[i] == 0xFF && result[i + 1] == 0xDA {
            found_sos = true;
            break;
        }
    }
    assert!(found_sos, "SOS marker not found");
}
