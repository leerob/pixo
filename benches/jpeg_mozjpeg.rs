//! JPEG-only benchmark: comprs vs mozjpeg comparison
//!
//! This benchmark focuses exclusively on JPEG encoding performance,
//! comparing comprs against mozjpeg on various test images.
//!
//! Run with:
//! ```
//! cargo bench --bench jpeg_mozjpeg
//! ```
//!
//! For size comparison only (no timing):
//! ```
//! cargo bench --bench jpeg_mozjpeg -- --summary-only
//! ```

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, BenchmarkId, Criterion, Throughput};

use comprs::{jpeg, ColorType};

// ============================================================================
// Test Image Loading
// ============================================================================

struct TestImage {
    name: &'static str,
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}

fn load_test_images() -> Vec<TestImage> {
    let mut images = Vec::new();

    // Load JPEG images from tests/fixtures/ directory
    let fixtures = [
        ("multi-agent", "tests/fixtures/multi-agent.jpg"),
        ("browser", "tests/fixtures/browser.jpg"),
        ("review", "tests/fixtures/review.jpg"),
        ("web", "tests/fixtures/web.jpg"),
    ];

    for (name, path) in fixtures {
        if let Ok(img) = image::open(path) {
            let rgb = img.to_rgb8();
            images.push(TestImage {
                name,
                pixels: rgb.to_vec(),
                width: rgb.width(),
                height: rgb.height(),
            });
        }
    }

    // Add synthetic test images for consistent baseline
    images.push(generate_gradient_image("gradient_512", 512, 512));
    images.push(generate_noisy_image("noise_512", 512, 512));
    images.push(generate_photo_like_image("photo_like_512", 512, 512));

    images
}

fn generate_gradient_image(name: &'static str, width: u32, height: u32) -> TestImage {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width) as u8;
            let g = ((y * 255) / height) as u8;
            let b = (((x + y) * 127) / (width + height)) as u8;
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    TestImage {
        name,
        pixels,
        width,
        height,
    }
}

fn generate_noisy_image(name: &'static str, width: u32, height: u32) -> TestImage {
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
    TestImage {
        name,
        pixels,
        width,
        height,
    }
}

/// Generate a photo-like image with smooth gradients and some edges
fn generate_photo_like_image(name: &'static str, width: u32, height: u32) -> TestImage {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            // Sky gradient (top half)
            let sky_blend = if y < height / 2 {
                1.0 - (y as f32 / (height as f32 / 2.0))
            } else {
                0.0
            };

            // Ground gradient (bottom half)
            let ground_blend = if y >= height / 2 {
                (y as f32 - height as f32 / 2.0) / (height as f32 / 2.0)
            } else {
                0.0
            };

            // Add some variation
            let variation = ((x as f32 * 0.1).sin() * 20.0) as i16;

            let r = ((135.0 * sky_blend
                + 34.0 * ground_blend
                + 60.0 * (1.0 - sky_blend - ground_blend)) as i16
                + variation)
                .clamp(0, 255) as u8;
            let g = ((206.0 * sky_blend
                + 139.0 * ground_blend
                + 80.0 * (1.0 - sky_blend - ground_blend)) as i16
                + variation)
                .clamp(0, 255) as u8;
            let b = ((235.0 * sky_blend
                + 34.0 * ground_blend
                + 100.0 * (1.0 - sky_blend - ground_blend)) as i16)
                .clamp(0, 255) as u8;

            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    TestImage {
        name,
        pixels,
        width,
        height,
    }
}

// ============================================================================
// JPEG Encoding Benchmarks
// ============================================================================

fn bench_jpeg_presets(c: &mut Criterion) {
    let images = load_test_images();
    if images.is_empty() {
        eprintln!("Warning: No test images loaded");
        return;
    }

    let mut group = c.benchmark_group("JPEG Presets");

    for img in &images {
        let pixel_bytes = (img.width as u64) * (img.height as u64) * 3;
        group.throughput(Throughput::Bytes(pixel_bytes));

        let mut buf = Vec::new();

        // Fast preset (no optimization)
        group.bench_with_input(BenchmarkId::new("fast", img.name), &img, |b, img| {
            b.iter(|| {
                jpeg::encode_with_options_into(
                    &mut buf,
                    black_box(&img.pixels),
                    img.width,
                    img.height,
                    85,
                    ColorType::Rgb,
                    &jpeg::JpegOptions::fast(85),
                )
                .unwrap()
            });
        });

        // Balanced preset (Huffman optimization)
        group.bench_with_input(BenchmarkId::new("balanced", img.name), &img, |b, img| {
            b.iter(|| {
                jpeg::encode_with_options_into(
                    &mut buf,
                    black_box(&img.pixels),
                    img.width,
                    img.height,
                    85,
                    ColorType::Rgb,
                    &jpeg::JpegOptions::balanced(85),
                )
                .unwrap()
            });
        });

        // Max preset (Huffman + 4:2:0)
        group.bench_with_input(BenchmarkId::new("max", img.name), &img, |b, img| {
            b.iter(|| {
                jpeg::encode_with_options_into(
                    &mut buf,
                    black_box(&img.pixels),
                    img.width,
                    img.height,
                    85,
                    ColorType::Rgb,
                    &jpeg::JpegOptions::max(85),
                )
                .unwrap()
            });
        });
    }

    group.finish();
}

fn bench_jpeg_vs_image_crate(c: &mut Criterion) {
    let images = load_test_images();
    if images.is_empty() {
        return;
    }

    let mut group = c.benchmark_group("JPEG vs image crate");

    for img in &images {
        let pixel_bytes = (img.width as u64) * (img.height as u64) * 3;
        group.throughput(Throughput::Bytes(pixel_bytes));

        let mut buf = Vec::new();

        // comprs max preset
        group.bench_with_input(BenchmarkId::new("comprs_max", img.name), &img, |b, img| {
            b.iter(|| {
                jpeg::encode_with_options_into(
                    &mut buf,
                    black_box(&img.pixels),
                    img.width,
                    img.height,
                    85,
                    ColorType::Rgb,
                    &jpeg::JpegOptions::max(85),
                )
                .unwrap()
            });
        });

        // image crate
        group.bench_with_input(BenchmarkId::new("image_crate", img.name), &img, |b, img| {
            b.iter(|| {
                use image::ImageEncoder;
                let mut output = Vec::new();
                let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, 85);
                encoder
                    .write_image(
                        black_box(&img.pixels),
                        img.width,
                        img.height,
                        image::ColorType::Rgb8,
                    )
                    .unwrap();
                output
            });
        });
    }

    group.finish();
}

// ============================================================================
// Size Comparison Report
// ============================================================================

fn print_size_comparison() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              JPEG SIZE COMPARISON: comprs vs mozjpeg                         ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let images = load_test_images();

    // Check if mozjpeg is available
    let cjpeg_paths = [
        "vendor/mozjpeg/build/cjpeg",
        "vendor/mozjpeg/cjpeg",
        "/usr/local/bin/cjpeg",
        "/opt/homebrew/bin/cjpeg",
    ];
    let cjpeg_bin = cjpeg_paths.iter().find(|p| Path::new(p).exists());

    println!("┌────────────────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬──────────┐");
    println!("│ Image                  │ Pixels    │ Fast      │ Balanced  │ Max       │ mozjpeg   │ Gap      │");
    println!("├────────────────────────┼───────────┼───────────┼───────────┼───────────┼───────────┼──────────┤");

    let tmp_dir = PathBuf::from("target/jpeg-bench");
    let _ = fs::create_dir_all(&tmp_dir);

    for img in &images {
        let pixel_count = img.width * img.height;

        // Encode with each preset
        let mut fast_buf = Vec::new();
        jpeg::encode_with_options_into(
            &mut fast_buf,
            &img.pixels,
            img.width,
            img.height,
            85,
            ColorType::Rgb,
            &jpeg::JpegOptions::fast(85),
        )
        .unwrap();

        let mut balanced_buf = Vec::new();
        jpeg::encode_with_options_into(
            &mut balanced_buf,
            &img.pixels,
            img.width,
            img.height,
            85,
            ColorType::Rgb,
            &jpeg::JpegOptions::balanced(85),
        )
        .unwrap();

        let mut max_buf = Vec::new();
        jpeg::encode_with_options_into(
            &mut max_buf,
            &img.pixels,
            img.width,
            img.height,
            85,
            ColorType::Rgb,
            &jpeg::JpegOptions::max(85),
        )
        .unwrap();

        // Try mozjpeg if available
        let mozjpeg_size = if let Some(cjpeg) = cjpeg_bin {
            encode_with_mozjpeg(
                cjpeg,
                &img.pixels,
                img.width,
                img.height,
                &tmp_dir,
                img.name,
            )
        } else {
            None
        };

        let gap_str = if let Some(moz_size) = mozjpeg_size {
            let gap = (max_buf.len() as f64 / moz_size as f64 - 1.0) * 100.0;
            format!("{gap:+.1}%")
        } else {
            "N/A".to_string()
        };

        let moz_str = mozjpeg_size
            .map(format_size)
            .unwrap_or_else(|| "N/A".to_string());

        println!(
            "│ {:<22} │ {:>9} │ {:>9} │ {:>9} │ {:>9} │ {:>9} │ {:>8} │",
            truncate_name(img.name, 22),
            format!("{}K", pixel_count / 1000),
            format_size(fast_buf.len()),
            format_size(balanced_buf.len()),
            format_size(max_buf.len()),
            moz_str,
            gap_str
        );
    }

    println!("└────────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴──────────┘");
    println!();

    if cjpeg_bin.is_none() {
        println!("Note: mozjpeg (cjpeg) not found. Install mozjpeg for comparison.");
        println!("  macOS: brew install mozjpeg");
        println!("  Or clone and build: git clone https://github.com/mozilla/mozjpeg.git");
        println!();
    }

    // Speed comparison
    println!("┌────────────────────────┬─────────────┬─────────────┬─────────────┬─────────────────────────────┐");
    println!("│ Image                  │ Fast        │ Balanced    │ Max         │ Notes                       │");
    println!("├────────────────────────┼─────────────┼─────────────┼─────────────┼─────────────────────────────┤");

    for img in &images {
        let fast_time = measure_encode_time(
            &img.pixels,
            img.width,
            img.height,
            &jpeg::JpegOptions::fast(85),
        );
        let balanced_time = measure_encode_time(
            &img.pixels,
            img.width,
            img.height,
            &jpeg::JpegOptions::balanced(85),
        );
        let max_time = measure_encode_time(
            &img.pixels,
            img.width,
            img.height,
            &jpeg::JpegOptions::max(85),
        );

        let speedup = fast_time.as_secs_f64() / max_time.as_secs_f64();
        let notes = if speedup > 1.5 {
            format!("fast {speedup:.1}x faster")
        } else {
            "similar speed".to_string()
        };

        println!(
            "│ {:<22} │ {:>11} │ {:>11} │ {:>11} │ {:<27} │",
            truncate_name(img.name, 22),
            format_duration(fast_time),
            format_duration(balanced_time),
            format_duration(max_time),
            notes
        );
    }

    println!("└────────────────────────┴─────────────┴─────────────┴─────────────┴─────────────────────────────┘");
    println!();

    println!("Presets:");
    println!("  Fast:     No optimization, 4:4:4 subsampling, baseline DCT");
    println!("  Balanced: Huffman optimization, 4:4:4 subsampling, baseline DCT");
    println!("  Max:      Huffman + progressive + trellis quantization, 4:2:0 subsampling");
    println!();
}

fn encode_with_mozjpeg(
    cjpeg_bin: &str,
    pixels: &[u8],
    width: u32,
    height: u32,
    tmp_dir: &Path,
    name: &str,
) -> Option<usize> {
    // Write PPM file for mozjpeg input
    let ppm_path = tmp_dir.join(format!("{name}.ppm"));
    let jpg_path = tmp_dir.join(format!("{name}.mozjpeg.jpg"));

    let mut file = fs::File::create(&ppm_path).ok()?;
    writeln!(file, "P6\n{width} {height}\n255").ok()?;
    file.write_all(pixels).ok()?;
    drop(file);

    // Run cjpeg
    let status = Command::new(cjpeg_bin)
        .args([
            "-quality",
            "85",
            "-optimize",
            "-progressive",
            "-outfile",
            jpg_path.to_str()?,
            ppm_path.to_str()?,
        ])
        .output()
        .ok()?;

    if status.status.success() {
        fs::metadata(&jpg_path).ok().map(|m| m.len() as usize)
    } else {
        None
    }
}

fn measure_encode_time(
    pixels: &[u8],
    width: u32,
    height: u32,
    opts: &jpeg::JpegOptions,
) -> Duration {
    let mut buf = Vec::new();

    // Warm up
    for _ in 0..3 {
        jpeg::encode_with_options_into(&mut buf, pixels, width, height, 85, ColorType::Rgb, opts)
            .unwrap();
    }

    // Measure
    let start = Instant::now();
    let iterations = 10;
    for _ in 0..iterations {
        jpeg::encode_with_options_into(&mut buf, pixels, width, height, 85, ColorType::Rgb, opts)
            .unwrap();
    }
    start.elapsed() / iterations
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

fn format_duration(duration: Duration) -> String {
    let micros = duration.as_micros();
    if micros >= 1000 {
        format!("{:.2} ms", duration.as_secs_f64() * 1000.0)
    } else {
        format!("{micros} µs")
    }
}

fn truncate_name(name: &str, max_len: usize) -> String {
    if name.len() <= max_len {
        name.to_string()
    } else {
        format!("{}...", &name[..max_len - 3])
    }
}

// ============================================================================
// Criterion Configuration
// ============================================================================

fn custom_criterion() -> Criterion {
    Criterion::default()
        .sample_size(50)
        .measurement_time(std::time::Duration::from_secs(5))
}

criterion_group! {
    name = benches;
    config = custom_criterion();
    targets = bench_jpeg_presets, bench_jpeg_vs_image_crate
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--summary-only") {
        print_size_comparison();
        return;
    }

    // Run benchmarks
    benches();

    // Print size comparison after benchmarks
    print_size_comparison();
}
