//! Comprehensive library comparison benchmark.
//!
//! Compares comprs against popular image compression libraries and produces
//! a detailed summary table similar to the benchmarks/README.md format.
//!
//! Run with: cargo bench --bench comparison
//!
//! For a quick summary without full benchmarks:
//!   cargo bench --bench comparison -- --quick

mod corpus;

use std::io::Write as IoWrite;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, BenchmarkId, Criterion, Throughput};
use flate2::{write::ZlibEncoder, Compression};
use image::ImageEncoder;
use serde::Serialize;

use comprs::compress::deflate::deflate_zlib;
use comprs::{jpeg, png, ColorType};
use corpus::{generate_gradient_rgb, generate_noisy_rgb, DEFAULT_SQUARE_SIZES};

// ============================================================================
// Test Data Generation
// ============================================================================

fn make_compressible(len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    while out.len() < len {
        out.extend_from_slice(pattern);
    }
    out.truncate(len);
    out
}

fn make_random(len: usize, mut seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        out.push((seed >> 16) as u8);
    }
    out.truncate(len);
    out
}

// ============================================================================
// PNG Encoding Comparison
// ============================================================================

fn bench_png_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("PNG Comparison");

    for &size in DEFAULT_SQUARE_SIZES {
        let gradient = generate_gradient_rgb(size, size);
        let noisy = generate_noisy_rgb(size, size);
        let pixel_bytes = (size as u64) * (size as u64) * 3;

        group.throughput(Throughput::Bytes(pixel_bytes));

        // --- Gradient Image ---
        let mut png_buf = Vec::new();
        group.bench_with_input(
            BenchmarkId::new("comprs_gradient", format!("{size}x{size}")),
            &gradient,
            |b, pixels| {
                b.iter(|| {
                    png::encode_into(
                        &mut png_buf,
                        black_box(pixels),
                        size,
                        size,
                        ColorType::Rgb,
                        &png::PngOptions::default(),
                    )
                    .unwrap()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("image_crate_gradient", format!("{size}x{size}")),
            &gradient,
            |b, pixels| {
                b.iter(|| {
                    let mut output = Vec::new();
                    let encoder = image::codecs::png::PngEncoder::new(&mut output);
                    encoder
                        .write_image(black_box(pixels), size, size, image::ColorType::Rgb8)
                        .unwrap();
                    output
                });
            },
        );

        // --- Noisy Image ---
        group.bench_with_input(
            BenchmarkId::new("comprs_noisy", format!("{size}x{size}")),
            &noisy,
            |b, pixels| {
                b.iter(|| {
                    png::encode_into(
                        &mut png_buf,
                        black_box(pixels),
                        size,
                        size,
                        ColorType::Rgb,
                        &png::PngOptions::default(),
                    )
                    .unwrap()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("image_crate_noisy", format!("{size}x{size}")),
            &noisy,
            |b, pixels| {
                b.iter(|| {
                    let mut output = Vec::new();
                    let encoder = image::codecs::png::PngEncoder::new(&mut output);
                    encoder
                        .write_image(black_box(pixels), size, size, image::ColorType::Rgb8)
                        .unwrap();
                    output
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// JPEG Encoding Comparison
// ============================================================================

fn bench_jpeg_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("JPEG Comparison");

    for &size in DEFAULT_SQUARE_SIZES {
        let gradient = generate_gradient_rgb(size, size);
        let pixel_bytes = (size as u64) * (size as u64) * 3;

        group.throughput(Throughput::Bytes(pixel_bytes));

        let mut jpeg_buf = Vec::new();

        // comprs with 4:4:4 subsampling
        group.bench_with_input(
            BenchmarkId::new("comprs_q85_444", format!("{size}x{size}")),
            &gradient,
            |b, pixels| {
                b.iter(|| {
                    jpeg::encode_with_options_into(
                        &mut jpeg_buf,
                        black_box(pixels),
                        size,
                        size,
                        85,
                        ColorType::Rgb,
                        &jpeg::JpegOptions {
                            quality: 85,
                            subsampling: jpeg::Subsampling::S444,
                            restart_interval: None,
                        },
                    )
                    .unwrap()
                });
            },
        );

        // comprs with 4:2:0 subsampling
        group.bench_with_input(
            BenchmarkId::new("comprs_q85_420", format!("{size}x{size}")),
            &gradient,
            |b, pixels| {
                b.iter(|| {
                    jpeg::encode_with_options_into(
                        &mut jpeg_buf,
                        black_box(pixels),
                        size,
                        size,
                        85,
                        ColorType::Rgb,
                        &jpeg::JpegOptions {
                            quality: 85,
                            subsampling: jpeg::Subsampling::S420,
                            restart_interval: None,
                        },
                    )
                    .unwrap()
                });
            },
        );

        // image crate
        group.bench_with_input(
            BenchmarkId::new("image_crate_q85", format!("{size}x{size}")),
            &gradient,
            |b, pixels| {
                b.iter(|| {
                    let mut output = Vec::new();
                    let encoder =
                        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, 85);
                    encoder
                        .write_image(black_box(pixels), size, size, image::ColorType::Rgb8)
                        .unwrap();
                    output
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// DEFLATE/zlib Comparison
// ============================================================================

fn bench_deflate_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("DEFLATE Comparison");

    let compressible = make_compressible(1 << 20);
    let random = make_random(1 << 20, 0xDEAD_BEEF);

    let cases = [("compressible_1mb", &compressible), ("random_1mb", &random)];

    for (name, data) in cases {
        let bytes = data.len() as u64;
        group.throughput(Throughput::Bytes(bytes));

        group.bench_with_input(BenchmarkId::new("comprs", name), data, |b, input| {
            b.iter(|| {
                let encoded = deflate_zlib(black_box(input), 6);
                black_box(encoded.len())
            });
        });

        group.bench_with_input(BenchmarkId::new("flate2", name), data, |b, input| {
            b.iter(|| {
                let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
                encoder.write_all(black_box(input)).unwrap();
                let encoded = encoder.finish().unwrap();
                black_box(encoded.len())
            });
        });
    }

    group.finish();
}

// ============================================================================
// Summary Report (printed after benchmarks) + structured export
// ============================================================================

#[derive(Debug, Serialize)]
struct BinarySizeEntry {
    library: &'static str,
    artifact: &'static str,
    bytes: Option<u64>,
    status: &'static str,
    notes: &'static str,
}

#[derive(Debug, Serialize)]
struct OutputSizeEntry {
    format: &'static str,
    library: &'static str,
    width: u32,
    height: u32,
    quality: Option<u8>,
    subsampling: Option<&'static str>,
    bytes: usize,
}

#[derive(Debug, Serialize)]
struct SpeedEntry {
    operation: &'static str,
    library: &'static str,
    width: u32,
    height: u32,
    quality: Option<u8>,
    subsampling: Option<&'static str>,
    duration_micros: f64,
}

#[derive(Debug, Serialize)]
struct DeflateEntry {
    library: &'static str,
    case: &'static str,
    output_bytes: usize,
    ratio: f64,
}

#[derive(Debug, Serialize)]
struct SummaryData {
    binary_sizes: Vec<BinarySizeEntry>,
    output_sizes: Vec<OutputSizeEntry>,
    speed: Vec<SpeedEntry>,
    deflate: Vec<DeflateEntry>,
    notes: Vec<&'static str>,
}

fn build_summary_data() -> SummaryData {
    let mut binary_sizes = Vec::new();
    let comprs_wasm_size = get_wasm_size().map(|v| v as u64);
    binary_sizes.push(BinarySizeEntry {
        library: "comprs",
        artifact: "wasm32-unknown-unknown",
        bytes: comprs_wasm_size,
        status: if comprs_wasm_size.is_some() {
            "measured"
        } else {
            "missing"
        },
        notes: "Zero deps, pure Rust",
    });
    binary_sizes.push(BinarySizeEntry {
        library: "image crate",
        artifact: "n/a",
        bytes: None,
        status: "unmeasured",
        notes: "Pure Rust, many codecs",
    });
    binary_sizes.push(BinarySizeEntry {
        library: "photon-rs",
        artifact: "wasm (planned)",
        bytes: None,
        status: "planned",
        notes: "WASM-optimized; not yet wired into bench",
    });
    binary_sizes.push(BinarySizeEntry {
        library: "zune-image",
        artifact: "wasm/native (planned)",
        bytes: None,
        status: "planned",
        notes: "SIMD optimized; encoding support varies",
    });
    binary_sizes.push(BinarySizeEntry {
        library: "wasm-mozjpeg",
        artifact: "wasm (planned)",
        bytes: None,
        status: "planned",
        notes: "Emscripten compiled; optional",
    });

    // Shared gradient corpus for summary metrics.
    let gradient = generate_gradient_rgb(512, 512);

    // Output sizes (512x512 gradient).
    let mut output_sizes = Vec::new();
    let mut png_buf_comprs = Vec::new();
    png::encode_into(
        &mut png_buf_comprs,
        &gradient,
        512,
        512,
        ColorType::Rgb,
        &png::PngOptions::default(),
    )
    .unwrap();
    output_sizes.push(OutputSizeEntry {
        format: "PNG",
        library: "comprs",
        width: 512,
        height: 512,
        quality: None,
        subsampling: None,
        bytes: png_buf_comprs.len(),
    });

    let mut png_buf_image = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(&mut png_buf_image);
    encoder
        .write_image(&gradient, 512, 512, image::ColorType::Rgb8)
        .unwrap();
    output_sizes.push(OutputSizeEntry {
        format: "PNG",
        library: "image crate",
        width: 512,
        height: 512,
        quality: None,
        subsampling: None,
        bytes: png_buf_image.len(),
    });

    let mut jpeg_buf_comprs = Vec::new();
    jpeg::encode_with_options_into(
        &mut jpeg_buf_comprs,
        &gradient,
        512,
        512,
        85,
        ColorType::Rgb,
        &jpeg::JpegOptions {
            quality: 85,
            subsampling: jpeg::Subsampling::S444,
            restart_interval: None,
        },
    )
    .unwrap();
    output_sizes.push(OutputSizeEntry {
        format: "JPEG",
        library: "comprs",
        width: 512,
        height: 512,
        quality: Some(85),
        subsampling: Some("4:4:4"),
        bytes: jpeg_buf_comprs.len(),
    });

    let mut jpeg_buf_image = Vec::new();
    let jpeg_encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg_buf_image, 85);
    jpeg_encoder
        .write_image(&gradient, 512, 512, image::ColorType::Rgb8)
        .unwrap();
    output_sizes.push(OutputSizeEntry {
        format: "JPEG",
        library: "image crate",
        width: 512,
        height: 512,
        quality: Some(85),
        subsampling: Some("4:4:4"),
        bytes: jpeg_buf_image.len(),
    });

    let mut jpeg_buf_comprs_420 = Vec::new();
    jpeg::encode_with_options_into(
        &mut jpeg_buf_comprs_420,
        &gradient,
        512,
        512,
        85,
        ColorType::Rgb,
        &jpeg::JpegOptions {
            quality: 85,
            subsampling: jpeg::Subsampling::S420,
            restart_interval: None,
        },
    )
    .unwrap();
    output_sizes.push(OutputSizeEntry {
        format: "JPEG",
        library: "comprs",
        width: 512,
        height: 512,
        quality: Some(85),
        subsampling: Some("4:2:0"),
        bytes: jpeg_buf_comprs_420.len(),
    });

    // Speed measurements (avg of 10 iterations)
    let mut speed = Vec::new();
    let png_comprs_time = measure_time(|| {
        let mut buf = Vec::new();
        png::encode_into(
            &mut buf,
            &gradient,
            512,
            512,
            ColorType::Rgb,
            &png::PngOptions::default(),
        )
        .unwrap();
    });
    speed.push(SpeedEntry {
        operation: "PNG encode",
        library: "comprs",
        width: 512,
        height: 512,
        quality: None,
        subsampling: None,
        duration_micros: png_comprs_time.as_secs_f64() * 1_000_000.0,
    });

    let png_image_time = measure_time(|| {
        let mut output = Vec::new();
        let encoder = image::codecs::png::PngEncoder::new(&mut output);
        encoder
            .write_image(&gradient, 512, 512, image::ColorType::Rgb8)
            .unwrap();
    });
    speed.push(SpeedEntry {
        operation: "PNG encode",
        library: "image crate",
        width: 512,
        height: 512,
        quality: None,
        subsampling: None,
        duration_micros: png_image_time.as_secs_f64() * 1_000_000.0,
    });

    let jpeg_comprs_time = measure_time(|| {
        let mut buf = Vec::new();
        jpeg::encode_with_options_into(
            &mut buf,
            &gradient,
            512,
            512,
            85,
            ColorType::Rgb,
            &jpeg::JpegOptions {
                quality: 85,
                subsampling: jpeg::Subsampling::S444,
                restart_interval: None,
            },
        )
        .unwrap();
    });
    speed.push(SpeedEntry {
        operation: "JPEG encode",
        library: "comprs",
        width: 512,
        height: 512,
        quality: Some(85),
        subsampling: Some("4:4:4"),
        duration_micros: jpeg_comprs_time.as_secs_f64() * 1_000_000.0,
    });

    let jpeg_image_time = measure_time(|| {
        let mut output = Vec::new();
        let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, 85);
        encoder
            .write_image(&gradient, 512, 512, image::ColorType::Rgb8)
            .unwrap();
    });
    speed.push(SpeedEntry {
        operation: "JPEG encode",
        library: "image crate",
        width: 512,
        height: 512,
        quality: Some(85),
        subsampling: Some("4:4:4"),
        duration_micros: jpeg_image_time.as_secs_f64() * 1_000_000.0,
    });

    // DEFLATE comparison (compressible payload)
    let compressible = make_compressible(1 << 20);
    let comprs_deflate = deflate_zlib(&compressible, 6);
    let mut flate2_enc = ZlibEncoder::new(Vec::new(), Compression::default());
    flate2_enc.write_all(&compressible).unwrap();
    let flate2_deflate = flate2_enc.finish().unwrap();

    let mut deflate = Vec::new();
    deflate.push(DeflateEntry {
        library: "comprs",
        case: "compressible_1mb",
        output_bytes: comprs_deflate.len(),
        ratio: compressible.len() as f64 / comprs_deflate.len() as f64,
    });
    deflate.push(DeflateEntry {
        library: "flate2",
        case: "compressible_1mb",
        output_bytes: flate2_deflate.len(),
        ratio: compressible.len() as f64 / flate2_deflate.len() as f64,
    });

    let notes = vec![
        "WASM sizes are approximate and depend on build configuration",
        "Speed measurements are short-run averages; use Criterion output for statistical analysis",
        "Compression ratios vary with image content",
        "Use --export-json to feed the cross-language aggregator",
    ];

    SummaryData {
        binary_sizes,
        output_sizes,
        speed,
        deflate,
        notes,
    }
}

fn print_summary_report(data: &SummaryData) {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                         COMPRS BENCHMARK SUMMARY                             ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // --- Binary Size Comparison ---
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ WASM / Binary Size Comparison                                                │");
    println!("├────────────────────┬─────────────┬────────────────────────────────────────────┤");
    println!("│ Library            │ Size        │ Notes                                      │");
    println!("├────────────────────┼─────────────┼────────────────────────────────────────────┤");
    for entry in &data.binary_sizes {
        println!(
            "│ {:<18} │ {:>11} │ {:<42} │",
            entry.library,
            format_size_opt(entry.bytes),
            format!("{} ({})", entry.notes, entry.status)
        );
    }
    println!("└────────────────────┴─────────────┴────────────────────────────────────────────┘");
    println!();

    // --- Output Size Comparison ---
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Compression Output Size (512x512 gradient image)                             │");
    println!("├────────────────────┬─────────────┬─────────────┬──────────────────────────────┤");
    println!("│ Format             │ comprs      │ image crate │ Ratio                        │");
    println!("├────────────────────┼─────────────┼─────────────┼──────────────────────────────┤");
    let png_comprs = find_output_size(data, "PNG", "comprs", None);
    let png_image = find_output_size(data, "PNG", "image crate", None);
    if let (Some(c), Some(i)) = (png_comprs, png_image) {
        println!(
            "│ {:<18} │ {:>11} │ {:>11} │ {:>28.2}x │",
            "PNG",
            format_size(c.bytes),
            format_size(i.bytes),
            c.bytes as f64 / i.bytes as f64
        );
    }

    let jpeg_comprs = find_output_size(data, "JPEG", "comprs", Some("4:4:4"));
    let jpeg_image = find_output_size(data, "JPEG", "image crate", Some("4:4:4"));
    if let (Some(c), Some(i)) = (jpeg_comprs, jpeg_image) {
        println!(
            "│ {:<18} │ {:>11} │ {:>11} │ {:>28.2}x │",
            "JPEG (q85, 4:4:4)",
            format_size(c.bytes),
            format_size(i.bytes),
            c.bytes as f64 / i.bytes as f64
        );
    }

    if let Some(c420) = find_output_size(data, "JPEG", "comprs", Some("4:2:0")) {
        println!(
            "│ {:<18} │ {:>11} │ {:>11} │ {:>28} │",
            "JPEG (q85, 4:2:0)",
            format_size(c420.bytes),
            "N/A",
            "-"
        );
    }
    println!("└────────────────────┴─────────────┴─────────────┴──────────────────────────────┘");
    println!();

    // --- DEFLATE Comparison ---
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ DEFLATE Compression (1 MB payload)                                           │");
    println!("├────────────────────┬─────────────┬─────────────┬──────────────────────────────┤");
    println!("│ Library            │ Output Size │ Ratio       │ Notes                        │");
    println!("├────────────────────┼─────────────┼─────────────┼──────────────────────────────┤");
    for entry in &data.deflate {
        println!(
            "│ {:<18} │ {:>11} │ {:>10.1}x │ {:<28} │",
            entry.library,
            format_size(entry.output_bytes),
            entry.ratio,
            if entry.library == "comprs" {
                "Pure Rust, zero deps"
            } else {
                "miniz_oxide backend"
            }
        );
    }
    println!("└────────────────────┴─────────────┴─────────────┴──────────────────────────────┘");
    println!();

    // --- Speed Summary ---
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Speed Summary (512x512 image, avg of 10 iterations)                          │");
    println!("├────────────────────┬─────────────┬─────────────┬──────────────────────────────┤");
    println!("│ Operation          │ comprs      │ image crate │ Relative                     │");
    println!("├────────────────────┼─────────────┼─────────────┼──────────────────────────────┤");
    let png_speed = find_speed(data, "PNG encode", "comprs");
    let png_speed_image = find_speed(data, "PNG encode", "image crate");
    if let (Some(c), Some(i)) = (png_speed, png_speed_image) {
        println!(
            "│ {:<18} │ {:>11} │ {:>11} │ {:>28} │",
            "PNG encode",
            format_duration_micros(c.duration_micros),
            format_duration_micros(i.duration_micros),
            format_relative_ratio(c.duration_micros, i.duration_micros, "comprs", "image")
        );
    }

    let jpeg_speed = find_speed(data, "JPEG encode", "comprs");
    let jpeg_speed_image = find_speed(data, "JPEG encode", "image crate");
    if let (Some(c), Some(i)) = (jpeg_speed, jpeg_speed_image) {
        println!(
            "│ {:<18} │ {:>11} │ {:>11} │ {:>28} │",
            "JPEG encode",
            format_duration_micros(c.duration_micros),
            format_duration_micros(i.duration_micros),
            format_relative_ratio(c.duration_micros, i.duration_micros, "comprs", "image")
        );
    }
    println!("└────────────────────┴─────────────┴─────────────┴──────────────────────────────┘");
    println!();

    // --- Summary Notes ---
    println!("Notes:");
    for note in &data.notes {
        println!("  • {note}");
    }
    println!();
}

fn write_json_summary(path: &PathBuf, data: &SummaryData) {
    match serde_json::to_string_pretty(data) {
        Ok(json) => {
            if let Err(e) = std::fs::write(path, json) {
                eprintln!("Failed to write JSON summary to {}: {e}", path.display());
            } else {
                println!("Wrote JSON summary to {}", path.display());
            }
        }
        Err(e) => eprintln!("Failed to serialize JSON summary: {e}"),
    }
}

fn find_output_size<'a>(
    data: &'a SummaryData,
    format: &str,
    library: &str,
    subsampling: Option<&str>,
) -> Option<&'a OutputSizeEntry> {
    data.output_sizes.iter().find(|entry| {
        entry.format == format && entry.library == library && entry.subsampling == subsampling
    })
}

fn find_speed<'a>(data: &'a SummaryData, operation: &str, library: &str) -> Option<&'a SpeedEntry> {
    data.speed
        .iter()
        .find(|entry| entry.operation == operation && entry.library == library)
}

fn get_wasm_size() -> Option<usize> {
    // Try to find the WASM binary if it exists
    let paths = [
        "target/wasm32-unknown-unknown/release/comprs.wasm",
        "target/wasm32-unknown-unknown/release/comprs_bg.wasm",
    ];

    for path in paths {
        if let Ok(metadata) = std::fs::metadata(path) {
            return Some(metadata.len() as usize);
        }
    }

    None
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        let mb = bytes as f64 / (1024.0 * 1024.0);
        format!("{mb:.1} MB")
    } else if bytes >= 1024 {
        let kb = bytes as f64 / 1024.0;
        format!("{kb:.1} KB")
    } else {
        format!("{bytes} B")
    }
}

fn format_size_opt(bytes: Option<u64>) -> String {
    match bytes {
        Some(b) => format_size(b as usize),
        None => "N/A".to_string(),
    }
}

fn format_duration(duration: Duration) -> String {
    let micros = duration.as_micros();
    if micros >= 1000 {
        let ms = duration.as_secs_f64() * 1000.0;
        format!("{ms:.2} ms")
    } else {
        format!("{micros} µs")
    }
}

fn format_duration_micros(micros: f64) -> String {
    format_duration(Duration::from_secs_f64(micros / 1_000_000.0))
}

fn format_relative_ratio(
    primary_micros: f64,
    reference_micros: f64,
    primary_label: &str,
    reference_label: &str,
) -> String {
    let ratio = primary_micros / reference_micros;
    if ratio < 1.0 {
        let faster = 1.0 / ratio;
        format!("{primary_label} {faster:.1}x faster")
    } else if ratio > 1.0 {
        format!("{reference_label} {ratio:.1}x faster")
    } else {
        "equal".to_string()
    }
}

fn measure_time<F: FnMut()>(mut f: F) -> Duration {
    // Warm up
    for _ in 0..3 {
        f();
    }

    // Measure average of 10 iterations
    let start = Instant::now();
    let iterations = 10;
    for _ in 0..iterations {
        f();
    }
    start.elapsed() / iterations
}

// ============================================================================
// Custom Criterion Configuration
// ============================================================================

fn custom_criterion() -> Criterion {
    Criterion::default()
        .sample_size(50)
        .measurement_time(std::time::Duration::from_secs(5))
}

criterion_group! {
    name = benches;
    config = custom_criterion();
    targets = bench_png_comparison, bench_jpeg_comparison, bench_deflate_comparison
}

#[derive(Debug)]
struct CliOptions {
    summary_only: bool,
    export_json: Option<PathBuf>,
}

fn parse_cli_options() -> CliOptions {
    let mut summary_only = false;
    let mut export_json = None;
    let mut args = std::env::args().skip(1).peekable();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--summary-only" => summary_only = true,
            "--export-json" => {
                if let Some(path) = args.next() {
                    export_json = Some(PathBuf::from(path));
                }
            }
            _ => {}
        }
    }

    CliOptions {
        summary_only,
        export_json,
    }
}

// Custom main that prints summary after benchmarks and can export JSON
fn main() {
    let cli = parse_cli_options();

    if !cli.summary_only {
        benches();
    }

    let summary = build_summary_data();
    print_summary_report(&summary);

    if let Some(path) = cli.export_json {
        write_json_summary(&path, &summary);
    }
}
