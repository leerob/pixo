//! Compare preset options for PNG and JPEG.
//!
//! Run:
//!   cargo bench --bench preset_compare

use comprs::{jpeg, png, ColorType};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn generate_gradient(width: u32, height: u32) -> Vec<u8> {
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

fn png_presets(c: &mut Criterion) {
    let mut group = c.benchmark_group("png_presets");
    let size = 256u32;
    let pixels = generate_gradient(size, size);
    let pixel_bytes = (size as u64) * (size as u64) * 3;
    group.throughput(Throughput::Bytes(pixel_bytes));

    for (label, opts) in [
        ("default", png::PngOptions::default()),
        ("fast", png::PngOptions::fast()),
        ("max", png::PngOptions::max_compression()),
    ] {
        let mut buf = Vec::new();
        group.bench_with_input(BenchmarkId::new("png", label), &pixels, |b, pixels| {
            b.iter(|| {
                png::encode_into(
                    &mut buf,
                    black_box(pixels),
                    size,
                    size,
                    ColorType::Rgb,
                    &opts,
                )
                .unwrap()
            })
        });
    }

    group.finish();
}

fn jpeg_presets(c: &mut Criterion) {
    let mut group = c.benchmark_group("jpeg_presets");
    let size = 256u32;
    let pixels = generate_gradient(size, size);
    let pixel_bytes = (size as u64) * (size as u64) * 3;
    group.throughput(Throughput::Bytes(pixel_bytes));

    for (label, opts) in [
        (
            "default",
            jpeg::JpegOptions {
                quality: 85,
                subsampling: jpeg::Subsampling::S444,
                restart_interval: None,
            },
        ),
        ("fast", jpeg::JpegOptions::fast()),
        ("max_quality", jpeg::JpegOptions::max_quality()),
    ] {
        let mut buf = Vec::new();
        group.bench_with_input(BenchmarkId::new("jpeg", label), &pixels, |b, pixels| {
            b.iter(|| {
                jpeg::encode_with_options_into(
                    &mut buf,
                    black_box(pixels),
                    size,
                    size,
                    opts.quality,
                    ColorType::Rgb,
                    &opts,
                )
                .unwrap()
            })
        });
    }

    group.finish();
}

criterion_group!(benches, png_presets, jpeg_presets);
criterion_main!(benches);
