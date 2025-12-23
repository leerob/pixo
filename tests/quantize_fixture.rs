use comprs::png::{self, FilterStrategy, PngOptions, QuantizationMode, QuantizationOptions};
use comprs::ColorType;
use image::GenericImageView;

#[test]
fn quantizing_fixture_hits_indexed_and_smaller_size() {
    let fixture = "tests/fixtures/uncompressed.png";
    let img = image::open(fixture).expect("open fixture");
    let (w, h) = img.dimensions();
    let rgba = img.to_rgba8();
    // Crop to a manageable sub-rectangle to keep the test fast.
    let sub_w = w.min(128);
    let sub_h = h.min(128);
    let mut pixels = Vec::with_capacity((sub_w * sub_h * 4) as usize);
    let stride = (w * 4) as usize;
    for row in 0..sub_h {
        let start = (row * stride as u32) as usize;
        let end = start + (sub_w * 4) as usize;
        pixels.extend_from_slice(&rgba.as_raw()[start..end]);
    }

    // Baseline without quantization
    let baseline_opts = PngOptions {
        compression_level: 6,
        filter_strategy: FilterStrategy::AdaptiveSampled { interval: 2 },
        ..Default::default()
    };
    let baseline =
        png::encode_with_options(&pixels, sub_w, sub_h, ColorType::Rgba, &baseline_opts).unwrap();

    // Quantized path (force)
    let q_opts = PngOptions {
        compression_level: 6,
        filter_strategy: FilterStrategy::AdaptiveSampled { interval: 2 },
        quantization: QuantizationOptions {
            mode: QuantizationMode::Force,
            max_colors: 256,
            dithering: false,
        },
    };
    let quantized =
        png::encode_with_options(&pixels, sub_w, sub_h, ColorType::Rgba, &q_opts).unwrap();

    // Expect indexed color type (byte 25 in IHDR).
    assert_eq!(quantized[25], 3, "quantized output should be indexed");

    let baseline_size = baseline.len();
    let quantized_size = quantized.len();
    // Expect significant reduction on the cropped region; guard well below baseline.
    assert!(
        quantized_size <= 20_000,
        "quantized size too large: {quantized_size}"
    );
    assert!(
        quantized_size < baseline_size,
        "quantized PNG should be smaller than baseline (q={} vs base={})",
        quantized_size,
        baseline_size
    );
}
