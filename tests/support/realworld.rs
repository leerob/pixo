use comprs::ColorType;
use image::{DynamicImage, GenericImageView, ImageEncoder};
use std::fs;

use super::jpeg_corpus::read_jpeg_corpus;

/// Real-world decoded image and precomputed channel variants for tests.
pub struct RealImage {
    pub name: String,
    pub width: u32,
    pub height: u32,
    rgb: Vec<u8>,
    rgba: Vec<u8>,
    gray: Vec<u8>,
    gray_alpha: Vec<u8>,
    #[allow(dead_code)]
    has_alpha: bool,
}

impl RealImage {
    /// True if any alpha sample is not fully opaque.
    #[allow(dead_code)]
    pub fn has_transparency(&self) -> bool {
        self.has_alpha
    }

    /// Get pixels for a requested color type, if available.
    pub fn pixels(&self, ct: ColorType) -> Option<&[u8]> {
        match ct {
            ColorType::Rgb => Some(&self.rgb),
            ColorType::Rgba => Some(&self.rgba),
            ColorType::Gray => Some(&self.gray),
            ColorType::GrayAlpha => Some(&self.gray_alpha),
        }
    }
}

/// Compute mean squared error between two equally sized byte buffers.
#[allow(dead_code)]
pub fn mse(a: &[u8], b: &[u8]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut acc: f64 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = *x as f64 - *y as f64;
        acc += diff * diff;
    }
    Some(acc / (a.len() as f64))
}

/// Compute PSNR (dB) assuming 8-bit samples. Returns None for mismatched lengths.
#[allow(dead_code)]
pub fn psnr(a: &[u8], b: &[u8]) -> Option<f64> {
    let mse = mse(a, b)?;
    if mse == 0.0 {
        return Some(f64::INFINITY);
    }
    let max_i = 255.0;
    Some(10.0 * ((max_i * max_i) / mse).log10())
}

/// Encode via the `image` crate PNG encoder for reference sizing.
#[allow(dead_code)]
pub fn encode_png_reference(
    data: &[u8],
    width: u32,
    height: u32,
    ct: ColorType,
) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    let color = match ct {
        ColorType::Gray => image::ColorType::L8,
        ColorType::GrayAlpha => image::ColorType::La8,
        ColorType::Rgb => image::ColorType::Rgb8,
        ColorType::Rgba => image::ColorType::Rgba8,
    };
    image::codecs::png::PngEncoder::new(&mut output)
        .write_image(data, width, height, color)
        .map_err(|e| e.to_string())?;
    Ok(output)
}

/// Encode via the `image` crate JPEG encoder for reference sizing.
#[allow(dead_code)]
pub fn encode_jpeg_reference(
    data: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, quality);
    encoder
        .write_image(data, width, height, image::ColorType::Rgb8)
        .map_err(|e| e.to_string())?;
    Ok(output)
}

/// Load local fixtures (if present) and JPEG corpus samples, producing RGB/RGBA/Gray variants.
///
/// Returns an error if no images could be loaded; callers can choose to skip in that case.
pub fn load_real_images() -> Result<Vec<RealImage>, String> {
    let mut images = Vec::new();

    let local_candidates = [
        ("multi-agent.jpg", "tests/fixtures/multi-agent.jpg"),
        ("playground.png", "tests/fixtures/playground.png"),
    ];

    for (name, path) in local_candidates {
        if let Ok(bytes) = fs::read(path) {
            if let Ok(img) = image::load_from_memory(&bytes) {
                images.push(decode_image(name, img));
            }
        }
    }

    if let Ok(corpus) = read_jpeg_corpus() {
        for (path, bytes) in corpus {
            if let Ok(img) = image::load_from_memory(&bytes) {
                let name = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("jpeg-corpus")
                    .to_string();
                images.push(decode_image(name, img));
            }
        }
    }

    if images.is_empty() {
        Err("No fixtures available for real-world tests".into())
    } else {
        Ok(images)
    }
}

fn decode_image(name: impl Into<String>, img: DynamicImage) -> RealImage {
    let name = name.into();
    let (width, height) = img.dimensions();
    let rgba_img = img.to_rgba8();
    let rgb_img = img.to_rgb8();
    let gray_img = img.to_luma8();
    let gray_alpha_img = img.to_luma_alpha8();
    let has_alpha = rgba_img
        .chunks_exact(4)
        .any(|px| px.get(3).copied().unwrap_or(255) != 255);

    RealImage {
        name,
        width,
        height,
        rgb: rgb_img.into_raw(),
        rgba: rgba_img.into_raw(),
        gray: gray_img.into_raw(),
        gray_alpha: gray_alpha_img.into_raw(),
        has_alpha,
    }
}
