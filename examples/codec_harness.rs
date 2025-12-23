//! Benchmark harness to compare `comprs` against external codecs (mozjpeg/oxipng).
//!
//! Usage:
//! ```
//! # PNG path (expects oxipng binary if available)
//! cargo run --example codec_harness
//! # Specify fixture dir or binaries
//! FIXTURES=tests/fixtures \
//! OXIPNG_BIN=vendor/oxipng/target/release/oxipng \
//! MOZJPEG_CJPEG=vendor/mozjpeg/cjpeg \
//! cargo run --example codec_harness
//! ```
//!
//! The harness:
//! - Walks the fixture directory (recursively) for images.
//! - For each image, encodes PNG and JPEG with comprs.
//! - If external binaries exist, it runs oxipng for PNG and cjpeg (mozjpeg) for JPEG.
//! - Reports size and elapsed time per encoder.

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use comprs::{jpeg, png, ColorType};
use image::{DynamicImage, GenericImageView, ImageEncoder};

const DEFAULT_FIXTURES: &str = "tests/fixtures";
const DEFAULT_OXIPNG: &str = "vendor/oxipng/target/release/oxipng";
const DEFAULT_CJPEG_BUILD: &str = "vendor/mozjpeg/build/cjpeg";
const DEFAULT_CJPEG_FALLBACK: &str = "vendor/mozjpeg/cjpeg";
const DEFAULT_COMPRS_PRESET_ENV: &str = "COMPRS_PNG_PRESET";
const DEFAULT_COMPRS_FILTER_ENV: &str = "COMPRS_PNG_FILTER";
const DEFAULT_COMPRS_LEVEL_ENV: &str = "COMPRS_PNG_LEVEL";
const DEFAULT_JPEG_OPT_ENV: &str = "COMPRS_JPEG_OPTIMIZE_HUFFMAN";
const DEFAULT_JPEG_SUB_ENV: &str = "COMPRS_JPEG_SUBSAMPLING"; // s444/s420
const DEFAULT_JPEG_RESTART_ENV: &str = "COMPRS_JPEG_RESTART";
const DEFAULT_JPEG_PRESET_ENV: &str = "COMPRS_JPEG_PRESET"; // faster/auto/smallest or 0/1/2

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixtures_dir = env::var("FIXTURES").unwrap_or_else(|_| DEFAULT_FIXTURES.to_string());
    let oxipng_bin = env::var("OXIPNG_BIN").unwrap_or_else(|_| DEFAULT_OXIPNG.to_string());
    let cjpeg_bin = env::var("MOZJPEG_CJPEG").unwrap_or_else(|_| pick_cjpeg().to_string());

    let fixtures = collect_fixtures(Path::new(&fixtures_dir))?;
    if fixtures.is_empty() {
        println!("No fixtures found under {fixtures_dir}");
        return Ok(());
    }

    let tmp_dir = PathBuf::from("target/codec-harness");
    fs::create_dir_all(&tmp_dir)?;

    println!("Fixtures: {}", fixtures_dir);
    println!("oxipng: {} ({})", oxipng_bin, availability(&oxipng_bin));
    println!("cjpeg: {} ({})", cjpeg_bin, availability(&cjpeg_bin));
    println!();

    for fixture in fixtures {
        match image::open(&fixture) {
            Ok(img) => {
                let (w, h) = img.dimensions();
                println!("=== {} ({}x{}) ===", fixture.display(), w, h);

                // Only run PNG tests on PNG files to avoid confusing comparisons
                let is_png = fixture
                    .extension()
                    .and_then(OsStr::to_str)
                    .map(|s| s.eq_ignore_ascii_case("png"))
                    .unwrap_or(false);

                if is_png {
                    run_png_section(&img, &fixture, &tmp_dir, &oxipng_bin)?;
                }
                run_jpeg_section(&img, &fixture, &tmp_dir, &cjpeg_bin)?;

                println!();
            }
            Err(err) => {
                println!("=== {} ===", fixture.display());
                println!("Skipped: failed to decode ({err})");
                println!();
            }
        }
    }

    Ok(())
}

fn pick_cjpeg() -> &'static str {
    if Path::new(DEFAULT_CJPEG_BUILD).exists() {
        DEFAULT_CJPEG_BUILD
    } else {
        DEFAULT_CJPEG_FALLBACK
    }
}

fn load_comprs_png_options() -> (png::PngOptions, String) {
    // Start from defaults
    let mut opts = png::PngOptions::default();
    let mut desc_parts = Vec::new();

    // Optional preset override (COMPRS_PNG_PRESET)
    if let Ok(preset_name) = env::var(DEFAULT_COMPRS_PRESET_ENV) {
        let lower = preset_name.to_ascii_lowercase();
        let preset = match lower.as_str() {
            "fast" | "0" => Some(png::PngOptions::fast()),
            "balanced" | "1" => Some(png::PngOptions::balanced()),
            "max" | "2" => Some(png::PngOptions::max()),
            _ => None,
        };
        if let Some(p) = preset {
            opts = p;
            desc_parts.push(format!("preset={}", lower));
        } else {
            desc_parts.push(format!("preset={lower}(invalid->default)"));
        }
    }

    // Optional compression level override (COMPRS_PNG_LEVEL)
    if let Ok(level_str) = env::var(DEFAULT_COMPRS_LEVEL_ENV) {
        if let Ok(level) = level_str.parse::<u8>() {
            if (1..=9).contains(&level) {
                opts.compression_level = level;
                desc_parts.push(format!("level={level}"));
            } else {
                desc_parts.push(format!("level={level}(invalid-range)"));
            }
        } else {
            desc_parts.push(format!("level={level_str}(parse-fail)"));
        }
    }

    // Optional filter override (COMPRS_PNG_FILTER)
    if let Ok(filter_name) = env::var(DEFAULT_COMPRS_FILTER_ENV) {
        if let Some(f) = parse_filter(&filter_name) {
            opts.filter_strategy = f;
            desc_parts.push(format!("filter={}", filter_name.to_ascii_lowercase()));
        } else {
            desc_parts.push(format!("filter={}(invalid)", filter_name));
        }
    }

    let desc = if desc_parts.is_empty() {
        "default".to_string()
    } else {
        desc_parts.join(",")
    };

    (opts, desc)
}

fn load_comprs_jpeg_options() -> (jpeg::JpegOptions, String) {
    let mut opts = jpeg::JpegOptions::fast(85);
    let mut desc_parts = Vec::new();

    // Optional preset override (COMPRS_JPEG_PRESET)
    if let Ok(preset_name) = env::var(DEFAULT_JPEG_PRESET_ENV) {
        let lower = preset_name.to_ascii_lowercase();
        let preset = match lower.as_str() {
            "fast" | "0" => Some(jpeg::JpegOptions::fast(85)),
            "balanced" | "1" => Some(jpeg::JpegOptions::balanced(85)),
            "max" | "2" => Some(jpeg::JpegOptions::max(85)),
            _ => None,
        };
        if let Some(p) = preset {
            opts = p;
            desc_parts.push(format!("preset={}", lower));
        } else {
            desc_parts.push(format!("preset={lower}(invalid->default)"));
        }
    }

    if let Ok(opt_str) = env::var(DEFAULT_JPEG_OPT_ENV) {
        let enable = matches!(opt_str.to_ascii_lowercase().as_str(), "1" | "true" | "yes");
        opts.optimize_huffman = enable;
        desc_parts.push(format!("opt_huff={enable}"));
    }

    if let Ok(sub_str) = env::var(DEFAULT_JPEG_SUB_ENV) {
        if let Some(sub) = parse_subsampling(&sub_str) {
            opts.subsampling = sub;
            desc_parts.push(format!("subsampling={}", sub_str.to_ascii_lowercase()));
        } else {
            desc_parts.push(format!("subsampling={sub_str}(invalid)"));
        }
    }

    if let Ok(restart_str) = env::var(DEFAULT_JPEG_RESTART_ENV) {
        if let Ok(v) = restart_str.parse::<u16>() {
            if v > 0 {
                opts.restart_interval = Some(v);
                desc_parts.push(format!("restart={v}"));
            } else {
                desc_parts.push("restart=0(disabled)".to_string());
            }
        } else {
            desc_parts.push(format!("restart={restart_str}(parse-fail)"));
        }
    }

    let desc = if desc_parts.is_empty() {
        "default".to_string()
    } else {
        desc_parts.join(",")
    };
    (opts, desc)
}

fn parse_filter(name: &str) -> Option<png::FilterStrategy> {
    match name.to_ascii_lowercase().as_str() {
        "none" => Some(png::FilterStrategy::None),
        "sub" => Some(png::FilterStrategy::Sub),
        "up" => Some(png::FilterStrategy::Up),
        "average" | "avg" => Some(png::FilterStrategy::Average),
        "paeth" => Some(png::FilterStrategy::Paeth),
        "minsum" => Some(png::FilterStrategy::MinSum),
        "adaptive" => Some(png::FilterStrategy::Adaptive),
        "adaptive-fast" | "adaptive_fast" => Some(png::FilterStrategy::AdaptiveFast),
        _ => None,
    }
}

fn parse_subsampling(name: &str) -> Option<jpeg::Subsampling> {
    match name.to_ascii_lowercase().as_str() {
        "s444" | "444" => Some(jpeg::Subsampling::S444),
        "s420" | "420" => Some(jpeg::Subsampling::S420),
        _ => None,
    }
}

fn collect_fixtures(root: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    recurse_dir(root, &mut files)?;
    // Only keep common image extensions
    files.retain(|p| {
        matches!(
            p.extension().and_then(OsStr::to_str).map(|s| s.to_ascii_lowercase()),
            Some(ref ext) if ["png", "jpg", "jpeg", "webp"].contains(&ext.as_str())
        )
    });
    files.sort();
    Ok(files)
}

fn recurse_dir(dir: &Path, acc: &mut Vec<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            recurse_dir(&path, acc)?;
        } else {
            acc.push(path);
        }
    }
    Ok(())
}

fn run_png_section(
    img: &DynamicImage,
    source_path: &Path,
    tmp_dir: &Path,
    oxipng_bin: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());

    let (comprs_opts, comprs_desc) = load_comprs_png_options();

    // comprs encode
    let mut comprs_buf = Vec::new();
    let t0 = Instant::now();
    png::encode_into(&mut comprs_buf, &rgba, w, h, ColorType::Rgba, &comprs_opts)?;
    let comprs_dur = t0.elapsed();
    println!(
        "PNG comprs ({desc}): {:>8} bytes, {:>6.2} ms",
        comprs_buf.len(),
        to_millis(comprs_dur),
        desc = comprs_desc
    );
    println!(
        "  options: level={}, filter={:?}, alpha_opt={}, reduce_color={}, reduce_palette={}, strip_meta={}",
        comprs_opts.compression_level,
        comprs_opts.filter_strategy,
        comprs_opts.optimize_alpha,
        comprs_opts.reduce_color_type,
        comprs_opts.reduce_palette,
        comprs_opts.strip_metadata
    );

    // Prepare a PNG file for external tools (reuse source if already PNG)
    let png_input = if source_path
        .extension()
        .and_then(OsStr::to_str)
        .map(|s| s.eq_ignore_ascii_case("png"))
        .unwrap_or(false)
    {
        source_path.to_path_buf()
    } else {
        let tmp_png = tmp_dir.join(format!(
            "{}.as_png.png",
            source_path
                .file_stem()
                .and_then(OsStr::to_str)
                .unwrap_or("tmp")
        ));
        let mut file = fs::File::create(&tmp_png)?;
        let encoder = image::codecs::png::PngEncoder::new(&mut file);
        encoder.write_image(rgba.as_raw(), w, h, image::ColorType::Rgba8)?;
        tmp_png
    };

    // oxipng comparison (if available)
    if Path::new(oxipng_bin).exists() {
        let oxi_out = tmp_dir.join(format!(
            "{}.oxipng.png",
            source_path
                .file_stem()
                .and_then(OsStr::to_str)
                .unwrap_or("out")
        ));
        let t1 = Instant::now();
        let status = Command::new(oxipng_bin)
            .args([
                "-o",
                "4",
                "--strip",
                "safe",
                "--out",
                oxi_out.to_str().unwrap(),
                png_input.to_str().unwrap(),
            ])
            .status()?;
        if status.success() {
            let oxi_dur = t1.elapsed();
            let size = fs::metadata(&oxi_out)?.len();
            let delta = pct_delta(size, comprs_buf.len() as u64);
            println!(
                "PNG oxipng (-o4 --strip safe): {:>8} bytes, {:>6.2} ms ({:+.2}%)",
                size,
                to_millis(oxi_dur),
                delta
            );
        } else {
            println!("PNG oxipng: failed with status {status:?}");
        }
    } else {
        println!("PNG oxipng: skipped (missing binary at {})", oxipng_bin);
    }

    Ok(())
}

fn run_jpeg_section(
    img: &DynamicImage,
    source_path: &Path,
    tmp_dir: &Path,
    cjpeg_bin: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let (comprs_jpeg_opts, jpeg_desc) = load_comprs_jpeg_options();

    // comprs JPEG
    let mut comprs_buf = Vec::new();
    let t0 = Instant::now();
    jpeg::encode_with_options_into(
        &mut comprs_buf,
        rgb.as_raw(),
        w,
        h,
        comprs_jpeg_opts.quality,
        ColorType::Rgb,
        &comprs_jpeg_opts,
    )?;
    let comprs_dur = t0.elapsed();
    println!(
        "JPG comprs ({desc}): {:>8} bytes, {:>6.2} ms",
        comprs_buf.len(),
        to_millis(comprs_dur),
        desc = jpeg_desc
    );

    if Path::new(cjpeg_bin).exists() {
        let ppm_path = tmp_dir.join(format!(
            "{}.ppm",
            source_path
                .file_stem()
                .and_then(OsStr::to_str)
                .unwrap_or("tmp")
        ));
        write_ppm(&ppm_path, &rgb)?;

        let mjpeg_out = tmp_dir.join(format!(
            "{}.mozjpeg.jpg",
            source_path
                .file_stem()
                .and_then(OsStr::to_str)
                .unwrap_or("out")
        ));
        let t1 = Instant::now();
        let status = Command::new(cjpeg_bin)
            .args([
                "-quality",
                "85",
                "-optimize",
                "-progressive",
                "-outfile",
                mjpeg_out.to_str().unwrap(),
                ppm_path.to_str().unwrap(),
            ])
            .status()?;
        if status.success() {
            let mjpeg_dur = t1.elapsed();
            let size = fs::metadata(&mjpeg_out)?.len();
            let delta = pct_delta(size, comprs_buf.len() as u64);
            println!(
                "JPG mozjpeg (cjpeg -quality 85 -optimize -progressive): {:>8} bytes, {:>6.2} ms ({:+.2}%)",
                size,
                to_millis(mjpeg_dur),
                delta
            );
        } else {
            println!("JPG mozjpeg: failed with status {status:?}");
        }
    } else {
        println!("JPG mozjpeg: skipped (missing binary at {})", cjpeg_bin);
    }

    Ok(())
}

fn write_ppm(path: &Path, rgb: &image::RgbImage) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = fs::File::create(path)?;
    writeln!(file, "P6\n{} {}\n255", rgb.width(), rgb.height())?;
    file.write_all(rgb.as_raw())?;
    Ok(())
}

fn to_millis(dur: std::time::Duration) -> f64 {
    dur.as_secs_f64() * 1000.0
}

fn availability(path: &str) -> &'static str {
    if Path::new(path).exists() {
        "found"
    } else {
        "missing"
    }
}

fn pct_delta(other: u64, baseline: u64) -> f64 {
    if baseline == 0 {
        0.0
    } else {
        (other as f64 / baseline as f64 - 1.0) * 100.0
    }
}
