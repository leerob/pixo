use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use reqwest::blocking::Client;
use sha2::{Digest, Sha256};

// Small curated JPEG fixtures (libjpeg-turbo test images).
pub const JPEG_FIXTURES: &[(&str, &str, &str)] = &[
    (
        "testorig.jpg",
        "https://raw.githubusercontent.com/libjpeg-turbo/libjpeg-turbo/main/testimages/testorig.jpg",
        "1dffbddf2e835d2ca2a3f8b80b9dcd51cb07cdbcdaad34b0bc9b0f0e821c9df5",
    ),
    (
        "testimg.jpg",
        "https://raw.githubusercontent.com/libjpeg-turbo/libjpeg-turbo/main/testimages/testimg.jpg",
        "5e7bba21657fb11e2b3b2bf9d4ac266bf9b1c6820707c91129f7bf3a9d0a1147",
    ),
    (
        "cram_bgr24.jpg",
        "https://raw.githubusercontent.com/libjpeg-turbo/libjpeg-turbo/main/testimages/cram_bgr24.jpg",
        "a4bd6d7e704901166a6ed422dfc95168d6b243326f6b5e9d626fae0f82b1bfc9",
    ),
];

pub fn fetch_jpeg_corpus(fixtures_dir: &Path) -> Result<(), String> {
    let client = Client::builder()
        .user_agent("comprs-test/0.1")
        .build()
        .map_err(|e| e.to_string())?;

    fs::create_dir_all(fixtures_dir).map_err(|e| e.to_string())?;

    for &(name, url, sha) in JPEG_FIXTURES {
        let dest = fixtures_dir.join(name);
        if dest.exists() {
            continue;
        }

        let resp = client.get(url).send().map_err(|e| e.to_string())?;
        let resp = resp.error_for_status().map_err(|e| e.to_string())?;
        let bytes = resp.bytes().map_err(|e| e.to_string())?.to_vec();

        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let digest = format!("{:x}", hasher.finalize());
        if digest != sha {
            return Err(format!(
                "SHA mismatch for {name}: expected {sha}, got {digest}"
            ));
        }

        fs::write(&dest, &bytes).map_err(|e| e.to_string())?;
    }

    Ok(())
}

pub fn read_jpeg_corpus() -> Result<Vec<(PathBuf, Vec<u8>)>, String> {
    let fixtures_dir = Path::new("tests/fixtures/jpeg_corpus");
    fetch_jpeg_corpus(fixtures_dir)?;

    let mut cases = Vec::new();
    for entry in fs::read_dir(fixtures_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("jpg") {
            let mut data = Vec::new();
            fs::File::open(&path)
                .map_err(|e| e.to_string())?
                .read_to_end(&mut data)
                .map_err(|e| e.to_string())?;
            cases.push((path, data));
        }
    }
    Ok(cases)
}
