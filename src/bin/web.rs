//! comprs-web - WebAssembly binary for image compression
//!
//! This binary provides the WASM entrypoint for the comprs library.
//! Build with: cargo build --bin comprs-web --target wasm32-unknown-unknown --release --features wasm

#[cfg(feature = "wasm")]
pub use comprs::wasm::*;

/// Re-export all WASM functions when building for web.
/// This binary exists to provide a dedicated entry point for WASM builds.
#[cfg(not(feature = "wasm"))]
fn main() {
    eprintln!("comprs-web requires the 'wasm' feature.");
    eprintln!("Build with: cargo build --bin comprs-web --target wasm32-unknown-unknown --release --features wasm");
    std::process::exit(1);
}

/// Dummy main for wasm32 target (wasm-bindgen handles the entry point).
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
fn main() {}

/// When running natively with wasm feature (for testing), print usage.
#[cfg(all(feature = "wasm", not(target_arch = "wasm32")))]
fn main() {
    eprintln!("comprs-web is designed for WASM targets.");
    eprintln!("Build with: cargo build --bin comprs-web --target wasm32-unknown-unknown --release --features wasm");
    std::process::exit(1);
}
