# Benchmarks

This directory contains performance benchmarks for the comprs library.

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark suites
cargo bench --bench encode_benchmark  # PNG/JPEG encoding vs image crate
cargo bench --bench deflate_micro     # DEFLATE compression vs flate2
cargo bench --bench components        # Component-level benchmarks (DCT, Huffman, etc.)
cargo bench --bench size_snapshot -- --nocapture  # Quick size/time snapshot vs image crate
cargo bench --bench preset_compare    # Compare preset throughput (PNG/JPEG)
```

## Benchmark Suites

### encode_benchmark

Compares comprs PNG/JPEG encoding against the `image` crate:

- **PNG Encoding**: Tests gradient images at 64x64, 128x128, 256x256, 512x512
- **JPEG Encoding**: Tests quality 85 with 4:4:4 and 4:2:0 subsampling
- **Compression Ratio**: Compares output sizes for gradient and noisy images

### deflate_micro

Micro-benchmarks for DEFLATE compression:

- Compares `deflate_zlib` against `flate2` on 1 MiB payloads
- Tests both compressible (repeated patterns) and incompressible (random) data
- Reports throughput in bytes/second

### components

Component-level benchmarks for internal algorithms:

- DCT (Discrete Cosine Transform)
- Huffman encoding/decoding
- LZ77 compression
- CRC32/Adler32 checksums

---

## Library Comparison

When choosing an image compression solution, consider these trade-offs:

### Rust Libraries

| Library           | WASM-friendly   | Binary Size | Throughput | Notes                                              |
| ----------------- | --------------- | ----------- | ---------- | -------------------------------------------------- |
| **comprs**        | Yes             | 92 KB       | Good       | Zero deps, pure Rust, simple WASM target           |
| `image`           | Yes             | ~2-4MB      | Good       | Pure Rust, many codecs included                    |
| `photon-rs`       | Yes             | ~200-400KB  | Excellent  | Pure Rust, designed for WASM, 13x faster than JS   |
| `zune-image`      | Yes             | ~500KB-1MB  | Excellent  | Pure Rust, SIMD optimized, 1.8x faster than libpng |
| `mozjpeg`         | Emscripten only | ~30-50KB    | Excellent  | C library, requires complex build setup            |
| `libpng` bindings | Emscripten only | N/A         | Excellent  | C library, requires Emscripten toolchain           |

_Note: photon-rs benchmarks show Gaussian blur at 180ms vs 2400ms in pure JS. zune-png benchmarks show 1.8x speedup over libpng on x86._

### JavaScript/Node.js Libraries

| Library                     | Environment    | Bundle Size (minified+gzip) | Throughput        | Notes                                          |
| --------------------------- | -------------- | --------------------------- | ----------------- | ---------------------------------------------- |
| `sharp`                     | Node.js only   | 7-12MB native binaries      | Excellent         | libvips bindings, 4-5x faster than ImageMagick |
| `jimp`                      | Browser + Node | ~4MB unpacked               | Slow (~0.7 img/s) | Pure JS, no native deps                        |
| `pngjs`                     | Browser + Node | ~180KB gzipped              | Slow              | PNG only, pure JS                              |
| `jpeg-js`                   | Browser + Node | ~35KB gzipped               | Slow              | JPEG only, pure JS                             |
| `browser-image-compression` | Browser        | ~50KB gzipped               | Varies            | Uses Canvas API internally                     |
| `squoosh-lib`               | Browser + Node | 30-100KB/codec gzipped      | Good              | Google's WASM codecs                           |

_Note: Sizes vary by version. sharp includes prebuilt libvips binaries (~534KB JS + native binaries). jimp-compact is ~27x smaller than jimp._

### Browser Native APIs

| API                  | Binary Size | Quality Control | Format Support  |
| -------------------- | ----------- | --------------- | --------------- |
| Canvas `toBlob()`    | 0           | Limited         | JPEG, PNG, WebP |
| Canvas `toDataURL()` | 0           | Limited         | JPEG, PNG, WebP |
| `OffscreenCanvas`    | 0           | Limited         | JPEG, PNG, WebP |

**Limitations of browser Canvas APIs:**

- JPEG quality is browser-dependent and not standardized
- No control over PNG compression level or filter selection
- No access to advanced features (progressive JPEG, interlacing)
- Output varies between browsers

---

## Why comprs for WASM?

comprs is designed from the ground up for easy WASM deployment:

### 1. Zero C Dependencies

Libraries like `mozjpeg`, `libpng`, and `zlib` are C/C++ code that requires the **Emscripten** toolchain to compile to WASM:

```bash
# Complex: C library to WASM
emcc mozjpeg.c -o mozjpeg.wasm -s WASM=1 -O3

# Simple: Pure Rust to WASM
wasm-pack build --target web --features wasm
```

comprs compiles with `wasm32-unknown-unknown` - the standard, simple target.

### 2. Small Binary Size

| What you need | comprs    | image crate (est.) |
| ------------- | --------- | ------------------ |
| PNG + JPEG    | **92 KB** | ~2-4MB             |

The `image` crate includes decoders and encoders for BMP, GIF, ICO, TIFF, WebP, AVIF, and more - even if you only need PNG. You can use feature flags to reduce size, but the core still includes significant code.

comprs achieves its small size through:

- Zero runtime dependencies (hand-implemented DEFLATE, DCT, Huffman)
- Release profile optimizations (LTO, `opt-level = "z"`, `panic = "abort"`, symbol stripping)

### 3. Predictable Output

Browser Canvas APIs produce different output across browsers:

```javascript
// Safari, Chrome, and Firefox produce different JPEG bytes
canvas.toBlob(callback, "image/jpeg", 0.85);
```

comprs produces identical output regardless of environment.

### 4. Full Control

| Feature            | comprs                 | Canvas API        |
| ------------------ | ---------------------- | ----------------- |
| JPEG quality       | 1-100, precise         | Browser-dependent |
| PNG compression    | 1-9 levels             | None              |
| PNG filters        | All 5 types + adaptive | None              |
| Chroma subsampling | 4:4:4 or 4:2:0         | None              |

---

## Binary Size Breakdown

Measured WASM binary sizes for comprs:

| Configuration                    | Size      |
| -------------------------------- | --------- |
| Default release                  | 127 KB    |
| **Optimized** (LTO, opt-level=z) | **92 KB** |

The optimizations in `Cargo.toml` provide a **28% size reduction**.

### Build Configuration

The release profile in `Cargo.toml` enables size optimizations:

```toml
[profile.release]
lto = true           # Link-time optimization
opt-level = "z"      # Optimize for size
codegen-units = 1    # Single codegen unit for better optimization
panic = "abort"      # Remove unwinding code
strip = true         # Strip symbols from binary
```

Additional WASM-specific settings in `.cargo/config.toml`:

```toml
[target.wasm32-unknown-unknown]
rustflags = ["-C", "target-feature=+bulk-memory"]
```

### Building for WASM

```bash
cargo build --release --target wasm32-unknown-unknown --features wasm
```

For additional size reduction, install `wasm-opt` (from binaryen) and run:

```bash
wasm-opt -Oz -o optimized.wasm target/wasm32-unknown-unknown/release/comprs.wasm
```

### WASM size checker

Use the helper script to build and report raw/gzipped sizes (and wasm-opt -Oz if installed):

```bash
./scripts/wasm-size.sh
# pass extra args to cargo if needed, e.g.:
# ./scripts/wasm-size.sh --locked
```
Current local size (no wasm-opt): raw 90,473 bytes; gzipped 30,065 bytes.

### Target metrics

See `docs/performance-targets.md` for the current performance, quality, and size targets we are tracking (including wasm size baselines and regression budgets).

### Baseline snapshot

For a quick sanity check of PNG/JPEG sizes and timings vs the `image` crate, see `benches/size_snapshot_baseline.md` (generated from `cargo bench --bench size_snapshot -- --nocapture`). Rerun on your hardware to track regressions.

You can also run and capture the snapshot via:

```bash
./scripts/size_snapshot.sh -- --nocapture
```

For preset throughput baselines, see `benches/preset_compare_baseline.md` and rerun on your hardware (`cargo bench --bench preset_compare -- --sample-size 20`, or `--sample-size 10` for quicker runs).

Or use the helper script (logs to `target/preset_compare.log`):

```bash
./scripts/preset_compare.sh
# pass extra args after -- to override defaults
```

### Comparison to Alternatives

| Library         | WASM Size (uncompressed) |
| --------------- | ------------------------ |
| **comprs**      | **92 KB**                |
| wasm-mozjpeg    | ~208KB                   |
| squoosh oxipng  | ~625KB                   |
| squoosh mozjpeg | ~803KB                   |
| image crate     | ~6-10MB                  |

_Note: Squoosh codec sizes from squoosh-browser-sense. Sizes depend on build configuration and optimization level._

---

## Performance Tips

### For Benchmarking

1. **Use release mode**: `cargo bench` automatically uses release optimizations
2. **Warm up**: Criterion handles warm-up automatically
3. **Consistent hardware**: Close other applications during benchmarks
4. **Multiple runs**: Let Criterion collect enough samples for statistical significance

### For Production

1. **Reuse buffers**: Use `encode_into` variants to avoid allocations
2. **Choose appropriate quality**: JPEG 75-85 is usually sufficient
3. **Consider subsampling**: 4:2:0 reduces JPEG size by ~30-40%
4. **Match filter to content**: PNG `None` for noise, `Adaptive` for photos

---

## Benchmark Results

Run benchmarks on your hardware to get accurate numbers:

```bash
cargo bench --bench encode_benchmark -- --save-baseline my-machine
```

Results are saved to `target/criterion/` with HTML reports.

---

## Data Sources

Library sizes and performance claims were gathered from:

- **npm packages**: npmjs.com package pages (unpacked sizes)
- **Bundlephobia**: bundlephobia.com for minified+gzipped bundle sizes
- **sharp**: sharp.pixelplumbing.com documentation
- **Squoosh**: squoosh-browser-sense package, wasm-mozjpeg project
- **zune-png**: Phoronix benchmarks (1.8x faster than libpng)
- **photon-rs**: Project documentation (13x speedup for Gaussian blur vs JS)
- **jimp**: skypack.dev benchmarks (~0.7 img/s vs sharp's ~11 img/s)

_Sizes are approximate and vary by version, build configuration, and optimization settings. Last updated: December 2025._
