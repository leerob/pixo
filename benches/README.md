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

| Library           | WASM-friendly   | Binary Size (gzip) | Throughput | Notes                                              |
| ----------------- | --------------- | ------------------ | ---------- | -------------------------------------------------- |
| **comprs**        | Yes             | ~100-150KB (est.)  | Good       | Zero deps, pure Rust, simple WASM target           |
| `image`           | Yes             | ~2-4MB             | Good       | Pure Rust, many codecs included                    |
| `photon-rs`       | Yes             | ~200-400KB         | Excellent  | Pure Rust, designed for WASM, 13x faster than JS   |
| `zune-image`      | Yes             | ~500KB-1MB         | Excellent  | Pure Rust, SIMD optimized, 1.8x faster than libpng |
| `mozjpeg`         | Emscripten only | ~30-50KB           | Excellent  | C library, requires complex build setup            |
| `libpng` bindings | Emscripten only | N/A                | Excellent  | C library, requires Emscripten toolchain           |

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

| What you need | comprs (est. gzip) | image crate (est. gzip) |
| ------------- | ------------------ | ----------------------- |
| PNG only      | ~60-80KB           | ~2MB                    |
| JPEG only     | ~50-70KB           | ~2MB                    |
| PNG + JPEG    | ~100-150KB         | ~2MB                    |

The `image` crate includes decoders and encoders for BMP, GIF, ICO, TIFF, WebP, AVIF, and more - even if you only need PNG. You can use feature flags to reduce size, but the core still includes significant code.

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

Estimated WASM binary sizes for comprs (gzipped, with `opt-level = "z"` and `wasm-opt -Oz`):

| Component                         | Approx. Size (gzip) |
| --------------------------------- | ------------------- |
| DEFLATE (LZ77 + Huffman)          | ~30-40KB            |
| PNG encoder                       | ~20-30KB            |
| JPEG encoder (DCT + quantization) | ~40-50KB            |
| Core utilities                    | ~10KB               |
| **Total (PNG + JPEG)**            | **~100-150KB**      |

_These are estimates. Actual sizes depend on build configuration. Run `wasm-pack build --release --features wasm` and check the output size to verify._

Compare to alternatives:

| Library            | WASM Size (uncompressed) | WASM Size (gzip) |
| ------------------ | ------------------------ | ---------------- |
| comprs (estimated) | ~300-400KB               | ~100-150KB       |
| squoosh mozjpeg    | ~803KB                   | ~30-50KB         |
| squoosh oxipng     | ~625KB                   | ~80-120KB        |
| wasm-mozjpeg       | ~208KB                   | ~34KB            |
| image crate        | ~6-10MB                  | ~2-4MB           |

_Note: Squoosh codec sizes from squoosh-browser-sense. wasm-mozjpeg is a minimal WASM build. Actual sizes depend on build configuration and optimization level._

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
