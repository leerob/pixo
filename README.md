# comprs

A minimal-dependency, high-performance image compression library written in Rust.

This is a work-in-progress and exploration of coding agents.

## Features

- **Zero runtime dependencies** — All encoding algorithms implemented from scratch
- **Entirely in Rust** — No C or C++ codecs required
- **PNG and JPEG support** — Lossless PNG, lossy PNG (quantization), and lossy JPEG
- **Small WASM binary** — ~135 KB with competitive compression ([benchmarks](./benches/BENCHMARKS.md))

## Usage

1. **[Playground](https://comprs.vercel.app/)** (recommended) — Try it in your browser
2. **[WASM](./docs/wasm.md)** — Use in browser or Node.js applications
3. **[CLI](./docs/cli.md)** — Compress images from the command line
4. **[Rust Crate](./docs/crate.md)** — Use as a library in your Rust projects

## Documentation

Comprehensive guides explaining the algorithms and compression strategies:

- [Documentation Index](./docs/README.md) — Start here for an overview
- [Introduction to Image Compression](./docs/introduction-to-image-compression.md) — Why and how we compress images

### Core Algorithms

- [Huffman Coding](./docs/huffman-coding.md) — Variable-length codes based on symbol frequency
- [LZ77 Compression](./docs/lz77-compression.md) — Dictionary-based compression with sliding windows
- [DEFLATE Algorithm](./docs/deflate.md) — How LZ77 and Huffman combine

### Image Formats

- [PNG Encoding](./docs/png-encoding.md) — Lossless compression with predictive filtering
- [JPEG Encoding](./docs/jpeg-encoding.md) — Lossy compression pipeline
- [Discrete Cosine Transform](./docs/dct.md) — The mathematical heart of JPEG
- [Quantization](./docs/quantization.md) — How JPEG achieves compression

## References

- [PNG Specification (RFC 2083)](https://www.w3.org/TR/PNG/)
- [DEFLATE Specification (RFC 1951)](https://www.rfc-editor.org/rfc/rfc1951)
- [zlib Specification (RFC 1950)](https://www.rfc-editor.org/rfc/rfc1950)
- [JPEG Standard (ITU-T T.81)](https://www.w3.org/Graphics/JPEG/itu-t81.pdf)
