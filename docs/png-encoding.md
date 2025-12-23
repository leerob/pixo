# PNG Encoding

PNG (Portable Network Graphics) is a lossless image format that combines predictive filtering with DEFLATE compression. Developed in the 1990s as a patent-free alternative to GIF, it has become the standard for lossless web graphics.

## Why PNG?

PNG excels at:

- **Screenshots and UI elements** — Sharp edges stay sharp
- **Graphics with text** — No blurry letters
- **Images with transparency** — Full alpha channel support
- **Diagrams and illustrations** — Solid colors compress well

PNG is **not** ideal for:

- Photographs (use JPEG for smaller files)
- Animation (use GIF, APNG, or WebP)

## The PNG Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Raw Pixels  │───▶│  Filtering  │───▶│   DEFLATE   │───▶│  PNG Chunks │
│  (RGB/RGBA) │    │ (per-row    │    │  (compress) │    │  (format)   │
│             │    │  prediction)│    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## Presets

Our library provides three presets that bundle the right combination of filter strategy, compression level, and optimizations:

### Fast (Preset 0)

```rust
PngOptions::fast()
```

Prioritizes encoding speed over compression ratio.

- **Compression level**: 2
- **Filter strategy**: AdaptiveFast (tries Sub, Up, Paeth with early cutoffs)
- **Optimizations**: None enabled

Best for: Development workflows, real-time processing, or when encoding time matters more than file size.

### Balanced (Preset 1)

```rust
PngOptions::balanced()
```

Good tradeoff between speed and compression. This is the recommended default.

- **Compression level**: 6
- **Filter strategy**: Adaptive (tries all 5 filters per row)
- **Optimizations**: All enabled
  - Palette reduction (≤256 colors → indexed)
  - Color type reduction (RGBA → RGB, RGB → Gray when lossless)
  - Alpha optimization (zero RGB for transparent pixels)
  - Metadata stripping (remove tEXt, tIME chunks)

Best for: General use, web assets, production builds.

### Max (Preset 2)

```rust
PngOptions::max()
```

Maximum compression, competitive with oxipng.

- **Compression level**: 9
- **Filter strategy**: MinSum (sum-of-absolute-values scoring)
- **Optimizations**: All enabled + optimal DEFLATE
  - Uses Zopfli-style iterative refinement for smaller output
  - Significantly slower, but produces the smallest files

Best for: Final distribution, assets where every byte counts.

## Usage

```rust
use comprs::png::{encode_with_options, PngOptions};
use comprs::ColorType;

// Fast encoding
let png = encode_with_options(&pixels, width, height, ColorType::Rgba, &PngOptions::fast())?;

// Balanced (recommended)
let png = encode_with_options(&pixels, width, height, ColorType::Rgba, &PngOptions::balanced())?;

// Maximum compression
let png = encode_with_options(&pixels, width, height, ColorType::Rgba, &PngOptions::max())?;

// Or use preset number (0=fast, 1=balanced, 2=max)
let png = encode_with_options(&pixels, width, height, ColorType::Rgba, &PngOptions::from_preset(1))?;
```

## How Filtering Works

Before compression, PNG applies **filtering** to make the data more compressible. The key insight: differences between adjacent pixels are usually small.

```
Original:     100, 102, 104, 106, 108, 110, 112, 114

Sub filter:   100,   2,   2,   2,   2,   2,   2,   2
(subtract left pixel)

The filtered version has many repeated values → compresses much better!
```

PNG defines five filter types:

| Filter  | Prediction                           | Best for                      |
| ------- | ------------------------------------ | ----------------------------- |
| None    | Raw value                            | Random/noisy data             |
| Sub     | Left pixel                           | Horizontal gradients          |
| Up      | Above pixel                          | Vertical gradients            |
| Average | Average of left and above            | Diagonal gradients            |
| Paeth   | Nearest of left, above, upper-left   | General-purpose, often best   |

The best filter varies by row, so adaptive strategies try multiple filters and pick the one that produces the most compressible output using a scoring heuristic.

## Filter Strategies

For advanced use cases, you can specify a filter strategy directly:

```rust
use comprs::png::{PngOptions, FilterStrategy};

let options = PngOptions {
    filter_strategy: FilterStrategy::MinSum,
    compression_level: 6,
    ..Default::default()
};
```

| Strategy      | Description                                         | Speed    |
| ------------- | --------------------------------------------------- | -------- |
| `None`        | Always use no filter                                | Fastest  |
| `Sub`         | Always use Sub filter                               | Fast     |
| `Up`          | Always use Up filter                                | Fast     |
| `Average`     | Always use Average filter                           | Fast     |
| `Paeth`       | Always use Paeth filter                             | Fast     |
| `MinSum`      | Per-row selection via sum-of-abs scoring            | Medium   |
| `AdaptiveFast`| Tries Sub/Up/Paeth with early cutoffs              | Medium   |
| `Adaptive`    | Tries all 5 filters, picks lowest score            | Slower   |

## Lossless Optimizations

The balanced and max presets enable several lossless optimizations:

### Palette Reduction

Images with ≤256 unique colors are converted to indexed color mode with a PLTE chunk. The palette is reordered using the Zeng algorithm to maximize DEFLATE compression.

### Color Type Reduction

- RGBA with all pixels opaque → RGB (saves 25%)
- RGB where R=G=B for all pixels → Grayscale (saves 67%)
- RGBA with varying alpha but R=G=B → GrayAlpha (saves 50%)

### Alpha Optimization

For transparent pixels (alpha=0), the RGB channels are zeroed. Since these colors are invisible, this is lossless but creates more repetition for DEFLATE.

### Bit Depth Reduction

Grayscale images using only certain values can be packed to lower bit depths (4-bit, 2-bit, 1-bit).

## File Structure

A PNG file is a sequence of **chunks**:

```
┌──────────────┐
│ PNG Signature│  8 bytes: 89 50 4E 47 0D 0A 1A 0A
├──────────────┤
│ IHDR Chunk   │  Image header (dimensions, color type)
├──────────────┤
│ PLTE Chunk   │  Palette (optional, for indexed color)
├──────────────┤
│ IDAT Chunk(s)│  Compressed image data
├──────────────┤
│ IEND Chunk   │  End marker
└──────────────┘
```

Each chunk has: Length (4 bytes) + Type (4 bytes) + Data + CRC-32 (4 bytes).

## Next Steps

Continue to [JPEG Encoding](./jpeg-encoding.md) to learn about lossy compression for photographs.

---

## References

- [RFC 2083 - PNG Specification](https://www.w3.org/TR/PNG/)
- [PNG Filter Algorithms (W3C)](https://www.w3.org/TR/PNG-Filters.html)
- See implementation: `src/png/mod.rs`, `src/png/filter.rs`
