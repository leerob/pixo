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

## File Structure

A PNG file is a sequence of **chunks**:

```
┌──────────────┐
│ PNG Signature│  8 bytes: 89 50 4E 47 0D 0A 1A 0A
├──────────────┤
│ IHDR Chunk   │  Image header (dimensions, color type)
├──────────────┤
│ IDAT Chunk(s)│  Compressed image data
├──────────────┤
│ IEND Chunk   │  End marker
└──────────────┘
```

### The PNG Signature

Every PNG file starts with these 8 bytes:

```
89 50 4E 47 0D 0A 1A 0A
│   │  │  │  │  │  │  │
│   P  N  G  │  │  │  │
│            CR LF │  LF
│                  │
High bit set       End-of-file character
(detect 7-bit      (detect file truncation)
transmission errors)
```

This signature detects common file transfer issues:

- Binary/text mode confusion (CR/LF changes)
- 7-bit channel corruption (high bit stripped)
- File truncation (EOF character)

```rust
// From src/png/mod.rs
const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
```

### Chunk Structure

Every chunk follows this format:

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│    Length    │     Type     │     Data     │     CRC      │
│   (4 bytes)  │   (4 bytes)  │  (variable)  │   (4 bytes)  │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

- **Length**: Size of data field (not including type or CRC)
- **Type**: 4 ASCII characters (e.g., "IHDR", "IDAT")
- **Data**: Chunk-specific payload
- **CRC**: CRC-32 checksum of type + data

```rust
// From src/png/chunk.rs
pub fn write_chunk(output: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    // Length (big-endian)
    output.extend_from_slice(&(data.len() as u32).to_be_bytes());

    // Type
    output.extend_from_slice(chunk_type);

    // Data
    output.extend_from_slice(data);

    // CRC (of type + data)
    let crc = crc32(&[chunk_type.as_slice(), data].concat());
    output.extend_from_slice(&crc.to_be_bytes());
}
```

## The IHDR Chunk

The image header contains essential metadata:

```
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│Width│Height│Bit │Color│Comp │Filter│Inter│
│4 bytes│4 bytes│Depth│Type│Method│Method│lace│
└─────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

| Field       | Size    | Values                             |
| ----------- | ------- | ---------------------------------- |
| Width       | 4 bytes | Image width in pixels              |
| Height      | 4 bytes | Image height in pixels             |
| Bit Depth   | 1 byte  | 8 (we support 8-bit only)          |
| Color Type  | 1 byte  | 0=Gray, 2=RGB, 4=GrayAlpha, 6=RGBA |
| Compression | 1 byte  | 0 (DEFLATE is the only option)     |
| Filter      | 1 byte  | 0 (adaptive filtering)             |
| Interlace   | 1 byte  | 0=None, 1=Adam7                    |

```rust
// From src/png/mod.rs
fn write_ihdr(output: &mut Vec<u8>, width: u32, height: u32, color_type: ColorType) {
    let mut ihdr_data = Vec::with_capacity(13);

    ihdr_data.extend_from_slice(&width.to_be_bytes());
    ihdr_data.extend_from_slice(&height.to_be_bytes());
    ihdr_data.push(color_type.png_bit_depth());   // Bit depth: 8
    ihdr_data.push(color_type.png_color_type());  // Color type
    ihdr_data.push(0);  // Compression: DEFLATE
    ihdr_data.push(0);  // Filter method: adaptive
    ihdr_data.push(0);  // Interlace: none

    chunk::write_chunk(output, b"IHDR", &ihdr_data);
}
```

## PNG Filtering: The Secret Sauce

Before compression, PNG applies **filtering** to make the data more compressible. The key insight: **differences between adjacent pixels are usually small**.

Consider a row of pixels in a gradient:

```
Original:     100, 102, 104, 106, 108, 110, 112, 114

Sub filter:     100,   2,   2,   2,   2,   2,   2,   2
(subtract left)

The filtered version has many repeated values → compresses much better!
```

### The Five Filter Types

PNG defines five filter types (0-4):

#### Filter Type 0: None

No filtering. Raw pixel values.

```
Filtered(x) = Original(x)
```

#### Filter Type 1: Sub

Difference from left pixel.

```
Filtered(x) = Original(x) - Left(x)

Example (3-byte RGB pixels):
Original: [100,150,200] [105,155,205] [110,160,210]
Left:     [  0,  0,  0] [100,150,200] [105,155,205]
Filtered: [100,150,200] [  5,  5,  5] [  5,  5,  5]
```

```rust
// From src/png/filter.rs
fn filter_sub(row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    for (i, &byte) in row.iter().enumerate() {
        let left = if i >= bpp { row[i - bpp] } else { 0 };
        output.push(byte.wrapping_sub(left));
    }
}
```

#### Filter Type 2: Up

Difference from above pixel.

```
Filtered(x) = Original(x) - Above(x)

Excellent for horizontal patterns (striped backgrounds, etc.)
```

```rust
// From src/png/filter.rs
fn filter_up(row: &[u8], prev_row: &[u8], output: &mut Vec<u8>) {
    for (i, &byte) in row.iter().enumerate() {
        output.push(byte.wrapping_sub(prev_row[i]));
    }
}
```

#### Filter Type 3: Average

Difference from average of left and above.

```
Filtered(x) = Original(x) - floor((Left(x) + Above(x)) / 2)

Good for gradients in both directions.
```

```rust
// From src/png/filter.rs
fn filter_average(row: &[u8], prev_row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    for (i, &byte) in row.iter().enumerate() {
        let left = if i >= bpp { row[i - bpp] as u16 } else { 0 };
        let above = prev_row[i] as u16;
        let avg = ((left + above) / 2) as u8;
        output.push(byte.wrapping_sub(avg));
    }
}
```

#### Filter Type 4: Paeth

Uses the Paeth predictor — a clever heuristic developed by Alan W. Paeth.

```
Given three neighbors:
    C | B
   ---+---
    A | X

Paeth selects the neighbor closest to (A + B - C):
- If A + B - C is close to A → predict A
- If A + B - C is close to B → predict B
- If A + B - C is close to C → predict C

The intuition: (A + B - C) estimates X assuming linear gradients.
```

```rust
// From src/png/filter.rs
fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    let a = a as i16;
    let b = b as i16;
    let c = c as i16;

    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();

    if pa <= pb && pa <= pc {
        a as u8
    } else if pb <= pc {
        b as u8
    } else {
        c as u8
    }
}
```

### Adaptive Filter Selection

The best filter varies by row! PNG allows choosing different filters for each scanline.

Our implementation tries all five filters and picks the one that produces the most compressible output:

```rust
// From src/png/filter.rs
fn adaptive_filter(row: &[u8], prev_row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    // Apply each filter
    let mut none_buf = Vec::new();
    let mut sub_buf = Vec::new();
    let mut up_buf = Vec::new();
    let mut avg_buf = Vec::new();
    let mut paeth_buf = Vec::new();

    none_buf.extend_from_slice(row);
    filter_sub(row, bpp, &mut sub_buf);
    filter_up(row, prev_row, &mut up_buf);
    filter_average(row, prev_row, bpp, &mut avg_buf);
    filter_paeth(row, prev_row, bpp, &mut paeth_buf);

    // Score each filter (lower = more compressible)
    let scores = [
        (FILTER_NONE, score_filter(&none_buf)),
        (FILTER_SUB, score_filter(&sub_buf)),
        (FILTER_UP, score_filter(&up_buf)),
        (FILTER_AVERAGE, score_filter(&avg_buf)),
        (FILTER_PAETH, score_filter(&paeth_buf)),
    ];

    // Pick the filter with lowest score
    let (best_filter, _) = scores.iter().min_by_key(|(_, score)| *score).unwrap();
    // ...
}
```

The scoring heuristic is **sum of absolute values** — smaller values suggest less entropy:

```rust
fn score_filter(filtered: &[u8]) -> u64 {
    filtered.iter().map(|&b| (b as i8).unsigned_abs() as u64).sum()
}
```

## Filtered Data Format

After filtering, each scanline is prefixed with a filter type byte:

```
┌────┬──────────────────────────┐
│ F  │    Filtered Row Data     │
│1byte│   (width × bytes/pixel)  │
└────┴──────────────────────────┘
```

For a 10×3 RGB image:

```
Row 0: [Filter byte] [30 bytes of filtered RGB data]
Row 1: [Filter byte] [30 bytes of filtered RGB data]
Row 2: [Filter byte] [30 bytes of filtered RGB data]
```

## IDAT Chunks

The filtered data is compressed with DEFLATE and split into IDAT chunks:

```rust
// From src/png/mod.rs
fn write_idat_chunks(output: &mut Vec<u8>, compressed: &[u8]) {
    const CHUNK_SIZE: usize = 8192;

    for chunk_data in compressed.chunks(CHUNK_SIZE) {
        chunk::write_chunk(output, b"IDAT", chunk_data);
    }
}
```

Splitting into 8KB chunks keeps individual chunks manageable for decoders.

## CRC-32 Checksums

PNG uses CRC-32 for chunk integrity. Our implementation uses a precomputed lookup table for speed:

```rust
// From src/compress/crc32.rs
const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFF_u32;
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC_TABLE[index];
    }
    crc ^ 0xFFFFFFFF
}
```

## Filter Selection Strategies

Our library supports multiple strategies:

```rust
// From src/png/mod.rs
pub enum FilterStrategy {
    None,     // Always use no filter (fastest)
    Sub,      // Always use Sub filter
    Up,       // Always use Up filter
    Average,  // Always use Average filter
    Paeth,    // Always use Paeth filter
    Adaptive, // Choose best per row (best compression)
}
```

**Trade-offs**:

- `None`: Fastest encoding, worst compression
- `Adaptive`: Best compression, slower encoding
- Single filter: Middle ground

## Complete Encoding Example

```rust
// Encode RGB pixels as PNG
let pixels: Vec<u8> = vec![255, 0, 0, 0, 255, 0, 0, 0, 255]; // 3 RGB pixels
let png = png::encode(&pixels, 3, 1, ColorType::Rgb).unwrap();
```

What happens internally:

1. **Validate** dimensions and data length
2. **Write PNG signature** (8 bytes)
3. **Write IHDR** with dimensions and color type
4. **Filter** each row (Sub, Up, Average, Paeth, or adaptive)
5. **DEFLATE** compress the filtered data
6. **Write IDAT** chunks with compressed data
7. **Write IEND** chunk

## Compression Performance

PNG compression effectiveness depends heavily on the image:

| Image Type      | Typical Compression            |
| --------------- | ------------------------------ |
| Solid color     | 99%+ (nearly nothing to store) |
| Simple graphics | 80-95%                         |
| Screenshots     | 50-80%                         |
| Photographs     | 10-30% (use JPEG instead!)     |

## Why Not Just DEFLATE?

You might wonder: why not just DEFLATE the raw pixels?

The filtering stage makes a **huge** difference:

```
Example: 100×100 gradient image

Without filtering:
  - Each pixel is unique value
  - DEFLATE finds few patterns
  - Compressed size: ~7 KB

With filtering (Sub):
  - Most values become small differences (~2)
  - DEFLATE finds many repeated patterns
  - Compressed size: ~0.5 KB

14× improvement from filtering!
```

## Summary

PNG encoding combines:

- **Clever filtering** that converts pixels to small differences
- **DEFLATE compression** that eliminates redundancy
- **Chunk structure** for integrity and metadata
- **Adaptive selection** for optimal per-row filtering

This pipeline achieves excellent lossless compression while remaining simple to implement and decode.

## Next Steps

Continue to [JPEG Encoding](./jpeg-encoding.md) to learn about lossy compression for photographs.

---

## References

- [RFC 2083 - PNG Specification](https://www.w3.org/TR/PNG/)
- [PNG Filter Algorithms (W3C)](https://www.w3.org/TR/PNG-Filters.html)
- Paeth, A.W. (1991). "Image File Compression Made Easy" in Graphics Gems II
- See implementation: `src/png/mod.rs`, `src/png/filter.rs`, `src/png/chunk.rs`
