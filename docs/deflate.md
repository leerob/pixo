# The DEFLATE Algorithm

DEFLATE is the compression algorithm behind PNG, gzip, ZIP, and HTTP compression. Specified in RFC 1951 (1996), it elegantly combines LZ77 dictionary compression with Huffman entropy coding for excellent compression ratios.

## The Big Picture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Input      │───▶│    LZ77     │───▶│  Huffman    │───▶ Compressed
│  Data       │    │ (find       │    │  Encoding   │     Output
│             │    │  patterns)  │    │  (optimal   │
│             │    │             │    │   codes)    │
└─────────────┘    └─────────────┘    └─────────────┘
```

1. **LZ77** finds repeated patterns and converts them to back-references
2. **Huffman coding** assigns optimal variable-length codes to the LZ77 tokens

This two-stage approach achieves better compression than either algorithm alone.

## Block Structure

DEFLATE organizes data into **blocks**. Each block can use different compression settings:

```
┌─────────────────────────────────────────────────────────────────┐
│ Block 1           │ Block 2           │ Block 3 (final)        │
│ (fixed Huffman)   │ (dynamic Huffman) │ (stored)               │
└─────────────────────────────────────────────────────────────────┘
```

Each block starts with a 3-bit header:

| Bits   | Meaning                                                          |
| ------ | ---------------------------------------------------------------- |
| 1 bit  | BFINAL: 1 if this is the last block                              |
| 2 bits | BTYPE: Block type (00=stored, 01=fixed, 10=dynamic, 11=reserved) |

```rust
// From src/compress/deflate.rs
fn encode_fixed_huffman(tokens: &[Token]) -> Vec<u8> {
    let mut writer = BitWriter::new();

    // Block header: BFINAL=1 (last block), BTYPE=01 (fixed Huffman)
    writer.write_bits(1, 1); // BFINAL
    writer.write_bits(1, 2); // BTYPE (01 = fixed Huffman, LSB first)
    // ...
}
```

### Block Types

**Type 0 (Stored)**: No compression. Just copy bytes verbatim.

- Used when data is incompressible (already compressed, random)
- 5-byte overhead per 65535 bytes

**Type 1 (Fixed Huffman)**: Use predefined Huffman tables.

- No need to transmit the code tables
- Good for small data or when encoding speed matters

**Type 2 (Dynamic Huffman)**: Custom Huffman tables optimized for this block.

- Tables are transmitted before the data
- Best compression for larger blocks

Our implementation uses fixed Huffman codes for simplicity:

```rust
// From src/compress/deflate.rs
pub fn deflate(data: &[u8], level: u8) -> Vec<u8> {
    // Use LZ77 to find matches
    let mut lz77 = Lz77Compressor::new(level);
    let tokens = lz77.compress(data);

    // For simplicity, we use fixed Huffman codes
    encode_fixed_huffman(&tokens)
}
```

## Encoding Literals and Lengths

DEFLATE uses a unified alphabet for literals (0-255), end-of-block (256), and lengths (257-285):

```
┌──────────────────────────────────────────────────────────────┐
│ 0-255: Literal bytes                                         │
│ 256:   End of block marker                                   │
│ 257-285: Match lengths (with extra bits for fine-tuning)     │
└──────────────────────────────────────────────────────────────┘
```

### Length Encoding

Match lengths (3-258) are encoded as codes 257-285 plus extra bits:

| Code | Length | Extra Bits | Range |
| ---- | ------ | ---------- | ----- |
| 257  | 3      | 0          | 3     |
| 258  | 4      | 0          | 4     |
| ...  | ...    | ...        | ...   |
| 265  | 11-12  | 1          | 11-12 |
| 266  | 13-14  | 1          | 13-14 |
| ...  | ...    | ...        | ...   |
| 285  | 258    | 0          | 258   |

```rust
// From src/compress/deflate.rs
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
    35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258,
];

const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
```

**Example**: Encoding length 18

1. Find the code: 18 is in range 17-18 (code 268)
2. Extra bits needed: 1 bit
3. Extra value: 18 - 17 = 1
4. Output: code 268 + extra bit 1

## Encoding Distances

Distances (1-32768) have their own alphabet (0-29) with extra bits:

| Code | Distance    | Extra Bits | Range       |
| ---- | ----------- | ---------- | ----------- |
| 0    | 1           | 0          | 1           |
| 1    | 2           | 0          | 2           |
| 2    | 3           | 0          | 3           |
| 3    | 4           | 0          | 4           |
| 4    | 5-6         | 1          | 5-6         |
| 5    | 7-8         | 1          | 7-8         |
| ...  | ...         | ...        | ...         |
| 29   | 24577-32768 | 13         | 24577-32768 |

```rust
// From src/compress/deflate.rs
const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
    257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577,
];

const DISTANCE_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
];
```

## Fixed Huffman Code Tables

Fixed (Type 1) blocks use these predefined Huffman code lengths:

```
Literal/Length (0-287):
  Symbols   0-143: 8 bits each
  Symbols 144-255: 9 bits each
  Symbols 256-279: 7 bits each
  Symbols 280-287: 8 bits each

Distance (0-29):
  All symbols: 5 bits each
```

These lengths were chosen to give reasonable compression without needing to transmit the tables.

## Encoding Example

Let's encode the tokens from compressing "ABRACADABRA":

```
Tokens: A, B, R, A, C, A, D, Match(length=4, distance=7)
```

Step by step:

1. **A** (literal 65): Look up code for symbol 65 → Huffman code
2. **B** (literal 66): Look up code for symbol 66 → Huffman code
3. **R** (literal 82): Look up code for symbol 82 → Huffman code
4. **A** (literal 65): Same as above
5. **C** (literal 67): Look up code for symbol 67 → Huffman code
6. **A** (literal 65): Same as above
7. **D** (literal 68): Look up code for symbol 68 → Huffman code
8. **Match(4,7)**:
   - Length 4 → code 258 (8 bits in fixed table) + 0 extra bits
   - Distance 7 → code 5 (5 bits) + 1 extra bit (value 0)
9. **End of block**: Symbol 256 (7 bits in fixed table)

```rust
// From src/compress/deflate.rs
for token in tokens {
    match *token {
        Token::Literal(byte) => {
            let code = lit_codes[byte as usize];
            writer.write_bits(reverse_bits(code.code, code.length), code.length);
        }
        Token::Match { length, distance } => {
            // Encode length
            let (len_symbol, len_extra_bits, len_extra_value) = length_code(length);
            let len_code = lit_codes[len_symbol as usize];
            writer.write_bits(reverse_bits(len_code.code, len_code.length), len_code.length);

            if len_extra_bits > 0 {
                writer.write_bits(len_extra_value as u32, len_extra_bits);
            }

            // Encode distance
            let (dist_symbol, dist_extra_bits, dist_extra_value) = distance_code(distance);
            let dist_code = dist_codes[dist_symbol as usize];
            writer.write_bits(reverse_bits(dist_code.code, dist_code.length), dist_code.length);

            if dist_extra_bits > 0 {
                writer.write_bits(dist_extra_value as u32, dist_extra_bits);
            }
        }
    }
}
```

## Bit Ordering: LSB First

A crucial detail: DEFLATE packs bits **LSB (Least Significant Bit) first** within bytes.

```
Bits written: 1, 0, 1, 1, 0
              ▲
              │ first bit goes to bit 0 (rightmost)

Resulting byte: 0b_???01011
                      ▲▲▲▲▲
                      │││││
                      │││││
                      5th bit ─┘│││
                      4th bit ──┘││
                      3rd bit ───┘│
                      2nd bit ────┘
                      1st bit ─────┘
```

However, Huffman codes themselves are **MSB first** within the code. So we reverse the bits when writing:

```rust
// From src/compress/deflate.rs
fn reverse_bits(code: u16, length: u8) -> u32 {
    let mut result = 0u32;
    let mut code = code as u32;
    for _ in 0..length {
        result = (result << 1) | (code & 1);
        code >>= 1;
    }
    result
}
```

## Stored Blocks

Sometimes data is incompressible. In this case, stored blocks are more efficient:

```
┌────┬────────┬────────┬─────────────────────────┐
│Hdr │  LEN   │  NLEN  │      Raw Data           │
│ 3b │ 16bit  │ 16bit  │   (up to 65535 bytes)   │
└────┴────────┴────────┴─────────────────────────┘
```

- **LEN**: Number of data bytes
- **NLEN**: One's complement of LEN (for error checking)

```rust
// From src/compress/deflate.rs
pub fn deflate_stored(data: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();

    for (i, chunk) in data.chunks(65535).enumerate() {
        let is_final = i == num_chunks - 1;
        let len = chunk.len() as u16;
        let nlen = !len;

        // Block header
        output.push(if is_final { 0x01 } else { 0x00 });

        // LEN and NLEN (little-endian)
        output.push(len as u8);
        output.push((len >> 8) as u8);
        output.push(nlen as u8);
        output.push((nlen >> 8) as u8);

        // Data
        output.extend_from_slice(chunk);
    }

    output
}
```

## Compression Ratios

DEFLATE typically achieves:

| Data Type                   | Compression Ratio           |
| --------------------------- | --------------------------- |
| English text                | 30-40% of original          |
| Source code                 | 20-30% of original          |
| Structured data (JSON, XML) | 10-20% of original          |
| Photographs                 | 80-100% (use JPEG instead!) |
| Random/encrypted data       | ~100% (incompressible)      |

## Why DEFLATE Works So Well

The combination of LZ77 + Huffman is synergistic:

1. **LZ77 creates skewed distributions**: After LZ77, the output has many more small distances and short matches than random data would. These skewed frequencies are perfect for Huffman coding.

2. **Huffman optimizes the final encoding**: The variable-length codes ensure common patterns (short distances, common literals) use fewer bits.

3. **Block structure allows adaptation**: Different parts of a file can use different strategies.

## Comparison with Other Formats

| Algorithm | Used In        | Approach                           |
| --------- | -------------- | ---------------------------------- |
| DEFLATE   | PNG, gzip, ZIP | LZ77 + Huffman                     |
| bzip2     | bz2 files      | BWT + MTF + Huffman                |
| LZMA      | 7z, xz         | LZ77 + Range coding                |
| Zstandard | zst files      | LZ77 + FSE + Huffman               |
| Brotli    | Web (HTTP)     | LZ77 + 2nd-order context + Huffman |

DEFLATE remains popular because:

- Excellent software support
- Good balance of speed and compression
- Well-understood and standardized

## Summary

DEFLATE combines:

- **LZ77** for pattern matching
- **Huffman coding** for optimal bit allocation
- **Block structure** for flexibility
- **Clever encoding** of lengths and distances

This 1996 algorithm remains one of the most widely-used compression methods in computing.

## Next Steps

Continue to [PNG Encoding](./png-encoding.md) to see how PNG uses DEFLATE along with predictive filtering for excellent image compression.

---

## References

- [RFC 1951 - DEFLATE Compressed Data Format Specification](https://www.rfc-editor.org/rfc/rfc1951)
- Deutsch, P. (1996). "DEFLATE Compressed Data Format Specification version 1.3"
- See implementation: `src/compress/deflate.rs`
