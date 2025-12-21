//! DEFLATE compression algorithm (RFC 1951).
//!
//! Combines LZ77 compression with Huffman coding.

use crate::bits::BitWriter;
use crate::compress::huffman;
use crate::compress::lz77::{Lz77Compressor, Token, MAX_MATCH_LENGTH, MIN_MATCH_LENGTH};

/// Length code base values (codes 257-285).
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];

/// Extra bits for length codes.
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

/// Distance code base values (codes 0-29).
const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

/// Extra bits for distance codes.
const DISTANCE_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Get the length code (257-285) for a match length.
fn length_code(length: u16) -> (u16, u8, u16) {
    debug_assert!(
        (MIN_MATCH_LENGTH as u16..=MAX_MATCH_LENGTH as u16).contains(&length),
        "Invalid length: {}",
        length
    );

    for (i, &base) in LENGTH_BASE.iter().enumerate() {
        let next_base = if i + 1 < LENGTH_BASE.len() {
            LENGTH_BASE[i + 1]
        } else {
            259
        };
        if length >= base && length < next_base {
            let extra_bits = LENGTH_EXTRA[i];
            let extra_value = length - base;
            return (257 + i as u16, extra_bits, extra_value);
        }
    }

    // Length 258
    (285, 0, 0)
}

/// Get the distance code (0-29) for a match distance.
fn distance_code(distance: u16) -> (u16, u8, u16) {
    debug_assert!(distance >= 1 && distance <= 32768, "Invalid distance");

    for (i, &base) in DISTANCE_BASE.iter().enumerate() {
        let next_base = if i + 1 < DISTANCE_BASE.len() {
            DISTANCE_BASE[i + 1]
        } else {
            32769
        };
        if distance >= base && distance < next_base {
            let extra_bits = DISTANCE_EXTRA[i];
            let extra_value = distance - base;
            return (i as u16, extra_bits, extra_value);
        }
    }

    unreachable!()
}

/// Compress data using DEFLATE algorithm.
///
/// # Arguments
/// * `data` - Raw data to compress
/// * `level` - Compression level 1-9
///
/// # Returns
/// Compressed data in raw DEFLATE format (no zlib/gzip wrapper).
pub fn deflate(data: &[u8], level: u8) -> Vec<u8> {
    if data.is_empty() {
        // Empty input: just output empty final block
        let mut writer = BitWriter::new();
        writer.write_bits(1, 1); // BFINAL = 1
        writer.write_bits(1, 2); // BTYPE = 01 (fixed Huffman)

        // Write end-of-block symbol (256)
        let lit_codes = huffman::fixed_literal_codes();
        let code = lit_codes[256];
        writer.write_bits(reverse_bits(code.code, code.length), code.length);

        return writer.finish();
    }

    // Use LZ77 to find matches
    let mut lz77 = Lz77Compressor::new(level);
    let tokens = lz77.compress(data);

    // For simplicity, we'll use fixed Huffman codes
    // Dynamic codes would give better compression but are more complex
    encode_fixed_huffman(&tokens)
}

/// Encode tokens using fixed Huffman codes.
fn encode_fixed_huffman(tokens: &[Token]) -> Vec<u8> {
    let lit_codes = huffman::fixed_literal_codes();
    let dist_codes = huffman::fixed_distance_codes();

    let mut writer = BitWriter::new();

    // Block header: BFINAL=1 (last block), BTYPE=01 (fixed Huffman)
    writer.write_bits(1, 1); // BFINAL
    writer.write_bits(1, 2); // BTYPE (01 = fixed Huffman, LSB first)

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
                writer.write_bits(
                    reverse_bits(dist_code.code, dist_code.length),
                    dist_code.length,
                );

                if dist_extra_bits > 0 {
                    writer.write_bits(dist_extra_value as u32, dist_extra_bits);
                }
            }
        }
    }

    // End of block symbol (256)
    let eob_code = lit_codes[256];
    writer.write_bits(reverse_bits(eob_code.code, eob_code.length), eob_code.length);

    writer.finish()
}

/// Reverse bits in a code (DEFLATE uses reversed bit order for Huffman codes).
#[inline]
fn reverse_bits(code: u16, length: u8) -> u32 {
    let mut result = 0u32;
    let mut code = code as u32;
    for _ in 0..length {
        result = (result << 1) | (code & 1);
        code >>= 1;
    }
    result
}

/// Compress data using DEFLATE with stored blocks (no compression).
/// Useful for already-compressed data or when speed is critical.
#[allow(dead_code)]
pub fn deflate_stored(data: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(data.len() + data.len() / 65535 * 5 + 10);
    let chunks = data.chunks(65535);
    let num_chunks = chunks.len();

    for (i, chunk) in data.chunks(65535).enumerate() {
        let is_final = i == num_chunks - 1;
        let len = chunk.len() as u16;
        let nlen = !len;

        // Block header
        output.push(if is_final { 0x01 } else { 0x00 }); // BFINAL + BTYPE=00

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_length_code() {
        assert_eq!(length_code(3), (257, 0, 0));
        assert_eq!(length_code(4), (258, 0, 0));
        assert_eq!(length_code(10), (264, 0, 0));
        assert_eq!(length_code(11), (265, 1, 0));
        assert_eq!(length_code(12), (265, 1, 1));
        assert_eq!(length_code(258), (285, 0, 0));
    }

    #[test]
    fn test_distance_code() {
        assert_eq!(distance_code(1), (0, 0, 0));
        assert_eq!(distance_code(2), (1, 0, 0));
        assert_eq!(distance_code(5), (4, 1, 0));
        assert_eq!(distance_code(6), (4, 1, 1));
    }

    #[test]
    fn test_deflate_empty() {
        let compressed = deflate(&[], 6);
        assert!(!compressed.is_empty());
    }

    #[test]
    fn test_deflate_simple() {
        let data = b"Hello, World!";
        let compressed = deflate(data, 6);

        // Should produce some output
        assert!(!compressed.is_empty());
        // For short data, compression might not reduce size much
    }

    #[test]
    fn test_deflate_repetitive() {
        let data = b"abcabcabcabcabcabcabcabcabcabc";
        let compressed = deflate(data, 6);

        // Repetitive data should compress well
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn test_deflate_stored() {
        let data = b"Hello, World!";
        let compressed = deflate_stored(data);

        // Stored blocks have 5 bytes overhead per 65535 bytes
        assert_eq!(compressed.len(), data.len() + 5);
    }

    #[test]
    fn test_reverse_bits() {
        assert_eq!(reverse_bits(0b101, 3), 0b101);
        assert_eq!(reverse_bits(0b100, 3), 0b001);
        assert_eq!(reverse_bits(0b11110000, 8), 0b00001111);
    }
}
