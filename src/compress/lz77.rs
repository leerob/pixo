//! LZ77 compression algorithm with sliding window.
//!
//! LZ77 finds repeated sequences in the input and replaces them with
//! (length, distance) pairs referring back to previous occurrences.

/// Maximum distance to look back for matches (32KB window).
pub const MAX_DISTANCE: usize = 32768;

/// Maximum match length (as per DEFLATE spec).
pub const MAX_MATCH_LENGTH: usize = 258;

/// Minimum match length worth encoding.
pub const MIN_MATCH_LENGTH: usize = 3;

/// Size of the hash table (power of 2 for fast modulo).
const HASH_SIZE: usize = 1 << 15; // 32768 entries

/// LZ77 token representing either a literal or a match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token {
    /// A literal byte that couldn't be compressed.
    Literal(u8),
    /// A back-reference: (length, distance).
    Match {
        /// Length of the match (3-258).
        length: u16,
        /// Distance back to the match (1-32768).
        distance: u16,
    },
}

/// Hash function for 3-byte sequences.
#[inline]
fn hash3(data: &[u8], pos: usize) -> usize {
    if pos + 2 >= data.len() {
        return 0;
    }
    let h = (data[pos] as u32)
        | ((data[pos + 1] as u32) << 8)
        | ((data[pos + 2] as u32) << 16);
    // Multiply by a prime and take high bits
    ((h.wrapping_mul(2654435769)) >> 17) as usize & (HASH_SIZE - 1)
}

/// LZ77 compressor with hash chain for fast matching.
pub struct Lz77Compressor {
    /// Hash table: maps hash -> most recent position
    head: Vec<i32>,
    /// Chain links: prev[pos % window] -> previous position with same hash
    prev: Vec<i32>,
    /// Compression level (affects search depth)
    max_chain_length: usize,
    /// Lazy matching: check if next position has better match
    lazy_matching: bool,
}

impl Lz77Compressor {
    /// Create a new LZ77 compressor.
    ///
    /// # Arguments
    /// * `level` - Compression level 1-9 (higher = better compression, slower)
    pub fn new(level: u8) -> Self {
        let level = level.clamp(1, 9);

        // Tune chain length and lazy matching based on level
        let (max_chain_length, lazy_matching) = match level {
            1 => (4, false),
            2 => (8, false),
            3 => (16, false),
            4 => (32, true),
            5 => (64, true),
            6 => (128, true),
            7 => (256, true),
            8 => (512, true),
            9 => (1024, true),
            _ => (128, true),
        };

        Self {
            head: vec![-1; HASH_SIZE],
            prev: vec![-1; MAX_DISTANCE],
            max_chain_length,
            lazy_matching,
        }
    }

    /// Compress data and return LZ77 tokens.
    pub fn compress(&mut self, data: &[u8]) -> Vec<Token> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut tokens = Vec::with_capacity(data.len());
        let mut pos = 0;

        // Reset hash tables
        self.head.fill(-1);
        self.prev.fill(-1);

        while pos < data.len() {
            let best_match = self.find_best_match(data, pos);

            if let Some((length, distance)) = best_match {
                // Check for lazy match if enabled
                if self.lazy_matching && pos + 1 < data.len() {
                    // Update hash for current position
                    self.update_hash(data, pos);

                    if let Some((next_length, _)) = self.find_best_match(data, pos + 1) {
                        if next_length > length + 1 {
                            // Better match at next position, emit literal
                            tokens.push(Token::Literal(data[pos]));
                            pos += 1;
                            continue;
                        }
                    }
                }

                tokens.push(Token::Match {
                    length: length as u16,
                    distance: distance as u16,
                });

                // Update hash for all positions in the match
                for i in 0..length {
                    self.update_hash(data, pos + i);
                }
                pos += length;
            } else {
                tokens.push(Token::Literal(data[pos]));
                self.update_hash(data, pos);
                pos += 1;
            }
        }

        tokens
    }

    /// Find the best match at the given position.
    fn find_best_match(&self, data: &[u8], pos: usize) -> Option<(usize, usize)> {
        if pos + MIN_MATCH_LENGTH > data.len() {
            return None;
        }

        let hash = hash3(data, pos);
        let mut chain_pos = self.head[hash];
        let mut best_length = MIN_MATCH_LENGTH - 1;
        let mut best_distance = 0;

        let max_distance = pos.min(MAX_DISTANCE);
        let mut chain_count = 0;

        while chain_pos >= 0 && chain_count < self.max_chain_length {
            let match_pos = chain_pos as usize;
            let distance = pos - match_pos;

            if distance > max_distance {
                break;
            }

            // Compare strings
            let length = self.match_length(data, match_pos, pos);

            if length > best_length {
                best_length = length;
                best_distance = distance;

                // Early exit if we found max length
                if length >= MAX_MATCH_LENGTH {
                    break;
                }
            }

            // Follow chain
            chain_pos = self.prev[match_pos % MAX_DISTANCE];
            chain_count += 1;
        }

        if best_length >= MIN_MATCH_LENGTH {
            Some((best_length, best_distance))
        } else {
            None
        }
    }

    /// Calculate match length between two positions.
    #[inline]
    fn match_length(&self, data: &[u8], pos1: usize, pos2: usize) -> usize {
        let max_len = (data.len() - pos2).min(MAX_MATCH_LENGTH);
        let mut length = 0;

        while length < max_len && data[pos1 + length] == data[pos2 + length] {
            length += 1;
        }

        length
    }

    /// Update hash table for a position.
    #[inline]
    fn update_hash(&mut self, data: &[u8], pos: usize) {
        if pos + 2 >= data.len() {
            return;
        }

        let hash = hash3(data, pos);
        self.prev[pos % MAX_DISTANCE] = self.head[hash];
        self.head[hash] = pos as i32;
    }
}

impl Default for Lz77Compressor {
    fn default() -> Self {
        Self::new(6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz77_no_matches() {
        let mut compressor = Lz77Compressor::new(6);
        let data = b"abcdefgh";
        let tokens = compressor.compress(data);

        // All literals
        assert_eq!(tokens.len(), 8);
        for (i, &token) in tokens.iter().enumerate() {
            assert_eq!(token, Token::Literal(data[i]));
        }
    }

    #[test]
    fn test_lz77_simple_repeat() {
        let mut compressor = Lz77Compressor::new(6);
        let data = b"abcabcabc";
        let tokens = compressor.compress(data);

        // Should have "abc" as literals, then matches
        assert!(tokens.len() < 9); // Less than all literals
    }

    #[test]
    fn test_lz77_long_repeat() {
        let mut compressor = Lz77Compressor::new(6);
        let data = b"abcdefghijabcdefghijabcdefghij";
        let tokens = compressor.compress(data);

        // Should compress well
        assert!(tokens.len() < 20);
    }

    #[test]
    fn test_lz77_empty() {
        let mut compressor = Lz77Compressor::new(6);
        let tokens = compressor.compress(&[]);
        assert!(tokens.is_empty());
    }
}
