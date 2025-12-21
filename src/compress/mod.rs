//! Compression algorithms.

pub mod crc32;
pub mod deflate;
pub mod huffman;
pub mod lz77;

pub use crc32::crc32;
pub use deflate::deflate;
