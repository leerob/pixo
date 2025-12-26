//! Bit-level I/O utilities for binary encoding.

/// A bit writer that packs bits into bytes, LSB first (for DEFLATE).
#[derive(Debug)]
pub struct BitWriter {
    buffer: Vec<u8>,
    current_byte: u8,
    bit_position: u8,
}

impl BitWriter {
    /// Create a new bit writer with default capacity.
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    /// Create a new bit writer with specified byte capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            current_byte: 0,
            bit_position: 0,
        }
    }

    /// Write bits to the stream, LSB first.
    ///
    /// # Arguments
    /// * `value` - The value to write (only lower `num_bits` are used)
    /// * `num_bits` - Number of bits to write (1-32)
    #[inline]
    pub fn write_bits(&mut self, value: u32, num_bits: u8) {
        debug_assert!(num_bits <= 32);

        let mut value = value;
        let mut remaining = num_bits;

        while remaining > 0 {
            let available = 8 - self.bit_position;
            let to_write = remaining.min(available);

            // Extract the bits we want to write
            let mask = (1u32 << to_write) - 1;
            let bits = (value & mask) as u8;

            // Add to current byte at the correct position
            self.current_byte |= bits << self.bit_position;

            self.bit_position += to_write;
            value >>= to_write;
            remaining -= to_write;

            // If byte is full, flush it
            if self.bit_position == 8 {
                self.buffer.push(self.current_byte);
                self.current_byte = 0;
                self.bit_position = 0;
            }
        }
    }

    #[inline]
    pub fn write_bit(&mut self, bit: bool) {
        self.write_bits(bit as u32, 1);
    }

    /// Flushes partial byte first with zeros if not byte-aligned.
    pub fn write_byte(&mut self, byte: u8) {
        if self.bit_position == 0 {
            self.buffer.push(byte);
        } else {
            self.write_bits(byte as u32, 8);
        }
    }

    pub fn write_bytes(&mut self, bytes: &[u8]) {
        if self.bit_position == 0 {
            self.buffer.extend_from_slice(bytes);
        } else {
            for &byte in bytes {
                self.write_bits(byte as u32, 8);
            }
        }
    }

    /// Flush any remaining bits, padding with zeros.
    pub fn flush(&mut self) {
        if self.bit_position > 0 {
            self.buffer.push(self.current_byte);
            self.current_byte = 0;
            self.bit_position = 0;
        }
    }

    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        self.flush();
        self.buffer
    }

    /// Returns length in bytes (not counting partial byte).
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty() && self.bit_position == 0
    }

    pub fn bit_position(&self) -> u8 {
        self.bit_position
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// A fast bit writer using a 64-bit accumulator (LSB-first) for DEFLATE.
#[derive(Debug)]
pub struct BitWriter64 {
    buffer: Vec<u8>,
    acc: u64,
    bits_in_acc: u8,
}

impl Default for BitWriter64 {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter64 {
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            acc: 0,
            bits_in_acc: 0,
        }
    }

    #[inline]
    pub fn write_bits(&mut self, value: u32, num_bits: u8) {
        debug_assert!(num_bits <= 32);
        debug_assert!(self.bits_in_acc <= 63);
        let val64 = (value & ((1u32 << num_bits) - 1)) as u64;
        self.acc |= val64 << self.bits_in_acc;
        self.bits_in_acc += num_bits;

        while self.bits_in_acc >= 8 {
            self.buffer.push(self.acc as u8);
            self.acc >>= 8;
            self.bits_in_acc -= 8;
        }
    }

    #[inline]
    pub fn write_bit(&mut self, bit: bool) {
        self.write_bits(bit as u32, 1);
    }

    /// Pads remaining bits with zeros.
    pub fn flush(&mut self) {
        if self.bits_in_acc > 0 {
            self.buffer.push(self.acc as u8);
            self.acc = 0;
            self.bits_in_acc = 0;
        }
    }

    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        self.flush();
        self.buffer
    }

    /// Returns length in bytes (not counting partial byte).
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty() && self.bits_in_acc == 0
    }
}

/// A bit writer that packs bits MSB first (for JPEG).
#[derive(Debug)]
pub struct BitWriterMsb {
    buffer: Vec<u8>,
    current_byte: u8,
    bit_position: u8, // Counts from 8 down to 0
}

impl BitWriterMsb {
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            current_byte: 0,
            bit_position: 8,
        }
    }

    /// Processes multiple bits at once instead of bit-by-bit.
    #[inline]
    pub fn write_bits(&mut self, value: u32, num_bits: u8) {
        debug_assert!(num_bits <= 32);

        let mut remaining = num_bits;
        let val = value;

        while remaining > 0 {
            let space = self.bit_position;
            let to_write = remaining.min(space);

            // Extract the top `to_write` bits from the remaining value
            // and place them in the current byte at the correct position
            let shift = remaining - to_write;
            let mask = (1u32 << to_write) - 1;
            let bits = ((val >> shift) & mask) as u8;

            self.bit_position -= to_write;
            self.current_byte |= bits << self.bit_position;
            remaining -= to_write;

            // If byte is full, flush it with JPEG byte stuffing
            if self.bit_position == 0 {
                self.flush_byte_with_stuffing();
            }
        }
    }

    /// Flush the current byte and apply JPEG byte stuffing if needed.
    #[inline]
    fn flush_byte_with_stuffing(&mut self) {
        self.buffer.push(self.current_byte);
        // JPEG byte stuffing: if we wrote 0xFF, add 0x00
        if self.current_byte == 0xFF {
            self.buffer.push(0x00);
        }
        self.current_byte = 0;
        self.bit_position = 8;
    }

    #[inline]
    pub fn write_bit(&mut self, bit: bool) {
        self.write_bits(bit as u32, 1);
    }

    /// Pads remaining bits with 1s (as per JPEG spec).
    pub fn flush(&mut self) {
        if self.bit_position < 8 {
            // Pad with 1s
            self.current_byte |= (1 << self.bit_position) - 1;
            self.buffer.push(self.current_byte);
            if self.current_byte == 0xFF {
                self.buffer.push(0x00);
            }
            self.current_byte = 0;
            self.bit_position = 8;
        }
    }

    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        self.flush();
        self.buffer
    }

    /// Must be byte-aligned.
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        debug_assert_eq!(self.bit_position, 8, "Must be byte-aligned");
        self.buffer.extend_from_slice(bytes);
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty() && self.bit_position == 8
    }
}

impl Default for BitWriterMsb {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_writer_single_bits() {
        let mut writer = BitWriter::new();
        // Write 8 bits: 10110100 LSB first
        writer.write_bit(false); // bit 0
        writer.write_bit(false); // bit 1
        writer.write_bit(true); // bit 2
        writer.write_bit(false); // bit 3
        writer.write_bit(true); // bit 4
        writer.write_bit(true); // bit 5
        writer.write_bit(false); // bit 6
        writer.write_bit(true); // bit 7

        let result = writer.finish();
        assert_eq!(result, vec![0b10110100]);
    }

    #[test]
    fn test_bit_writer_multi_bits() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b101, 3);
        writer.write_bits(0b11, 2);
        writer.write_bits(0b001, 3);

        let result = writer.finish();
        // LSB first: 101 + 11 + 001 = 00111101
        assert_eq!(result, vec![0b00111101]);
    }

    #[test]
    fn test_bit_writer_cross_byte() {
        let mut writer = BitWriter::new();
        writer.write_bits(0xFF, 8);
        writer.write_bits(0x0F, 4);

        let result = writer.finish();
        assert_eq!(result, vec![0xFF, 0x0F]);
    }

    #[test]
    fn test_bit_writer_msb() {
        let mut writer = BitWriterMsb::new();
        writer.write_bits(0b10110100, 8);

        let result = writer.finish();
        assert_eq!(result, vec![0b10110100]);
    }

    #[test]
    fn test_bit_writer_msb_partial() {
        let mut writer = BitWriterMsb::new();
        writer.write_bits(0b101, 3);

        let result = writer.finish();
        // MSB first, padded with 1s: 101_11111 = 0xBF
        assert_eq!(result, vec![0b10111111]);
    }

    #[test]
    fn test_bit_writer_write_byte_aligned() {
        let mut writer = BitWriter::new();
        writer.write_byte(0xAB);
        let result = writer.finish();
        assert_eq!(result, vec![0xAB]);
    }

    #[test]
    fn test_bit_writer_write_byte_unaligned() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b1, 1); // one bit set
        writer.write_byte(0xFF);
        let result = writer.finish();
        // First bit 1, then 0xFF = 11111111, becomes 1_1111111|1 = 0xFF, 0x01
        assert_eq!(result, vec![0xFF, 0x01]);
    }

    #[test]
    fn test_bit_writer_write_bytes_aligned() {
        let mut writer = BitWriter::new();
        writer.write_bytes(&[0xAB, 0xCD]);
        let result = writer.finish();
        assert_eq!(result, vec![0xAB, 0xCD]);
    }

    #[test]
    fn test_bit_writer_write_bytes_unaligned() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b1, 1);
        writer.write_bytes(&[0xFF, 0xFF]);
        let result = writer.finish();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_bit_writer_len_and_is_empty() {
        let mut writer = BitWriter::new();
        assert!(writer.is_empty());
        assert_eq!(writer.len(), 0);

        writer.write_bits(0xFF, 8);
        assert!(!writer.is_empty());
        assert_eq!(writer.len(), 1);
    }

    #[test]
    fn test_bit_writer_bit_position() {
        let mut writer = BitWriter::new();
        assert_eq!(writer.bit_position(), 0);

        writer.write_bits(0b101, 3);
        assert_eq!(writer.bit_position(), 3);

        writer.write_bits(0b11111, 5);
        assert_eq!(writer.bit_position(), 0); // full byte written
    }

    #[test]
    fn test_bit_writer_default() {
        let writer: BitWriter = Default::default();
        assert!(writer.is_empty());
    }

    #[test]
    fn test_bit_writer64_single_bits() {
        let mut writer = BitWriter64::new();
        writer.write_bit(true);
        writer.write_bit(false);
        writer.write_bit(true);
        writer.write_bit(false);
        writer.write_bit(true);
        writer.write_bit(true);
        writer.write_bit(false);
        writer.write_bit(true);

        let result = writer.finish();
        assert_eq!(result, vec![0b10110101]);
    }

    #[test]
    fn test_bit_writer64_len_is_empty() {
        let mut writer = BitWriter64::new();
        assert!(writer.is_empty());
        assert_eq!(writer.len(), 0);

        writer.write_bits(0xFF, 8);
        assert!(!writer.is_empty());
        assert_eq!(writer.len(), 1);
    }

    #[test]
    fn test_bit_writer64_default() {
        let writer: BitWriter64 = Default::default();
        assert!(writer.is_empty());
    }

    #[test]
    fn test_bit_writer_msb_single_bits() {
        let mut writer = BitWriterMsb::new();
        writer.write_bit(true);
        writer.write_bit(false);
        writer.write_bit(true);
        writer.write_bit(false);
        writer.write_bit(true);
        writer.write_bit(false);
        writer.write_bit(true);
        writer.write_bit(false);

        let result = writer.finish();
        assert_eq!(result, vec![0b10101010]);
    }

    #[test]
    fn test_bit_writer_msb_write_bytes() {
        let mut writer = BitWriterMsb::new();
        writer.write_bytes(&[0xAB, 0xCD]);
        let result = writer.finish();
        assert_eq!(result, vec![0xAB, 0xCD]);
    }

    #[test]
    fn test_bit_writer_msb_len_is_empty() {
        let mut writer = BitWriterMsb::new();
        assert!(writer.is_empty());
        assert_eq!(writer.len(), 0);

        // Write a non-0xFF byte to avoid byte stuffing
        writer.write_bits(0x42, 8);
        assert!(!writer.is_empty());
        assert_eq!(writer.len(), 1);
    }

    #[test]
    fn test_bit_writer_msb_default() {
        let writer: BitWriterMsb = Default::default();
        assert!(writer.is_empty());
    }

    #[test]
    fn test_bit_writer_msb_byte_stuffing_0xff() {
        let mut writer = BitWriterMsb::new();
        writer.write_bits(0xFF, 8);
        let result = writer.finish();
        // 0xFF should be followed by 0x00 for JPEG byte stuffing
        assert_eq!(result, vec![0xFF, 0x00]);
    }

    #[test]
    fn test_bit_writer_msb_flush_with_0xff_pad() {
        let mut writer = BitWriterMsb::new();
        // Write 7 ones, leaving one bit to pad
        writer.write_bits(0b1111111, 7);
        let result = writer.finish();
        // Padding with 1s gives 0xFF, which triggers byte stuffing
        assert_eq!(result, vec![0xFF, 0x00]);
    }
}
