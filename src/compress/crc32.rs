//! CRC32 checksum implementation (PNG uses CRC-32/ISO-HDLC).

/// CRC32 lookup table using polynomial 0xEDB88320 (reversed 0x04C11DB7).
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

/// Calculate CRC32 checksum of data.
///
/// Uses the CRC-32/ISO-HDLC algorithm (polynomial 0x04C11DB7 reflected).
/// This is the CRC used by PNG, gzip, and many other formats.
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFF_u32;
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC_TABLE[index];
    }
    crc ^ 0xFFFFFFFF
}

/// Calculate CRC32 incrementally.
pub struct Crc32 {
    crc: u32,
}

impl Crc32 {
    /// Create a new CRC32 calculator.
    pub fn new() -> Self {
        Self { crc: 0xFFFFFFFF }
    }

    /// Update the CRC with more data.
    #[inline]
    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            let index = ((self.crc ^ byte as u32) & 0xFF) as usize;
            self.crc = (self.crc >> 8) ^ CRC_TABLE[index];
        }
    }

    /// Finalize and return the CRC value.
    #[inline]
    pub fn finalize(self) -> u32 {
        self.crc ^ 0xFFFFFFFF
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32_empty() {
        assert_eq!(crc32(&[]), 0x00000000);
    }

    #[test]
    fn test_crc32_check_value() {
        // Standard test: CRC32 of "123456789" should be 0xCBF43926
        let data = b"123456789";
        assert_eq!(crc32(data), 0xCBF43926);
    }

    #[test]
    fn test_crc32_incremental() {
        let data = b"123456789";

        // Full calculation
        let full_crc = crc32(data);

        // Incremental calculation
        let mut crc = Crc32::new();
        crc.update(&data[..4]);
        crc.update(&data[4..]);
        let incremental_crc = crc.finalize();

        assert_eq!(full_crc, incremental_crc);
    }

    #[test]
    fn test_crc32_png_iend() {
        // PNG IEND chunk has type "IEND" (no data)
        // CRC should be 0xAE426082
        let chunk_type = b"IEND";
        assert_eq!(crc32(chunk_type), 0xAE426082);
    }
}
