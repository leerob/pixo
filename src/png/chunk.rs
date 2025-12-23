//! PNG chunk handling.

/// Write a PNG chunk (length, type, data, CRC32) to the output buffer.
pub fn write_chunk(output: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    output.reserve(12 + data.len());

    let mut crc = crate::compress::crc32::Crc32::new();
    crc.update(chunk_type);
    crc.update(data);
    let crc = crc.finalize();

    output.extend_from_slice(&(data.len() as u32).to_be_bytes());
    output.extend_from_slice(chunk_type);
    output.extend_from_slice(data);
    output.extend_from_slice(&crc.to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_iend_chunk() {
        let mut output = Vec::new();
        write_chunk(&mut output, b"IEND", &[]);

        assert_eq!(output.len(), 12);
        assert_eq!(&output[0..4], &[0, 0, 0, 0]);
        assert_eq!(&output[4..8], b"IEND");
        assert_eq!(&output[8..12], &0xAE426082_u32.to_be_bytes());
    }

    #[test]
    fn test_write_chunk_with_data() {
        let mut output = Vec::new();
        write_chunk(&mut output, b"tEXt", b"hello");

        assert_eq!(output.len(), 17);
        assert_eq!(&output[0..4], &[0, 0, 0, 5]);
        assert_eq!(&output[4..8], b"tEXt");
        assert_eq!(&output[8..13], b"hello");
    }
}
