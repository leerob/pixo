//! Adler-32 checksum (RFC 1950) used for zlib wrappers.

/// Calculate Adler-32 checksum of data.
///
/// Optimized to defer modulo operations to chunk boundaries for better performance.
/// Uses NMAX = 5552 which is the largest n such that 255*n*(n+1)/2 + (n+1)*(65520) <= 2^32-1.
///
/// When the `simd` feature is enabled, uses SIMD acceleration for improved throughput.
#[inline]
#[must_use]
pub fn adler32(data: &[u8]) -> u32 {
    #[cfg(feature = "simd")]
    {
        crate::simd::adler32(data)
    }

    #[cfg(not(feature = "simd"))]
    {
        adler32_scalar(data)
    }
}

/// Scalar implementation of Adler-32.
#[inline]
#[cfg_attr(feature = "simd", allow(dead_code))]
fn adler32_scalar(data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65_521;
    // NMAX is the largest n such that we can accumulate n bytes without overflow
    // 255*n*(n+1)/2 + (n+1)*(65520) <= 2^32-1
    const NMAX: usize = 5552;

    let mut s1: u32 = 1;
    let mut s2: u32 = 0;

    // Process in chunks, only applying modulo at chunk boundaries
    for chunk in data.chunks(NMAX) {
        for &b in chunk {
            s1 += b as u32;
            s2 += s1;
        }
        // Apply modulo only once per chunk instead of per byte
        s1 %= MOD_ADLER;
        s2 %= MOD_ADLER;
    }

    (s2 << 16) | s1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adler32_empty() {
        assert_eq!(adler32(&[]), 1);
    }

    #[test]
    fn test_adler32_known_values() {
        assert_eq!(adler32(b"hello"), 0x062C0215);
        assert_eq!(adler32(b"Adler-32"), 0x0C34027B);
        assert_eq!(adler32(b"123456789"), 0x091E01DE);
    }

    #[test]
    fn test_adler32_scalar_directly() {
        assert_eq!(adler32_scalar(&[]), 1);
        assert_eq!(adler32_scalar(b"hello"), 0x062C0215);
        assert_eq!(adler32_scalar(b"123456789"), 0x091E01DE);
    }

    #[test]
    fn test_adler32_scalar_large_input() {
        let large_data = vec![0xAB; 10000];
        let result = adler32_scalar(&large_data);
        assert_ne!(result, 0);
    }

    #[test]
    fn test_adler32_scalar_exactly_nmax() {
        let data = vec![0xFF; 5552];
        let result = adler32_scalar(&data);
        assert_ne!(result, 0);
    }

    #[test]
    fn test_adler32_scalar_multiple_chunks() {
        let data = vec![0x55; 5552 * 3];
        let result = adler32_scalar(&data);
        assert_ne!(result, 0);
    }
}
