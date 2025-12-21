//! Color type definitions and conversions.

/// Supported color types for image encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorType {
    /// Grayscale, 1 byte per pixel.
    Gray,
    /// Grayscale with alpha, 2 bytes per pixel.
    GrayAlpha,
    /// RGB, 3 bytes per pixel.
    Rgb,
    /// RGBA, 4 bytes per pixel.
    Rgba,
}

impl ColorType {
    /// Returns the number of bytes per pixel for this color type.
    #[inline]
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            ColorType::Gray => 1,
            ColorType::GrayAlpha => 2,
            ColorType::Rgb => 3,
            ColorType::Rgba => 4,
        }
    }

    /// Returns the PNG color type value.
    #[inline]
    pub(crate) const fn png_color_type(self) -> u8 {
        match self {
            ColorType::Gray => 0,
            ColorType::GrayAlpha => 4,
            ColorType::Rgb => 2,
            ColorType::Rgba => 6,
        }
    }

    /// Returns the bit depth for PNG encoding.
    #[inline]
    pub(crate) const fn png_bit_depth(self) -> u8 {
        8 // We only support 8-bit depth
    }
}

/// Convert RGB to YCbCr color space (used by JPEG).
///
/// Returns (Y, Cb, Cr) where each component is in range 0-255.
#[inline]
pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = r as f32;
    let g = g as f32;
    let b = b as f32;

    // ITU-R BT.601 conversion
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
    let cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;

    (
        y.round().clamp(0.0, 255.0) as u8,
        cb.round().clamp(0.0, 255.0) as u8,
        cr.round().clamp(0.0, 255.0) as u8,
    )
}

/// Convert RGBA to YCbCr, ignoring alpha channel.
#[inline]
pub fn rgba_to_ycbcr(r: u8, g: u8, b: u8, _a: u8) -> (u8, u8, u8) {
    rgb_to_ycbcr(r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_per_pixel() {
        assert_eq!(ColorType::Gray.bytes_per_pixel(), 1);
        assert_eq!(ColorType::GrayAlpha.bytes_per_pixel(), 2);
        assert_eq!(ColorType::Rgb.bytes_per_pixel(), 3);
        assert_eq!(ColorType::Rgba.bytes_per_pixel(), 4);
    }

    #[test]
    fn test_rgb_to_ycbcr_black() {
        let (y, cb, cr) = rgb_to_ycbcr(0, 0, 0);
        assert_eq!(y, 0);
        assert_eq!(cb, 128);
        assert_eq!(cr, 128);
    }

    #[test]
    fn test_rgb_to_ycbcr_white() {
        let (y, cb, cr) = rgb_to_ycbcr(255, 255, 255);
        assert_eq!(y, 255);
        assert_eq!(cb, 128);
        assert_eq!(cr, 128);
    }

    #[test]
    fn test_rgb_to_ycbcr_red() {
        let (y, cb, cr) = rgb_to_ycbcr(255, 0, 0);
        // Red should have high Y, low Cb, high Cr
        assert!(y > 50 && y < 100);
        assert!(cb < 128);
        assert!(cr > 200);
    }
}
