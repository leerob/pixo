//! PNG filtering implementation.
//!
//! PNG uses filtering to improve compression by exploiting correlations
//! between adjacent pixels.

use super::{FilterStrategy, PngOptions};

/// Filter type bytes as defined by PNG specification.
const FILTER_NONE: u8 = 0;
const FILTER_SUB: u8 = 1;
const FILTER_UP: u8 = 2;
const FILTER_AVERAGE: u8 = 3;
const FILTER_PAETH: u8 = 4;

/// Apply PNG filtering to raw image data.
///
/// Returns filtered data with a filter type byte prepended to each row.
pub fn apply_filters(
    data: &[u8],
    width: u32,
    height: u32,
    bytes_per_pixel: usize,
    options: &PngOptions,
) -> Vec<u8> {
    let row_bytes = width as usize * bytes_per_pixel;
    let filtered_row_size = row_bytes + 1; // +1 for filter type byte

    let mut output = Vec::with_capacity(filtered_row_size * height as usize);

    // Previous row (starts as zeros)
    let mut prev_row = vec![0u8; row_bytes];

    for y in 0..height as usize {
        let row_start = y * row_bytes;
        let row = &data[row_start..row_start + row_bytes];

        match options.filter_strategy {
            FilterStrategy::None => {
                output.push(FILTER_NONE);
                output.extend_from_slice(row);
            }
            FilterStrategy::Sub => {
                output.push(FILTER_SUB);
                filter_sub(row, bytes_per_pixel, &mut output);
            }
            FilterStrategy::Up => {
                output.push(FILTER_UP);
                filter_up(row, &prev_row, &mut output);
            }
            FilterStrategy::Average => {
                output.push(FILTER_AVERAGE);
                filter_average(row, &prev_row, bytes_per_pixel, &mut output);
            }
            FilterStrategy::Paeth => {
                output.push(FILTER_PAETH);
                filter_paeth(row, &prev_row, bytes_per_pixel, &mut output);
            }
            FilterStrategy::Adaptive => {
                // Try all filters and pick the best one
                adaptive_filter(row, &prev_row, bytes_per_pixel, &mut output);
            }
        }

        // Update previous row
        prev_row.copy_from_slice(row);
    }

    output
}

/// Sub filter: difference from left pixel.
fn filter_sub(row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    for (i, &byte) in row.iter().enumerate() {
        let left = if i >= bpp { row[i - bpp] } else { 0 };
        output.push(byte.wrapping_sub(left));
    }
}

/// Up filter: difference from above pixel.
fn filter_up(row: &[u8], prev_row: &[u8], output: &mut Vec<u8>) {
    for (i, &byte) in row.iter().enumerate() {
        output.push(byte.wrapping_sub(prev_row[i]));
    }
}

/// Average filter: difference from average of left and above.
fn filter_average(row: &[u8], prev_row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    for (i, &byte) in row.iter().enumerate() {
        let left = if i >= bpp { row[i - bpp] as u16 } else { 0 };
        let above = prev_row[i] as u16;
        let avg = ((left + above) / 2) as u8;
        output.push(byte.wrapping_sub(avg));
    }
}

/// Paeth filter: difference from Paeth predictor.
fn filter_paeth(row: &[u8], prev_row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    for (i, &byte) in row.iter().enumerate() {
        let left = if i >= bpp { row[i - bpp] } else { 0 };
        let above = prev_row[i];
        let upper_left = if i >= bpp { prev_row[i - bpp] } else { 0 };
        let predicted = paeth_predictor(left, above, upper_left);
        output.push(byte.wrapping_sub(predicted));
    }
}

/// Paeth predictor function.
///
/// Selects the value (a, b, or c) closest to p = a + b - c.
#[inline]
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

/// Adaptive filter selection: try all filters and pick the best.
fn adaptive_filter(row: &[u8], prev_row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    let row_len = row.len();

    // Buffers for each filter type
    let mut none_buf = Vec::with_capacity(row_len);
    let mut sub_buf = Vec::with_capacity(row_len);
    let mut up_buf = Vec::with_capacity(row_len);
    let mut avg_buf = Vec::with_capacity(row_len);
    let mut paeth_buf = Vec::with_capacity(row_len);

    // Apply each filter
    none_buf.extend_from_slice(row);
    filter_sub(row, bpp, &mut sub_buf);
    filter_up(row, prev_row, &mut up_buf);
    filter_average(row, prev_row, bpp, &mut avg_buf);
    filter_paeth(row, prev_row, bpp, &mut paeth_buf);

    // Score each filter (sum of absolute differences - lower is better for compression)
    let scores = [
        (FILTER_NONE, score_filter(&none_buf)),
        (FILTER_SUB, score_filter(&sub_buf)),
        (FILTER_UP, score_filter(&up_buf)),
        (FILTER_AVERAGE, score_filter(&avg_buf)),
        (FILTER_PAETH, score_filter(&paeth_buf)),
    ];

    // Find the filter with the lowest score
    let (best_filter, _) = scores.iter().min_by_key(|(_, score)| *score).unwrap();

    // Output the best filter result
    output.push(*best_filter);
    match *best_filter {
        FILTER_NONE => output.extend_from_slice(&none_buf),
        FILTER_SUB => output.extend_from_slice(&sub_buf),
        FILTER_UP => output.extend_from_slice(&up_buf),
        FILTER_AVERAGE => output.extend_from_slice(&avg_buf),
        FILTER_PAETH => output.extend_from_slice(&paeth_buf),
        _ => unreachable!(),
    }
}

/// Score a filtered row using sum of absolute values.
///
/// Lower scores typically result in better compression.
#[inline]
fn score_filter(filtered: &[u8]) -> u64 {
    filtered.iter().map(|&b| (b as i8).unsigned_abs() as u64).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paeth_predictor() {
        // When all are equal, should return that value
        assert_eq!(paeth_predictor(100, 100, 100), 100);

        // When a=0, b=0, c=0, should return 0
        assert_eq!(paeth_predictor(0, 0, 0), 0);

        // Test typical case: a=10, b=20, c=15
        // p = a + b - c = 10 + 20 - 15 = 15
        // pa = |p - a| = |15 - 10| = 5
        // pb = |p - b| = |15 - 20| = 5
        // pc = |p - c| = |15 - 15| = 0
        // pc is smallest, so return c (15)
        assert_eq!(paeth_predictor(10, 20, 15), 15);
    }

    #[test]
    fn test_filter_sub() {
        let row = vec![10, 20, 30, 40, 50, 60];
        let mut output = Vec::new();
        filter_sub(&row, 3, &mut output);

        // First 3 bytes: no left pixel, so unchanged
        assert_eq!(output[0], 10);
        assert_eq!(output[1], 20);
        assert_eq!(output[2], 30);

        // Next 3 bytes: difference from 3 bytes back
        assert_eq!(output[3], 40u8.wrapping_sub(10)); // 30
        assert_eq!(output[4], 50u8.wrapping_sub(20)); // 30
        assert_eq!(output[5], 60u8.wrapping_sub(30)); // 30
    }

    #[test]
    fn test_filter_up() {
        let row = vec![50, 60, 70];
        let prev = vec![10, 20, 30];
        let mut output = Vec::new();
        filter_up(&row, &prev, &mut output);

        assert_eq!(output[0], 40); // 50 - 10
        assert_eq!(output[1], 40); // 60 - 20
        assert_eq!(output[2], 40); // 70 - 30
    }

    #[test]
    fn test_apply_filters_none() {
        let data = vec![100, 150, 200, 50, 100, 150];
        let options = PngOptions {
            filter_strategy: FilterStrategy::None,
            ..Default::default()
        };

        let filtered = apply_filters(&data, 2, 1, 3, &options);

        // Should be filter byte (0) + original data
        assert_eq!(filtered[0], FILTER_NONE);
        assert_eq!(&filtered[1..], &data[..]);
    }

    #[test]
    fn test_apply_filters_multiple_rows() {
        let data = vec![
            10, 20, 30, 40, 50, 60, // Row 1
            70, 80, 90, 100, 110, 120, // Row 2
        ];
        let options = PngOptions {
            filter_strategy: FilterStrategy::None,
            ..Default::default()
        };

        let filtered = apply_filters(&data, 2, 2, 3, &options);

        // Should have 2 rows, each with filter byte
        assert_eq!(filtered.len(), 2 * (1 + 6)); // 2 rows * (1 filter + 6 data)
        assert_eq!(filtered[0], FILTER_NONE);
        assert_eq!(filtered[7], FILTER_NONE);
    }
}
