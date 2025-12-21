//! Discrete Cosine Transform (DCT) implementation for JPEG.
//!
//! Uses the separable 2D DCT approach for efficiency.

use std::f32::consts::PI;

/// Precomputed cosine values for DCT.
/// cos_table[i][j] = cos((2*i + 1) * j * PI / 16)
const COS_TABLE: [[f32; 8]; 8] = precompute_cos_table();

/// Precompute the cosine table at compile time.
const fn precompute_cos_table() -> [[f32; 8]; 8] {
    let mut table = [[0.0f32; 8]; 8];
    let mut i = 0;
    while i < 8 {
        let mut j = 0;
        while j < 8 {
            // cos((2*i + 1) * j * PI / 16)
            // We need to compute this at compile time
            let angle = ((2 * i + 1) * j) as f32 * PI / 16.0;
            table[i][j] = cos_approx(angle);
            j += 1;
        }
        i += 1;
    }
    table
}

/// Approximate cosine for const fn (Taylor series).
const fn cos_approx(x: f32) -> f32 {
    // Normalize x to [-PI, PI]
    let mut x = x;
    while x > PI {
        x -= 2.0 * PI;
    }
    while x < -PI {
        x += 2.0 * PI;
    }

    // Taylor series for cos(x)
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let x8 = x4 * x4;

    1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0 + x8 / 40320.0
}

/// Normalization factors for DCT.
/// alpha(0) = 1/sqrt(2), alpha(k) = 1 for k > 0
const ALPHA: [f32; 8] = [
    0.7071067811865476, // 1/sqrt(2)
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
];

/// Perform 2D DCT on an 8x8 block.
///
/// Uses the separable property: 2D DCT = 1D DCT on rows, then 1D DCT on columns.
pub fn dct_2d(block: &[f32; 64]) -> [f32; 64] {
    let mut temp = [0.0f32; 64];
    let mut result = [0.0f32; 64];

    // 1D DCT on rows
    for row in 0..8 {
        let row_start = row * 8;
        dct_1d(
            &block[row_start..row_start + 8],
            &mut temp[row_start..row_start + 8],
        );
    }

    // 1D DCT on columns
    for col in 0..8 {
        let mut col_in = [0.0f32; 8];
        let mut col_out = [0.0f32; 8];

        for row in 0..8 {
            col_in[row] = temp[row * 8 + col];
        }

        dct_1d(&col_in, &mut col_out);

        for row in 0..8 {
            result[row * 8 + col] = col_out[row];
        }
    }

    result
}

/// Perform 1D DCT on 8 values.
fn dct_1d(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), 8);
    debug_assert_eq!(output.len(), 8);

    for k in 0..8 {
        let mut sum = 0.0f32;
        for n in 0..8 {
            sum += input[n] * COS_TABLE[n][k];
        }
        output[k] = 0.5 * ALPHA[k] * sum;
    }
}

/// Perform inverse 2D DCT on an 8x8 block.
#[allow(dead_code)]
pub fn idct_2d(block: &[f32; 64]) -> [f32; 64] {
    let mut temp = [0.0f32; 64];
    let mut result = [0.0f32; 64];

    // 1D IDCT on columns
    for col in 0..8 {
        let mut col_in = [0.0f32; 8];
        let mut col_out = [0.0f32; 8];

        for row in 0..8 {
            col_in[row] = block[row * 8 + col];
        }

        idct_1d(&col_in, &mut col_out);

        for row in 0..8 {
            temp[row * 8 + col] = col_out[row];
        }
    }

    // 1D IDCT on rows
    for row in 0..8 {
        let row_start = row * 8;
        idct_1d(
            &temp[row_start..row_start + 8],
            &mut result[row_start..row_start + 8],
        );
    }

    result
}

/// Perform 1D inverse DCT on 8 values.
fn idct_1d(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), 8);
    debug_assert_eq!(output.len(), 8);

    for n in 0..8 {
        let mut sum = 0.0f32;
        for k in 0..8 {
            sum += ALPHA[k] * input[k] * COS_TABLE[n][k];
        }
        output[n] = 0.5 * sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_dc_component() {
        // All zeros should give all zeros
        let block = [0.0f32; 64];
        let result = dct_2d(&block);
        for &val in &result {
            assert!((val).abs() < 0.001);
        }
    }

    #[test]
    fn test_dct_constant_block() {
        // Constant block: DC component should be non-zero, AC should be small
        let block = [100.0f32; 64];
        let result = dct_2d(&block);

        // DC component (index 0) should be large
        assert!(result[0].abs() > 100.0);

        // AC components should be relatively small compared to DC
        // (some numerical error is expected with the const fn cos approximation)
        for &val in result.iter().skip(1) {
            assert!(val.abs() < 5.0, "AC component too large: {}", val);
        }
    }

    #[test]
    fn test_dct_idct_roundtrip() {
        let mut block = [0.0f32; 64];
        for i in 0..64 {
            block[i] = (i as f32 * 4.0) - 128.0;
        }

        let dct = dct_2d(&block);
        let recovered = idct_2d(&dct);

        // Allow some numerical error due to const fn cos approximation
        for i in 0..64 {
            assert!(
                (block[i] - recovered[i]).abs() < 5.0,
                "Mismatch at {}: {} vs {}",
                i,
                block[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_cos_table_values() {
        // cos(0) = 1
        assert!((COS_TABLE[0][0] - 1.0).abs() < 0.0001);

        // cos(pi/4) = 1/sqrt(2) ≈ 0.707
        // This is cos((2*0 + 1) * 2 * PI / 16) = cos(PI/8)
        assert!((COS_TABLE[0][2] - (PI / 8.0).cos()).abs() < 0.001);
    }
}
