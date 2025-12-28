---
name: Rust Code Review
overview: Code review improvements focusing on `#[must_use]`, builder patterns for cleaner APIs, type aliases, named constants, safety documentation, and cleanup of unused parameters.
todos:
  - id: must-use
    content: "Add #[must_use] to public encode/resize functions"
    status: completed
  - id: builders
    content: Expand builder patterns for resize and other multi-arg functions
    status: completed
  - id: type-aliases
    content: Add type aliases for complex coefficient tuple types
    status: completed
  - id: named-constants
    content: Extract magic numbers to named constantsDocument safety invariants on SIMD unsafe functions
    status: completed
  - id: unused-params
    content: Remove or validate unused parameters
    status: completed
---

# Rust Code Review

## 1. Add `#[must_use]` to Public Functions

Add `#[must_use] `to public functions returning `Result` or values that shouldn't be ignored.

**Files:** [`src/lib.rs`](src/lib.rs), [`src/resize.rs`](src/resize.rs)

```rust
#[must_use]
pub fn encode_png(data: &[u8], width: u32, height: u32, options: &PngOptions) -> Result<Vec<u8>>
```

---

## 2. Expand Builder Pattern Usage

Use builder patterns to eliminate positional argument confusion and address `#[allow(clippy::too_many_arguments)]`. Follow the existing `JpegOptionsBuilder` pattern in [`src/jpeg/mod.rs:168-216`](src/jpeg/mod.rs).

### 2.1 Public API: Resize Functions

**File:** [`src/resize.rs`](src/resize.rs) (lines 69-90, 107-117)

**Current:**

```rust
pub fn resize(data: &[u8], src_width: u32, src_height: u32,
              dst_width: u32, dst_height: u32, color_type: ColorType,
              algorithm: ResizeAlgorithm) -> Result<Vec<u8>>

#[allow(clippy::too_many_arguments)]
pub fn resize_into(output: &mut Vec<u8>, data: &[u8],
                   src_width: u32, src_height: u32, dst_width: u32, dst_height: u32,
                   color_type: ColorType, algorithm: ResizeAlgorithm) -> Result<()>
```

**Proposed:**

```rust
#[derive(Debug, Clone)]
pub struct ResizeOptions {
    pub src_width: u32,
    pub src_height: u32,
    pub dst_width: u32,
    pub dst_height: u32,
    pub color_type: ColorType,
    pub algorithm: ResizeAlgorithm,
}

impl ResizeOptions {
    pub fn builder(src_width: u32, src_height: u32) -> ResizeOptionsBuilder {
        ResizeOptionsBuilder::new(src_width, src_height)
    }
}

#[derive(Debug, Clone)]
pub struct ResizeOptionsBuilder {
    src_width: u32,
    src_height: u32,
    dst_width: Option<u32>,
    dst_height: Option<u32>,
    color_type: ColorType,
    algorithm: ResizeAlgorithm,
}

impl ResizeOptionsBuilder {
    pub fn new(src_width: u32, src_height: u32) -> Self { ... }
    pub fn dst(mut self, width: u32, height: u32) -> Self { ... }
    pub fn color_type(mut self, ct: ColorType) -> Self { ... }
    pub fn algorithm(mut self, alg: ResizeAlgorithm) -> Self { ... }
    #[must_use]
    pub fn build(self) -> Result<ResizeOptions> { ... }
}

// New API
pub fn resize(data: &[u8], options: &ResizeOptions) -> Result<Vec<u8>>
pub fn resize_into(output: &mut Vec<u8>, data: &[u8], options: &ResizeOptions) -> Result<()>

// Usage:
let opts = ResizeOptions::builder(800, 600)
    .dst(400, 300)
    .algorithm(ResizeAlgorithm::Lanczos3)
    .build()?;
let resized = resize(&pixels, &opts)?;
```

### 2.2 Internal: JPEG Encoding Context

These are internal functions; use a config struct (not a full builder) to group related params.

**File:** [`src/jpeg/mod.rs`](src/jpeg/mod.rs)

| Function | Line | Params |

|----------|------|--------|

| `encode_progressive` | 802 | 9 |

| `encode_scan` | 1358 | 9 |

| `process_block_444` | 924 | 8 |

**Proposed:** Create an internal `EncodeContext` struct:

```rust
struct EncodeContext<'a> {
    data: &'a [u8],
    width: usize,
    height: usize,
    color_type: ColorType,
    subsampling: Subsampling,
    quant_tables: &'a QuantizationTables,
    huff_tables: &'a HuffmanTables,
    use_trellis: bool,
}
```

### 2.3 Internal: Progressive Scan Encoding

**File:** [`src/jpeg/progressive.rs`](src/jpeg/progressive.rs)

| Function | Line | Params |

|----------|------|--------|

| `encode_ac_first` | 126 | 8 |

| `encode_ac_refine` | 201 | 8 |

**Proposed:** Group scan parameters:

```rust
struct ScanParams {
    ss: u8,        // spectral selection start
    se: u8,        // spectral selection end
    al: u8,        // successive approximation low
    is_luminance: bool,
}

pub fn encode_ac_first(
    writer: &mut BitWriterMsb,
    block: &[i16; 64],
    scan: &ScanParams,
    eob_run: &mut u16,
    tables: &HuffmanTables,
)
```

### 2.4 Internal: Huffman Table Construction

**File:** [`src/jpeg/huffman.rs`](src/jpeg/huffman.rs) (line 133)

**Current:**

```rust
fn from_bits_vals(
    dc_lum_bits: [u8; 16], dc_lum_vals: Vec<u8>,
    dc_chrom_bits: [u8; 16], dc_chrom_vals: Vec<u8>,
    ac_lum_bits: [u8; 16], ac_lum_vals: Vec<u8>,
    ac_chrom_bits: [u8; 16], ac_chrom_vals: Vec<u8>,
) -> Option<Self>
```

**Proposed:** Group by table type:

```rust
struct HuffmanSpec {
    bits: [u8; 16],
    vals: Vec<u8>,
}

fn from_specs(
    dc_lum: HuffmanSpec,
    dc_chrom: HuffmanSpec,
    ac_lum: HuffmanSpec,
    ac_chrom: HuffmanSpec,
) -> Option<Self>
```

---

## 3. Add Type Aliases for Complex Tuples

Improve readability of functions returning complex coefficient tuples.

**File:** [`src/jpeg/mod.rs`](src/jpeg/mod.rs)

```rust
// Before
fn compute_all_coefficients(...) -> (Vec<[i16; 64]>, Vec<[i16; 64]>, Vec<[i16; 64]>)

// After
type DctCoefficients = Vec<[i16; 64]>;
type YCbCrCoefficients = (DctCoefficients, DctCoefficients, DctCoefficients);
fn compute_all_coefficients(...) -> YCbCrCoefficients
```

---

## 4. Extract Magic Numbers to Named Constants

Replace inline numeric literals with descriptive constants.

**File:** [`src/compress/deflate.rs`](src/compress/deflate.rs)

// Before

if improvement < 50 { break; }

// After

const CONVERGENCE_THRESHOLD: usize = 50;

if improvement < CONVERGENCE_THRESHOLD { break; }

---

## 5. Remove/Fix Unused Parameters

Address unused `color_type` parameter in JPEG encoding.

**File:** [`src/jpeg/mod.rs`](src/jpeg/mod.rs) - either remove the parameter or add validation that rejects unsupported color types early.
