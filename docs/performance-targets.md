# Performance & size targets (PNG/JPEG)

These targets define what “competitive” looks like for comprs across native and WASM builds. They are checkpoints to guide optimizations and regressions.

## WASM size targets
- **PNG + JPEG (optimized profile)**: ≤ 95 KB uncompressed (`cargo build --release --target wasm32-unknown-unknown --features wasm` with current size profile). Current measured (local, no wasm-opt): **90,473 bytes raw, 30,065 bytes gzipped**.
- **After wasm-opt -Oz (if used in pipelines)**: track and hold the gzipped size within ±5% of baseline.
- **Acceptable regression budget**: Any change that increases the optimized wasm size by >3% requires justification and a size offsetting change or a new baseline.

## Native throughput targets
Benchmarks are run in release mode on x86_64; arm64 variants should be tracked similarly. Measure with existing benches and forthcoming competitor benches.

- **PNG encode** (64–512 px gradients and screenshots):
  - Match or exceed `image` crate encode throughput (parity baseline).
  - Provide a “fast” preset that is at least **1.5× faster** than current default on gradient workloads with ≤5% size loss.
  - Provide a “max” preset that matches or beats libpng-size output (within 1%) while staying within 1.1× the current runtime.

- **JPEG encode** (Q=75 and Q=85, 4:4:4 and 4:2:0):
  - Achieve **≥1.5×** speedup vs current scalar path on x86_64 once SIMD is enabled.
  - Keep output size within ±3% of libjpeg-turbo at equal quality; SSIM within 0.002 of libjpeg-turbo reference for the same quality level.

## Quality & ratio guardrails
- **PNG**: Compression ratio should not regress more than 1% versus current default for adaptive/default settings on gradient and screenshot fixtures when adding speedups.
- **JPEG**: SSIM (or PSNR) against libjpeg-turbo reference at the same quality must remain within 0.002 SSIM (or 0.1 dB PSNR) for Q=75/85; file size within ±3%.

## ARM64 parity
- Track arm64 (NEON) throughput; aim for parity with x86_64 relative improvements (e.g., SIMD-enabled speedups should deliver at least 1.4× over scalar on arm64 as well).

## Regression policy
- Add perf/size checks to CI where feasible:
  - WASM size check via `scripts/wasm-size.sh` (raw + gz; wasm-opt if available).
  - Criterion baselines for key PNG/JPEG encode benches; fail on >5% regression unless explicitly accepted.

## Benchmark coverage to maintain
- PNG encode: gradients, screenshots, noisy images at 64/128/256/512 px.
- JPEG encode: photo-like fixtures at Q=75/85, 4:4:4 and 4:2:0.
- Deflate micro: compressible vs incompressible 1 MiB payloads.
- WASM build size: tracked per commit or at least per release candidate.
