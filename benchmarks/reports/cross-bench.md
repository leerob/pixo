# Cross-language benchmark summary

## Speed (PNG, 512x512 gradient)
| Rank | Library | Env | Time (ms) | Note |
| --- | --- | --- | --- | --- |
| 1 | sharp | js | 8.20 ms |  |
| 2 | comprs | rust | 8.45 ms |  |
| 3 | image crate | rust | 10.74 ms |  |
| 4 | pngjs | js | 20.05 ms |  |
| 5 | jimp | js | 22.62 ms |  |

## Speed (JPEG q85, 512x512 gradient)
| Rank | Library | Env | Time (ms) | Note |
| --- | --- | --- | --- | --- |
| 1 | sharp | js | 2.62 ms |  |
| 2 | comprs | rust | 10.37 ms | subsampling=4:4:4 |
| 3 | image crate | rust | 12.89 ms | subsampling=4:4:4 |
| 4 | jpeg-js | js | 19.87 ms |  |
| 5 | jimp | js | 50.42 ms |  |

## Output size (PNG, 512x512 gradient)
| Rank | Library | Env | Size | Note |
| --- | --- | --- | --- | --- |
| 1 | comprs | rust | 8.68 KB |  |
| 2 | jimp | js | 37.66 KB |  |
| 3 | pngjs | js | 37.66 KB |  |
| 4 | image crate | rust | 76.82 KB |  |
| 5 | sharp | js | 191.83 KB |  |

## Output size (JPEG q85, 512x512 gradient)
| Rank | Library | Env | Size | Note |
| --- | --- | --- | --- | --- |
| 1 | sharp | js | 7.84 KB |  |
| 2 | comprs | rust | 11.64 KB | subsampling=4:2:0 |
| 3 | image crate | rust | 16.68 KB | subsampling=4:4:4 |
| 4 | jimp | js | 17.30 KB |  |
| 5 | jpeg-js | js | 17.30 KB |  |
| 6 | comprs | rust | 17.33 KB | subsampling=4:4:4 |

## Binary/package size
| Rank | Library | Env | Size | Note |
| --- | --- | --- | --- | --- |
| 1 | jpeg-js | js | 74.25 KB | jpeg-js |
| 2 | sharp | js | 491.04 KB | sharp |
| 3 | pngjs | js | 634.86 KB | pngjs |
| 4 | jimp | js | 4.72 MB | jimp |

## Notes
- WASM sizes are approximate and depend on build configuration
- Speed measurements are short-run averages; use Criterion output for statistical analysis
- Compression ratios vary with image content
- Use --export-json to feed the cross-language aggregator
- QUICK=1 reduces iterations to speed up the run.
- Package sizes are measured from installed node_modules when available.
- JS results use quick iterations by default; set BENCH_ITERATIONS for longer runs.
