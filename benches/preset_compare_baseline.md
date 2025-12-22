# Preset compare baseline (local)

Command:
```
cargo bench --bench preset_compare -- --sample-size 10 --warm-up-time 0.5 --measurement-time 2 --noplot
```

Environment: x86_64, release bench profile, 256x256 gradient.

Output:
```
png_presets/png/default time:   [6.1959 ms 6.3089 ms 6.4258 ms]
                        thrpt:  [29.179 MiB/s 29.720 MiB/s 30.262 MiB/s]
png_presets/png/fast    time:   [4.4317 ms 4.4351 ms 4.4397 ms]
                        thrpt:  [42.232 MiB/s 42.276 MiB/s 42.309 MiB/s]
png_presets/png/max     time:   [8.0334 ms 8.0918 ms 8.1327 ms]
                        thrpt:  [23.055 MiB/s 23.171 MiB/s 23.340 MiB/s]

jpeg_presets/jpeg/default       time: [2.5349 ms 2.5434 ms 2.5510 ms]
                               thrpt: [73.502 MiB/s 73.720 MiB/s 73.966 MiB/s]
jpeg_presets/jpeg/fast          time: [1.4988 ms 1.4992 ms 1.4998 ms]
                               thrpt: [125.02 MiB/s 125.07 MiB/s 125.10 MiB/s]
jpeg_presets/jpeg/max_quality   time: [2.5406 ms 2.5434 ms 2.5470 ms]
                               thrpt: [73.617 MiB/s 73.720 MiB/s 73.802 MiB/s]
```

Notes:
- Presets behave as expected: fast > default > max for speed in PNG; JPEG fast (4:2:0) is fastest; max_quality roughly matches default speed (both 4:4:4 with higher quality on max_quality).
