# Preset compare baseline (local)

Command:
```
cargo bench --bench preset_compare -- --sample-size 20
```

Environment: x86_64, release bench profile, 256x256 gradient.

Output (partial; png default/fast shown before abort):
```
png_presets/png/default time:   [6.1705 ms 6.2464 ms 6.3364 ms]
                        thrpt:  [29.591 MiB/s 30.017 MiB/s 30.387 MiB/s]
Found 2 outliers among 20 measurements (10.00%)
  1 (5.00%) high mild
  1 (5.00%) high severe
```

Re-run locally for complete data once the bench finishes; this snapshot shows the default PNG throughput for reference. Use `--sample-size 10` if you need a quicker run.
