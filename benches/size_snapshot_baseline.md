# Size snapshot baseline (local)

Command:
```
cargo bench --bench size_snapshot -- --nocapture
```

Environment: x86_64 (release bench profile), Q=85 for JPEG.

Output (post 4:2:0 fix):
```
==> gradient 256x256
PNG comprs: 6.292 ms, 2175 bytes
PNG image:  2.829 ms, 35214 bytes
JPEG comprs q85 444: 2.488 ms, 6485 bytes
JPEG comprs q85 420: 1.467 ms, 3998 bytes
JPEG image q85:      3.270 ms, 6403 bytes

==> noisy 256x256
PNG comprs: 12.320 ms, 195632 bytes
PNG image:  3.830 ms, 196947 bytes
JPEG comprs q85 444: 6.135 ms, 104372 bytes
JPEG comprs q85 420: 3.157 ms, 50144 bytes
JPEG image q85:      7.722 ms, 104302 bytes
```

Use this as a rough reference; rerun on your hardware to track regressions.```
