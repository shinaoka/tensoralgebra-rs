# Fused Elementwise Benchmark

Compares chained elementwise evaluation with one reused temporary per op
against `strided_kernel::fused_elementwise_into` runtime-DAG plans. The first
three cases match static specializations; `interpreter_fallback` intentionally
does not.

Cases:

- `add_mul`: `(a + b) * a`
- `broadcast_exp_mul_add`: `exp(a * b + c)`, where `c` is a stride-0 broadcast
  scalar
- `long_chain`: `rsqrt(sqrt(min(max(a / b, lo), hi)))`
- `interpreter_fallback`: `exp(-(a + b))`

The runner validates fused output against the per-op reference before timing.
All temporaries are allocated outside timed regions.

## Run

Run thread-count variants sequentially. Do not run benchmark processes in
parallel.

```bash
STRIDED_RS_HASH=$(cd ../strided-rs && git rev-parse HEAD)

RUSTFLAGS="-C target-cpu=native" \
  STRIDED_RS_HASH="$STRIDED_RS_HASH" \
  RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo run --release --bin fused_elementwise

RUSTFLAGS="-C target-cpu=native" \
  STRIDED_RS_HASH="$STRIDED_RS_HASH" \
  RAYON_NUM_THREADS=4 OMP_NUM_THREADS=1 \
  cargo run --release --features parallel --bin fused_elementwise
```

For README result updates on Linux, pin cores as required by `AGENTS.md`, for
example `taskset -c 0` for 1T and `taskset -c 0-3` for 4T.

Useful local knobs:

```bash
SIZES=512 WARMUP=1 NRUNS=3 \
  RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo run --release --bin fused_elementwise
```

## Results (Apple M5 Max MacBook Pro)

Measured on macOS, 2026-07-04. macOS CPU pinning was unavailable.

- benchmark command: `SIZES=256,512,1024 WARMUP=3 NRUNS=11`
- compiler flags: `RUSTFLAGS="-C target-cpu=native"`
- strided-rs: `fac38f1b3b03408891d796e180689d661d16c3cf`
- benchmark suite: this commit

Median milliseconds with IQR in parentheses.

### 1T

`RAYON_NUM_THREADS=1`, `OMP_NUM_THREADS=1`.

| Case | n | Elements | per-op reused buffers | fused_elementwise_into | Speedup |
|---|---:|---:|---:|---:|---:|
| add_mul | 256 | 65,536 | 0.034 (0.000) | **0.016 (0.000)** | 2.06x |
| broadcast_exp_mul_add | 256 | 65,536 | 0.151 (0.004) | **0.123 (0.004)** | 1.23x |
| long_chain | 256 | 65,536 | 0.096 (0.003) | **0.062 (0.000)** | 1.56x |
| interpreter_fallback | 256 | 65,536 | **0.150 (0.000)** | 0.492 (0.022) | 0.31x |
| add_mul | 512 | 262,144 | 0.111 (0.002) | **0.054 (0.001)** | 2.07x |
| broadcast_exp_mul_add | 512 | 262,144 | 0.471 (0.016) | **0.409 (0.009)** | 1.15x |
| long_chain | 512 | 262,144 | 0.325 (0.013) | **0.197 (0.002)** | 1.65x |
| interpreter_fallback | 512 | 262,144 | **0.485 (0.004)** | 1.572 (0.107) | 0.31x |
| add_mul | 1024 | 1,048,576 | 0.382 (0.009) | **0.208 (0.015)** | 1.84x |
| broadcast_exp_mul_add | 1024 | 1,048,576 | 1.767 (0.032) | **1.563 (0.035)** | 1.13x |
| long_chain | 1024 | 1,048,576 | 1.226 (0.052) | **0.794 (0.018)** | 1.54x |
| interpreter_fallback | 1024 | 1,048,576 | **1.945 (0.052)** | 6.286 (0.264) | 0.31x |

### 4T

`RAYON_NUM_THREADS=4`, `OMP_NUM_THREADS=1`.

| Case | n | Elements | per-op reused buffers | fused_elementwise_into | Speedup |
|---|---:|---:|---:|---:|---:|
| add_mul | 256 | 65,536 | 0.073 (0.033) | **0.031 (0.016)** | 2.34x |
| broadcast_exp_mul_add | 256 | 65,536 | 0.163 (0.007) | **0.118 (0.016)** | 1.39x |
| long_chain | 256 | 65,536 | 0.186 (0.012) | **0.070 (0.010)** | 2.65x |
| interpreter_fallback | 256 | 65,536 | **0.165 (0.012)** | 0.544 (0.046) | 0.30x |
| add_mul | 512 | 262,144 | 0.059 (0.066) | **0.048 (0.017)** | 1.23x |
| broadcast_exp_mul_add | 512 | 262,144 | 0.171 (0.005) | **0.136 (0.054)** | 1.26x |
| long_chain | 512 | 262,144 | 0.206 (0.092) | **0.090 (0.065)** | 2.28x |
| interpreter_fallback | 512 | 262,144 | **0.239 (0.042)** | 0.857 (0.029) | 0.28x |
| add_mul | 1024 | 1,048,576 | 0.182 (0.007) | **0.084 (0.015)** | 2.17x |
| broadcast_exp_mul_add | 1024 | 1,048,576 | 0.559 (0.066) | **0.438 (0.033)** | 1.28x |
| long_chain | 1024 | 1,048,576 | 0.671 (0.102) | **0.258 (0.027)** | 2.60x |
| interpreter_fallback | 1024 | 1,048,576 | **0.699 (0.174)** | 3.080 (0.075) | 0.23x |
