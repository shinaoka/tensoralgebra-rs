# Environment-Specific Setup

## OpenBLAS 0.3.29 (Required for BLAS benchmarks)

The system OpenBLAS on Ubuntu 20.04 is version 0.3.8, which is ~2x slower than Julia's bundled OpenBLAS 0.3.29 on AMD EPYC. A locally built OpenBLAS 0.3.29 is required for fair BLAS benchmarks.

### Installation

```bash
curl -L -o /tmp/OpenBLAS-0.3.29.tar.gz \
  https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.29/OpenBLAS-0.3.29.tar.gz
cd /tmp && tar xf OpenBLAS-0.3.29.tar.gz && cd OpenBLAS-0.3.29
make -j$(nproc) USE_OPENMP=0 NO_LAPACK=1
make PREFIX=$HOME/opt/openblas-0.3.29 install
```

### Usage

```bash
export OPENBLAS_LIB_DIR=$HOME/opt/openblas-0.3.29/lib
export LD_LIBRARY_PATH=$HOME/opt/openblas-0.3.29/lib:$LD_LIBRARY_PATH
```

`scripts/run_all.sh` auto-detects `$HOME/opt/openblas-0.3.29/` if `OPENBLAS_LIB_DIR` is not already set.

### Why not system OpenBLAS?

| OpenBLAS version | mera_closed_120 opt_flops 1T | Source |
|---|---:|---|
| 0.3.8 (system) | 2946 ms | `apt install libopenblas-dev` |
| 0.3.29 (local) | 1471 ms | built from source |
| 0.3.29 (Julia) | 1488 ms | Julia-bundled via lbt |

The 2x difference is due to improved GEMM kernels for AMD EPYC (Zen3) in newer OpenBLAS releases.

### Verification

```bash
# Check which library is linked
ldd target/release/strided-rs-benchmark-suite | grep openblas
# Should show: $HOME/opt/openblas-0.3.29/lib/libopenblas.so.0
```
