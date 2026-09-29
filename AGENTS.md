# Project guidance

This is a research repository, not a production tensor-algebra library. Read `README.md`, `docs/research-map.md`, `docs/experiments.md`, and `docs/provenance.md` before adding an experiment.

- Keep BLIS-style GEMM, TBLIS-style contraction, batched linalg, and executor/FFI choices provisional until comparable measurements support a decision.
- Before timing a numerical path, check known values and reconstruction or residuals. Record the provider version, build flags, CPU, layout, dtype, shape, batch size, thread count, and timed boundary with results.
- Cite the original paper and any implementation consulted in the source file when code is written. Clearly label a port or close translation, preserve upstream notices, and review imported tests file by file.
- Keep host-controlled threading explicit. Do not introduce an ambient global pool in a library experiment without identifying it as the behavior under test.
- Do not publish a package or present benchmark claims from unrecorded measurements.
