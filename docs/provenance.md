# Provenance, licenses, and citation

This repository began with research notes. In Phase 0 (2026-09-29) three
projects were imported with `git subtree add` without `--squash`, so their
history and authorship are preserved:

| Directory | Upstream | Commit | License | Notices kept |
| --- | --- | --- | --- | --- |
| `strided/` (removed) | [tensor4all/strided-rs](https://github.com/tensor4all/strided-rs) | `71b7cb9` | MIT OR Apache-2.0; `strided-perm` also BSD-3-Clause (HPTT-derived) | removed with the directory |
| `tensorprimitives/` | [lkdvos/tensorprimitives-rs](https://github.com/lkdvos/tensorprimitives-rs) (Lukas Devos) | `8cda75e` | MIT OR Apache-2.0 | `tensorprimitives/LICENSE-*`, per-crate `LICENSE-*` |
| `benchmarks/` | [tensor4all/strided-rs-benchmark-suite](https://github.com/tensor4all/strided-rs-benchmark-suite) | `0550611` | MIT | `benchmarks/LICENSE` |

On 2026-09-30 `strided/` and the strided and einsum benchmarks under
`benchmarks/` were removed: strided-rs is again an external dependency
(pinned to its v0.4.4 commit, as in tenferro-rs), and its benchmarks live in
strided-rs-benchmark-suite. The only tprims addition to strided,
`run_with_exec`, moved to `tprims_exec::strided`. Their history stays
reachable through the import merges.

`crates/tprims-core/include/dlpack/dlpack.h` is DLPack v1.1
(<https://github.com/dmlc/dlpack>, tag `v1.1`), copied verbatim under the
Apache License 2.0 (`crates/tprims-core/include/dlpack/LICENSE`); the Rust
`#[repr(C)]` mirror in `tprims-core/src/dlpack.rs` follows it.

The history of an imported file is reachable through the import merge
commit's second parent (for example `git log 6a84228^2 -- strided-perm/src/lib.rs`).

## Three different uses of prior work

1. **Published algorithm or documented API:** Implement independently from a paper or public operation contract. Record the paper and the important implementation choices in the module and research notes. The algorithm itself does not bring the upstream software license into independently written code. This idea/expression distinction is stated by the [U.S. Copyright Office](https://www.copyright.gov/register/tx-programs.html); local law and any patent claims are separate questions.
2. **Code port or close translation:** Treat the affected code as derived from the original. Preserve the original file-level copyright and license notices, name the source file/version in the new file, and include the full required license text with source and binary distributions. Changing language or variable names does not make it independent.
3. **External oracle or benchmark:** Running an installed BLIS, TBLIS, faer, OpenBLAS, or LAPACK implementation without copying its files does not put its source in this repository. If its code or test vectors are imported, reclassify that import under item 2 or examine the exact file's terms.

The [shared tensor4all provenance policy](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/provenance.md) calls for recording source provenance when code is written, rather than reconstructing it later. Scientific credit is due even when no code license transfers. Cite the original algorithm papers and relevant implementation papers in documentation and publications.

## Verified upstream license starting points

| Project | Project license file | Review note |
| --- | --- | --- |
| faer | [MIT](https://github.com/sarah-quinones/faer-rs/blob/main/LICENSE) (GitHub mirror of Codeberg project) | Keep the original copyright and permission notice when copying substantial code. |
| BLIS | [BSD 3-Clause](https://github.com/flame/blis/blob/master/LICENSE) | The root license explicitly says to inspect file-level copyright headers. |
| TBLIS | [BSD 3-Clause](https://github.com/MatthewsResearchGroup/tblis/blob/master/LICENSE) | Its license says “except where otherwise indicated”; inspect each imported file. |
| OpenBLAS | [BSD 3-Clause](https://github.com/OpenMathLib/OpenBLAS/blob/develop/LICENSE) | Inspect the exact imported test/source file and bundled subproject; the root text alone is not a file inventory. |
| Reference LAPACK | [LICENSE](https://github.com/Reference-LAPACK/lapack/blob/master/LICENSE) | Check the selected file and version before reuse. |
| gemmkit | [MIT or Apache-2.0](https://github.com/SomeB1oody/gemmkit) | Verify the chosen release's package contents if integrating. |
| tenferro-rs | MIT OR Apache-2.0 | Planned port (Phase 1): permute plus batched GEMM contraction and the CPU GEMM driver from `tenferro-cpu`. Same maintainers; record the source commit. |
| tensorprimitives-rs | MIT OR Apache-2.0 | Imported (Phase 0) under `tensorprimitives/` at `8cda75e` with `git subtree add`, history and authorship preserved: `tensorcontract` (TBLIS-style direct contraction, packing, microkernels) by Lukas Devos. |

[MIT](https://opensource.org/license/mit) requires its copyright and permission notices in copies or substantial portions. [BSD 3-Clause](https://opensource.org/license/BSD-3-clause) requires retaining the copyright notice, conditions, and disclaimer in source redistributions, reproducing them in binary distribution materials, and not implying endorsement. A `NOTICE` file is useful as an index, but its name alone does not satisfy these conditions. Including such permissively licensed portions does not, by itself, relicense unrelated original project code.

For any future imported file, record: upstream project, URL, version or commit, path, relationship (verbatim/ported/adapted/test fixture), applicable license, retained copyright headers, and the distribution path for license texts. Review package artifacts before publication.

## Intel oneMKL distinction

The [Intel oneMKL product](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2025-0/intel-oneapi-math-kernel-library-onemkl.html) and the open-source oneMKL/oneMath interface project are distinct. Public oneMKL APIs and academic papers can inform an independent implementation; do not assume optimized product internals are available as open-source implementation material. Check the applicable terms before using any Intel source or binary.
