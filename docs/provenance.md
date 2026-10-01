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

`crates/tprims-core/include/tapp.h` and `include/tapp/*.h` are the TAPP C API
headers of <https://github.com/TAPPorg/reference-implementation> at commit
`77c32d744ee6d339f504620cc80b8679601669bc` (BSD 3-Clause), copied verbatim with
the upstream `LICENSE.md` and `AUTHORS.md`; `include/tapp/README.md` records the
commit and the SHA-256 of each file, which `cargo test -p tprims-bundle`
checks. The ABI baseline is this commit, not moving `main`.

The history of an imported file is reachable through the import merge
commit's second parent (for example `git log 6a84228^2 -- strided-perm/src/lib.rs`).

## Kernel-layer move (2026-09-30)

Pure-rename commit `d5776b8` moved Lukas Devos's tensorcontract files into
`crates/tprims-gemm-kernel` (`element`, `scatter`, `pack`, `writeback`,
`kernel/cache`) and `crates/tprims-kernel-tensorcontract`
(`kernel/{scalar,simd,x86,aarch64}`). The type/selection split of
`kernel/mod.rs` retains the original bodies and relocates their tests.
Both crates retain Lukas Devos in `authors` and copies of the imported
MIT/Apache-2.0 license texts. `git log --follow` reaches the original files;
no imported history was rewritten. Plain `git subtree pull` no longer
updates these moved files: upstream changes must be reconciled explicitly.
The kernel provider's path-only tensorcontract dev-dependency preserves
its existing custom-scalar contraction doctest, not a runtime dependency.

### Downstream kernels (issue #28)

A downstream crate that supplies its own kernels through a caller-scoped
`KernelCatalog` reports itself as `Origin::External { crate_name, license }`.
tprims prints those strings in diagnostics (`SelectedGemm::origin`,
`list_kernels`/`KernelCatalog::list`) and never labels them `Portable` or
`Tensorcontract`; it does not vouch for their license or correctness, which
is the provider's `unsafe` promise at admission. `tprims-custom-kernel-test`
is the test stand-in: its kernels are new MIT OR Apache-2.0 code written for
the test, and its complex family calls the project's own public reference loop
(`portable::cplx_tile`), so no upstream kernel body is copied.

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
| tensorprimitives-rs | MIT OR Apache-2.0 | Imported (Phase 0) under `tensorprimitives/` at `8cda75e` with `git subtree add`, history and authorship preserved: `tensorcontract` (TBLIS-style direct contraction and driver) and, after the 2026-09-30 pure-rename split, `tprims-kernel-tensorcontract` (scalar and SIMD microkernels) by Lukas Devos. |
| gemm-f64, gemm-f32, gemm-common | [MIT](https://github.com/sarah-ek/gemm/blob/main/LICENSE) | Called, not copied (the adapter `tprims-kernel-gemm` was removed in #37): it reached the public microkernel tables through a shim. The microkernel module is undocumented upstream, so the versions are pinned exactly (`=0.19.0`) and a bump has to be re-checked. |
| private-gemm-x86 | [MIT](https://github.com/sarah-ek/gemm-x64-v2/blob/main/LICENSE) | Called, not copied (the adapter `tprims-kernel-pgx86` was removed in #37): it wrapped the single `gemm` entry point, pinned exactly (`=0.1.20`) for the same reason. |
| `tprims-kernel-cplx` (project-owned, MIT OR Apache-2.0) | n/a | Independent intrinsics written from the interleaved complex arithmetic `Cr += Ar*Br - Ai*Bi; Ci += Ai*Br + Ar*Bi` with a signed swap formed once per A vector, as stated in issue #30. Not a port: no source of `gemm-c32/c64` (0.19.0: `mod microkernel` is private, so no callable tile entry exists) or `private-gemm-x86` (0.1.20: `call_microkernel`/`millikernel_*` take a custom register/parameter ABI, unusable as a `TileUkrFn` without a larger adapter) was read into or copied. Reuse check done 2026-10-01 against the pinned registry sources; the independent-intrinsics route was chosen for that reason. Panels, scatter, write-back and scheduling are the existing driver's (Lukas Devos's tensorcontract driver). Its families report `Origin::Cplx` (crate `tprims-kernel-cplx`, `MIT OR Apache-2.0`); `Origin::External` is reserved for downstream third-party kernels. |

### Per-origin crate rule

A crate that contains imported code states its origin, authorship and license
in its manifest and keeps the imported license texts beside it
(`tprims-kernel-tensorcontract`, `tensorcontract`). A crate that only *calls*
another project carries the same provenance in this file and in its header, but
its own license is the project's (as `tprims-kernel-gemm` and
`tprims-kernel-pgx86` were, before #37 removed them). The kernel contract crate
(`tprims-gemm-kernel`) is project code: it was written here, from the design and
the research notes, and only the *moved* kernel files carry Lukas Devos's
authorship.

### The no-port rule (issue #23)

The switchable-engine work is a **contract around** other people's kernels, not
a port of them: a family holds a function pointer to a compiled upstream kernel
or to one of ours, and the driver calls it. No upstream kernel body was copied
into `tprims-gemm-kernel` (or the removed `tprims-kernel-gemm` and `tprims-kernel-pgx86`), and
`faer` remains a called dependency under its upstream license, which is not the
project's: nothing here relicenses it. Any future port would be a new decision
with its own provenance record.

[MIT](https://opensource.org/license/mit) requires its copyright and permission notices in copies or substantial portions. [BSD 3-Clause](https://opensource.org/license/BSD-3-clause) requires retaining the copyright notice, conditions, and disclaimer in source redistributions, reproducing them in binary distribution materials, and not implying endorsement. A `NOTICE` file is useful as an index, but its name alone does not satisfy these conditions. Including such permissively licensed portions does not, by itself, relicense unrelated original project code.

For any future imported file, record: upstream project, URL, version or commit, path, relationship (verbatim/ported/adapted/test fixture), applicable license, retained copyright headers, and the distribution path for license texts. Review package artifacts before publication.

## Intel oneMKL distinction

The [Intel oneMKL product](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2025-0/intel-oneapi-math-kernel-library-onemkl.html) and the open-source oneMKL/oneMath interface project are distinct. Public oneMKL APIs and academic papers can inform an independent implementation; do not assume optimized product internals are available as open-source implementation material. Check the applicable terms before using any Intel source or binary.
