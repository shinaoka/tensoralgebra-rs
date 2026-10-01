# BLIS and TBLIS: how they organize switchable GEMM engines

Research notes for a Rust switchable GEMM/tensor-contraction engine. Date: 2026-09-30.

## Sources and pinned revisions

Everything below was read from source at these commits (shallow clones in the scratchpad), unless marked otherwise.

- **BLIS master** `913242ced922e9749983adb6b95fc620ddb895b2` — base `B = https://github.com/flame/blis/blob/913242ced922e9749983adb6b95fc620ddb895b2`
- **BLIS 0.9.0** (tag, 2022-04-01) `14c86f66b20901b60ee276da355c1b62642c18d2` — base `B09 = https://github.com/flame/blis/blob/14c86f66b20901b60ee276da355c1b62642c18d2`
- **TBLIS master (2.x line)** `eb719e718976572e0ab53975f4e0c799faeb35f2` (2025-11-21) — base `T2 = https://github.com/devinamatthews/tblis/blob/eb719e718976572e0ab53975f4e0c799faeb35f2`. The repo now redirects to `MatthewsResearchGroup/tblis`. It pins a BLIS commit in `blis-git-tag` = `358e689cadd6757f564a2992cf46a2f7d6fa6bb0`.
- **TBLIS 1.x branch** `c4f81e08b2827e72335baa7bf91a245f72c43970` (same as tag v1.3.0) — base `T1 = https://github.com/devinamatthews/tblis/blob/c4f81e08b2827e72335baa7bf91a245f72c43970`. This is the version with the standalone `src/configs/*` that the task refers to.

Papers:

- [VvdG15] Van Zee & van de Geijn, "BLIS: A Framework for Rapidly Instantiating BLAS Functionality", ACM TOMS 41(3), 2015. doi:10.1145/2764454
- [VZ+16] Van Zee, Smith, Marker, Low, van de Geijn, Igual, Smelyanskiy, Zhang, Kistler, Austel, Gunnels, Killough, "The BLIS Framework: Experiments in Portability", ACM TOMS 42(2), 2016. doi:10.1145/2755561
- [VS17] Van Zee & Smith, "Implementing High-performance Complex Matrix Multiplication via the 3m and 4m Methods", ACM TOMS 44(1), 2017. https://dl.acm.org/doi/10.1145/3086466
- [VZ20] Van Zee, "Implementing High-Performance Complex Matrix Multiplication via the 1M Method", SIAM J. Sci. Comput. 42(5):C221–C244, 2020. Preprint: https://www.cs.utexas.edu/~flame/pubs/blis6_sisc_rev1.pdf
- [LISQ16] Low, Igual, Smith, Quintana-Ortí, "Analytical Modeling Is Enough for High-Performance BLIS", ACM TOMS 43(2):12, 2016. https://dl.acm.org/doi/10.1145/2925987, PDF https://www.cs.utexas.edu/~flame/pubs/TOMS-BLIS-Analytical.pdf
- [M18] Matthews, "High-Performance Tensor Contraction without Transposition", SIAM J. Sci. Comput. 40(1):C1–C24, 2018. arXiv:1607.00291, https://arxiv.org/abs/1607.00291

Items marked **[inference]** are my interpretation, not a statement found in a source.

---

# Part I — BLIS

## 1. Central objects

### 1a. `cntx_t` (context): one per sub-configuration, holding the kernel and blocksize tables

- **Current master** (`B/frame/include/bli_type_defs.h#L1458`): `cntx_t` holds six growable stacks (`stck_t`):
  - `blkszs`: blocksize objects
  - `bmults`: for each blocksize, the id of the blocksize it must be a multiple of
  - `ukrs`: single-type kernels (`func_t`)
  - `ukr2s`: two-type kernels (`func2_t`)
  - `ukr_prefs`: preference booleans (`mbool_t`)
  - `l3_sup_handlers`

  The stacks let plugins append new kernel and blocksize ids at runtime (see 1c).
- **BLIS 0.9.0** (`B09/frame/include/bli_type_defs.h#L1428`) used fixed arrays instead: `blkszs[]`, `bmults[]`, `l3_vir_ukrs[]`, `l3_nat_ukrs[]`, `l3_nat_ukrs_prefs[]`, the sup thresholds, blocksizes and kernels, `l1f_kers`, `l1v_kers`, `packm_kers`, `unpackm_kers`, plus an `ind_t method` field. That release was already 1m-only: its `ind_t` enum is `{BLIS_1M, BLIS_NAT}` (`B09/.../bli_type_defs.h#L611`). So the virtual-ukr and `method` fields held 1m, not 3m/4m, state.

The supporting types (`B/frame/include/bli_type_defs.h#L1131-L1210`):

- **`blksz_t`**: two arrays indexed by datatype (s, d, c, z).
  - `v[]` is the default ("primary") value.
  - `e[]` is the extension ("max") value.
  - For MR/NR the max is PACKMR/PACKNR, the leading dimension of a packed micropanel. For MC/KC/NC the max is the largest block allowed, so a small leftover can be merged into the previous iteration ([ConfigurationHowTo](https://github.com/flame/blis/blob/913242ced922e9749983adb6b95fc620ddb895b2/docs/ConfigurationHowTo.md)).
- **`func_t`**: one `void_fp` per datatype.
- **`func2_t`**: an array of `void_fp` indexed by (datatype, datatype), used for mixed-type kernels such as packm (source type → packed type) and gemm (computation type, C type).
- **`mbool_t`**: one bool per datatype. It stores preferences such as "this gemm microkernel prefers row-stored C".
- **`auxinfo_t`**: passed to every microkernel. It carries:
  - the pack schemas of A and B
  - `a_next` / `b_next` for prefetching
  - the imaginary strides `is_a` / `is_b`
  - the panel strides `ps_a` / `ps_b`
  - the micro-tile offsets `off_m` / `off_n`
  - `void_fp ukr` plus `const void* params`, which point to the virtual microkernel and its parameters

Kernel ids (`ukr_t`, `B/.../bli_type_defs.h#L638`) put the type-arity (1-type or 2-type) in the top bits of the id and use the low bits as a linear index:

- 1-type kernels: l1v, l1f, trsm, gemmtrsm, gemmtrsm1m, gemmsup.
- 2-type kernels: `BLIS_PACKM_KER`, `BLIS_PACKM_1ER_KER`, `BLIS_PACKM_RO_KER`, the diag variants, `BLIS_UNPACKM_KER`, `BLIS_GEMM_UKR`, `BLIS_GEMM1M_UKR`, and the mixed-domain wrappers `BLIS_GEMM_{CCR,RCC,CRR}_UKR`.

Preference ids are in `ukr_pref_t` (`#L722`), e.g. `BLIS_GEMM_UKR_ROW_PREF`. Blocksize ids are in `bszid_t` (`#L890`):

- level-3: KR, MR, NR, MC, KC, NC
- packing broadcast factors: BBM, BBN
- level-2 and level-1f: M2, N2, AF, DF, XF
- sup thresholds: MT, NT, KT
- sup blocksizes: *_SUP

### 1b. `gemm_cntl_t` / `gemm_var_cntl_t`: a per-call control tree

This is the second level of state. It is built on each call by `bli_gemm_cntl_init` (`B/frame/3/gemm/bli_gemm_cntl.c#L75`).

- `gemm_cntl_t` (`B/frame/3/gemm/bli_gemm_cntl.h#L190`) is a fixed chain of nodes: `part_jc → part_pc → pack_b → part_ic → pack_a → ker → ir_loop`.
- The kernel node `gemm_var_cntl_t` (`#L40`) holds:
  - `dt_comp`, `dt_out`
  - `ukr`: the microkernel actually called, possibly a virtual one such as 1m
  - `real_ukr`: the underlying real-domain kernel
  - `params`, `real_params`
  - `mr`, `nr`: after any 1m scaling
  - `mr_scale`, `nr_scale`
  - `row_pref`
- The packing node `packm_def_cntl_t` (`B/frame/1m/packm/bli_packm_cntl.h#L72`) holds:
  - the packm ukr pointer
  - `bmult_m_def`, `bmult_m_pack`, `bmult_m_bcast`, the scales
  - `pack_schema`: `BLIS_PACKED_PANELS`, `_1E`, `_1R` or `_RO`
  - `params`
- Setters let external code swap the pack variant, the macrokernel variant and their params. TBLIS 2.x uses these (§7).

### 1c. How kernels register

Each sub-configuration provides `bli_cntx_init_<conf>(cntx)`. Example: haswell, `B/config/haswell/bli_cntx_init_haswell.c`. It runs in three steps.

1. **Fill with reference kernels.** It calls the auto-generated `bli_cntx_init_<conf>_ref(cntx)`, which installs reference kernels and default blocksizes for every datatype.
2. **Override the optimized entries.** Variadic setters take `(kernel-id, datatype, function-pointer)` triples ended by a `BLIS_VA_END` sentinel:
   - `bli_cntx_set_ukrs(cntx, …)` for kernels
   - `bli_cntx_set_ukr_prefs(cntx, …)` for `(pref-id, dt, bool)` triples
3. **Set blocksizes.** It fills a local `blksz_t blkszs[BLIS_NUM_BLKSZS]` with `bli_blksz_init_easy` (default = max) or `bli_blksz_init` (separate default and max). It then commits them with `bli_cntx_set_blkszs(cntx, id, &blksz, multiple-id, …, BLIS_VA_END)`. For example, NC must be a multiple of NR, KC of KR, and MC of MR. On haswell, d/z use the 6×8 and 3×4 kernels with row preference TRUE, and MC/KC/NC are hard-coded numbers.

Global registration is `bli_gks_register_cntx(arch_id, nat_init_fp, ref_init_fp)`. `bli_gks_init` calls it for every configuration compiled into the library through the `INSERT_GENTCONF` X-macro (`B/frame/base/bli_gks.c#L58-L90`, `#L174`).

After the native init runs, registration validates the context (`B/frame/base/bli_gks.c#L249-L272`):

- MC % MR, NC % NR, KC % KR must be 0.
- MC % NR and NC % MR must be 0, unless `BLIS_RELAX_MCNR_NCMR_CONSTRAINTS` is set; this is for trsm.
- For each real type whose kernel is column-preferring, MR and PACKMR must be even (`B/frame/base/bli_check.c#L961`). For row-preferring kernels, NR must be even. **[inference]** This is what makes the 1m halving legal.
- The register tile must fit in `BLIS_STACK_BUF_MAX_SIZE`, since wrappers put an MR×NR temporary tile on the stack.

**Plugin registration (current master only).** Out-of-tree code can allocate new ids at runtime:

- `bli_gks_register_ukr(&id)`, `bli_gks_register_ukr2(&id)`, `bli_gks_register_blksz(&id)`, `bli_gks_register_ukr_pref(&id)` (`B/frame/base/bli_gks.c#L486-L600`).
- Each call pushes a new slot into every compiled context and checks that all contexts return the same new id (the `id != next_id` check fails with `BLIS_INVALID_UKR_ID`).
- The plugin then fills the slots per architecture with `bli_cntx_set_ukr(id, &func_t, cntx)` and `bli_cntx_set_blksz(id, &blksz, mult, cntx)`.

## 2. Runtime selection

**Configure time: choose which sub-configurations get compiled.**

- `config_registry` defines families. An umbrella family lists several sub-configurations, e.g. `intel64: skx knl haswell sandybridge penryn generic`. A sub-configuration can also borrow kernel sets from others with `/`, e.g. `zen: zen/haswell/sandybridge` ([ConfigurationHowTo](https://github.com/flame/blis/blob/913242ced922e9749983adb6b95fc620ddb895b2/docs/ConfigurationHowTo.md)).
- Each sub-configuration's kernels are compiled with that sub-configuration's own flags, from its `make_defs.mk`.

**Run time: pick one architecture id, once.**

- `bli_arch_query_id` calls `bli_arch_set_id_once`, which runs `bli_arch_set_id` exactly once under a `bli_pthread_once` (`B/frame/base/bli_arch.c#L80-L110`).
- `bli_arch_set_id` (`#L125-L215`) first reads `BLIS_ARCH_DEBUG`, which enables a stderr "selecting sub-configuration" log.
- It then reads `BLIS_ARCH_TYPE`, which may be a number or a name. If set, the value must be in range and its context must have been compiled in, or BLIS aborts. These checks only run when error checking is enabled, which is the default (`B/frame/base/bli_arch.c#L150-L190`).
- Otherwise, for an umbrella family, it calls `bli_cpuid_query_id()`. Singleton builds hard-wire the id through `BLIS_FAMILY_*` macros.
- `bli_cpuid_query_id` (`B/frame/base/bli_cpuid.c#L85-L175`) tests enabled sub-configurations newest to oldest per vendor (skx, knl, haswell, …; zen3, zen2, zen, …). If nothing matches or the vendor is unknown, it returns `BLIS_ARCH_GENERIC`.
- skx vs haswell is decided from a hand-maintained CPU-model list, because the number of FMA units is not directly queryable ([HardwareSupport.md](https://github.com/flame/blis/blob/913242ced922e9749983adb6b95fc620ddb895b2/docs/HardwareSupport.md)).

**Initialization and thread safety.**

- `bli_init_once` → `bli_init_apis` (`B/frame/base/bli_init.c`) uses `bli_pthread_switch_on`, a once-style guarded switch.
  - Process-global: gks, thread, memsys.
  - Thread-local (`BLIS_THREAD_LOCAL`): ind and rntm. Each application thread gets its own induced-method state and its own default rntm.
- `bli_gks_init` states it runs on only one thread per init/finalize cycle, so it needs no mutex (`B/frame/base/bli_gks.c#L60-L64`).
- When `BLIS_ENABLE_GKS_CACHING` is on, the native context pointer is cached, making `bli_gks_query_cntx()` a plain load. Caching is on by default (`B/frame/include/bli_config_macro_defs.h#L90-L96`).
- History: BLIS 0.9.0 kept one context per (arch, induced method). Induced-method contexts were built lazily under a `gks_mutex` (`B09/frame/base/bli_gks.c#L506-L601`, `bli_gks_query_ind_cntx`). Current master dropped that: 1m adjustments are computed per call in the cntl tree (§3).

**Global or per call?** Architecture selection is process-global. Callers can still pass an explicit context:

- Every expert API (`bli_gemm_ex(…, cntx, rntm)` etc.) accepts a `const cntx_t*` and uses the global one only when it is NULL (`B/frame/3/bli_l3_oapi_ex.c#L101`).
- **[inference]** A caller could build or modify its own `cntx_t` and pass it per call. That is the per-plan override hook.
- Threading goes through `rntm_t`, again per call via `_ex` or globally (§6).
- **Sandboxes** replace `bli_gemm_ex()` wholesale at configure time (`./configure -s gemmlike …`, [Sandboxes.md](https://github.com/flame/blis/blob/913242ced922e9749983adb6b95fc620ddb895b2/docs/Sandboxes.md)). That is a compile-time switch, not a runtime one.

## 3. Complex arithmetic: native vs induced

**Which methods exist today.** Only native and 1m.

- `ind_t = {BLIS_1M, BLIS_NAT}` (`B/.../bli_type_defs.h#L581`).
- 3m and 4m (3mh, 3m1, 4mh, 4m1a, 4m1b) were removed in commit `f065a8070f187739ec2b34417b8ab864a7de5d7e` (2021-10-28), described as "rarely used and posed code maintenance challenges" (BLIS `CHANGELOG`, around line 3773). [VS17] is historical design only.
- No split/planar complex layout exists. The pack schemas are:
  - `BLIS_PACKED_PANELS`: native, interleaved
  - `_1E` and `_1R`: 1m
  - `_RO`: real part only, for mixed-domain

  (`B/.../bli_type_defs.h#L473`)

**Default policy.** `bli_ind_init` (`B/frame/base/bli_ind.c#L45-L72`) enables 1m for c or z when both hold:

- that type's gemm ukr is the reference kernel, detected by comparing function pointers against the reference context (`bli_gks_cntx_ukr_is_ref`);
- the matching real kernel is optimized.

So an optimized native complex kernel wins automatically. If both are reference, native is used.

**Switch API.**

- `bli_ind_enable/disable(method)`
- `bli_ind_enable_dt/disable_dt(method, dt)`
- `bli_ind_disable_all[_dt]`
- `bli_ind_oper_enable_only(oper, method, dt)`
- queries: `bli_ind_oper_find_avail(oper, dt)`, `bli_ind_oper_get_avail_impl_string`

The state is `bli_l3_ind_oper_st[method][level3-op][c|z]`, declared `BLIS_THREAD_LOCAL` "so one application thread's change doesn't affect another". Without TLS it is protected by a mutex (`B/frame/3/bli_l3_ind.c#L55-L75`). `find_avail` returns the first method that is both implemented and enabled, and falls back to NAT (`#L114`).

**How 1m changes packing and blocksizes.** All of this happens per call in `bli_gemm_cntl_init` (`B/frame/3/gemm/bli_gemm_cntl.c#L90-L330`):

- `dt_comp` becomes the real type when the method is induced or the domains are mixed. `row_pref` is read from the *real* kernel's preference.
- If C's storage disagrees with the kernel's preference, the whole problem is transposed: A and B swap, and op(C) becomes C^T. The same trick orients the 1m variant.
- If `im == BLIS_1M`:
  - Column-preferring kernel: `schema_a = 1E`, `schema_b = 1R`, and `mr_scale = mc_scale = 2`.
  - Row-preferring kernel: `schema_a = 1R`, `schema_b = 1E`, and `nr_scale = nc_scale = 2`.
  - Always `kc_scale = 2`.
  - `mr_pack_scale` stays 1, because PACKMR/PACKNR are *not* halved: the 1e format doubles the elements, which cancels the halving. [VZ20] §3.3 states the same rule.
  - `gemm_ukr` is set to the virtual `BLIS_GEMM1M_UKR` from `ukr2s`. `real_ukr` stays the real `BLIS_GEMM_UKR`.
- **Mixed domain uses the same mechanism.**
  - C complex += A complex × B real: the complex operand is packed normally but with rescaled blocksizes, then run through a `GEMM_CCR` wrapper.
  - C real += A complex × B complex: both operands packed 1R, `kc_scale = 2`, one operand conjugated, `GEMM_RCC` wrapper.
  - C complex += A real × B real: `GEMM_CRR` wrapper.
  - C real += A complex × B real: `RO` packing.
- Packing picks its kernel from the schema: `BLIS_PACKM_1ER_KER` for 1e/1r, `BLIS_PACKM_RO_KER` for real-only.
- The kernel node stores `mr_def/mr_scale`, `nr_def/nr_scale`, `row_pref`, `ukr` and `real_ukr`. `bli_gemm_var_cntl_set_params(&cntl->ker, &cntl->ker)` makes the node its own `params` (`#L429`), so the virtual ukr can find the real kernel through `auxinfo->params`.

**How one microkernel signature serves every method.** Every gemm ukr, whether native, reference, 1m-virtual or mixed-domain wrapper, has the same C signature (`B/frame/3/bli_l3_ukr_params.h#L39`, `bli_l3_ukr_prot.h`):

    (m, n, k, alpha, a, b, beta, c, rs_c, cs_c, auxinfo, cntx)

The scalars and panels are `void*`.

The 1m virtual ukr (`B/ref_kernels/ind/bli_gemm1m_ref.c`):

1. reads `real_ukr`, `row_pref`, `mr` and `nr` from `auxinfo->params`;
2. doubles the dimensions: `m_r = 2m` for column preference or `n_r = 2n` for row preference, and always `k_r = 2k`;
3. calls the real kernel directly on C, treated as a real matrix with doubled strides, when possible;
4. otherwise computes into a stack tile `ct` and accumulates. This fallback covers complex alpha or beta with nonzero imaginary part, general-stride C, C stored against the preference, or mismatched precisions.

The real kernel is never modified. [VZ20] (§3.1–3.6) describes the 1e format as elements "expanded (duplicated…, swapped and imaginary negated)" and 1r as "merely reordered", and discusses bypassing the virtual ukr when that is profitable.

## 4. Reference vs optimized kernels, and heterogeneous MR/NR

**Reference kernels are compiled once per sub-configuration.**

- The sources in `ref_kernels/` are compiled with `BLIS_CNAME_INFIX` and `BLIS_REF_SUFFIX` name-mangling, giving names like `bli_dgemm_haswell_ref`, plus that sub-configuration's compiler flags.
- `bli_cntx_init_<conf>_ref` installs them (`B/ref_kernels/bli_cntx_ref.c`).
- MR/NR for the reference kernel come from per-configuration macros in `bli_kernel_defs_<conf>.h`, e.g. haswell has `BLIS_MR_d 6`, `BLIS_NR_d 8`. Where a configuration doesn't define them, framework defaults apply (`B/frame/include/bli_kernel_macro_defs.h#L260` onward), and `BLIS_PACKMR_* = MR*BBM`.
- `bli_cntx_ref.c#L377-L397` initializes MR/NR blocksizes from those macros and uses generic MC/KC/NC (256/128/128/64, 256, 4096).

**Two reference gemm variants** (`B/ref_kernels/3/bli_gemm_ref.c`):

- A fixed-size variant with constant MR/NR from the macros and `#pragma omp simd`. It unrolls well.
- A `gemm_gen` fallback that reads PACKMR/PACKNR/BBM/BBN from the context at runtime. It is used when `BLIS_MR_x == -1` or when broadcast packing is in effect (`#L185-L215`).

**"Is this kernel reference?"** is answered by pointer comparison against a reference context rebuilt for the same arch (`bli_gks_cntx_ukr_is_ref`, `bli_gks_l3_ukr_impl_type` → `kimpl_t` {REFERENCE, VIRTUAL, OPTIMIZED}; `B/frame/base/bli_gks.c#L409-L480`).

**Heterogeneous MR/NR across datatypes and configs** is handled because:

- All blocksizes are data in the context, per datatype.
- The loops read MR/NR/MC/KC/NC per call from `cntx` for `dt_comp`, then apply the scales.
- `bmults` enforces divisibility at registration.
- Packing writes micropanels with leading dimension PACKMR/PACKNR (the max value) and zero-pads edges.

**Edge handling moved into the microkernel.**

- Since commit `54fa28b`, the gemm ukr receives `m ≤ MR`, `n ≤ NR` and must handle partial tiles itself. Most assembly kernels do it with the `GEMM_UKR_SETUP_CT` / `GEMM_UKR_FLUSH_CT` macros (`B/frame/include/bli_edge_case_macro_defs.h#L71`), which switch to an aligned stack tile when the tile is partial or C's strides don't suit the kernel ([KernelsHowTo.md](https://github.com/flame/blis/blob/913242ced922e9749983adb6b95fc620ddb895b2/docs/KernelsHowTo.md), "Edge cases in MR, NR dimensions").
- trsm ukrs still assume full MR×NR.
- The kernel must also accept any `rs_c`/`cs_c`, i.e. general stride. The row/column preference is only a performance hint used to orient the problem.

## 5. Blocksizes

- **Storage:** each blocksize is a `blksz_t` {default, max} per datatype, held in `cntx->blkszs` and paired with a multiple-id in `cntx->bmults` (§1).
  - MR and NR: max = PACKMR, PACKNR.
  - MC, KC, NC: max = merge threshold.
  - KR: k-unroll multiple.
  - BBM, BBN: broadcast factors.
- **Derivation:** static constants in each `bli_cntx_init_<conf>.c`, e.g. haswell `NC = 4080`. Neither BLIS nor TBLIS probes caches at runtime.
  - [LISQ16] gives the analytical recipe used to pick them: `mr·nr ≥ Nvec·Lvfma·Nvfma` to hide FMA latency within the register budget, KC from L1 size and associativity (the B micropanel stays in L1 while A streams), MC from L2, NC from L3 (§4).
  - **[inference]** A Rust engine could compute these at startup from CPUID cache descriptors using the same model and store them in the descriptor, but the libraries themselves only hard-code them.
- **Passing to loops:**
  - Current master: `bli_gemm_cntl_init` reads def and max for `dt_comp` once per call, applies the 1m and mixed-domain scales, and stores the results in the `part_cntl_t` nodes (`B/frame/3/gemm/bli_gemm_cntl.c`, the `bli_part_cntl_init_node` calls).
  - The partition variants use def, and use max to decide whether a leftover merges into the last block.
  - Before the cntl rework: `bli_cntx_get_blksz_def_dt` / `bli_cntx_get_blksz_max_dt` looked up inside the loops.
- **sup (small/skinny) path:** separate thresholds MT/NT/KT and blocksizes *_SUP select an unpacked "sup" code path with its own kernel table (`BLIS_GEMMSUP_{RRR…CCC}_UKR`) and per-storage-combination preferences.

## 6. Threading, kept separate from kernel selection

- `rntm_t` (`B/.../bli_type_defs.h#L1476`) holds:
  - `thread_impl`: single, OpenMP, pthreads or HPX
  - `auto_factor`
  - `num_threads`
  - `thrloop[6]`: ways for JC, PC, IC, JR, IR, PR
  - `pack_a`, `pack_b`, `l3_sup`
- Neither `cntx_t` nor the kernels contain threading information. The only coupling is that factorization uses m, n, k and the MR/NR/MC granularity.
- **Sources of settings:**
  - Environment, read once at init: `BLIS_NUM_THREADS` / `BLIS_NT` / `OMP_NUM_THREADS` for automatic mode, or `BLIS_{JC,PC,IC,JR,IR}_NT` for manual mode (`B/frame/base/bli_rntm.c#L144-L165`).
  - Global runtime: `bli_thread_set_num_threads`, `bli_thread_set_ways(jc, pc, ic, jr, ir)`, `bli_thread_set_thread_impl`. This state is per application thread.
  - Per call: a local `rntm_t` passed to `_ex` APIs (`bli_rntm_set_ways`, `bli_rntm_set_num_threads`) ([Multithreading.md](https://github.com/flame/blis/blob/913242ced922e9749983adb6b95fc620ddb895b2/docs/Multithreading.md)).
- The PC (k) loop is "Unavailable; always 1", because iterations update the same C (same doc).
- The auto factorization is `bli_rntm_factorize(m, n, k, rntm)`. The thread tree `thrinfo_t` is built from rntm plus cntl (`bli_l3_thrinfo_create`) and is independent of which ukr was chosen.
- [VZ+16] reports the multi-loop parallelization results.

---

# Part II — TBLIS

There are two different designs.

- **TBLIS 1.x** (branch `1.x`, [M18]): a standalone C++ framework with its own config structs that reuses BLIS *assembly microkernels*. [M18] §7.1: "use a BLIS-like framework and the actual BLIS micro-kernels"; "we did not pursue using the BLIS framework directly" because it lacked flexibility inside the macrokernel.
- **TBLIS 2.x** (current master): a BLIS **plugin** that registers extra kernel ids in BLIS's gks and reuses BLIS's context, control tree, 1m logic and threading.

## 7a. TBLIS 1.x configuration

**`tblis::config`** (`T1/src/configs/configs.hpp#L95-L231`) is a flat struct. Each field is an array of four entries, one per datatype s/d/c/z, indexed through `type_idx<T>`.

- **`blocksize`** (`#L40`) holds `_def[4]`, `_max[4]`, `_iota[4]`, `_extent[4]`:
  - `iota` is the multiple: the register blocksize for cache blocksizes.
  - `extent` is the packed leading dimension. This is PACKMR/PACKNR under another name: the default packing kernels step by `ME = extent` (`T1/src/kernels/3m/packm.hpp`).
  - `def` and `max` work as in BLIS.

  Fields: `gemm_mr`, `gemm_nr`, `gemm_kr`, `gemm_mc`, `gemm_nc`, `gemm_kc`, plus 1f and transpose blocksizes.
- **`microkernel<ukr_t>`** (`#L60`) is `void(*)(void) _ukr[4]` with a typed `call<T>(…)` that casts back.
  - Fields: `gemm_ukr`; the 1v, 1f and transpose kernels; and 16 packing kernels. The packing kernels come in MR and NR flavors for each input layout:
    - `pack_nn`: normal strides
    - `pack_nnd`: diagonal-scaled
    - `pack_sn`, `pack_ns`, `pack_ss`: scatter vectors on rows, columns or both
    - `pack_nb`, `pack_sb`: block-scatter
    - `pack_ss_scal`: scatter with scaling vectors
- **`parameter<U>`**: `gemm_row_major`, `gemm_flip_ukr` (bool), and the threading heuristics `m_thread_ratio`, `n_thread_ratio`, `mr_max_thread`, `nr_max_thread`.
- `check_fn_t check` and `const char* name`.

**gemm ukr signature** (`T1/src/kernels/3m/gemm.hpp#L22`):

    (k, alpha*, a*, b*, beta*, c*, rs_c, cs_c, auxinfo_t*)

This is the *older* BLIS signature: fixed MR×NR, no m/n. TBLIS 1.x links BLIS's assembly kernels by name through `EXTERN_GEMM_UKR`, e.g. `bli_dgemm_asm_6x8`.

**Registration** (`T1/src/configs/config_builder.hpp`) is macro plus template metaprogramming:

- `TBLIS_BEGIN_CONFIG(name)` declares `name_config : config_template<name_config>`. `config_template` supplies a default for every field.
- Overrides are per-type macros, where `_` means "keep default":
  - `TBLIS_CONFIG_GEMM_MR(s,d,c,z)`, `..._NR`, `..._KR`, `..._MC`, `..._NC`, `..._KC`
  - `..._MC_MAX`, and the `…_EXTENT` variants
  - `TBLIS_CONFIG_GEMM_UKR(s,d,c,z)`
  - `TBLIS_CONFIG_GEMM_ROW_MAJOR(…)`
  - `TBLIS_CONFIG_CHECK(fn)`
- These become static template members: `register_blocksize` (`#L217`), `cache_blocksize` (`#L233`) whose `iota` is the register blocksize, and `static_microkernel`.
- `TBLIS_CONFIG_INSTANTIATE` builds one function-local static `config` instance by converting the traits into the runtime struct.
- **Reference kernels come from the same templates.** An unset ukr defaults to `gemm_ukr_def<Config, T>` (`T1/src/kernels/3m/gemm.hpp#L31`), which reads MR/NR from the config at compile time. So each config gets its own reference kernel instantiation for its own MR/NR, much like BLIS compiling `ref_kernels` per configuration.
  - Example: haswell (`T1/src/configs/haswell/config.hpp#L82-L99`) sets `MR = (6,6,_,_)`, `NR = (16,8,_,_)`, `KC = 256`, `MC = (144,72,_,_)`, `NC = 4080`, the s/d asm ukrs, and `ROW_MAJOR(true,true,_,_)`.
- **Complex** c and z are left as `_`. They use the templated reference kernel with native `std::complex` arithmetic. **TBLIS 1.x has no induced methods**; complex is always native, reference-quality unless a config supplies a kernel.
- The set of configs compiled in comes from `configure --enable-config=…` (default `auto`), which generates `foreach_config.h` from `FOREACH_CONFIG` (`T1/configure.ac#L144-L170`, `#L438`).

**Runtime selection** (`T1/src/configs/configs.cxx`):

- `default_config` (`#L37`) is a function-local static, so C++11 initialization gives thread safety.
- It runs every compiled config's `check()`. Each returns a priority, or −1 if it can't run on this machine. For example `haswell_check` tests the Intel vendor plus AVX and AVX2 via cpuid and returns 3; `reference_check` returns 0.
- It picks the highest priority and aborts if none is usable.
- `TBLIS_VERBOSE` logs the priorities and the chosen config (`T1/src/util/env.cxx#L19`).
- There is **no env var to force a config**. Instead:
  - every public API takes `const tblis_config* cfg` per call, where NULL means the default (`get_config(cfg)`);
  - `get_config(name)` (`#L93`) returns a named config after checking that it is usable.
- So selection is global by default and overridable per call, and the config objects are immutable.

**Microkernel wrapper, edges and scatter** (`T1/src/nodes/gemm_ukr.hpp#L128-L360`), `gemm_micro_kernel::operator()` overloaded per C matrix type:

- If the tile is full (`m == MR && n == NR`) and C has constant strides in this block (`rs_c && cs_c`), it calls the ukr directly on C.
- Otherwise it calls the ukr with `beta = 0` into a stack tile `T p_ab[512]` and runs `accum_utile(…)`. There are four overloads, for normal or scatter rows × normal or scatter columns: `p_c[rscat[i] + cscat[j]] = p_ab + beta·p_c`.
- `gemm_flip_ukr` calls `ukr(k, α, b, a, β, c, cs_c, rs_c)` to reuse an NR×MR kernel as MR×NR.
- `gemm_row_major` sets the temporary tile's strides and the whole-problem transpose decision in `nodes/gemm.hpp` (`trans = C.stride(!row_major) == 1`).
- Note the hard cap of 512 elements on the tile, where BLIS uses a stack buffer sized from the SIMD register count.

**Tensor packing with scatter and block-scatter vectors** ([M18] §6):

- Scatter vectors: the tensor-to-matrix mapping is stored as `rscat(T)` / `cscat(T)`, the offset of each matrix row and column in the unmodified tensor.
- Block-scatter vectors: `rbs` / `cbs` hold, for each block of size b (MR for rows of A and C, NR for columns of B and C, and a small `kP` for the contracted dimension), the constant stride within that block, or 0 if the block isn't uniform.
- Packing a sliver is split into MR×kP (or kP×NR) micro-tiles. A tile with a constant stride uses the strided `pack_nb`/`pack_sb` kernels; otherwise the gather kernels `pack_sn`/`pack_ss` run.
- "The micro-kernel does not need to be modified for scatter-matrix layouts of A and B because the packing operations write the values in … a fixed layout."
- For C, micro-tiles with constant strides go "directly into the micro-kernel without writing to a temporary buffer". About 53% of tiles in the paper's example qualify, and more for large tensors. The rest use temp tile + scatter.
- Scatter vectors are generated late, once blocks have bounded size: at packing for A and B, just before the macrokernel for C. They are allocated from the pack-buffer pool, so there is no per-call unbounded malloc (§6.3).
- The loop nest is a C++ variadic template, `GEMM<PartitionN<NC>, PartitionK<KC>, MatrifyAndPackB<KP,NR>, PartitionM<MC>, MatrifyAndPackA<MR,KP>, MatrifyC<MR,NR>, PartitionN<NR>, PartitionM<MR>, MicroKernel<MR,NR>>`, with overloads chosen by matrix type (TensorMatrix → BlockScatterMatrix → packed Matrix) instead of BLIS's runtime control tree ([M18] §7.1, Fig. 5).

**Threading** (`T1/src/util/gemm_thread.hpp`):

- `make_gemm_thread_config` splits nthreads into ic×jc with `partition_2x2(nt, m·m_thread_ratio, n·n_thread_ratio)`.
- It carves ir out of ic and jr out of jc, up to `mr_max_thread` / `nr_max_thread`.
- `BLIS_{JC,IC,JR,IR}_NT` env vars override the result.
- There is no pc parallelism (kc gang = 1, `nodes/gemm.hpp`).
- The ratios are per-config data, but they sit outside the kernel pointers.

## 7b. TBLIS 2.x: a BLIS plugin

**Registration** (`T2/tblis/plugin/bli_plugin_tblis.cxx#L16`, `register_plugin()`):

1. `bli_init()`.
2. Allocate new ids through gks: `bli_gks_register_ukr2(&PACKM_BSMTC_UKR)`, `bli_gks_register_ukr(&GEMM_BSMTC_UKR)`, the MULT/REDUCE/SHIFT/TRANS kernels, and new blocksizes `MRT_BSZ`, `NRT_BSZ`, `KE_BSZ`.
3. For every compiled sub-configuration, call `plugin_init_<conf>_ref()`. That function looks up the BLIS context for the arch and calls `bli_cntx_set_ukr2` / `bli_cntx_set_ukr` / `bli_cntx_set_blksz` with reference `func_t`/`func2_t` tables built by `TBLIS_INIT_REF_KERNEL` (`T2/tblis/plugin/ref_kernels/plugin_init_ref.cxx`, `T2/tblis/plugin/kernel.hpp`).
4. The per-arch optimized `bli_plugin_init_<conf>.cxx` files exist but are **empty stubs**; none of them calls a `bli_cntx_set*` setter (checked with grep across `plugin/config/*`). The layout mirrors BLIS's `config_registry` and `bli_kernel_defs_<conf>.h`.

**Kernel interfaces** (`T2/tblis/plugin/bli_plugin_tblis.h#L34-L70`):

- `gemm_bsmtc_ft` is BLIS's gemm signature with scatter added for C:

      (m, n, k, alpha, a, b, beta, c, rs_c, rscat_c, cs_c, cscat_c, auxinfo, cntx)

- `packm_bsmtc_ft` takes `(conj, schema, panel_dim, panel_len, dim_max, len_max, bcast, kappa, c, rscat, rbs, cscat, cbs, p, ldp)`.

**How optimized GEMM is reached anyway.** The reference `gemm_bsmtc` (`T2/tblis/plugin/ref_kernels/gemm_bsmtc_ref.cxx`) is a scatter-aware wrapper around BLIS's own ukr:

- It fetches `bli_gemm_var_cntl_ukr(cntl)`, the row preference and MR/NR from `auxinfo->params` (the BLIS cntl node).
- If C has constant strides, it calls that ukr directly.
- Otherwise it calls the ukr with β = 0 into `ct[BLIS_STACK_BUF_MAX_SIZE/sizeof(T)]` and scatters through `rscat`/`cscat`.

So optimized SIMD comes from whatever BLIS selected for the architecture, including the 1m virtual ukr for complex. TBLIS itself registers only reference-level wrappers and packers.

**Complex and packing.**

- `tblis::mult` builds a normal BLIS `gemm_cntl_t` with `bli_gemm_cntl_init(bli_ind_oper_find_avail(BLIS_GEMM, dt) or BLIS_NAT, …)`, then swaps in its own pack variant (`packm_blk_bsmtc`), macrokernel (`gemm_ker_bsmtc`) and params (`bsmtc_params` for A, B, C) through the `bli_gemm_cntl_set_*` setters (`T2/tblis/frame/3t/dense/mult.cxx#L200-L335`).
- Because the schema comes from BLIS, the reference bsmtc packer implements `BLIS_PACKED_PANELS`, `_1R`, `_1E` and `_RO` (`T2/tblis/plugin/ref_kernels/packm_bsmtc_ref.cxx#L67`, `#L148`, `#L244`, `#L340`).
- This is how TBLIS 2.x picks up 1m for free: the tensor packing writes the 1e/1r format directly from scatter vectors.

**Runtime selection.**

- `tblis_config` is `typedef void` (`T2/tblis/frame/base/basic_types.h#L126`).
- `tblis_tensor_mult(comm, cntx, …)` ignores that argument and passes `bli_gks_query_cntx()` (`T2/tblis/frame/3t/mult.cxx#L22-L160`).
- So TBLIS 2.x inherits BLIS's architecture selection entirely, including CPUID detection, `BLIS_ARCH_TYPE` and `BLIS_ARCH_DEBUG`.

**Threading** (`T2/tblis/frame/base/thread.cxx#L110-L150`):

- `thread_blis` builds a local `rntm_t` from the global one and sets `num_threads` from the TBLIS communicator.
- It calls `bli_rntm_factorize(m, n, k)` and `bli_thrcomm_create`.
- Each TBLIS thread then enters `bli_l3_int` with its own `thrinfo`.

**[inference]** The plugin API used here (`bli_gks_register_ukr*`, `func2_t` ukr2s, the `gemm_cntl_t` setters, `bli_gemm_cntl_init` returning a transpose flag) is post-0.9 BLIS master. That is presumably why TBLIS pins a BLIS commit (`blis-git-tag` = `358e689c…`) rather than a release; I did not check whether a BLIS 2.0 release contains it.

---

# Part III — Design lessons for a Rust switchable engine

## What maps well

1. **A two-level split: immutable arch descriptor + per-plan control.** This is the core idea in both libraries.
   - *Level 1, selected once:* the equivalent of `cntx_t`. It holds, per datatype, the ukr pointer, packm pointers (native, 1er, ro), the row/column preference, and {def, max, multiple} blocksizes for MR/NR/KR/MC/KC/NC/BBM/BBN. In Rust: a `&'static ArchDesc` chosen through `OnceLock` from CPUID (priority scan à la TBLIS 1.x, or newest-first à la BLIS), overridable by env or API, with a guaranteed portable fallback. That fallback corresponds to `generic` in BLIS and `reference` with priority 0 in TBLIS 1.x.
   - *Level 2, per plan:* the equivalent of `gemm_cntl_t`. `bli_gemm_cntl_init` is the template. From (descriptor, dtypes, complex method, C storage) it derives, once:
     - whether to transpose, to match the kernel's C-storage preference;
     - the computation dtype (real for 1m);
     - the pack schemas for A and B (1e/1r assigned by `row_pref`);
     - the scaled blocksizes (`mr/mr_scale`, `kc/2`, PACKMR unscaled);
     - the concrete `ukr` plus the underlying `real_ukr`.

     Keep it immutable after planning and execute many times. This matches "function pointers selected once, immutable per plan".
2. **The complex method as data, not code paths.** 1m touches only three things: pack schema, blocksize scales, and which ukr pointer is called. The real kernel is reused untouched and the loop nest never branches on the method.
   - A Rust `enum ComplexMethod { Native, OneM, /* Planar, FourM, ThreeM */ }` can map to `(pack_fn_a, pack_fn_b, scales, ukr_fn)` in the planner.
   - Planar/split or 4m/3m would fit the same slots: 4m and 3m need extra packed copies or multiple ukr calls per tile. BLIS removed them for maintenance reasons, not because they didn't fit.
   - Registration must enforce BLIS's even-MR/NR check where 1m is allowed.
3. **Virtual microkernels as wrappers with the same signature.** Examples: the 1m ukr, the mixed-domain CCR/RCC/CRR wrappers, TBLIS's bsmtc scatter wrapper, TBLIS 1.x's flip. The wrapper finds its inner kernel through `auxinfo.params`, which is C's substitute for a closure. In Rust, a plan-owned struct `{inner: UkrFn, mr, nr, row_pref, …}` passed by reference does the same, with no type erasure beyond an `fn` pointer.
4. **Automatic reference-vs-optimized policy by comparing kernel identity.** `bli_ind_init` enables 1m only when the complex kernel is the reference one and the real kernel is optimized. A Rust descriptor can carry an explicit `KernelImpl::{Reference, Optimized, Virtual}` tag per entry instead of comparing pointers.
5. **Reference kernels generic over MR/NR.** BLIS compiles its reference kernels per configuration with constant MR/NR; TBLIS 1.x uses a `template<Config,T>` default kernel. In Rust: `fn ref_ukr<T, const MR: usize, const NR: usize>` with a generic runtime-MR/NR fallback, like BLIS `gemm_gen`. Every descriptor, including SIMD ones with no complex kernel, then has a complete table.
6. **Keep threading orthogonal.** In both libraries the loop ways (`jc, pc=1, ic, jr, ir`) live in a separate runtime object (`rntm_t`, or TBLIS's `gemm_thread_config` from `partition_2x2` plus per-config ratio hints) and are factorized from m, n, k. Kernel tables carry at most *hints*, like TBLIS 1.x `m_thread_ratio` / `nr_max_thread`.
7. **Scatter/tensor support as extra pack kernels plus a C-update wrapper.** The gemm ukr doesn't change:
   - Pack kernels are selected by input kind (strided, scatter, block-scatter, diagonal-scaled).
   - C uses a per-tile check: constant stride goes direct to the ukr, otherwise temp tile + scatter.
   - Scatter vectors are generated per MC/NC block from pooled workspace, not per call.

## Pitfalls to avoid

- **Global or thread-local mutable switches.** BLIS's induced-method enable state is `BLIS_THREAD_LOCAL`, the global rntm is per application thread, and `bli_thread_set_ways` and the env vars are read once. The same call can therefore behave differently depending on the calling thread. Prefer explicit plan parameters with an immutable process-default descriptor.
- **Macro-heavy registration.**
  - BLIS: `INSERT_GENTCONF` X-macros, `GENTFUNC*` per-type expansion, and variadic `(id, dt, fp, …, BLIS_VA_END)` setters, which already hit a variadic enum-width bug (see the comment on `kerid_t`, BLIS issue #839).
  - TBLIS 1.x: preprocessor "empty-argument" detection plus template traits.
  - In Rust, use plain `const` or `static` descriptor tables built by functions, and let the type system check signatures instead of casting through `void(*)()` (TBLIS `microkernel::call<T>`) or `void_fp`.
- **Runtime id allocation across contexts.** BLIS's plugin ids must come out identical in every compiled context (the `id != next_id` check). A Rust enum of kernel slots known at compile time avoids this.
- **Duplicate per-call derivation.** BLIS 0.9 cached induced contexts under a mutex; master recomputes the cntl tree on every call. A plan object amortizes that cost. **[inference]** This matters for small contractions.
- **Hard-coded temporary-tile caps.** TBLIS 1.x uses `p_ab[512]`. BLIS sizes `BLIS_STACK_BUF_MAX_SIZE` from SIMD register count × width and validates at registration that the register tile fits. Size the edge tile from the descriptor, or a const-generic bound, and assert at registration.
- **Edge-case contract drift.** BLIS moved edge handling into the ukr (m ≤ MR, n ≤ NR); TBLIS 1.x kept full-tile kernels plus an outer wrapper. Pick one contract. **[inference]** Full-tile kernels plus a shared framework wrapper keep SIMD kernels simpler and make portable/SIMD switching cheaper.
- **Preference handled only by transposing the whole problem.** This works for GEMM. With scatter C, a tile can have row stride 0 in the scatter sense (non-constant), so a fallback path is always needed.

## A minimal uniform microkernel contract (synthesis)

One signature for native, 1m-virtual and portable kernels:

- **Inputs:**
  - `k` (and optionally `m ≤ MR`, `n ≤ NR` if edge handling lives in the kernel);
  - `alpha: *const T`;
  - `a`: packed micropanel, column-major MR×k with leading dimension PACKMR;
  - `b`: packed micropanel, row-major k×NR with leading dimension PACKNR;
  - `beta: *const T`;
  - `c: *mut T` with `rs_c`, `cs_c`, where general stride must be accepted and the preferred orientation is declared in the descriptor;
  - `aux: &AuxInfo`.
- **AuxInfo:** `a_next`, `b_next` (prefetch); pack schemas; imaginary strides `is_a` / `is_b` (planar/split layouts would use these); panel strides; tile offsets; and an opaque `params` pointer through which virtual kernels reach the inner real kernel and its MR/NR/row_pref.
- **Descriptor-side metadata per kernel:**
  - MR, NR, PACKMR, PACKNR, KR, broadcast factors;
  - row/column preference;
  - datatype(s): computation type and C type;
  - implementation kind: reference, optimized, or virtual.

  The planner checks divisibility of MC/NC/KC against these at registration.
- **Scatter C** stays outside the kernel. The framework calls the kernel with β = 0 into an aligned MR×NR accumulator (a slice owned by the plan) and applies `c[rscat[i]+cscat[j]] = acc + β·c`. It skips that step when the tile's block-scatter strides are constant. BLIS's `gemm_bsmtc` in TBLIS 2.x shows this can be a thin wrapper with the same signature plus `rscat`/`cscat`.

---

# Part IV — Parallel partitioning boundaries (added scope)

Same pinned commits and bases (`B`, `T1`, `T2`) as above.

## IV.1 BLIS

**Which loops are parallel.**
- Of the five loops, four can be parallel: JC (n by NC), IC (m by MC), JR (n by NR) and IR (m by MR). PC (k by KC) is always 1, because iterations write the same C ([Multithreading.md](https://github.com/flame/blis/blob/913242ced922e9749983adb6b95fc620ddb895b2/docs/Multithreading.md) loop table).
- The ways come from `rntm_t.thrloop[]`. Sources: `BLIS_{JC,PC,IC,JR,IR}_NT`, or the automatic `bli_rntm_factorize(m, n, k)` (`B/frame/base/bli_rntm.c#L144-L165`).
- The sup path explicitly splits jc, pc, pb, ic, pa, jr and ir (`B/frame/3/bli_l3_thrinfo.c`, `bli_l3_sup_thrinfo_grow`).

**Communicator tree.**
- Each control-tree node records which loop's ways parallelize it via `BLIS_THREAD_{NC,KC,MC,NR,…}` bits, set by `bli_cntl_attach_sub_node` in `B/frame/3/gemm/bli_gemm_cntl.c#L470-L540`.
- `bli_l3_thrinfo_grow` walks the cntl tree and calls `bli_thrinfo_split(n_way, parent)` at each node (`B/frame/3/bli_l3_thrinfo.c#L39-L95`). The split rule (`B/frame/thread/bli_thrinfo.c#L158-L230`):
  - `child_nt = parent_nt / n_way`; the parent count must be divisible or BLIS aborts.
  - `child_tid = parent_tid % child_nt`, `work_id = parent_tid / child_nt`.
  - For `1 < n_way < parent_nt`, the parent's chief allocates the child `thrcomm_t`s and broadcasts them. `n_way == 1` reuses the parent communicator; `n_way == parent_nt` uses `BLIS_SINGLE_COMM`.
- `thrinfo_t` (`B/frame/thread/bli_thrinfo.h#L41-L75`) holds `comm`, `thread_id`, `n_way`, `work_id`, `free_comm`, `sba_pool`, `pba`, and a `mem_t mem` (the pack buffer this node owns).
- Threads are launched per call:
  - `bli_l3_thread_decorator` → `bli_thread_launch`, which is `omp parallel num_threads(nt)` (`B/frame/thread/bli_thread_openmp.c#L49`) or `pthread_create` (`B/frame/thread/bli_thread_pthreads.c#L100`).
  - It checks an `array_t` of per-thread sba pools out of the sba under a lock (`B/frame/3/bli_l3_decor.c#L161-L185`).

**Boundary alignment.**
- JC and IC use `bli_thread_range_ndim` / `_mdim`, which calls `bli_thread_range_sub(work_id, n_way, n, bf, handle_edge_low)` (`B/frame/thread/bli_thread_range.c#L38-L180`, `#L710-L830`).
- `bf` is the node's blocksize *multiple*, not MC/NC: `nr_def/nr_scale` for JC and `mr_def/mr_scale` for IC. These are the `b_mult` arguments to `bli_part_cntl_init_node` in `B/frame/3/gemm/bli_gemm_cntl.c`.
- So thread boundaries fall on MR/NR multiples. Each thread then walks its own subrange in MC/NC steps (`B/frame/3/gemm/bli_gemm_blk_var1.c`).
- Nothing aligns to cache lines. **[inference]** When MR·sizeof(T) is not a multiple of 64 B, two threads can share a C cache line at the boundary.
- Triangular and structured cases use `bli_thread_range_weighted_sub`, which balances area instead of width (`#L397`).

**Ragged edges.**
- Thread split: `bli_thread_range_sub` gives each thread `n_bf_whole/n_way` whole blocks. The remainder whole blocks go one each to the lowest (or highest) work_ids. The sub-`bf` leftover (`n % bf`) goes only to the last thread, or the first when `handle_edge_low`, i.e. BWD direction. The table in the comment at `#L55-L75` shows this.
- Cache-block level: `bli_determine_blocksize(direct, i, dim, b_alg, b_max)` returns the remainder whenever the rest of the range is ≤ `b_max`. So the last MC/NC/KC block can grow up to max instead of leaving a tiny tail (`B/frame/base/bli_blksz.c#L236-L290`).
- Register level: `m_iter = ceil(m/MR)`. The last microtile is partial, the ukr handles `m < MR`, and packing zero-pads.

**JR/IR partitioning** is chosen at configure time with `--thread-part-jrir=slab|rr|tlb`; the default is `slab` (`B/configure#L388-L420`).
- `slab`: contiguous ranges of micropanels, via `bli_thread_range_sub` with bf = 1.
- `rr`: start = tid, stride = nt.
- `tlb`: tile-level balancing at single-microtile granularity, `bli_thread_range_tlb_*`.

All three are defined in `B/frame/thread/bli_thread_range_slab_rr.h#L38-L100` and used in `B/frame/3/gemm/bli_gemm_ker_var2.c#L161-L200`. packm uses slab or rr to match (`B/frame/1m/packm/bli_packm_blk_var1.c#L135-L170`); tlb falls back to slab there.

**Barriers and sync points** (per call):
- `bli_thrinfo_split`: a broadcast (implied barrier) whenever a new communicator is created.
- `bli_packm_int`: a barrier *before* packing, so the previous consumers of the same buffer are done, and *after* packing, so the pack is complete before compute (`B/frame/1m/packm/bli_packm_int.c#L52`, `#L64`). The barrier scope is the packm node's communicator:
  - B panel: every thread in one JC group.
  - A block: every thread in one IC group.
- `bli_packm_alloc_ex`: broadcast + barrier, only when the buffer must grow (`B/frame/1m/packm/bli_packm_alloc.c#L78-L112`).
- There is no barrier after the macrokernel; the next `packm_int` pre-barrier serves that role.
- End of call: the thread decorator barriers before freeing the thrinfo tree, "prevents memory being released while others still use it" (`B/frame/3/bli_l3_decor.c#L84-L88`), then the OpenMP region or pthread join ends.

## IV.2 TBLIS

**TBLIS 1.x.**
- Communicator tree: `comm.gang(TCI_EVENLY, jc_nt)` → `kc` (always 1 way) → `ic_nt` → `jr_nt` → `ir_nt` (`T1/src/nodes/gemm.hpp#L126-L130`). The ways come from `make_gemm_thread_config` (§7a).
- Partition nodes (`T1/src/nodes/partm.hpp#L25-L60`) call `subcomm->distribute_over_gangs({len, M_iota}, …)`, where the grain `M_iota` is the register blocksize. So boundaries again fall on MR/NR multiples.
- The TCI split (`T1/src/external/tci/tci/communicator.c#L169-L185`) is `first = idx·ngrain/n`, `last = (idx+1)·ngrain/n`, in whole grains. The partial last grain falls in the final range, so remainder grains spread evenly with no low/high bias.
- Inside a gang the loop steps by `M_def`, and uses `M_max − M_def` overshoot to absorb a small tail. This is the same idea as BLIS max blocksizes.
- Packing splits the micropanel range over threads with `comm.distribute_over_threads({m, MR}, {k, KR}, …)`, a 2-D split in MR × KR grains (`T1/src/matrix/normal_matrix.hpp#L72`, `T1/src/matrix/block_scatter_matrix.hpp#L387`).
- Barriers: after pack and after the child run in `pack_and_run` (`T1/src/nodes/packm.hpp#L20-L75`). Building the block-scatter matrix is master-only then barrier: the master thread fills all scatter and block-scatter vectors by itself (`T1/src/matrix/block_scatter_matrix.hpp#L273-L281`).

**TBLIS 2.x.** It reuses the BLIS tree unchanged: `bli_l3_thrinfo_create(tid, gl_comm, array, &rntm, cntl)`, with ways from `bli_rntm_factorize` (`T2/tblis/frame/base/thread.cxx#L110-L140`). It adds two parallel steps of its own:
- Scatter vectors for C, per macrokernel call (`T2/tblis/frame/3m/gemm/gemm_ker_bsmtc.cxx#L170-L213`). Space comes from `bli_packm_alloc_ex(…, BLIS_BUFFER_FOR_GEN_USE, thread_par)`, so it is shared by the jr×ir team. `rscat`, `cscat`, `rbs` and `cbs` are filled cooperatively (slab over `ir_nt·jr_nt` threads with `bli_thread_range_sl`), followed by a barrier.
- Scatter vectors for A and B, in the pack routine (`T2/tblis/frame/1m/packm/packm_blk_bsmtc.cxx#L40-L115`). They sit in the *same* pack buffer, just after the panels. The buffer size is extended by `n_iter·(panel_dim_max+1) + k_blocks·(panel_len_block+1)` stride entries. They are filled slab-wise by all packing threads, then a barrier, then the micropanels are packed slab/rr.

**How tensor dimensions map onto the partition** (both versions).
- The partition runs purely in the *fused* matrix index. `m = ∏ len_AC · nblock_AC`, and likewise for n and k (`T2/tblis/frame/3t/dense/mult.cxx#L240-L260`).
- The loop code never sees tensor dimensions. A thread's range `[off, off+size)` is turned into scatter and block-scatter entries by `fill_block_scatter(…, nblock, block_off, ndim, len, stride, BS, off, size, scat, bs, pack_3d)`, which decodes linear indices into multi-indices and flags which BS-sized blocks have constant stride (`T2/tblis/frame/base/block_scatter.hpp#L19-L30`).
- **[inference]** Boundaries therefore land at MR/NR multiples of the fused index, not at tensor-mode boundaries. A block crossing a mode "jump" just gets `bs = 0` and takes the scatter path. [M18] §6.2 reports that about 53% of C micro-tiles are constant-stride in its small example and more for real sizes.
- Blocked, DPD and indexed tensors add an outer task layer: independent sub-contractions are scheduled with `comm.do_tasks_deferred(ntask, cost = dense_size·inout_ratio, …)` (`inout_ratio = 200000` in `T2/tblis/frame/base/thread.cxx#L104`; uses in `T2/tblis/frame/3t/indexed/mult.cxx#L98`, `#L203`).

---

# Part V — Buffer memory locality (added scope)

## V.1 BLIS

**Who allocates pack buffers.** There is one process-global packing block allocator (pba) with a mutex and three pools: A blocks, B panels, C panels (`B/frame/base/bli_pba.c#L39-L40`, struct in `B/frame/include/bli_type_defs.h#L1088`).
- The pool block size is set at init as the maximum over all datatypes of the max-blocksize-derived size (roughly MC_max × (KC_max + max(MR,NR)) × dtype size for A, scaled up by PACKMR/MR or PACKNR/NR so the MR and NR roles can swap for right-side trsm; the `+max(MR,NR)` is a "nudging" margin), "so that new pools do not need to be allocated if the user switches datatypes" (`#L388-L500`).
- The pools start with 0 blocks, `block_ptrs_len` 80/80/0 (`#L322-L361`). On checkout, an empty pool grows by one block, and a pool whose blocks are too small is reinitialized larger (`B/frame/base/bli_pool.c#L234-L280`).
- Pools are enabled by default: `--enable-pba-pools` and `--enable-sba-pools` default to yes (`B/configure#L3064-L3065`). With pools disabled, every request becomes an aligned `malloc` (`bli_pba.c#L95-L118`).
- A separate small-block allocator (sba) serves thrinfo and communicator structures. It is an `apool_t` of per-thread `pool_t` arrays, checked out per call under a lock (`B/frame/base/bli_sba.c#L36-L80`, `B/frame/3/bli_l3_decor.c#L161-L185`).

**Sharing between threads.**
- One A block per IC group, shared by that group's jr×ir threads. One B panel per JC group, shared by all ic×jr×ir threads in it. This follows from where the `pack_a` and `pack_b` nodes sit in the cntl tree.
- In `bli_packm_alloc_ex` only the *chief* of the packm communicator calls `bli_pba_acquire_m` (mutex-protected checkout). It broadcasts the `mem_t`; every member stores it in its `thrinfo.mem`, then everyone barriers (`B/frame/1m/packm/bli_packm_alloc.c#L57-L115`).
- The block stays attached to the thrinfo node for the whole call and is only re-acquired when a larger size is needed. At `bli_thrinfo_free` the chief releases it to the global pool (`B/frame/thread/bli_thrinfo.c#L143-L150`).
- Pool blocks are **reused across calls** and live until `bli_finalize`.

**Alignment.**
- Pool blocks are aligned to `BLIS_POOL_ADDR_ALIGN_SIZE_{A,B,C}` = `BLIS_PAGE_SIZE` = 4096, with offset `BLIS_POOL_ADDR_OFFSET_SIZE_*` = 0 by default (`B/frame/include/bli_kernel_macro_defs.h#L160`, `#L218-L235`). General allocations use `BLIS_POOL_ADDR_ALIGN_SIZE_GEN` (page).
- Only the *first* micropanel is page-aligned. Later micropanels are aligned to `sizeof(T)`, or to PACKMR·sizeof(T) when the alignment is a multiple of PACKMR ([KernelsHowTo.md](https://github.com/flame/blis/blob/913242ced922e9749983adb6b95fc620ddb895b2/docs/KernelsHowTo.md), "Alignment of a1 and b1").
- The pool allocator is `BLIS_MALLOC_POOL`, which is `hbw_malloc` when libmemkind is enabled (`B/frame/include/bli_kernel_macro_defs.h#L89-L120`).

**First touch and NUMA.**
- BLIS has no NUMA-aware code: grep finds no numa or first-touch logic in `frame/`. Blocks are malloc'd, not zeroed; `bli_pool.c` has no memset or calloc.
- **[inference]** Under Linux first-touch, a block's physical pages are placed by the *packing writes* of the first call that uses it. Those writes are spread across every thread in the pack communicator (slab of micropanels per thread). Later calls reuse the block wherever it was first placed, possibly by a different team.
- Thread placement is left to OpenMP or OS affinity. The docs only suggest `GOMP_CPU_AFFINITY` (Multithreading.md).

**Is the packer the consumer?** Generally not. **[inference, from the code]**
- The packing range is split over *all* `nt` members of the pack communicator (`bli_packm_blk_var1.c#L137-L145`). Consumption is split differently: the A block by the IR ways (usually 1, so every thread reads every A micropanel), the B panel by the JR ways only.
- So most micropanels are written by one core and read by others, through shared L2/L3. This is the intended design in [VZ+16]: the A block shared in L2 by the jr threads, the B panel in L3 by the ic group.

## V.2 TBLIS

**TBLIS 1.x.**
- Three global `MemoryPool`s, `BuffersForA(4096)`, `BuffersForB(4096)` and `BuffersForScatter(4096)`, each 4096-byte aligned (`T1/src/internal/3m/mult.cxx#L13-L14`, `T1/src/internal/3t/dense/mult.cxx#L19`).
- Each pool is a mutex-protected free list. A block is reused if it is big enough and aligned; otherwise it is freed and reallocated. Allocation uses `hbw_malloc` or `malloc`, and `flush()` frees everything (`T1/src/memory/memory_pool.hpp#L18-L125`).
- The pack node allocates once per call per gang. `comm.master()` calls `Pool.allocate(m_p·k_p + max(m_p,k_p)·TBLIS_MAX_UNROLL)` and broadcasts the pointer (`T1/src/nodes/packm.hpp#L79-L120`). Scatter buffers are allocated the same way (`T1/src/nodes/matrify.hpp#L60-L126`).
- Blocks return to the pool when the `Block` RAII handle is destroyed at the end of the call, so they are reused across calls.
- There is no NUMA handling. Packing writes are split over the gang's threads (`distribute_over_threads`); consumption happens in the jr/ir gangs.

**TBLIS 2.x.** It inherits everything from BLIS: the pba pools, chief allocation, page alignment and barriers. The scatter and block-scatter vectors for A and B are appended to the same pba pack block, and C's vectors come from a general-use pba allocation (§IV.2). So locality is identical to BLIS, and the scatter metadata is co-located with the panels it describes.

## V.3 Lessons for the Rust engine (partition and memory)

- Partition in whole MR/NR (register-multiple) grains per thread and let the last thread absorb the sub-grain tail, as BLIS `bli_thread_range_sub` and TCI `distribute` both do. At the cache-block level, use the def/max merge rule. If false sharing of C matters, add a cache-line-rounded grain for the JC/IC split **[inference]**.
- A pack buffer is owned by a *team* (the packm communicator), not by a thread:
  - the chief acquires it from a pool and broadcasts it;
  - barrier before and after packing;
  - it is reused within the call and returned to a process pool at the end.
- A Rust plan can pre-size these from the descriptor's max blocksizes. BLIS sizes pools from max(def, max) over all datatypes to avoid regrowth.
- NUMA is unaddressed in both libraries. A Rust engine that cares should decide first-touch explicitly, for example per-socket team pools where each team's own threads touch its buffer during a warm-up pack **[inference]**, and should not assume a packer-equals-consumer affinity.
- Put tensor scatter metadata in the same pooled block as the panels (TBLIS 2.x) and fill it cooperatively before a barrier (TBLIS 2.x). Avoid TBLIS 1.x's master-only fill, which serializes that step.
