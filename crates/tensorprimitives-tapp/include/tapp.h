/*
 * tapp.h — the C ABI exported by `tensorprimitives-tapp`.
 *
 * This declares the Tensor Algebra Processing Primitives interface
 * (arXiv:2601.07827, https://github.com/TAPPorg/reference-implementation) as
 * implemented by this crate, so that a C or C++ project can link the produced
 * `cdylib`/`staticlib` without obtaining anything else.
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * ---------------------------------------------------------------------------
 * Provenance, and why this file exists at all
 * ---------------------------------------------------------------------------
 *
 * TAPP is a standard, and the natural thing for a provider to do is tell
 * callers to include the standard's own headers under `api/include/tapp`.
 * That is still
 * supported and still correct — this file declares the same ABI, and if you
 * already have the upstream headers, use them.
 *
 * It is not sufficient, though, because the upstream reference implementation
 * has **no releases and no tags** (see D35 in `DECISIONS.md`), so "the header"
 * is an untagged `main`. A consumer that pins this crate would be pinning its
 * declarations to a moving target, and the drift would be silent — precisely
 * the failure mode TBLIS's `type_t` enumerator swap already cost this project
 * once. Shipping a header that versions with the implementation removes that.
 *
 * Every prototype below was checked against the upstream headers at
 * `TAPPorg/reference-implementation@main`, and is *independently* pinned by
 * `tests/abi_layout.rs`, which re-declares the whole interface in Rust and
 * drives a complete contraction through it. A signature that disagrees with
 * this file is a link or behaviour failure in that test, not a downstream
 * surprise. The C side of the same claim is checked by `examples/c-consumer`,
 * which includes *this* header and verifies numerical results.
 *
 * Two deliberate departures from upstream, both documented where they occur:
 *   - `TAPP_IN_PLACE`, which upstream leaves an open `//TODO`.
 *   - the non-zero `TAPP_error` codes, which upstream does not specify at all.
 *
 * ---------------------------------------------------------------------------
 * Linking
 * ---------------------------------------------------------------------------
 *
 *   cargo build --release -p tensorprimitives-tapp
 *   cc myprog.c -I<this dir> \
 *      -L target/release -ltensorprimitives_tapp -Wl,-rpath,$PWD/target/release
 *
 * Prefer the `cdylib` (`libtensorprimitives_tapp.so` / `.dylib`) over the
 * `staticlib`: the static archive bundles the Rust standard library, needs
 * `-lpthread -ldl -lm` on Linux, and exports internal symbols that collide if
 * two Rust static libraries end up in one binary.
 *
 * From CMake, prefer corrosion — see `examples/c-consumer/CMakeLists.txt`,
 * which handles the platform link line for you.
 *
 * Nothing here requires a Rust toolchain at run time, and nothing here
 * involves the C++ ABI: no name mangling, no exceptions cross this boundary,
 * no libstdc++ version coupling.
 */

#ifndef TENSORPRIMITIVES_TAPP_H
#define TENSORPRIMITIVES_TAPP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef TAPP_EXPORT
#define TAPP_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------- version */

/*
 * The version of *this header*. `TAPP_implementation_version()` below returns
 * the version of the library that was actually linked. They are the same file's
 * two ends and they can disagree — a distribution that ships the header and the
 * shared library as separate packages, a stale `-I`, an `LD_PRELOAD` — so both
 * exist and `examples/c-consumer` checks that they agree.
 *
 * These track the crate version, which is `MIT OR Apache-2.0`-licensed
 * `tensorprimitives-tapp`, not any version of the TAPP standard: TAPP has no
 * releases and no tags at all, which is why this header exists (see above).
 */
#define TAPP_VERSION_MAJOR 0
#define TAPP_VERSION_MINOR 1
#define TAPP_VERSION_PATCH 0
#define TAPP_VERSION_STRING "0.1.0"

/* Non-zero if this header is at least the given version. */
#define TAPP_VERSION_AT_LEAST(major, minor, patch)                             \
    ((TAPP_VERSION_MAJOR) > (major) ||                                         \
     ((TAPP_VERSION_MAJOR) == (major) &&                                       \
      ((TAPP_VERSION_MINOR) > (minor) ||                                       \
       ((TAPP_VERSION_MINOR) == (minor) && (TAPP_VERSION_PATCH) >= (patch)))))

/* ------------------------------------------------------------------ types */

/*
 * Every TAPP handle is an `intptr_t`. In this implementation they are
 * `Box::into_raw` pointers, so they are process-local: do not serialise one,
 * send one to another process, or mix one with a handle from a different TAPP
 * provider linked into the same binary.
 */
typedef intptr_t TAPP_handle;
typedef intptr_t TAPP_executor;
typedef intptr_t TAPP_status;
typedef intptr_t TAPP_tensor_info;
typedef intptr_t TAPP_tensor_product;
typedef intptr_t TAPP_attr;

typedef int TAPP_error;
typedef int TAPP_datatype;
typedef int TAPP_prectype;
typedef int TAPP_element_op;

/* -------------------------------------------------------------- datatypes */

enum {
    TAPP_F32 = 0,  /* 32-bit IEEE-754 real                                  */
    TAPP_F64 = 1,  /* 64-bit IEEE-754 real                                  */
    TAPP_C32 = 2,  /* interleaved complex float,  == C99 `float _Complex`   */
    TAPP_C64 = 3,  /* interleaved complex double, == C99 `double _Complex`  */
    TAPP_F16 = 4,  /* rejected by this implementation (TAPP_ERROR_DATATYPE) */
    TAPP_BF16 = 5, /* rejected by this implementation (TAPP_ERROR_DATATYPE) */

    /* Upstream aliases. */
    TAPP_FLOAT = TAPP_F32,
    TAPP_DOUBLE = TAPP_F64,
    TAPP_SCOMPLEX = TAPP_C32,
    TAPP_DCOMPLEX = TAPP_C64
};

/*
 * `TAPP_C32`/`TAPP_C64` buffers are interleaved with the real part first, and
 * are layout-identical to C99 `float _Complex` / `double _Complex`. Passing a
 * `double _Complex*` as the `void*` data pointer is a plain reinterpret cast
 * with no conversion. Pinned by `complex_datatypes_are_interleaved_and_c_compatible`.
 */

/*
 * Computational precision. This implementation **accepts any value and
 * ignores it**, computing at the storage precision of the operands; it is
 * declared so that portable caller code compiles. Pass `TAPP_DEFAULT_PREC`.
 *
 * Values 2 and 3 are genuinely absent from the upstream enum — complex has no
 * separate precision enumerator — rather than omitted here.
 */
enum {
    TAPP_DEFAULT_PREC = -1,
    TAPP_F32F32_ACCUM_F32 = TAPP_F32,
    TAPP_F64F64_ACCUM_F64 = TAPP_F64,
    TAPP_F16F16_ACCUM_F16 = TAPP_F16,
    TAPP_F16F16_ACCUM_F32 = 5,
    TAPP_BF16BF16_ACCUM_F32 = 6
};

/* Per-operand element operation. Conjugation is free here: it is folded into
 * packing as a negation of the imaginary plane. */
enum { TAPP_IDENTITY = 0, TAPP_CONJUGATE = 1 };

/* ----------------------------------------------------------------- errors */

/*
 * Upstream `error.h` is a bare `typedef int TAPP_error` and fixes **only**
 * `TAPP_SUCCESS == 0`. Every non-zero code below is therefore this
 * implementation's own numbering. Portable code must test success with
 * `TAPP_check_success()` rather than comparing against these; they are pinned
 * by `enumerator_values_match_the_upstream_headers` so that callers who do
 * depend on them are not broken silently.
 */
enum {
    TAPP_SUCCESS = 0,
    TAPP_ERROR_NULL = 1,        /* null pointer, or a zero handle            */
    TAPP_ERROR_DATATYPE = 2,    /* unsupported or mismatched element type    */
    TAPP_ERROR_SHAPE = 3,       /* extents disagree, or an extent overflows  */
    TAPP_ERROR_LABELS = 4,      /* index labels are inconsistent             */
    TAPP_ERROR_UNSUPPORTED = 5, /* well-formed, not supported (see below)    */
    TAPP_ERROR_INTERNAL = 6     /* a bug, or an allocation failure           */
};

TAPP_EXPORT bool TAPP_check_success(TAPP_error error);

/*
 * Write a human-readable explanation of `error` into `message`, truncated to
 * `maxlen` bytes including the NUL. Returns the length that would have been
 * written had there been room, so a return >= maxlen means truncation.
 */
TAPP_EXPORT size_t TAPP_explain_error(TAPP_error error, size_t maxlen, char* message);

/* ---------------------------------------------------------------- handles */

/*
 * The handle discipline, which every call below inherits:
 *
 *   - A handle argument must be either 0 or a live value from the matching
 *     `TAPP_create_*` that has not been destroyed. 0 is rejected with
 *     `TAPP_ERROR_NULL`; a stale or foreign non-zero value is undefined
 *     behaviour, because nothing distinguishes it from a live one.
 *   - Each handle may be destroyed once, with no other call in flight against
 *     it. Nothing here is internally synchronised.
 *   - Data pointers must be valid for every offset the extents and strides in
 *     the tensor infos generate.
 */

TAPP_EXPORT TAPP_error TAPP_create_handle(TAPP_handle* handle);
TAPP_EXPORT TAPP_error TAPP_destroy_handle(TAPP_handle handle);

TAPP_EXPORT TAPP_error TAPP_create_executor(TAPP_executor* exec);
TAPP_EXPORT TAPP_error TAPP_destroy_executor(TAPP_executor exec);

/*
 * `TAPP_execute_product` accepts a null `status` out-parameter, which is what
 * this implementation expects: upstream defines no semantics for the status
 * object beyond its destructor, so nothing is produced to be destroyed. If you
 * do receive a non-zero status, destroy it with this.
 */
TAPP_EXPORT TAPP_error TAPP_destroy_status(TAPP_status status);

/*
 * Attributes. Upstream `attributes.h` declares these but specifies no keys, so
 * this implementation exports them and refuses every key with
 * `TAPP_ERROR_UNSUPPORTED`. `TAPP_attr_get` always stores a defined value
 * (NULL) before returning, so a caller that ignores the return code does not
 * read an uninitialised slot.
 */
TAPP_EXPORT TAPP_error TAPP_attr_set(TAPP_attr attr, int key, void* value);
TAPP_EXPORT TAPP_error TAPP_attr_get(TAPP_attr attr, int key, void** value);
TAPP_EXPORT TAPP_error TAPP_attr_clear(TAPP_attr attr, int key);

/* ----------------------------------------------------------------- tensors */

/*
 * Describe one tensor: element type, number of modes, extents, and strides in
 * **elements** (not bytes). `strides` may be null only when `nmode == 0`.
 * The descriptor does not own or reference the data; buffers are supplied at
 * execute time.
 */
TAPP_EXPORT TAPP_error TAPP_create_tensor_info(TAPP_tensor_info* info,
                                               TAPP_datatype type,
                                               int nmode,
                                               const int64_t* extents,
                                               const int64_t* strides);

TAPP_EXPORT TAPP_error TAPP_destroy_tensor_info(TAPP_tensor_info info);

TAPP_EXPORT int TAPP_get_nmodes(TAPP_tensor_info info);
TAPP_EXPORT TAPP_error TAPP_set_nmodes(TAPP_tensor_info info, int nmodes);

/* `extents`/`strides` must have room for `TAPP_get_nmodes(info)` entries. */
TAPP_EXPORT void TAPP_get_extents(TAPP_tensor_info info, int64_t* extents);
TAPP_EXPORT TAPP_error TAPP_set_extents(TAPP_tensor_info info, const int64_t* extents);
TAPP_EXPORT void TAPP_get_strides(TAPP_tensor_info info, int64_t* strides);
TAPP_EXPORT TAPP_error TAPP_set_strides(TAPP_tensor_info info, const int64_t* strides);

/* ---------------------------------------------------------------- products */

/*
 * Plan  D := alpha * op_A(A) * op_B(B) + beta * op_C(C).
 *
 * Each `idx_*` is an array of `nmode` integer labels, one per mode of that
 * tensor; modes sharing a label are contracted or batched according to where
 * the label appears. A label array may be null only for a scalar (`nmode == 0`).
 *
 * Supported (see the crate docs for the full table):
 *   - case 1 simple contraction, case 2 Hadamard/batch, case 3 repeated
 *     indices (diagonals), case 4 isolated input indices (reductions)
 *   - `TAPP_CONJUGATE` on any operand, at no cost
 *
 * Rejected with `TAPP_ERROR_UNSUPPORTED` or `TAPP_ERROR_DATATYPE`:
 *   - case 5 isolated *output* indices (broadcast), which TAPP permits an
 *     implementation to refuse
 *   - mixed *storage* types across operands: this engine plans one element
 *     type per product, so all four infos must agree
 *
 * `prec` is accepted and ignored; computation happens at storage precision.
 */
TAPP_EXPORT TAPP_error TAPP_create_tensor_product(TAPP_tensor_product* plan,
                                                  TAPP_handle handle,
                                                  TAPP_element_op op_A,
                                                  TAPP_tensor_info A,
                                                  const int64_t* idx_A,
                                                  TAPP_element_op op_B,
                                                  TAPP_tensor_info B,
                                                  const int64_t* idx_B,
                                                  TAPP_element_op op_C,
                                                  TAPP_tensor_info C,
                                                  const int64_t* idx_C,
                                                  TAPP_element_op op_D,
                                                  TAPP_tensor_info D,
                                                  const int64_t* idx_D,
                                                  TAPP_prectype prec);

TAPP_EXPORT TAPP_error TAPP_destroy_tensor_product(TAPP_tensor_product plan);

/*
 * `alpha` and `beta` point at one scalar of the product's element type — for
 * `TAPP_C64`, a `double _Complex`, not a `double`.
 *
 * `C` may be `TAPP_IN_PLACE` (null), meaning "no C term". This implementation
 * accepts that **only with `beta == 0`**: a non-zero `beta` against a null `C`
 * is refused with `TAPP_ERROR_UNSUPPORTED` rather than being reinterpreted as
 * `C == D`, because guessing which the caller meant is how a silent wrong
 * answer happens. Pass `C = D` explicitly for accumulation in place.
 *
 * `status` may be null, and passing null is recommended — see
 * `TAPP_destroy_status`.
 */
TAPP_EXPORT TAPP_error TAPP_execute_product(TAPP_tensor_product plan,
                                            TAPP_executor exec,
                                            TAPP_status* status,
                                            const void* alpha,
                                            const void* A,
                                            const void* B,
                                            const void* beta,
                                            const void* C,
                                            void* D);

/*
 * The same product over `num_batches` independent buffer sets. `alpha` and
 * `beta` are single scalars shared by every batch; `A`, `B`, `C` and `D` are
 * arrays of `num_batches` pointers.
 */
TAPP_EXPORT TAPP_error TAPP_execute_batched_product(TAPP_tensor_product plan,
                                                    TAPP_executor exec,
                                                    TAPP_status* status,
                                                    int num_batches,
                                                    const void* alpha,
                                                    const void** A,
                                                    const void** B,
                                                    const void* beta,
                                                    const void** C,
                                                    void** D);

/*
 * Upstream `product.h` leaves `TAPP_IN_PLACE` an open `//TODO`, so this
 * definition is this implementation's. It is guarded: if you include an
 * upstream header that defines it first, that definition wins and this one is
 * skipped.
 */
#ifndef TAPP_IN_PLACE
#define TAPP_IN_PLACE NULL
#endif

/* -------------------------------------------------------------- extensions */

/*
 * Not part of TAPP. Returns a static NUL-terminated name identifying the
 * linked provider, so a program that could be built against several TAPP
 * implementations can log which one it actually got. Never returns null; do
 * not free the result.
 */
TAPP_EXPORT const char* TAPP_implementation_name(void);

/*
 * Not part of TAPP. Returns the `"major.minor.patch"` version of the linked
 * library as a static NUL-terminated string. Never returns null; do not free the
 * result.
 *
 * Compare against `TAPP_VERSION_STRING` to detect a header that does not match
 * the library it is describing, which is the failure a separately-packaged
 * header and shared library make possible.
 */
TAPP_EXPORT const char* TAPP_implementation_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TENSORPRIMITIVES_TAPP_H */
