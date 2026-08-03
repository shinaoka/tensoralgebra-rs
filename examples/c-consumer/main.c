/*
 * A C consumer of the TAPP surface, doing real work.
 *
 * This exists to make "a C or C++ project can link this" a *tested* claim
 * rather than an inferred one. `tests/abi_layout.rs` already re-declares the
 * whole interface and drives a contraction through it — but it does that from
 * Rust, so no C compiler has ever seen `include/tapp.h`, and a header that
 * disagreed with the library would not have been caught anywhere.
 *
 * So this program is compiled by a C compiler, against the shipped header,
 * linked against the built library, and checks numerical results rather than
 * just return codes. A wrong stride convention, a mis-declared prototype or a
 * complex layout mismatch all show up as a wrong number here.
 *
 * It covers: real and complex element types, non-trivial extents, alpha/beta,
 * an explicit C term, conjugation on an operand, the in-place refusal, and the
 * error-explanation path.
 */

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <tapp.h>

/* D is M x N, A is M x K, B is K x N, all row-major. */
#define M 3
#define K 4
#define N 2

static int failures = 0;

static void check(int cond, const char* what) {
    if (cond) {
        printf("  ok    %s\n", what);
    } else {
        printf("  FAIL  %s\n", what);
        failures++;
    }
}

/* Report a TAPP_error with the library's own explanation, and abort the run:
 * once a call has failed there is nothing meaningful to compare against. */
static void must(TAPP_error err, const char* what) {
    if (!TAPP_check_success(err)) {
        char msg[256];
        TAPP_explain_error(err, sizeof msg, msg);
        fprintf(stderr, "%s failed: %s (code %d)\n", what, msg, err);
        exit(1);
    }
}

/* Labels: D[i,j] = sum_k A[i,k] * B[k,j]. Mode 2 is the contracted one. */
static const int64_t idx_a[2] = {0, 2};
static const int64_t idx_b[2] = {2, 1};
static const int64_t idx_cd[2] = {0, 1};

static const int64_t ext_a[2] = {M, K};
static const int64_t ext_b[2] = {K, N};
static const int64_t ext_d[2] = {M, N};

static const int64_t str_a[2] = {K, 1};
static const int64_t str_b[2] = {N, 1};
static const int64_t str_d[2] = {N, 1};

/* ---------------------------------------------------------------- f64 case */

static void test_f64(TAPP_handle handle, TAPP_executor exec) {
    double a[M * K], b[K * N], c[M * N], d[M * N], want[M * N];
    const double alpha = 2.5, beta = -0.5;

    for (int i = 0; i < M * K; i++) a[i] = 0.5 * (i + 1);
    for (int i = 0; i < K * N; i++) b[i] = 1.0 - 0.25 * i;
    for (int i = 0; i < M * N; i++) c[i] = 3.0 + i;

    /* Reference, computed here so the check is independent of the engine. */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) acc += a[i * K + k] * b[k * N + j];
            want[i * N + j] = alpha * acc + beta * c[i * N + j];
        }
    }

    TAPP_tensor_info ia, ib, ic, id;
    must(TAPP_create_tensor_info(&ia, TAPP_F64, 2, ext_a, str_a), "create A");
    must(TAPP_create_tensor_info(&ib, TAPP_F64, 2, ext_b, str_b), "create B");
    must(TAPP_create_tensor_info(&ic, TAPP_F64, 2, ext_d, str_d), "create C");
    must(TAPP_create_tensor_info(&id, TAPP_F64, 2, ext_d, str_d), "create D");

    TAPP_tensor_product plan;
    must(TAPP_create_tensor_product(&plan, handle, TAPP_IDENTITY, ia, idx_a,
                                    TAPP_IDENTITY, ib, idx_b, TAPP_IDENTITY, ic,
                                    idx_cd, TAPP_IDENTITY, id, idx_cd,
                                    TAPP_DEFAULT_PREC),
         "plan f64");

    must(TAPP_execute_product(plan, exec, NULL, &alpha, a, b, &beta, c, d),
         "execute f64");

    int good = 1;
    for (int i = 0; i < M * N; i++)
        if (fabs(d[i] - want[i]) > 1e-12 * (1.0 + fabs(want[i]))) good = 0;
    check(good, "f64 D = alpha*A*B + beta*C matches a C reference loop");

    /* Round-trip the descriptor accessors while a live info is to hand. */
    int64_t got_ext[2] = {0, 0};
    TAPP_get_extents(ia, got_ext);
    check(TAPP_get_nmodes(ia) == 2 && got_ext[0] == M && got_ext[1] == K,
          "tensor-info accessors round-trip");

    must(TAPP_destroy_tensor_product(plan), "destroy plan");
    must(TAPP_destroy_tensor_info(ia), "destroy A");
    must(TAPP_destroy_tensor_info(ib), "destroy B");
    must(TAPP_destroy_tensor_info(ic), "destroy C");
    must(TAPP_destroy_tensor_info(id), "destroy D");
}

/* ---------------------------------------------------------------- c64 case */

/*
 * The complex case is the one that matters most for the ABI: it asserts that a
 * C `double _Complex*` is the *same bytes* as what the engine reads, with no
 * conversion. Conjugation on A is exercised because it is applied inside
 * packing rather than by a pre-pass, so a sign error would not show up
 * anywhere else in this program.
 */
static void test_c64_conj(TAPP_handle handle, TAPP_executor exec) {
    double _Complex a[M * K], b[K * N], c[M * N], d[M * N], want[M * N];
    const double _Complex alpha = 1.5 - 0.5 * I, beta = 0.25 + 0.75 * I;

    for (int i = 0; i < M * K; i++) a[i] = (0.5 * (i + 1)) + (0.25 * i) * I;
    for (int i = 0; i < K * N; i++) b[i] = (1.0 - 0.25 * i) + (0.5 - 0.1 * i) * I;
    for (int i = 0; i < M * N; i++) c[i] = (3.0 + i) - (0.5 * i) * I;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double _Complex acc = 0.0;
            for (int k = 0; k < K; k++)
                acc += conj(a[i * K + k]) * b[k * N + j];
            want[i * N + j] = alpha * acc + beta * c[i * N + j];
        }
    }

    TAPP_tensor_info ia, ib, ic, id;
    must(TAPP_create_tensor_info(&ia, TAPP_C64, 2, ext_a, str_a), "create A");
    must(TAPP_create_tensor_info(&ib, TAPP_C64, 2, ext_b, str_b), "create B");
    must(TAPP_create_tensor_info(&ic, TAPP_C64, 2, ext_d, str_d), "create C");
    must(TAPP_create_tensor_info(&id, TAPP_C64, 2, ext_d, str_d), "create D");

    TAPP_tensor_product plan;
    must(TAPP_create_tensor_product(&plan, handle, TAPP_CONJUGATE, ia, idx_a,
                                    TAPP_IDENTITY, ib, idx_b, TAPP_IDENTITY, ic,
                                    idx_cd, TAPP_IDENTITY, id, idx_cd,
                                    TAPP_DEFAULT_PREC),
         "plan c64");

    must(TAPP_execute_product(plan, exec, NULL, &alpha, a, b, &beta, c, d),
         "execute c64");

    int good = 1;
    for (int i = 0; i < M * N; i++)
        if (cabs(d[i] - want[i]) > 1e-12 * (1.0 + cabs(want[i]))) good = 0;
    check(good, "c64 with TAPP_CONJUGATE on A matches a C reference loop");

    must(TAPP_destroy_tensor_product(plan), "destroy plan");
    must(TAPP_destroy_tensor_info(ia), "destroy A");
    must(TAPP_destroy_tensor_info(ib), "destroy B");
    must(TAPP_destroy_tensor_info(ic), "destroy C");
    must(TAPP_destroy_tensor_info(id), "destroy D");
}

/* ------------------------------------------------------- documented refusals */

/*
 * Two behaviours the header promises in prose. If either stopped holding, a
 * caller would get a silently different answer rather than an error, so they
 * are worth a C-level check rather than only a Rust one.
 */
static void test_refusals(TAPP_handle handle, TAPP_executor exec) {
    double a[M * K] = {0}, b[K * N] = {0}, d[M * N] = {0};
    const double alpha = 1.0, nonzero_beta = 1.0;

    TAPP_tensor_info ia, ib, ic, id;
    must(TAPP_create_tensor_info(&ia, TAPP_F64, 2, ext_a, str_a), "create A");
    must(TAPP_create_tensor_info(&ib, TAPP_F64, 2, ext_b, str_b), "create B");
    must(TAPP_create_tensor_info(&ic, TAPP_F64, 2, ext_d, str_d), "create C");
    must(TAPP_create_tensor_info(&id, TAPP_F64, 2, ext_d, str_d), "create D");

    TAPP_tensor_product plan;
    must(TAPP_create_tensor_product(&plan, handle, TAPP_IDENTITY, ia, idx_a,
                                    TAPP_IDENTITY, ib, idx_b, TAPP_IDENTITY, ic,
                                    idx_cd, TAPP_IDENTITY, id, idx_cd,
                                    TAPP_DEFAULT_PREC),
         "plan");

    /* A null C with a non-zero beta is refused, not reinterpreted as C == D. */
    TAPP_error err = TAPP_execute_product(plan, exec, NULL, &alpha, a, b,
                                          &nonzero_beta, TAPP_IN_PLACE, d);
    check(!TAPP_check_success(err), "TAPP_IN_PLACE with beta != 0 is refused");

    /* Every attribute key is refused, but the symbols must exist to link. */
    void* slot = (void*)0x1234;
    check(TAPP_attr_get(0, 0, &slot) == TAPP_ERROR_UNSUPPORTED && slot == NULL,
          "attribute API links, refuses, and leaves a defined value");

    must(TAPP_destroy_tensor_product(plan), "destroy plan");
    must(TAPP_destroy_tensor_info(ia), "destroy A");
    must(TAPP_destroy_tensor_info(ib), "destroy B");
    must(TAPP_destroy_tensor_info(ic), "destroy C");
    must(TAPP_destroy_tensor_info(id), "destroy D");
}

int main(void) {
    printf("TAPP provider: %s\n", TAPP_implementation_name());

    TAPP_handle handle;
    TAPP_executor exec;
    must(TAPP_create_handle(&handle), "create handle");
    must(TAPP_create_executor(&exec), "create executor");

    test_f64(handle, exec);
    test_c64_conj(handle, exec);
    test_refusals(handle, exec);

    must(TAPP_destroy_executor(exec), "destroy executor");
    must(TAPP_destroy_handle(handle), "destroy handle");

    if (failures) {
        printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    printf("\nall checks passed\n");
    return 0;
}
