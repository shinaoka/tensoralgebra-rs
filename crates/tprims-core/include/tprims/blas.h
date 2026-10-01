/* tprims C ABI: BLAS-like operations. */
#ifndef TPRIMS_BLAS_H
#define TPRIMS_BLAS_H
#include "tprims/core.h"
#ifdef __cplusplus
extern "C" {
#endif

/* `exec` is any TAPP_executor of this library: 0 (serial), one from
   TAPP_create_executor, or one from tprims_tapp_executor_create_rayon.
   C = alpha * op(A) * op(B) + beta * C; rank-2 operands of C's dtype. */
tprims_status tprims_blas_gemm(TAPP_executor exec, const void *alpha,
                               tprims_tensor a, int32_t conj_a,
                               tprims_tensor b, int32_t conj_b,
                               const void *beta, tprims_tensor c);

/* Batched GEMM over [rows, cols, batch]; strategy 0 auto, 1 faer loop,
   2 TBLIS-style; *selected (if non-null): 0/1 faer (serial/spread items),
   2/3 TBLIS (serial/spread items). */
tprims_status tprims_blas_gemm_batched(TAPP_executor exec, const void *alpha,
                                       tprims_tensor a, int32_t conj_a,
                                       tprims_tensor b, int32_t conj_b,
                                       const void *beta, tprims_tensor c,
                                       int32_t strategy, int32_t *selected);
#ifdef __cplusplus
}
#endif
#endif
