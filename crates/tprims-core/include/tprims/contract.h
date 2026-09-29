/* tprims C ABI: binary contraction plans (dot_general semantics; output
   axes [lhs_free..., rhs_free..., batch...]). */
#ifndef TPRIMS_CONTRACT_H
#define TPRIMS_CONTRACT_H
#include "tprims/core.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TPRIMS_NO_MATERIALIZE 1u

typedef struct {
  const size_t *lhs_contract; const size_t *rhs_contract; size_t n_contract;
  const size_t *lhs_batch; const size_t *rhs_batch; size_t n_batch;
} tprims_dot_general;

typedef struct tprims_contract_plan tprims_contract_plan;

/* strategy 0 auto, 1 permute + batched GEMM, 2 TBLIS-style. */
tprims_status tprims_contract_plan_create(const tprims_dot_general *cfg,
                                          tprims_tensor a, tprims_tensor b, tprims_tensor c,
                                          int32_t conj_a, int32_t conj_b,
                                          int32_t strategy, uint32_t flags,
                                          tprims_contract_plan **out);
tprims_status tprims_contract_plan_execute(const tprims_contract_plan *plan, tprims_exec *exec,
                                           const void *alpha, tprims_tensor a, tprims_tensor b,
                                           const void *beta, tprims_tensor c);
/* 0 permute+GEMM without copies, 1 with copies, 2 TBLIS-style, 3 elementwise. */
int32_t tprims_contract_plan_selected(const tprims_contract_plan *plan);
void tprims_contract_plan_destroy(tprims_contract_plan *plan);
#ifdef __cplusplus
}
#endif
#endif
