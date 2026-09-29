/* tprims C ABI: core (DLPack operands, status, executors). */
#ifndef TPRIMS_CORE_H
#define TPRIMS_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TPRIMS_ABI_VERSION 100 /* 0.1.0 */

/* DLPack 1.x. The vendored copy (v1.1, Apache-2.0, include/dlpack/) is used
   unless the host already included its own dlpack.h (same include guard). */
#include "dlpack/dlpack.h"

/* Operand contract for every tprims call:
   - tensors are CPU (kDLCPU or kDLCUDAHost), lanes == 1, f32/f64/complex64/
     complex128, data + byte_offset aligned to the element's real type;
   - the whole span from the lowest to the highest addressed element (padding
     included) is initialized memory that no other thread writes during the
     call, and an output is not accessed by anyone else during the call;
   - an output whose span overlaps an input's span is rejected
     (TPRIMS_ERR_ALIASED), even when the element sets are disjoint;
   - DLManagedTensorVersioned with a major version other than 1 is rejected. */

typedef int32_t tprims_status;
#define TPRIMS_OK 0
#define TPRIMS_ERR_INVALID_ARGUMENT 1
#define TPRIMS_ERR_SHAPE 2
#define TPRIMS_ERR_DTYPE 3
#define TPRIMS_ERR_DEVICE 4
#define TPRIMS_ERR_READ_ONLY 5
#define TPRIMS_ERR_ALIASED 6
#define TPRIMS_ERR_WOULD_MATERIALIZE 7
#define TPRIMS_ERR_NUMERICAL 8
#define TPRIMS_BUSY 9
#define TPRIMS_ERR_WOULD_DEADLOCK 10
#define TPRIMS_ERR_CLOSED 11
#define TPRIMS_ERR_PANIC 12
#define TPRIMS_ERR_INTERNAL 13

/* Borrowed operand: valid for the duration of a call; never freed by tprims. */
typedef struct { DLTensor *view; uint64_t flags; } tprims_tensor;

tprims_tensor tprims_tensor_borrow_versioned(struct DLManagedTensorVersioned *m);
tprims_tensor tprims_tensor_borrow_raw(DLTensor *t, uint64_t flags);

const char *tprims_last_error(void);
uint32_t tprims_abi_version(void);
int32_t tprims_has_part(const char *part);

typedef struct tprims_exec tprims_exec;
typedef struct { size_t stack_size; } tprims_rayon_opts;

tprims_exec *tprims_exec_serial(void);
tprims_exec *tprims_exec_rayon_create(size_t nthreads, const tprims_rayon_opts *opts);
size_t tprims_exec_num_threads(tprims_exec *exec);
tprims_status tprims_exec_set_budget(tprims_exec *exec, size_t max_threads);
tprims_status tprims_exec_close(tprims_exec *exec);
void tprims_exec_retain(tprims_exec *exec);
void tprims_exec_release(tprims_exec *exec);

#ifdef __cplusplus
}
#endif
#endif
