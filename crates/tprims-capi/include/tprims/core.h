/* tprims C ABI: core (DLPack operands, status codes). Executors are the TAPP
   executors of <tapp.h>; the Rayon extension is in <tprims/tapp_ext.h>. */
#ifndef TPRIMS_CORE_H
#define TPRIMS_CORE_H

#include <stddef.h>
#include <stdint.h>

/* The pinned upstream TAPP headers: include/tapp.h and the include/tapp directory. */
#include "tapp.h"

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
#define TPRIMS_BUSY 9               /* executor has calls in flight */
#define TPRIMS_ERR_WOULD_DEADLOCK 10 /* destroyed from its own worker */
/* 11 retired (was TPRIMS_ERR_CLOSED) */
#define TPRIMS_ERR_PANIC 12
#define TPRIMS_ERR_INTERNAL 13
#define TPRIMS_ERR_LABELS 14        /* labels do not describe a contraction */
#define TPRIMS_ERR_UNSUPPORTED 15   /* declined precision, element op, index class */

/* TAPP_error values returned by libtprims are these statuses: zero is
   success (TAPP_check_success), the others are provider-defined. */

/* Borrowed operand: valid for the duration of a call; never freed by tprims. */
typedef struct { DLTensor *view; uint64_t flags; } tprims_tensor;

tprims_tensor tprims_tensor_borrow_versioned(struct DLManagedTensorVersioned *m);
tprims_tensor tprims_tensor_borrow_raw(DLTensor *t, uint64_t flags);

const char *tprims_last_error(void);
uint32_t tprims_abi_version(void);
int32_t tprims_has_part(const char *part);

/* Library identity (non-standard TAPP extensions): a static NUL-terminated name
   and the crate version "major.minor.patch" of the loaded library. Never NULL;
   do not free. Compare with TPRIMS_ABI_VERSION to detect a header and a library
   from different versions. */
const char *TAPP_implementation_name(void);
const char *TAPP_implementation_version(void);

#ifdef __cplusplus
}
#endif
#endif
