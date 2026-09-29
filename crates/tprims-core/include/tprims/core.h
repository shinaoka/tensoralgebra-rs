/* tprims C ABI: core (DLPack operands, status, executors). */
#ifndef TPRIMS_CORE_H
#define TPRIMS_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TPRIMS_ABI_VERSION 100 /* 0.1.0 */

/* DLPack 1.x (subset used by tprims), layout-compatible with dlpack.h. */
#ifndef DLPACK_VERSION_MAJOR
typedef struct { int32_t device_type; int32_t device_id; } DLDevice;
typedef struct { uint8_t code; uint8_t bits; uint16_t lanes; } DLDataType;
typedef struct {
  void *data; DLDevice device; int32_t ndim; DLDataType dtype;
  int64_t *shape; int64_t *strides; uint64_t byte_offset;
} DLTensor;
typedef struct { uint32_t major; uint32_t minor; } DLPackVersion;
typedef struct DLManagedTensorVersioned {
  DLPackVersion version; void *manager_ctx;
  void (*deleter)(struct DLManagedTensorVersioned *self);
  uint64_t flags; DLTensor dl_tensor;
} DLManagedTensorVersioned;
#define kDLCPU 1
#define kDLCUDAHost 3
#define kDLFloat 2
#define kDLComplex 5
#define DLPACK_FLAG_BITMASK_READ_ONLY (1ULL << 0)
#endif

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

tprims_tensor tprims_tensor_borrow_versioned(DLManagedTensorVersioned *m);
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
