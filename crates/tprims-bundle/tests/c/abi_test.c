/* End-to-end test of libtprims from C: zero copy, layouts, pools, cross-part
   handles, error reporting. Exit status 0 on success. */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "tprims/tprims.h"

#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__, msg, tprims_last_error()); return 1; } } while (0)

static DLTensor f64_tensor(double *data, int32_t ndim, int64_t *shape, int64_t *strides) {
  DLTensor t;
  memset(&t, 0, sizeof t);
  t.data = data; t.device.device_type = kDLCPU; t.ndim = ndim;
  t.dtype.code = kDLFloat; t.dtype.bits = 64; t.dtype.lanes = 1;
  t.shape = shape; t.strides = strides; t.byte_offset = 0;
  return t;
}

int main(void) {
  CHECK(tprims_abi_version() == TPRIMS_ABI_VERSION, "abi version");
  CHECK(tprims_has_part("blas") && tprims_has_part("contract") && tprims_has_part("core"), "parts");
  CHECK(!tprims_has_part("linalg"), "linalg not in this slice");

  /* A: 2x3 row-major (NULL strides), B: 3x2 column-major, C: 2x2 column-major. */
  double a[6] = {1, 3, 5, 2, 4, 6}; /* rows [1,3,5], [2,4,6] */
  double b[6] = {1, 3, 5, 2, 4, 6}; /* columns [1,3,5], [2,4,6] */
  double c[4] = {NAN, NAN, NAN, NAN};
  int64_t sa[2] = {2, 3}, sb[2] = {3, 2}, sc[2] = {2, 2};
  int64_t stb[2] = {1, 3}, stc[2] = {1, 2};
  DLTensor ta = f64_tensor(a, 2, sa, NULL), tb = f64_tensor(b, 2, sb, stb), tc = f64_tensor(c, 2, sc, stc);
  double one = 1.0, zero = 0.0;

  tprims_exec *pool = tprims_exec_rayon_create(4, NULL);
  CHECK(pool != NULL, "pool create");
  CHECK(tprims_exec_num_threads(pool) == 4, "pool size");
  tprims_exec *serial = tprims_exec_serial();
  tprims_exec *execs[2] = {serial, pool};
  for (int i = 0; i < 2; i++) {
    tprims_status st = tprims_blas_gemm(execs[i], &one, tprims_tensor_borrow_raw(&ta, 0), 0,
                                        tprims_tensor_borrow_raw(&tb, 0), 0, &zero,
                                        tprims_tensor_borrow_raw(&tc, 0));
    CHECK(st == TPRIMS_OK, "gemm");
    /* written in place into the caller's buffer (zero copy) */
    CHECK(c[0] == 35 && c[1] == 44 && c[2] == 44 && c[3] == 56, "gemm values");
  }

  /* Same product through a contraction plan, executed on the pool: the
     executor handle from core is accepted by the contract part. */
  size_t lc[1] = {1}, rc[1] = {0};
  tprims_dot_general cfg = {lc, rc, 1, NULL, NULL, 0};
  tprims_contract_plan *plan = NULL;
  memset(c, 0, sizeof c);
  CHECK(tprims_contract_plan_create(&cfg, tprims_tensor_borrow_raw(&ta, 0), tprims_tensor_borrow_raw(&tb, 0),
                                    tprims_tensor_borrow_raw(&tc, 0), 0, 0, 0, TPRIMS_NO_MATERIALIZE, &plan) == TPRIMS_OK,
        "plan create");
  CHECK(tprims_contract_plan_selected(plan) == 0, "copy-free plan");
  CHECK(tprims_contract_plan_execute(plan, pool, &one, tprims_tensor_borrow_raw(&ta, 0), tprims_tensor_borrow_raw(&tb, 0),
                                     &zero, tprims_tensor_borrow_raw(&tc, 0)) == TPRIMS_OK,
        "plan execute");
  CHECK(c[0] == 35 && c[3] == 56, "contract values");
  tprims_contract_plan_destroy(plan);

  /* Complex operands whose byte_offset is one real, not one element (#16):
     complex elements align to their real type, so these are valid. */
  {
    int64_t s1[2] = {1, 1};
    double za[4] = {0, 2, 1, 0}, zb[4] = {0, 3, 4, 0}, zc[4] = {0, 0, 0, 0};
    double zone[2] = {1, 0}, zzero[2] = {0, 0};
    DLTensor x[3];
    double *zd[3] = {za, zb, zc};
    for (int i = 0; i < 3; i++) {
      x[i] = f64_tensor(zd[i], 2, s1, NULL);
      x[i].dtype.code = kDLComplex; x[i].dtype.bits = 128; x[i].byte_offset = 8;
    }
    /* (2+1i)(3+4i) = 2+11i */
    CHECK(tprims_blas_gemm(serial, zone, tprims_tensor_borrow_raw(&x[0], 0), 0, tprims_tensor_borrow_raw(&x[1], 0), 0,
                           zzero, tprims_tensor_borrow_raw(&x[2], 0)) == TPRIMS_OK,
          "complex128 byte_offset 8");
    CHECK(zc[1] == 2 && zc[2] == 11, "complex128 values");
    float fa[4] = {0, 2, 1, 0}, fb[4] = {0, 3, 4, 0}, fc[4] = {0, 0, 0, 0};
    float fone[2] = {1, 0}, fzero[2] = {0, 0};
    float *fd[3] = {fa, fb, fc};
    for (int i = 0; i < 3; i++) {
      x[i].data = fd[i]; x[i].dtype.bits = 64; x[i].byte_offset = 4;
    }
    CHECK(tprims_blas_gemm(serial, fone, tprims_tensor_borrow_raw(&x[0], 0), 0, tprims_tensor_borrow_raw(&x[1], 0), 0,
                           fzero, tprims_tensor_borrow_raw(&x[2], 0)) == TPRIMS_OK,
          "complex64 byte_offset 4");
    CHECK(fc[1] == 2 && fc[2] == 11, "complex64 values");
    /* A genuinely misaligned effective address is still rejected. */
    x[0].byte_offset = 2;
    CHECK(tprims_blas_gemm(serial, fone, tprims_tensor_borrow_raw(&x[0], 0), 0, tprims_tensor_borrow_raw(&x[1], 0), 0,
                           fzero, tprims_tensor_borrow_raw(&x[2], 0)) == TPRIMS_ERR_INVALID_ARGUMENT,
          "misaligned complex64 rejected");
  }

  /* Errors: read-only output, output aliasing an input, closed pool. */
  CHECK(tprims_blas_gemm(serial, &one, tprims_tensor_borrow_raw(&ta, 0), 0, tprims_tensor_borrow_raw(&tb, 0), 0,
                         &zero, tprims_tensor_borrow_raw(&tc, DLPACK_FLAG_BITMASK_READ_ONLY)) == TPRIMS_ERR_READ_ONLY,
        "read-only rejected");
  CHECK(strlen(tprims_last_error()) > 0, "error message");
  CHECK(tprims_blas_gemm(serial, &one, tprims_tensor_borrow_raw(&tc, 0), 0, tprims_tensor_borrow_raw(&tc, 0), 0,
                         &zero, tprims_tensor_borrow_raw(&tc, 0)) == TPRIMS_ERR_ALIASED,
        "aliasing rejected");
  CHECK(tprims_exec_close(pool) == TPRIMS_OK, "close joins");
  CHECK(tprims_exec_close(pool) == TPRIMS_OK, "second close");
  CHECK(tprims_blas_gemm(pool, &one, tprims_tensor_borrow_raw(&ta, 0), 0, tprims_tensor_borrow_raw(&tb, 0), 0,
                         &zero, tprims_tensor_borrow_raw(&tc, 0)) == TPRIMS_ERR_CLOSED,
        "closed pool");
  tprims_exec_release(pool);
  tprims_exec_release(serial);
  printf("abi_test ok\n");
  return 0;
}
