/* End-to-end test of libtprims from C: zero copy, layouts, pools, TAPP and BLAS
   sharing one executor, error reporting. Exit status 0 on success. */
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
  CHECK(tprims_has_part("blas") && tprims_has_part("tapp") && tprims_has_part("core"), "parts");
  CHECK(!tprims_has_part("linalg"), "linalg not in this slice");

  /* A: 2x3 row-major (NULL strides), B: 3x2 column-major, C: 2x2 column-major. */
  double a[6] = {1, 3, 5, 2, 4, 6}; /* rows [1,3,5], [2,4,6] */
  double b[6] = {1, 3, 5, 2, 4, 6}; /* columns [1,3,5], [2,4,6] */
  double c[4] = {NAN, NAN, NAN, NAN};
  int64_t sa[2] = {2, 3}, sb[2] = {3, 2}, sc[2] = {2, 2};
  int64_t stb[2] = {1, 3}, stc[2] = {1, 2};
  DLTensor ta = f64_tensor(a, 2, sa, NULL), tb = f64_tensor(b, 2, sb, stb), tc = f64_tensor(c, 2, sc, stc);
  double one = 1.0, zero = 0.0;

  /* One executor type for every part: 0 (serial), TAPP_create_executor, and an
     owned Rayon pool all go to the BLAS entry points. */
  TAPP_executor pool = 0, serial = 0;
  CHECK(tprims_tapp_executor_create_rayon(&pool, 4, NULL) == TPRIMS_OK && pool != 0, "pool create");
  size_t psize = 99, pbudget = 99;
  CHECK(tprims_tapp_executor_get_threads(pool, &psize, &pbudget) == TPRIMS_OK && psize == 4 && pbudget == 4, "pool size");
  CHECK(tprims_tapp_executor_set_budget(pool, 2) == TPRIMS_OK, "budget");
  CHECK(tprims_tapp_executor_get_threads(pool, &psize, &pbudget) == TPRIMS_OK && psize == 4 && pbudget == 2, "budget set");
  CHECK(tprims_tapp_executor_set_budget(pool, 0) == TPRIMS_ERR_INVALID_ARGUMENT, "zero budget");
  CHECK(tprims_tapp_executor_set_budget(pool, 4) == TPRIMS_OK, "budget reset");
  CHECK(TAPP_check_success(TAPP_create_executor(&serial)), "serial executor");
  CHECK(tprims_tapp_executor_get_threads(serial, &psize, &pbudget) == TPRIMS_OK && psize == 0 && pbudget == 1, "serial threads");
  TAPP_executor execs[3] = {0, serial, pool};
  for (int i = 0; i < 3; i++) {
    c[0] = c[1] = c[2] = c[3] = NAN;
    tprims_status st = tprims_blas_gemm(execs[i], &one, tprims_tensor_borrow_raw(&ta, 0), 0,
                                        tprims_tensor_borrow_raw(&tb, 0), 0, &zero,
                                        tprims_tensor_borrow_raw(&tc, 0));
    CHECK(st == TPRIMS_OK, "gemm");
    /* written in place into the caller's buffer (zero copy) */
    CHECK(c[0] == 35 && c[1] == 44 && c[2] == 44 && c[3] == 56, "gemm values");
  }

  /* The same product through TAPP, on every executor: D = A * B with labels
     i,k / k,j / i,j; A is row-major (strides {3,1}), B and D column-major. */
  {
    TAPP_handle handle;
    CHECK(TAPP_check_success(TAPP_create_handle(&handle)), "handle");
    int64_t ea[2] = {2, 3}, sa_[2] = {3, 1}, eb[2] = {3, 2}, sb_[2] = {1, 3}, ed[2] = {2, 2}, sd_[2] = {1, 2};
    TAPP_tensor_info ia, ib, id;
    CHECK(TAPP_check_success(TAPP_create_tensor_info(&ia, TAPP_F64, 2, ea, sa_)), "info A");
    CHECK(TAPP_check_success(TAPP_create_tensor_info(&ib, TAPP_F64, 2, eb, sb_)), "info B");
    CHECK(TAPP_check_success(TAPP_create_tensor_info(&id, TAPP_F64, 2, ed, sd_)), "info D");
    int64_t la[2] = {'i', 'k'}, lb[2] = {'k', 'j'}, ld[2] = {'i', 'j'};
    TAPP_tensor_product plan;
    CHECK(TAPP_check_success(TAPP_create_tensor_product(&plan, handle, TAPP_IDENTITY, ia, la, TAPP_IDENTITY, ib, lb,
                                                        TAPP_IDENTITY, id, ld, TAPP_IDENTITY, id, ld, TAPP_DEFAULT_PREC)),
          "plan create");
    CHECK(TAPP_create_tensor_product(&plan, 0, TAPP_IDENTITY, ia, la, TAPP_IDENTITY, ib, lb, TAPP_IDENTITY, id, ld,
                                     TAPP_IDENTITY, id, ld, TAPP_DEFAULT_PREC) == TPRIMS_ERR_INVALID_ARGUMENT,
          "zero handle rejected");
    CHECK(TAPP_check_success(TAPP_create_tensor_product(&plan, handle, TAPP_IDENTITY, ia, la, TAPP_IDENTITY, ib, lb,
                                                        TAPP_IDENTITY, id, ld, TAPP_IDENTITY, id, ld, TAPP_DEFAULT_PREC)),
          "plan create again");
    double ra[6] = {1, 3, 5, 2, 4, 6}; /* row-major rows [1,3,5], [2,4,6] */
    for (int i = 0; i < 3; i++) {
      double d[4] = {NAN, NAN, NAN, NAN};
      TAPP_status status = -1;
      tprims_status st = TAPP_execute_product(plan, execs[i], &status, &one, ra, b, &zero, TAPP_IN_PLACE, d);
      CHECK(st == TPRIMS_OK && status == 0, "TAPP execute");
      CHECK(d[0] == 35 && d[1] == 44 && d[2] == 44 && d[3] == 56, "TAPP values");
    }
    /* Output aliasing an input is rejected before any write. */
    CHECK(TAPP_execute_product(plan, 0, NULL, &one, ra, b, &zero, TAPP_IN_PLACE, ra) == TPRIMS_ERR_ALIASED,
          "aliasing rejected");
    CHECK(strlen(tprims_last_error()) > 0, "aliasing message");
    CHECK(TAPP_check_success(TAPP_destroy_tensor_product(plan)), "plan destroy");
    CHECK(TAPP_check_success(TAPP_destroy_tensor_info(ia)) && TAPP_check_success(TAPP_destroy_tensor_info(ib)) &&
              TAPP_check_success(TAPP_destroy_tensor_info(id)),
          "info destroy");
    CHECK(TAPP_check_success(TAPP_destroy_handle(handle)), "handle destroy");
  }

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
  /* Destruction: an owned pool is stopped and joined; the default executor is a no-op. */
  CHECK(TAPP_check_success(TAPP_destroy_executor(pool)), "destroy pool joins");
  CHECK(TAPP_check_success(TAPP_destroy_executor(serial)), "destroy serial");
  CHECK(TAPP_check_success(TAPP_destroy_executor(0)), "destroy default executor");
  CHECK(tprims_tapp_executor_create_rayon(&pool, 0, NULL) == TPRIMS_ERR_INVALID_ARGUMENT, "zero threads");
  char msg[64];
  CHECK(TAPP_explain_error(TPRIMS_BUSY, sizeof msg, msg) == strlen(msg) && strlen(msg) > 0, "explain");
  printf("abi_test ok\n");
  return 0;
}
