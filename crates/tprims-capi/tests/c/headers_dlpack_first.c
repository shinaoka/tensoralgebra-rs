/* A host that includes the real dlpack.h before tprims. */
#include "dlpack/dlpack.h"
#include "tprims/tprims.h"
int main(void) { DLTensor t; tprims_tensor x = tprims_tensor_borrow_raw(&t, DLPACK_FLAG_BITMASK_READ_ONLY); (void)x; return kDLCPU == 1 ? 0 : 1; }
