/* A host that includes tprims before the real dlpack.h. */
#include "tprims/tprims.h"
#include "dlpack/dlpack.h"
#include <stddef.h>
_Static_assert(sizeof(DLTensor) == 48, "DLTensor layout");
_Static_assert(offsetof(struct DLManagedTensorVersioned, dl_tensor) == 32, "DLManagedTensorVersioned layout");
_Static_assert(sizeof(tprims_tensor) == 16, "tprims_tensor layout");
int main(void) { return DLPACK_MAJOR_VERSION == 1 ? 0 : 1; }
