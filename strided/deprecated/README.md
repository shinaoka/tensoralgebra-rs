# Frozen strided-rs code

These crates came with the strided-rs import (upstream `71b7cb9`) and are
outside the tprims stack: the `strided-rs` facade (the stack has no facade),
`strided-einsum2` (binary contraction moves to `tprims-contract`),
`strided-opteinsum` and the `mdarray-`/`ndarray-opteinsum` adapters (N-ary
einsum stays above the stack). They are not workspace members and are not
built or tested. Do not edit them; the published 0.4.x releases remain on
crates.io and in the upstream repository.

`benches/` was already frozen upstream.
