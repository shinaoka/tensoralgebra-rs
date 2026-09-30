//! Any partition must be bitwise identical to the serial run: every output
//! element owns one thread and accumulates over the whole of `K` in order.
mod common;

use common::{run_with_width, Shape};

const SHAPE: Shape = Shape {
    m: 203,
    n: 157,
    k: 311,
};

#[test]
fn every_width_is_bitwise_identical_to_width_one() {
    tprims_kernel_tensorcontract::register();
    for id in [
        "tc.scalar.f64.4x4",
        "portable.f64.4x4",
        "portable.f64.4x4.direct",
        "portable.f64.4x4.direct-b",
    ] {
        let base = run_with_width::<f64>(id, 1, SHAPE, false);
        for w in [2usize, 3, 4, 8] {
            let got = run_with_width::<f64>(id, w, SHAPE, false);
            assert!(
                base.iter()
                    .zip(&got)
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
                "{id} width {w}"
            );
        }
    }
}

#[test]
fn aligned_c_lines_are_bitwise_identical_too() {
    let base = run_with_width::<f64>("portable.f64.4x4", 1, SHAPE, true);
    for w in [2usize, 4, 8] {
        let got = run_with_width::<f64>("portable.f64.4x4", w, SHAPE, true);
        assert!(
            base.iter()
                .zip(&got)
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "aligned width {w}"
        );
    }
    // Alignment must not change the answer either, only the boundaries.
    let unaligned = run_with_width::<f64>("portable.f64.4x4", 4, SHAPE, false);
    let aligned = run_with_width::<f64>("portable.f64.4x4", 4, SHAPE, true);
    assert!(unaligned
        .iter()
        .zip(&aligned)
        .all(|(x, y)| x.to_bits() == y.to_bits()));
}
