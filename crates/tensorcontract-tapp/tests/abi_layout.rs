//! TAPP C-ABI conformance: the layout and linkage assumptions underneath it.
//!
//! The rest of the suite calls the crate's `extern "C"` functions by Rust path,
//! which checks their behaviour but not their *linkage*: a `#[no_mangle]` that
//! was renamed, dropped or given a signature the header does not describe would
//! sail through. This file therefore re-declares the whole of the upstream
//! header in an `extern "C"` block and drives a complete contraction through
//! those declarations, so the symbols have to exist, have to be `extern "C"`,
//! and have to take the argument list `api/include/tapp/*.h` publishes.
//!
//! Everything asserted here about the upstream headers was read from
//! `TAPPorg/reference-implementation` at `main`, not from the paper — the paper
//! does not print the prototypes. The numeric enumerator values in
//! `enumerator_values_match_the_upstream_headers` are quoted from
//! `api/include/tapp/datatype.h` and `api/include/tapp/product.h`.

mod common;

use std::ffi::{c_void, CStr};
use std::mem::{align_of, offset_of, size_of};
use std::os::raw::{c_char, c_int};
use std::slice;

use num_complex::Complex;

use common::{explain_status, TAPP_DEFAULT_PREC};

/// The upstream header, transcribed. Types follow `tapp/*.h`: every handle is
/// `intptr_t`, `TAPP_error` / `TAPP_datatype` / `TAPP_prectype` /
/// `TAPP_element_op` are `int`, label and shape arrays are `const int64_t*`.
mod c_abi {
    use std::ffi::c_void;
    use std::os::raw::{c_char, c_int};

    extern "C" {
        // tapp/error.h
        pub fn TAPP_check_success(error: c_int) -> bool;
        pub fn TAPP_explain_error(error: c_int, maxlen: usize, message: *mut c_char) -> usize;

        // tapp/handle.h
        pub fn TAPP_create_handle(handle: *mut isize) -> c_int;
        pub fn TAPP_destroy_handle(handle: isize) -> c_int;

        // tapp/executor.h
        pub fn TAPP_create_executor(exec: *mut isize) -> c_int;
        pub fn TAPP_destroy_executor(exec: isize) -> c_int;

        // tapp/status.h
        pub fn TAPP_destroy_status(status: isize) -> c_int;

        // tapp/tensor.h
        pub fn TAPP_create_tensor_info(
            info: *mut isize,
            ty: c_int,
            nmode: c_int,
            extents: *const i64,
            strides: *const i64,
        ) -> c_int;
        pub fn TAPP_destroy_tensor_info(info: isize) -> c_int;
        pub fn TAPP_get_nmodes(info: isize) -> c_int;
        pub fn TAPP_set_nmodes(info: isize, nmodes: c_int) -> c_int;
        pub fn TAPP_get_extents(info: isize, extents: *mut i64);
        pub fn TAPP_set_extents(info: isize, extents: *const i64) -> c_int;
        pub fn TAPP_get_strides(info: isize, strides: *mut i64);
        pub fn TAPP_set_strides(info: isize, strides: *const i64) -> c_int;

        // tapp/product.h
        #[allow(clippy::too_many_arguments)]
        pub fn TAPP_create_tensor_product(
            plan: *mut isize,
            handle: isize,
            op_a: c_int,
            a: isize,
            idx_a: *const i64,
            op_b: c_int,
            b: isize,
            idx_b: *const i64,
            op_c: c_int,
            c: isize,
            idx_c: *const i64,
            op_d: c_int,
            d: isize,
            idx_d: *const i64,
            prec: c_int,
        ) -> c_int;
        pub fn TAPP_destroy_tensor_product(plan: isize) -> c_int;
        pub fn TAPP_execute_product(
            plan: isize,
            exec: isize,
            status: *mut isize,
            alpha: *const c_void,
            a: *const c_void,
            b: *const c_void,
            beta: *const c_void,
            c: *const c_void,
            d: *mut c_void,
        ) -> c_int;
        pub fn TAPP_execute_batched_product(
            plan: isize,
            exec: isize,
            status: *mut isize,
            num_batches: c_int,
            alpha: *const c_void,
            a: *const *const c_void,
            b: *const *const c_void,
            beta: *const c_void,
            c: *const *const c_void,
            d: *mut *mut c_void,
        ) -> c_int;

        // Non-standard extension, documented in the crate.
        pub fn TAPP_implementation_name() -> *const c_char;
    }
}

// `TAPP_attr_set` / `TAPP_attr_get` / `TAPP_attr_clear` from
// `api/include/tapp/attributes.h` are *not* declared above, because this crate
// does not export them: a C program that includes `<tapp.h>` and calls them
// fails to link. They are omitted here deliberately — declaring an unresolved
// symbol would break the build rather than record the gap — and the gap is in
// the report instead.

// -------------------------------------------------------- linkage and symbols

/// A complete contraction driven only through the transcribed header: create
/// handle, executor and four tensor infos, build a plan, execute it, tear
/// everything down, and check the arithmetic.
///
/// If any `#[no_mangle]` in `src/lib.rs` were renamed or removed, this file
/// would not link. If any signature drifted from the header, this test would
/// pass garbage and fail on the numbers.
#[test]
fn every_documented_symbol_resolves_and_runs() {
    // D[i,j] = A[i,k] B[k,j] with A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    // column-major, so D = [[19,22],[43,50]].
    let ea = [2i64, 2];
    let sa = [1i64, 2];
    let eb = [2i64, 2];
    let sb = [1i64, 2];
    let ed = [2i64, 2];
    let sd = [1i64, 2];
    let la = [0i64, 2];
    let lb = [2i64, 1];
    let ld = [0i64, 1];
    let av = [1.0f64, 3.0, 2.0, 4.0];
    let bv = [5.0f64, 7.0, 6.0, 8.0];
    let want = [19.0f64, 43.0, 22.0, 50.0];

    unsafe {
        let mut handle = 0isize;
        assert_eq!(c_abi::TAPP_create_handle(&mut handle), 0);
        let mut exec = 0isize;
        assert_eq!(c_abi::TAPP_create_executor(&mut exec), 0);

        let mut ia = 0isize;
        let mut ib = 0isize;
        let mut ic = 0isize;
        let mut id = 0isize;
        for (h, e, s) in [
            (&mut ia, &ea, &sa),
            (&mut ib, &eb, &sb),
            (&mut ic, &ed, &sd),
            (&mut id, &ed, &sd),
        ] {
            // TAPP_F64 == 1, from datatype.h.
            assert_eq!(
                c_abi::TAPP_create_tensor_info(h, 1, 2, e.as_ptr(), s.as_ptr()),
                0
            );
        }
        assert_eq!(c_abi::TAPP_get_nmodes(ia), 2);

        // Exercise the mutators on a spare info before using it: set it to
        // something wrong, read it back, then set it right again.
        let bogus = [3i64, 3];
        let bogus_s = [1i64, 3];
        assert_eq!(c_abi::TAPP_set_extents(ic, bogus.as_ptr()), 0);
        assert_eq!(c_abi::TAPP_set_strides(ic, bogus_s.as_ptr()), 0);
        let mut got_e = [0i64; 2];
        let mut got_s = [0i64; 2];
        c_abi::TAPP_get_extents(ic, got_e.as_mut_ptr());
        c_abi::TAPP_get_strides(ic, got_s.as_mut_ptr());
        assert_eq!(got_e, bogus);
        assert_eq!(got_s, bogus_s);
        assert_eq!(c_abi::TAPP_set_nmodes(ic, 2), 0);
        assert_eq!(c_abi::TAPP_set_extents(ic, ed.as_ptr()), 0);
        assert_eq!(c_abi::TAPP_set_strides(ic, sd.as_ptr()), 0);

        let mut plan = 0isize;
        // TAPP_IDENTITY == 0, from product.h.
        let e = c_abi::TAPP_create_tensor_product(
            &mut plan,
            handle,
            0,
            ia,
            la.as_ptr(),
            0,
            ib,
            lb.as_ptr(),
            0,
            ic,
            ld.as_ptr(),
            0,
            id,
            ld.as_ptr(),
            TAPP_DEFAULT_PREC,
        );
        assert_eq!(e, 0, "{}", explain_status(e));
        assert!(c_abi::TAPP_check_success(e));

        let alpha = 1.0f64;
        let beta = 0.0f64;
        let mut dv = [0.0f64; 4];
        let mut status = 0isize;
        let e = c_abi::TAPP_execute_product(
            plan,
            exec,
            &mut status,
            &alpha as *const f64 as *const c_void,
            av.as_ptr() as *const c_void,
            bv.as_ptr() as *const c_void,
            &beta as *const f64 as *const c_void,
            std::ptr::null(),
            dv.as_mut_ptr() as *mut c_void,
        );
        assert_eq!(e, 0, "{}", explain_status(e));
        assert_eq!(dv, want);

        // The batched entry point over one batch, into a second buffer.
        let mut dv2 = [0.0f64; 4];
        let ap = [av.as_ptr() as *const c_void];
        let bp = [bv.as_ptr() as *const c_void];
        let mut dp = [dv2.as_mut_ptr() as *mut c_void];
        let e = c_abi::TAPP_execute_batched_product(
            plan,
            exec,
            &mut status,
            1,
            &alpha as *const f64 as *const c_void,
            ap.as_ptr(),
            bp.as_ptr(),
            &beta as *const f64 as *const c_void,
            std::ptr::null(),
            dp.as_mut_ptr(),
        );
        assert_eq!(e, 0, "{}", explain_status(e));
        assert_eq!(dv2, want);

        assert_eq!(c_abi::TAPP_destroy_status(status), 0);
        assert_eq!(c_abi::TAPP_destroy_tensor_product(plan), 0);
        for i in [ia, ib, ic, id] {
            assert_eq!(c_abi::TAPP_destroy_tensor_info(i), 0);
        }
        assert_eq!(c_abi::TAPP_destroy_executor(exec), 0);
        assert_eq!(c_abi::TAPP_destroy_handle(handle), 0);

        // The remaining two symbols.
        let mut buf: [c_char; 64] = [0; 64];
        assert!(c_abi::TAPP_explain_error(3, buf.len(), buf.as_mut_ptr()) > 0);
        let name = c_abi::TAPP_implementation_name();
        assert!(!name.is_null());
        assert!(CStr::from_ptr(name).to_str().is_ok());
    }
}

/// The C symbols and the Rust paths are the same functions, not two
/// implementations that happen to agree.
#[test]
fn the_c_symbols_are_the_crate_functions() {
    assert_eq!(
        c_abi::TAPP_implementation_name as *const () as usize,
        tensorcontract_tapp::TAPP_implementation_name as *const () as usize
    );
    assert_eq!(
        c_abi::TAPP_check_success as *const () as usize,
        tensorcontract_tapp::TAPP_check_success as *const () as usize
    );
    assert_eq!(
        c_abi::TAPP_execute_product as *const () as usize,
        tensorcontract_tapp::TAPP_execute_product as *const () as usize
    );
}

// ------------------------------------------------------------ enumerator values

/// Every enumerator the crate exports, against the numbers in the upstream
/// headers.
///
/// `api/include/tapp/datatype.h`: `TAPP_F32 = 0`, `TAPP_F64 = 1`, `TAPP_C32 =
/// 2`, `TAPP_C64 = 3`, `TAPP_F16 = 4`, `TAPP_BF16 = 5`, and
/// `TAPP_DEFAULT_PREC = -1`. `api/include/tapp/product.h`: `TAPP_IDENTITY = 0`,
/// `TAPP_CONJUGATE = 1`.
///
/// This matters more than a normal constant test: the project has already been
/// bitten by two libraries that swapped two enumerators between releases (TBLIS
/// 1.3 and 2.0 exchange `TYPE_DOUBLE` and `TYPE_SCOMPLEX`), which produces
/// plausible wrong numbers and no error at all. A datatype tag is exactly the
/// kind of constant that must be pinned rather than assumed.
#[test]
fn enumerator_values_match_the_upstream_headers() {
    use tensorcontract_tapp::*;
    assert_eq!(TAPP_F32, 0);
    assert_eq!(TAPP_F64, 1);
    assert_eq!(TAPP_C32, 2);
    assert_eq!(TAPP_C64, 3);
    assert_eq!(TAPP_F16, 4);
    assert_eq!(TAPP_BF16, 5);
    assert_eq!(TAPP_IDENTITY, 0);
    assert_eq!(TAPP_CONJUGATE, 1);
    assert_eq!(TAPP_DEFAULT_PREC, -1);
    // `TAPP_error` is only specified at zero: upstream `error.h` declares
    // `typedef int TAPP_error` and no enumerators, so the non-zero codes below
    // are this implementation's own and a portable caller must go through
    // `TAPP_check_success`. Pinned so they stay stable for callers that do
    // depend on them.
    assert_eq!(TAPP_SUCCESS, 0);
    assert_eq!(TAPP_ERROR_NULL, 1);
    assert_eq!(TAPP_ERROR_DATATYPE, 2);
    assert_eq!(TAPP_ERROR_SHAPE, 3);
    assert_eq!(TAPP_ERROR_LABELS, 4);
    assert_eq!(TAPP_ERROR_UNSUPPORTED, 5);
    assert_eq!(TAPP_ERROR_INTERNAL, 6);
}

// ------------------------------------------------------------------- data layout

/// `TAPP_C32` and `TAPP_C64` are interleaved, real part first, and
/// layout-identical to C99 `float _Complex` / `double _Complex` and to
/// `num_complex::Complex`.
///
/// The whole complex side of the ABI is a reinterpret cast — a C caller's
/// `double _Complex*` becomes a `*const Complex<f64>` with no conversion — so
/// this is the assumption everything else rests on.
#[test]
fn complex_datatypes_are_interleaved_and_c_compatible() {
    assert_eq!(size_of::<Complex<f32>>(), 2 * size_of::<f32>());
    assert_eq!(size_of::<Complex<f64>>(), 2 * size_of::<f64>());
    assert_eq!(align_of::<Complex<f32>>(), align_of::<f32>());
    assert_eq!(align_of::<Complex<f64>>(), align_of::<f64>());

    // Field offsets, which is what "interleaved, real first" actually means.
    assert_eq!(offset_of!(Complex<f32>, re), 0);
    assert_eq!(offset_of!(Complex<f32>, im), 4);
    assert_eq!(offset_of!(Complex<f64>, re), 0);
    assert_eq!(offset_of!(Complex<f64>, im), 8);

    // A C caller's array of `double _Complex` is an array of pairs of doubles;
    // reading it as `Complex<f64>` must recover the same numbers in the same
    // order, and writing it back must be byte-identical.
    let raw = [1.0f64, -2.0, 3.0, -4.0];
    let zs: &[Complex<f64>] =
        unsafe { slice::from_raw_parts(raw.as_ptr() as *const Complex<f64>, 2) };
    assert_eq!(zs, &[Complex::new(1.0, -2.0), Complex::new(3.0, -4.0)]);
    let back: &[f64] = unsafe { slice::from_raw_parts(zs.as_ptr() as *const f64, 4) };
    assert_eq!(back, raw);
}

/// The scalar types that cross the boundary are the ones the header names.
///
/// TAPP passes no aggregates by value at all: every argument is a scalar, an
/// `intptr_t` handle, or a pointer to caller-owned memory, so there is no struct
/// whose field offsets could disagree with a C compiler's. What has to match is
/// the width of the handles, of `int`, and of the `int64_t` shape and label
/// arrays.
#[test]
fn boundary_scalar_types_have_the_widths_the_header_names() {
    // `intptr_t`: handles must round-trip a pointer without losing bits.
    assert_eq!(size_of::<isize>(), size_of::<*mut c_void>());
    assert_eq!(align_of::<isize>(), align_of::<*mut c_void>());
    // `int`, the type of `TAPP_error`, `TAPP_datatype`, `TAPP_prectype`,
    // `TAPP_element_op` and `nmode`.
    assert_eq!(size_of::<c_int>(), 4);
    // `int64_t`, the type of every extent, stride and index label.
    assert_eq!(size_of::<i64>(), 8);
    // `bool`, the return type of `TAPP_check_success`, is one byte in both
    // Rust and C23 / `<stdbool.h>`.
    assert_eq!(size_of::<bool>(), 1);
    // `size_t`, `TAPP_explain_error`'s `maxlen` and return type.
    assert_eq!(size_of::<usize>(), size_of::<*const c_void>());
}

/// The handle types that carry state are distinct, non-zero, pointer-aligned
/// live allocations, so a C caller can hold several at once and tell them apart.
#[test]
fn stateful_handles_are_distinct_nonzero_and_aligned() {
    unsafe {
        // Executors carry a thread count, tensor infos a layout, so both are
        // real allocations.
        let mut xs = [0isize; 4];
        for x in xs.iter_mut() {
            assert_eq!(c_abi::TAPP_create_executor(x), 0);
            assert_ne!(*x, 0, "zero is the ABI's invalid-handle value");
            assert_eq!(
                (*x as usize) % align_of::<usize>(),
                0,
                "a handle is a cast pointer and must stay aligned"
            );
        }
        for (i, &x) in xs.iter().enumerate() {
            for &y in &xs[i + 1..] {
                assert_ne!(x, y, "two live executors must be distinguishable");
            }
        }
        for x in xs {
            assert_eq!(c_abi::TAPP_destroy_executor(x), 0);
        }

        let e = [4i64, 3];
        let s = [1i64, 4];
        let mut is = [0isize; 4];
        for i in is.iter_mut() {
            assert_eq!(
                c_abi::TAPP_create_tensor_info(i, 3, 2, e.as_ptr(), s.as_ptr()),
                0
            );
            assert_ne!(*i, 0);
            assert_eq!((*i as usize) % align_of::<usize>(), 0);
        }
        for (i, &x) in is.iter().enumerate() {
            for &y in &is[i + 1..] {
                assert_ne!(x, y, "two live tensor infos must be distinguishable");
            }
        }
        for i in is {
            assert_eq!(c_abi::TAPP_destroy_tensor_info(i), 0);
        }
    }
}

/// The *library* handle is not a distinct allocation, and every one this
/// implementation hands out is the same value.
///
/// `TAPP_create_handle` boxes `struct HandleState { _private: () }`, which is
/// zero-sized, and a `Box` of a zero-sized type allocates nothing:
/// `Box::into_raw` yields `NonNull::dangling()`, i.e. the type's alignment. So
/// the handle is `1`, always, and `src/lib.rs`'s stated reason for boxing it —
/// "kept as a real allocation so that handle validity can be checked and future
/// state has somewhere to live" — does not hold today. The only handle value the
/// ABI can reject is `0`.
///
/// This is sound rather than a bug: creating and destroying a zero-sized box is
/// well defined, and the shim ignores the library handle anyway (see
/// `a_zero_library_handle_is_accepted_by_create_tensor_product`). It is pinned
/// because the day `HandleState` gains a field the allocation becomes real, two
/// handles stop being equal, and this test is where somebody finds out that a C
/// program which creates two handles and destroys both has been double-freeing
/// harmlessly all along.
#[test]
fn the_library_handle_is_a_zero_sized_box_and_so_always_the_same_value() {
    unsafe {
        let mut h1 = 0isize;
        let mut h2 = 0isize;
        assert_eq!(c_abi::TAPP_create_handle(&mut h1), 0);
        assert_eq!(c_abi::TAPP_create_handle(&mut h2), 0);
        assert_ne!(h1, 0);
        assert_eq!(
            h1, h2,
            "library handles have become distinct: re-read this test's comment"
        );
        // Destroyed once, not twice: with `h1 == h2` a second destroy is a
        // double free of the same value, which is a no-op only for as long as
        // the state stays zero-sized.
        assert_eq!(c_abi::TAPP_destroy_handle(h1), 0);
    }
}
