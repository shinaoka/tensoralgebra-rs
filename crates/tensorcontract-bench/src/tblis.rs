//! Minimal hand-written FFI to the C++ TBLIS baseline.
//!
//! We bind TBLIS directly rather than using the published `tblis`/`tblis-ffi`
//! crates because the benchmark needs to control *which* TBLIS is measured
//! (version, branch, and above all which BLIS configuration its kernels were
//! built for). Those crates vendor their own build. The surface we need is
//! four functions wide, so the trade is clearly worth it.
//!
//! Struct layout was verified against a `sizeof`/`offsetof` probe compiled
//! with the same headers (`tblis_tensor` is 64 bytes:
//! `type@0 conj@4 scalar@8 data@32 ndim@40 len@48 stride@56`). The layout is
//! identical in 1.3.0 and 2.0 (`ndim` is `unsigned` in 1.3 and `int` in 2.0,
//! same size and offset).
//!
//! # Version skew — read before changing anything here
//!
//! The `type_t` enumerators are **swapped** between the two TBLIS releases:
//!
//! | | `TYPE_SINGLE` | | | `TYPE_DCOMPLEX` |
//! |---|---|---|---|---|
//! | v1.3.0 | 0 | `TYPE_DOUBLE` = 1 | `TYPE_SCOMPLEX` = 2 | 3 |
//! | v2.0   | 0 | `TYPE_SCOMPLEX` = 1 | `TYPE_DOUBLE` = 2 | 3 |
//!
//! This is a silent ABI break: linking code built against one against the
//! other computes single-complex where double was asked for, with no error and
//! no crash — the tensor sizes still line up because the harness passes the
//! extents separately. Hence the `tblis13` feature, and hence the runtime
//! self-check in [`verify_type_tags`] which multiplies a known matrix and
//! refuses to proceed if the answer is wrong.

#![allow(non_camel_case_types)]
#![allow(dead_code)] // surface is used only under the `tblis` / `blas` features

use std::ffi::c_void;
use std::os::raw::{c_char, c_int, c_uint};

#[cfg(not(feature = "tblis13"))]
mod tags {
    use std::os::raw::c_int;
    pub const TYPE_SINGLE: c_int = 0;
    pub const TYPE_SCOMPLEX: c_int = 1;
    pub const TYPE_DOUBLE: c_int = 2;
    pub const TYPE_DCOMPLEX: c_int = 3;
    pub const VERSION: &str = "2.x";
}

#[cfg(feature = "tblis13")]
mod tags {
    use std::os::raw::c_int;
    pub const TYPE_SINGLE: c_int = 0;
    pub const TYPE_DOUBLE: c_int = 1;
    pub const TYPE_SCOMPLEX: c_int = 2;
    pub const TYPE_DCOMPLEX: c_int = 3;
    pub const VERSION: &str = "1.3";
}

#[allow(unused_imports)]
pub use tags::{TYPE_DCOMPLEX, TYPE_DOUBLE, TYPE_SCOMPLEX, TYPE_SINGLE, VERSION};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct tblis_scalar {
    /// The union payload, widest member is `dcomplex` (two f64).
    pub data: [f64; 2],
    pub ty: c_int,
    pub _pad: c_int,
}

impl tblis_scalar {
    pub fn f32(v: f32) -> Self {
        let mut s = tblis_scalar {
            data: [0.0; 2],
            ty: TYPE_SINGLE,
            _pad: 0,
        };
        // The union aliases a `float` at offset 0.
        unsafe { *(s.data.as_mut_ptr() as *mut f32) = v };
        s
    }
    pub fn f64(v: f64) -> Self {
        tblis_scalar {
            data: [v, 0.0],
            ty: TYPE_DOUBLE,
            _pad: 0,
        }
    }
    pub fn c32(re: f32, im: f32) -> Self {
        let mut s = tblis_scalar {
            data: [0.0; 2],
            ty: TYPE_SCOMPLEX,
            _pad: 0,
        };
        unsafe {
            let p = s.data.as_mut_ptr() as *mut f32;
            *p = re;
            *p.add(1) = im;
        }
        s
    }
    pub fn c64(re: f64, im: f64) -> Self {
        tblis_scalar {
            data: [re, im],
            ty: TYPE_DCOMPLEX,
            _pad: 0,
        }
    }
}

#[repr(C)]
pub struct tblis_tensor {
    pub ty: c_int,
    pub conj: c_int,
    pub scalar: tblis_scalar,
    pub data: *mut c_void,
    pub ndim: c_int,
    pub _pad: c_int,
    pub len: *mut isize,
    pub stride: *mut isize,
}

extern "C" {
    pub fn tblis_tensor_mult(
        comm: *const c_void,
        cntx: *const c_void,
        a: *const tblis_tensor,
        idx_a: *const c_char,
        b: *const tblis_tensor,
        idx_b: *const c_char,
        c: *mut tblis_tensor,
        idx_c: *const c_char,
    );
    pub fn tblis_set_num_threads(n: c_uint);
    pub fn tblis_get_num_threads() -> c_uint;
}

/// Owned scratch for one TBLIS operand (TBLIS wants mutable `len`/`stride`).
pub struct Operand {
    pub len: Vec<isize>,
    pub stride: Vec<isize>,
    pub labels: Vec<c_char>,
}

impl Operand {
    pub fn new(extents: &[i64], strides: &[i64], labels: &str) -> Self {
        let mut l: Vec<c_char> = labels.bytes().map(|b| b as c_char).collect();
        l.push(0);
        Operand {
            len: extents.iter().map(|&x| x as isize).collect(),
            stride: strides.iter().map(|&x| x as isize).collect(),
            labels: l,
        }
    }

    pub fn tensor(&mut self, ty: c_int, scalar: tblis_scalar, data: *mut c_void) -> tblis_tensor {
        tblis_tensor {
            ty,
            conj: 0,
            scalar,
            data,
            ndim: self.len.len() as c_int,
            _pad: 0,
            len: self.len.as_mut_ptr(),
            stride: self.stride.as_mut_ptr(),
        }
    }

    pub fn labels(&self) -> *const c_char {
        self.labels.as_ptr()
    }
}

/// Confirm at runtime that the compiled-in `type_t` enumerators match the
/// TBLIS actually linked.
///
/// Because 1.3 and 2.0 swap `TYPE_DOUBLE` and `TYPE_SCOMPLEX`, a mismatch is
/// silent: TBLIS happily reinterprets an `f64` buffer as `Complex<f32>` and
/// returns numbers. Multiplying a known 2x2 identity-ish pair in each of the
/// four dtypes and checking the result catches it immediately.
///
/// # Safety
/// Requires a linked TBLIS.
#[cfg(feature = "tblis")]
pub unsafe fn verify_type_tags() -> Result<(), String> {
    fn ident2<T: Copy>(one: T, zero: T) -> Vec<T> {
        vec![one, zero, zero, one]
    }

    // A = [[1,0],[0,1]], B = [[2,3],[4,5]] column-major, expect C == B.
    unsafe fn run<T: Copy + PartialEq + std::fmt::Debug>(
        ty: c_int,
        one: T,
        zero: T,
        b: &[T],
        alpha: tblis_scalar,
        beta: tblis_scalar,
        name: &str,
    ) -> Result<(), String> {
        let a = ident2(one, zero);
        let mut c = vec![zero; 4];
        let ext = [2i64, 2];
        let str_ = [1i64, 2];
        let mut oa = Operand::new(&ext, &str_, "ik");
        let mut ob = Operand::new(&ext, &str_, "kj");
        let mut oc = Operand::new(&ext, &str_, "ij");
        let ta = oa.tensor(ty, alpha, a.as_ptr() as *mut c_void);
        let tb = ob.tensor(ty, alpha, b.as_ptr() as *mut c_void);
        let mut tc = oc.tensor(ty, beta, c.as_mut_ptr() as *mut c_void);
        tblis_tensor_mult(
            std::ptr::null(),
            std::ptr::null(),
            &ta,
            oa.labels(),
            &tb,
            ob.labels(),
            &mut tc,
            oc.labels(),
        );
        if c != b {
            return Err(format!(
                "TBLIS type tag mismatch for {name}: identity * {b:?} gave {c:?}.\n\
                 The harness was built for TBLIS {VERSION} but linked against a \
                 different version (1.3 and 2.0 swap TYPE_DOUBLE and TYPE_SCOMPLEX).\n\
                 Toggle the `tblis13` cargo feature."
            ));
        }
        Ok(())
    }

    use num_complex::Complex;
    run::<f32>(
        TYPE_SINGLE,
        1.0,
        0.0,
        &[2.0, 3.0, 4.0, 5.0],
        tblis_scalar::f32(1.0),
        tblis_scalar::f32(0.0),
        "f32",
    )?;
    run::<f64>(
        TYPE_DOUBLE,
        1.0,
        0.0,
        &[2.0, 3.0, 4.0, 5.0],
        tblis_scalar::f64(1.0),
        tblis_scalar::f64(0.0),
        "f64",
    )?;
    run::<Complex<f32>>(
        TYPE_SCOMPLEX,
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        &[
            Complex::new(2.0, 1.0),
            Complex::new(3.0, 1.0),
            Complex::new(4.0, 1.0),
            Complex::new(5.0, 1.0),
        ],
        tblis_scalar::c32(1.0, 0.0),
        tblis_scalar::c32(0.0, 0.0),
        "c32",
    )?;
    run::<Complex<f64>>(
        TYPE_DCOMPLEX,
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        &[
            Complex::new(2.0, 1.0),
            Complex::new(3.0, 1.0),
            Complex::new(4.0, 1.0),
            Complex::new(5.0, 1.0),
        ],
        tblis_scalar::c64(1.0, 0.0),
        tblis_scalar::c64(0.0, 0.0),
        "c64",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_layout_matches_c() {
        assert_eq!(std::mem::size_of::<tblis_scalar>(), 24);
        assert_eq!(std::mem::size_of::<tblis_tensor>(), 64);
        let t = tblis_tensor {
            ty: 0,
            conj: 0,
            scalar: tblis_scalar::f64(0.0),
            data: std::ptr::null_mut(),
            ndim: 0,
            _pad: 0,
            len: std::ptr::null_mut(),
            stride: std::ptr::null_mut(),
        };
        let base = &t as *const _ as usize;
        assert_eq!(&t.scalar as *const _ as usize - base, 8);
        assert_eq!(&t.data as *const _ as usize - base, 32);
        assert_eq!(&t.ndim as *const _ as usize - base, 40);
        assert_eq!(&t.len as *const _ as usize - base, 48);
        assert_eq!(&t.stride as *const _ as usize - base, 56);
    }
}
