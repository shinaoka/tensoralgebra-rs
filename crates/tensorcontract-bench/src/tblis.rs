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
//! `type@0 conj@4 scalar@8 data@32 ndim@40 len@48 stride@56`).

#![allow(non_camel_case_types)]
#![allow(dead_code)] // surface is used only under the `tblis` / `blas` features

use std::ffi::c_void;
use std::os::raw::{c_char, c_int, c_uint};

pub const TYPE_SINGLE: c_int = 0;
pub const TYPE_SCOMPLEX: c_int = 1;
pub const TYPE_DOUBLE: c_int = 2;
pub const TYPE_DCOMPLEX: c_int = 3;

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
