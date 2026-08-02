//! CBLAS bindings for the TTGT baseline's GEMM step.

#![allow(non_camel_case_types)]
#![allow(dead_code)] // surface is used only under the `tblis` / `blas` features

use std::ffi::c_void;
use std::os::raw::c_int;

pub const CBLAS_COL_MAJOR: c_int = 102;
pub const CBLAS_NO_TRANS: c_int = 111;

extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn cblas_sgemm(
        layout: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );
    #[allow(clippy::too_many_arguments)]
    pub fn cblas_dgemm(
        layout: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f64,
        a: *const f64,
        lda: c_int,
        b: *const f64,
        ldb: c_int,
        beta: f64,
        c: *mut f64,
        ldc: c_int,
    );
    #[allow(clippy::too_many_arguments)]
    pub fn cblas_cgemm(
        layout: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        lda: c_int,
        b: *const c_void,
        ldb: c_int,
        beta: *const c_void,
        c: *mut c_void,
        ldc: c_int,
    );
    #[allow(clippy::too_many_arguments)]
    pub fn cblas_zgemm(
        layout: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        lda: c_int,
        b: *const c_void,
        ldb: c_int,
        beta: *const c_void,
        c: *mut c_void,
        ldc: c_int,
    );
    pub fn openblas_set_num_threads(n: c_int);
}

/// `C = A * B` with all matrices column-major and no transposition.
///
/// # Safety
/// The three buffers must hold `m*k`, `k*n`, `m*n` elements respectively.
pub unsafe trait GemmScalar: Copy {
    unsafe fn gemm(m: usize, n: usize, k: usize, a: *const Self, b: *const Self, c: *mut Self);
}

unsafe impl GemmScalar for f32 {
    unsafe fn gemm(m: usize, n: usize, k: usize, a: *const Self, b: *const Self, c: *mut Self) {
        cblas_sgemm(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as c_int,
            n as c_int,
            k as c_int,
            1.0,
            a,
            m as c_int,
            b,
            k as c_int,
            0.0,
            c,
            m as c_int,
        );
    }
}

unsafe impl GemmScalar for f64 {
    unsafe fn gemm(m: usize, n: usize, k: usize, a: *const Self, b: *const Self, c: *mut Self) {
        cblas_dgemm(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as c_int,
            n as c_int,
            k as c_int,
            1.0,
            a,
            m as c_int,
            b,
            k as c_int,
            0.0,
            c,
            m as c_int,
        );
    }
}

unsafe impl GemmScalar for num_complex::Complex<f32> {
    unsafe fn gemm(m: usize, n: usize, k: usize, a: *const Self, b: *const Self, c: *mut Self) {
        let one = num_complex::Complex::<f32>::new(1.0, 0.0);
        let zero = num_complex::Complex::<f32>::new(0.0, 0.0);
        cblas_cgemm(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as c_int,
            n as c_int,
            k as c_int,
            &one as *const _ as *const c_void,
            a as *const c_void,
            m as c_int,
            b as *const c_void,
            k as c_int,
            &zero as *const _ as *const c_void,
            c as *mut c_void,
            m as c_int,
        );
    }
}

unsafe impl GemmScalar for num_complex::Complex<f64> {
    unsafe fn gemm(m: usize, n: usize, k: usize, a: *const Self, b: *const Self, c: *mut Self) {
        let one = num_complex::Complex::<f64>::new(1.0, 0.0);
        let zero = num_complex::Complex::<f64>::new(0.0, 0.0);
        cblas_zgemm(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as c_int,
            n as c_int,
            k as c_int,
            &one as *const _ as *const c_void,
            a as *const c_void,
            m as c_int,
            b as *const c_void,
            k as c_int,
            &zero as *const _ as *const c_void,
            c as *mut c_void,
            m as c_int,
        );
    }
}
