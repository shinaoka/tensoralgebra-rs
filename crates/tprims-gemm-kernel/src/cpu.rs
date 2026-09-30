//! CPU requirements for kernel-family selection; no execution or threading policy.

/// Instruction-set capabilities used by the registry.
///
/// # Examples
/// ```
/// use tprims_gemm_kernel::CpuFeatures;
/// assert!(CpuFeatures::detect().contains(CpuFeatures::NONE));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CpuFeatures {
    /// AVX2 instructions.
    pub avx2: bool,
    /// Fused multiply-add instructions.
    pub fma: bool,
    /// AVX-512 foundation instructions.
    pub avx512f: bool,
    /// AArch64 NEON instructions.
    pub neon: bool,
}

impl CpuFeatures {
    /// No instruction-set requirements.
    pub const NONE: Self = Self {
        avx2: false,
        fma: false,
        avx512f: false,
        neon: false,
    };

    /// Detect capabilities once, or return none when `std` is disabled.
    ///
    /// # Examples
    /// ```
    /// use tprims_gemm_kernel::CpuFeatures;
    /// assert_eq!(CpuFeatures::detect(), CpuFeatures::detect());
    /// ```
    pub fn detect() -> Self {
        #[cfg(feature = "std")]
        {
            static CPU: std::sync::OnceLock<CpuFeatures> = std::sync::OnceLock::new();
            *CPU.get_or_init(|| {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    Self {
                        avx2: std::is_x86_feature_detected!("avx2"),
                        fma: std::is_x86_feature_detected!("fma"),
                        avx512f: std::is_x86_feature_detected!("avx512f"),
                        neon: false,
                    }
                }
                #[cfg(target_arch = "aarch64")]
                {
                    Self {
                        neon: std::arch::is_aarch64_feature_detected!("neon"),
                        ..Self::NONE
                    }
                }
                #[cfg(not(any(
                    target_arch = "x86",
                    target_arch = "x86_64",
                    target_arch = "aarch64"
                )))]
                {
                    Self::NONE
                }
            })
        }
        #[cfg(not(feature = "std"))]
        {
            Self::NONE
        }
    }

    /// Whether all required capabilities are present.
    ///
    /// # Examples
    /// ```
    /// use tprims_gemm_kernel::CpuFeatures;
    /// assert!(CpuFeatures::NONE.contains(CpuFeatures::NONE));
    /// assert!(!CpuFeatures::NONE.contains(CpuFeatures { avx2: true, ..CpuFeatures::NONE }));
    /// ```
    pub const fn contains(self, required: Self) -> bool {
        (!required.avx2 || self.avx2)
            && (!required.fma || self.fma)
            && (!required.avx512f || self.avx512f)
            && (!required.neon || self.neon)
    }

    /// Required capabilities absent from this CPU, for selection errors.
    ///
    /// # Examples
    /// ```
    /// use tprims_gemm_kernel::CpuFeatures;
    /// let req = CpuFeatures { avx2: true, ..CpuFeatures::NONE };
    /// assert_eq!(CpuFeatures::NONE.missing(req), req);
    /// ```
    pub const fn missing(self, required: Self) -> Self {
        Self {
            avx2: required.avx2 && !self.avx2,
            fma: required.fma && !self.fma,
            avx512f: required.avx512f && !self.avx512f,
            neon: required.neon && !self.neon,
        }
    }
}

/// ISA label for diagnostics; `CpuFeatures` defines the actual requirements.
///
/// # Examples
/// ```
/// use tprims_gemm_kernel::Isa;
/// assert_eq!(Isa::Portable, Isa::Portable);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Isa {
    /// Baseline scalar code.
    Portable,
    /// AVX2 (families normally also require FMA).
    Avx2,
    /// AVX-512 foundation.
    Avx512,
    /// AArch64 NEON.
    Neon,
}
