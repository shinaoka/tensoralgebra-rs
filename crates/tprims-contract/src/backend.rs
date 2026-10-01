//! The tprims implementation of the neutral contraction interface.

use strided_view::{StridedView, StridedViewMut};
use tprims_blas::{Conj, GemmConfig, Scalar};
use tprims_contract_traits as tr;
use tprims_contract_traits::{
    BoxedPlan, ContractionBackend, Diagnostics, HostExecution, PlanningBudget, PreparedContraction,
    Problem, Requirements, Result,
};

use crate::host::exec_of;
use crate::{ContractPlan, Flags, Selected, Strategy};

const ID: &str = "tprims-contract";

fn conj(c: tr::Conj) -> Conj {
    match c {
        tr::Conj::No => Conj::No,
        tr::Conj::Yes => Conj::Yes,
    }
}

/// The tprims contraction implementation behind the neutral interface:
/// permute+GEMM, packed direct or elementwise, chosen per [`Strategy`] with the
/// library defaults unchanged.
///
/// # Examples
///
/// ```
/// use tprims_contract::{ExecHost, TprimsBackend};
/// use tprims_contract_traits::*;
/// use tprims_exec::Exec;
/// use strided_view::{StridedView, StridedViewMut};
///
/// let backend: Box<dyn ContractionBackend<f64>> = Box::new(TprimsBackend::default());
/// let lay = Layout::new(&[2, 2], &[1, 2]);
/// let problem = Problem::new(
///     DotGeneral::new(&[1], &[0], &[], &[]), lay.clone(), lay.clone(), lay, (Conj::No, Conj::No));
/// let plan = backend.prepare(&problem, &Requirements::new(), &PlanningBudget::serial()).unwrap();
/// let (a, b, mut c) = ([1.0, 2.0, 3.0, 4.0], [1.0, 0.0, 0.0, 1.0], [0.0; 4]);
/// let av = StridedView::new(&a, &[2, 2], &[1, 2], 0).unwrap();
/// let bv = StridedView::new(&b, &[2, 2], &[1, 2], 0).unwrap();
/// let mut cv = StridedViewMut::new(&mut c, &[2, 2], &[1, 2], 0).unwrap();
/// plan.execute_into_accum(&ExecHost::new(&Exec::serial()), 1.0, &av, &bv, 0.0, &mut cv).unwrap();
/// assert_eq!(c, a);
/// ```
#[derive(Clone, Debug)]
pub struct TprimsBackend {
    /// Which implementation plans use.
    pub strategy: Strategy,
    /// Matrix engine and kernel choice, resolved when the plan is prepared.
    pub gemm: GemmConfig,
}

impl Default for TprimsBackend {
    fn default() -> Self {
        Self {
            strategy: Strategy::Auto,
            gemm: GemmConfig::default(),
        }
    }
}

struct TprimsPlan<T> {
    plan: ContractPlan<T>,
    problem: Problem,
    diag: Diagnostics,
}

impl<T: Scalar> ContractionBackend<T> for TprimsBackend {
    fn id(&self) -> &'static str {
        ID
    }

    fn prepare(
        &self,
        problem: &Problem,
        requirements: &Requirements,
        _budget: &PlanningBudget,
    ) -> Result<BoxedPlan<T>> {
        let plan = ContractPlan::<T>::new_with(
            &self.gemm,
            &problem.dot,
            (&problem.a.dims, &problem.a.strides),
            (&problem.b.dims, &problem.b.strides),
            (&problem.c.dims, &problem.c.strides),
            (conj(problem.conj.0), conj(problem.conj.1)),
            self.strategy,
            Flags {
                no_materialize: requirements.no_materialize,
            },
        )?;
        let (algorithm, materialized) = match plan.selected() {
            Selected::PermuteGemm { materialized } => ("permute-gemm", materialized),
            Selected::Tblis => ("packed-direct", [false; 3]),
            Selected::Elementwise => ("elementwise", [false; 3]),
        };
        Ok(Box::new(TprimsPlan {
            plan,
            // The plan snapshots layouts itself; this copy serves the shared
            // pre-write check so a mismatch is reported in neutral terms.
            problem: problem.clone(),
            diag: Diagnostics::new(ID, algorithm).with_materialized(materialized),
        }))
    }
}

impl<T: Scalar> PreparedContraction<T> for TprimsPlan<T> {
    fn execute_into_accum(
        &self,
        host: &dyn HostExecution,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        beta: T,
        c: &mut StridedViewMut<'_, T>,
    ) -> Result<()> {
        self.problem.check_views(
            (a.dims(), a.strides()),
            (b.dims(), b.strides()),
            (c.dims(), c.strides()),
        )?;
        let exec = exec_of(host)?;
        self.plan.execute(&exec, alpha, a, b, beta, c)
    }

    fn diagnostics(&self) -> &Diagnostics {
        &self.diag
    }
}
