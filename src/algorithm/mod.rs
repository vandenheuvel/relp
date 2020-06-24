//! # Algorithms
use crate::data::linear_algebra::vector::Sparse;

pub mod two_phase;
pub mod criss_cross;
pub mod primal_dual;
pub mod utilities;

/// A problem formulation of which a relaxation can be solved.
/// 
/// Implementations of this trait are specialized according to specific properties. E.g. if a 
/// specific matrix problem has a trivial basic feasible solution, this can be made clear by 
/// implementing a trait that provides this basic feasible solution, and then a specialized
/// implementation of this trait can use that basic feasible solution to bypass the search for one.
pub trait SolveRelaxation<OF, OFZ> {
    /// Solve the relaxed version of this problem.
    /// 
    /// In the case of linear programming, that means that integer constraints are ignored.
    /// 
    /// # Return value
    /// 
    /// Whether the problem is feasible, and if so, a solution if the problem is bounded.
    fn solve_relaxation(&self) -> OptimizationResult<OF, OFZ>;
}

/// A linear program is either infeasible, unbounded or has a finite optimum.
///
/// This is determined as the result of an algorithm
#[allow(missing_docs)]
#[derive(Eq, PartialEq, Debug)]
pub enum OptimizationResult<F, FZ> {
    Infeasible,
    FiniteOptimum(Sparse<F, FZ, F>),
    Unbounded,
}
