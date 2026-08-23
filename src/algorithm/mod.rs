//! # Algorithms
use relp_num::Absorb;

use crate::algorithm::two_phase::matrix_provider::MatrixProvider;
use crate::algorithm::two_phase::matrix_provider::column::Column;
use crate::algorithm::two_phase::tableau::inverse_maintenance::InverseMaintainer;
use crate::algorithm::two_phase::tableau::inverse_maintenance::ops as im_ops;
use crate::algorithm::two_phase::tableau::kind::artificial::Cost as ArtificialCost;
use crate::data::linear_algebra::vector::{DenseVector, SparseVector};

pub mod two_phase;

/// A problem formulation of which a relaxation can be solved.
///
/// Implementations of this trait are specialized according to specific properties. E.g. if a
/// specific matrix problem has a trivial basic feasible solution, this can be made clear by
/// implementing a trait that provides this basic feasible solution, and then a specialized
/// implementation of this trait can use that basic feasible solution to bypass the search for one.
pub trait SolveRelaxation: MatrixProvider {
    /// Solve the relaxed version of this problem.
    ///
    /// In the case of linear programming, that means that integer constraints are ignored.
    ///
    /// # Return value
    ///
    /// Whether the problem is feasible, and if so, a solution if the problem is bounded.
    fn solve_relaxation<'provider, IM>(&'provider self) -> OptimizationResult<IM::F>
    where
        IM: InverseMaintainer<F: im_ops::FieldHR>,
        IM::F: Absorb<<Self::Column as Column>::F>,
        IM::F: Absorb<ArtificialCost>,
        IM::F: Absorb<Self::Cost<'provider>>,
        IM::F: Absorb<Self::Rhs>;
}

/// A linear program is either infeasible, unbounded or has a finite optimum.
///
/// This is determined as the result of an algorithm
#[allow(missing_docs)]
#[derive(Eq, PartialEq, Debug)]
pub enum OptimizationResult<F> {
    Infeasible,
    FiniteOptimum(TableauSolution<F>),
    Unbounded,
}

/// An optimal primal and dual pair for the problem a [`MatrixProvider`] presents.
///
/// All indices count rows and columns of the provider as it stands, including the ones it only
/// simulates, and including any row the algorithm found redundant and set aside.
///
/// [`MatrixProvider`]: crate::algorithm::two_phase::matrix_provider::MatrixProvider
#[derive(Eq, PartialEq, Debug)]
pub struct TableauSolution<F> {
    /// The value of each column. Zeros are left out.
    pub primal: SparseVector<F, F>,
    /// The multiplier of each row: `pi = c_B^T B^-1`.
    ///
    /// A row the algorithm removed as redundant gets a zero, which is a valid multiplier for it:
    /// such a row is a combination of the rows that were kept, so leaving it out of the dual
    /// solution changes nothing that the kept rows do not already say.
    pub dual: DenseVector<F>,
    /// Whether each column is in the basis.
    pub basic: Vec<bool>,
    /// Rows set aside in the first phase because the constraint matrix did not have full rank.
    ///
    /// Sorted and without repeats.
    pub redundant_rows: Vec<usize>,
}

impl<F> TableauSolution<F> {
    /// The value of each column, leaving out the zeros.
    pub fn primal(&self) -> &SparseVector<F, F> {
        &self.primal
    }
}
