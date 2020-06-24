//! # Representing linear programs for rapid read access
//!
//! The Simplex method algorithms work on a tableau. Because this tableau is very sparse in
//! practice, we store in a matrix that describes the current basis together with the original
//! (also sparse) matrix data. This module contains structures that can provide a matrix.
use std::fmt::Debug;

use crate::data::linear_algebra::traits::SparseElementZero;
use crate::data::linear_algebra::vector::{Dense, Sparse as SparseVector};
use crate::data::linear_program::elements::BoundDirection;

pub mod matrix_data;
pub mod remove_rows;
pub mod variable;

/// Abstract interface for a matrix and constraint vector.
///
/// This is the data of the "problem relative to the initial basis"; that is, nothing in
/// data structures implementing this trait determines a basis. The implementors of this trait
/// should be primarily read-only, with basis changes, the `CarryMatrix` fields of the `Tableau`
/// should change instead.
///
/// Note that a this trait doesn't have to be implemented by a (sparse) matrix data structure per
/// se; it could also be implemented by a graph, which lets itself be represented by data in a
/// matrix.
/// The indexing for the variables and constraints is as follows:
///
/// /                 || Vars of which we want a solution | Constraint slack vars | Bound slack vars |
/// ==================||==================================|=======================|==================|-----
/// Constraints       ||            constants             |       constants       |         0        || b |
/// ------------------||----------------------------------|-----------------------|------------------||---|
///                   ||                                  |                       | +/- 1            |
/// Bound constraints ||    constants (one 1 per row)     |           0           |       +/- 1      |
///                   ||                                  |                       |            +/- 1 |
/// --------------------------------------------------------------------------------------------------
///
/// TODO: Use existential types to avoid allocations, for example by using `impl Iterator` as the
///  return type of the `column` method.
pub trait MatrixProvider<F, FZ> {
    /// Column of the problem.
    ///
    /// # Arguments
    ///
    /// * `j`: Column index.
    ///
    /// # Return value
    ///
    /// A sparse vector.
    fn column(&self, j: usize) -> Column<&F, FZ, F>;

    /// Cost of a variable.
    ///
    /// # Arguments
    ///
    /// * `j`: Column index.
    ///
    /// # Return value
    ///
    /// Cost value.
    fn cost_value(&self, j: usize) -> &F;

    /// Constraint values.
    ///
    /// Note: constraint values of both the constraints and bounds. Lengths should be
    /// `self.nr_rows()`.
    ///
    /// # Return value
    ///
    /// A dense vector of constraint values, often called `b` in mathematical notation.
    fn constraint_values(&self) -> Dense<F>;

    /// Index of the row of a virtual bound, if any.
    ///
    /// TODO(ARCHITECTURE): Currently, the return value is a row index. Make this relative to
    ///  `self.nr_constraints`?
    ///
    /// # Arguments
    ///
    /// * `j`: Variable index for the bound, if it exists.
    /// * `bound_type`: Whether it concerns a lower or upper bound.
    ///
    /// # Return value
    ///
    /// The index of the row in which the bound is virtually represented, if the bound exists.
    fn bound_row_index(&self, j: usize, bound_type: BoundDirection) -> Option<usize>;

    /// The lower and upper bound for a variable.
    ///
    /// # Arguments
    ///
    /// * `j`: Variable index for the bounds, if any exist.
    ///
    /// # Return value
    ///
    /// A tuple for the (lower, upper) bounds.
    fn bounds(&self, j: usize) -> (&F, Option<&F>);

    /// The number of constraints in the problem. This excludes simple variable bounds.
    fn nr_constraints(&self) -> usize;

    /// The number of simple variable bounds in the problem. This excludes more complicated
    /// constraints.
    fn nr_bounds(&self) -> usize;

    /// The total number of rows in the provided virtual matrix.
    fn nr_rows(&self) -> usize {
        self.nr_constraints() + self.nr_bounds()
    }

    /// The total number of columns in the provided virtual matrix. This does not include artificial
    /// variables; those are virtually represented by the `Artificial` `TableauType`.
    fn nr_columns(&self) -> usize;

    /// Reconstruct a solution.
    ///
    /// Not all variables that a provider presents to the solution algorithms might be relevant for
    /// the final solution. Free variables that are split, for example, could be recombined here.
    ///
    /// # Arguments
    ///
    /// * `column_values`: A solution for each of the variables that this provider presents.
    ///
    /// # Return value
    ///
    /// A solution that might be smaller than the number of variables in this problem.
    fn reconstruct_solution<FZ2: SparseElementZero<F>>(
        &self,
        column_values: SparseVector<F, FZ2, F>,
    ) -> SparseVector<F, FZ2, F>;
}

/// Either a vector with (probably several) elements, or a slack column.
///
/// Introduced for the interface of
#[derive(Eq, PartialEq, Debug)]
pub enum Column<F, FZ, C> {
    /// A (probably non-slack) column.
    Sparse(SparseVector<F, FZ, C>),
    /// A slack column.
    Slack(usize, BoundDirection),
}
