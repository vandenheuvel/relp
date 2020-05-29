//! # Representing linear programs for rapid read access
//!
//! The Simplex method algorithms work on a tableau. Because this tableau is very sparse in
//! practice, we store in a matrix that describes the current basis together with the original
//! (also sparse) matrix data. This module contains structures that can provide a matrix.
use crate::data::linear_algebra::matrix::{ColumnMajorOrdering, MatrixOrder, SparseMatrix};
use crate::data::linear_algebra::vector::{DenseVector, SparseVector, Vector};
use crate::data::number_types::traits::Field;

pub mod matrix_data;
// TODO
//pub mod network;

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
pub trait MatrixProvider<F: Field> {
    /// Column of the problem.
    ///
    /// # Arguments
    ///
    /// * `j`: Column index.
    ///
    /// # Return value
    ///
    /// A sparse vector.
    fn column(&self, j: usize) -> SparseVector<F>;

    /// Cost of a variable.
    ///
    /// # Arguments
    ///
    /// * `j`: Column index.
    ///
    /// # Return value
    ///
    /// Cost value.
    fn cost_value(&self, j: usize) -> F;

    /// Constraint values.
    ///
    /// # Return value
    ///
    /// A dense vector of constraint values, often called `b` in mathematical notation.
    fn constraint_values(&self) -> DenseVector<F>;

    /// Index of the row of a virtual bound, if any.
    ///
    /// # Arguments
    ///
    /// * `j`: Variable index for the bound, if it exists.
    /// * `bound_type`: Whether it concerns a lower or upper bound.
    ///
    /// # Return value
    ///
    /// The index of the row in which the bound is virtually represented, if the bound exists.
    fn bound_row_index(&self, j: usize, bound_type: BoundType) -> Option<usize>;

    /// The lower and upper bound for a variable.
    ///
    /// # Arguments
    ///
    /// * `j`: Variable index for the bounds, if any exist.
    ///
    /// # Return value
    ///
    /// A tuple for the (lower, upper) bounds.
    fn bounds(&self, j: usize) -> (F, Option<F>);

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

    ///
    fn as_sparse_matrix(&self) -> SparseMatrix<F, ColumnMajorOrdering> {
        ColumnMajorOrdering::new(
            (0..self.nr_columns())
                .map(|j| self.column(j).iter_values().copied().collect())
                .collect(),
            self.nr_rows(),
            self.nr_columns(),
        )
    }

    fn reconstruct_solution(&self, column_values: SparseVector<F>) -> SparseVector<F>;
}

/// A bound introduced in branch and bound is of type '<=' or '>='.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BoundType {
    /// 0 <= x <= b
    Upper,
    /// x >= b >= 0
    Lower,
}

/// Logic for testing whether variables are feasible.
///
/// Defined as a separate trait from `MatrixProvider`. Matrices are defined over fields, and so
/// the `MatrixProvider` be. Some of the logic of variable feasibility is more part of linear
/// programming algorithms specifically, which are only defined over ordered fields. This logic is
/// thus separated into a different trait, which depends on the other trait.
pub trait VariableFeasibilityLogic<F: Field>: MatrixProvider<F> {
    /// Whether a variable is integer.
    ///
    /// # Arguments
    ///
    /// * `j`: Variable index.
    ///
    /// # Return value
    ///
    /// `true` if the variable is integer, `false` otherwise.
    fn is_feasible(&self, j: usize, value: F) -> bool;

    /// Closest feasible variable to the left and right.
    ///
    /// # Arguments
    ///
    /// * `j`: Variable index.
    ///
    /// # Return value
    ///
    /// Two `Option`s, one for the closest feasible value to the left, one for the closest feasible
    /// value to the right. Note that these values might be equal, if there is only one feasible
    /// value.
    fn closest_feasible(&self, j: usize, value: F) -> (Option<F>, Option<F>);
}
