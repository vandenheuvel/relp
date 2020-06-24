//! # Maintaining a basis inverse
//!
//! The simplex method requires us to keep track of the basis inverse. The latest constraint values
//! and a variable called `minus_pi` are also tracked, as is the objective value.
//!
//! It is currently only possible to track this inverse using a sparse, row-major matrix (no
//! factorization).
//!
//! TODO(ENHANCEMENT): A better inverse maintenance algorithm. Start with factorization?
use std::fmt::Display;

use crate::algorithm::two_phase::matrix_provider::{Column, MatrixProvider};
use crate::algorithm::two_phase::matrix_provider::remove_rows::RemoveRows;
use crate::data::linear_algebra::vector::{Dense as DenseVector, Sparse as SparseVector};

pub mod carry;

/// Maintain a basis inverse.
///
/// Should facilitate quick solving of a linear system.
pub trait InverseMaintenance<F, FZ>: Display {
    /// Create a `CarryMatrix` for a tableau with only artificial variables.
    ///
    /// Only used if it is known in advance that there will be no positive slacks.
    ///
    /// # Arguments
    ///
    /// * `b`: Constraint values of the original problem, i.e. the original `b` with respect to the
    /// unit basis.
    ///
    /// # Return value
    ///
    /// `CarryMatrix` with a `minus_pi` equal to -1's and the standard basis.
    fn create_for_fully_artificial(b: DenseVector<F>) -> Self;

    /// Create a `CarryMatrix` for a tableau with some artificial variables.
    ///
    /// Only used if unknown in advance how many positive slacks there will be.
    ///
    /// # Arguments
    ///
    /// * `artificial`: Indices of rows where an artificial variable is needed.
    /// * `basis`: (row index, column index) tuples of given basis variables.
    /// * `b`: Constraint values of the original problem, i.e. the original `b` with respect to the
    /// unit basis.
    ///
    /// # Return value
    ///
    /// `CarryMatrix` with a `minus_pi` equal to -1's and the standard basis.
    fn create_for_partially_artificial(
        artificial: &Vec<usize>,
        basis: &Vec<(usize, usize)>,
        b: DenseVector<F>,
    ) -> Self;

    /// Create a basis inverse when only the basis indices are known.
    ///
    /// Often, one would invert a matrix right at the start here to build up the basis inverse
    /// representation.
    ///
    /// # Arguments
    ///
    /// * `basis`: Indices of columns that are to be in the basis. Should match the number of rows
    /// of the provider. Values should be unique, could have been a set.
    /// * `provider`: Problem representation.
    fn from_basis(basis: &Vec<usize>, provider: &impl MatrixProvider<F, FZ>) -> Self;

    /// Create a basis inverse when the basis indices and their pivot rows are known.
    ///
    /// Often, one would invert a matrix right at the start here to build up the basis inverse
    /// representation.
    ///
    /// # Arguments
    ///
    /// * `basis`: Indices of columns that are to be in the basis. Should match the number of rows
    /// of the provider. Values should be unique, could have been a set.
    /// * `provider`: Problem representation.
    fn from_basis_pivots(
        basis: &Vec<(usize, usize)>,
        provider: &impl MatrixProvider<F, FZ>,
    ) -> Self;

    /// When a previous basis inverse representation was used to find a basic feasible solution.
    ///
    /// Artificial variables need to be removed somewhere in the process that calls this method.
    ///
    /// TODO(ARCHITECTURE): Can this method be removed? Aren't there too many constructors?
    ///
    /// # Arguments
    /// * `artificial`: Indices of rows where an artificial variable is needed.
    /// * `provider`: Original problem representation.
    /// * `basis`: (row index, column index) tuples of given basis variables.
    fn from_artificial<MP: MatrixProvider<F, FZ>>(
        artificial: Self,
        provider: &MP,
        basis: &Vec<usize>,
    ) -> Self;

    /// When a previous basis inverse representation was used to find a basic feasible solution.
    ///
    /// Artificial variables need to be removed somewhere in the process that calls this method.
    ///
    /// TODO(ARCHITECTURE): Can this method be removed? Aren't there too many constructors?
    ///
    /// # Arguments
    /// * `artificial`: Indices of rows where an artificial variable is needed.
    /// * `provider`: Original problem representation.
    /// * `basis`: (row index, column index) tuples of given basis variables.
    fn from_artificial_remove_rows<MP: MatrixProvider<F, FZ>>(
        artificial: Self,
        rows_removed: &RemoveRows<F, FZ, MP>,
        basis_indices: &Vec<usize>,
    ) -> Self;

    /// Update the basis by representing one row reduction operation.
    ///
    /// Supply a column, that should become part of the basis, to this function. Then this
    /// `CarryMatrix` gets updated such that if one were to matrix-multiply the concatenation column
    /// `[cost, c for c in column]` with this `CarryMatrix`, an (m + 1)-dimensional unitvector would
    /// be the result.
    ///
    /// # Arguments
    ///
    /// * `pivot_row_index`: Index of the pivot row.
    /// * `column`: Column relative to the current basis to be entered into that basis.
    /// * `cost`: Relative cost of that column. The objective function value will change by this
    /// amount.
    fn change_basis(&mut self, pivot_row_index: usize, column: &SparseVector<F, FZ, F>, cost: F);

    /// Calculates the cost difference `c_j`.
    ///
    /// This cost difference is the inner product of `minus_pi` and the column.
    fn cost_difference(&self, original_column: Column<&F, FZ, F>) -> F;

    /// Multiplies the submatrix consisting of `minus_pi` and B^-1 by a original_column.
    ///
    /// # Arguments
    ///
    /// * `original_column`: A `SparseVector<T>` of length `m`.
    ///
    /// # Return value
    ///
    /// A `SparseVector<T>` of length `m`.
    fn generate_column(&self, original_column: Column<&F, FZ, F>) -> SparseVector<F, FZ, F>;

    /// Generate a single element in the tableau with respect to the current basis.
    ///
    /// # Arguments
    ///
    /// * `i`: Row index
    /// * `original_column`: Column with respect to the original basis.
    fn generate_element(&self, i: usize, original_column: Column<&F, FZ, F>) -> F;

    /// Clone the latest constraint vector.
    ///
    /// TODO: Can this be avoided?
    fn b(&self) -> DenseVector<F>;

    /// Get the objective function value for the current basis.
    ///
    /// # Return value
    ///
    /// The objective value.
    fn get_objective_function_value(&self) -> F;

    /// Get the `i`th constraint value relative to the current basis.
    ///
    /// # Arguments
    ///
    /// * `i`: Index of the constraint to retrieve value from, in range `0` until `m`.
    ///
    /// # Return value
    ///
    /// The constraint value.
    fn get_constraint_value(&self, i: usize) -> &F;
}
