//! # Tableau types: artificial or not
//!
//! A tableau can contain artificial variables. They can be used to find a feasible solution in a
//! two-phase algorithm: the first phase finds a basic feasible solution, the second improves it.
//!
//! The `Tableau` type and algorithm logic in the parent modules is independent or whether a tableau
//! contains artificial variables, or not. This module enables those abstractions.

use crate::algorithm::two_phase::matrix_provider::column::Column;

pub mod artificial;
pub mod non_artificial;

/// The tableau type provides two different ways for the `Tableau` to function, depending on whether
/// any virtual artificial variables should be included in the problem.
pub trait Kind<'provider> {
    /// Representation of the column of the tableau.
    ///
    /// TODO(ENHANCEMENT): Drop the Ordered requirement once specialization on generic type
    ///  type arguments of trait methods is possible.
    type Column: Column<'provider>;
    /// Cost row type.
    ///
    /// For artificial tableaus, this type is always the zero-one cost type in the artificial
    /// module.
    type Cost;

    /// Coefficient of variable `j` in the objective function.
    ///
    /// # Arguments
    ///
    /// * `j`: Column index of the variable, in range `0` until `self.nr_columns()`.
    ///
    /// # Return value
    ///
    /// The cost of variable `j`.
    fn initial_cost_value(&self, j: usize) -> Self::Cost;

    /// Get the column from the original problem.
    ///
    /// Depending on whether the tableau is artificial or not, this requires either an artificial
    /// basis column, or a column from the original problem.
    fn original_column(&self, j: usize) -> Self::Column;

    /// Number of rows in the matrix and tableau.
    ///
    /// This method is there to facilitate calls to the `MatrixProvider` that is held by the struct
    /// implementing this trait.
    fn nr_rows(&self) -> usize;

    /// Number of columns in the tableau.
    ///
    /// This number includes any artificial variables.
    fn nr_columns(&self) -> usize;
}
