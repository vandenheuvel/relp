//! # Tableau types: artificial or not
//!
//! A tableau can contain artificial variables. They can be used to find a feasible solution in a
//! two-phase algorithm: the first phase finds a basic feasible solution, the second improves it.
//!
//! The `Tableau` type and algorithm logic in the parent modules is independent or whether a tableau
//! contains artificial variables, or not. This module enables those abstractions.
use std::collections::HashSet;
use std::marker::PhantomData;

use crate::algorithm::simplex::matrix_provider::{Column, MatrixProvider};
use crate::algorithm::simplex::matrix_provider::remove_rows::RemoveRows;
use crate::algorithm::simplex::tableau::inverse_maintenance::CarryMatrix;
use crate::algorithm::simplex::tableau::Tableau;
use crate::algorithm::utilities::remove_indices;
use crate::data::linear_algebra::traits::SparseElementZero;
use crate::data::linear_program::elements::BoundDirection;
use crate::data::number_types::traits::{Field, FieldRef};

/// The tableau type provides two different ways for the `Tableau` to function, depending on whether
/// any virtual artificial variables should be included in the problem.
///
/// This is not a typical trait, as the only two reasonable implementations are already implemented.
pub trait Kind<F: Field, FZ: SparseElementZero<F>>: Sized + PartialEq {
    /// Coefficient of variable `j` in the objective function.
    ///
    /// # Arguments
    ///
    /// * `j`: Column index of the variable, in range 0 until self.nr_columns().
    ///
    /// # Return value
    ///
    /// The cost of variable `j`.
    fn initial_cost_value(&self, j: usize) -> &F;

    /// Get the column from the original problem.
    ///
    /// Depending on whether the tableau is artificial or not, this requires either an artificial
    /// basis column, or a column from the original problem.
    fn original_column(&self, j: usize) -> Column<&F, FZ, F>;

    /// How many artificial variables are in the tableau.
    ///
    /// This number varies, because slack variables might have been recognized as practical
    /// candidates for basic feasible solutions by the `MatrixProvider` (the
    /// `positive_slack_indices` method).
    ///
    /// # Return value
    ///
    /// This number can be zero (for non artificial tableau's, represented by the `NonArtificial`
    /// struct), or any number through the number of rows (`self.nr_rows`).
    fn nr_artificial_variables(&self) -> usize;

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

/// The `TableauType` in case the `Tableau` contains artificial variables.
#[derive(Eq, PartialEq, Debug)]
pub struct Artificial<'a, F: Field, FZ: SparseElementZero<F>, MP: MatrixProvider<F, FZ>> {
    /// For the `i`th artificial variable was originally a simple basis vector with the coefficient
    /// at index `column_to_row[i]`.
    column_to_row: Vec<usize>,

    /// Values that can be referred to when unsized constants need to be returned.
    ///
    /// TODO(ARCHITECTURE): Replace with values that are Copy, or an enum?
    ONE: F,
    ZERO: F,
    /// Supplies data about the problem.
    ///
    /// This data doesn't change throughout the lifetime of this `Tableau`, and it is independent of
    /// the current basis as described by the `carry` and `basis_columns` attributes.
    provider: &'a MP,
    phantom_zero: PhantomData<FZ>,
}
impl<'a, F, FZ, MP> Kind<F, FZ> for Artificial<'a, F, FZ, MP>
    where
        F: Field,
        FZ: SparseElementZero<F>,
        MP: MatrixProvider<F, FZ>,
{
    /// Coefficient of variable `j` in the objective function.
    ///
    /// # Arguments
    ///
    /// * `j`: Column index of the variable, in range 0 until self.nr_columns().
    ///
    /// # Return value
    ///
    /// The cost of variable `j`.
    fn initial_cost_value(&self, j: usize) -> &F {
        debug_assert!(j < self.nr_artificial_variables() + self.provider.nr_columns());

        if j < self.nr_artificial_variables() {
            &self.ONE
        } else {
            &self.ZERO
        }
    }

    /// Retrieve an original column.
    ///
    /// # Arguments
    ///
    /// * `tableau`: Tableau to retrieve the column from.
    /// * `j`: Column index.
    ///
    /// # Return value
    ///
    /// The generated column, relative to the basis represented in the `Tableau`.
    fn original_column(&self, j: usize) -> Column<&F, FZ, F> {
        debug_assert!(j < self.nr_columns());

        if j < self.nr_artificial_variables() {
            Column::Slack(self.column_to_row[j], BoundDirection::Upper)
        } else {
            self.provider.column(j - self.nr_artificial_variables())
        }
    }

    fn nr_artificial_variables(&self) -> usize {
        self.column_to_row.len()
    }

    fn nr_rows(&self) -> usize {
        self.provider.nr_rows()
    }

    fn nr_columns(&self) -> usize {
        self.nr_artificial_variables() + self.provider.nr_columns()
    }
}

impl<'provider, F, FZ, MP> Tableau<F, FZ, Artificial<'provider, F, FZ, MP>>
    where
        F: Field + 'provider,
        for<'r> &'r F: FieldRef<F>,
        FZ: SparseElementZero<F>,
        MP: MatrixProvider<F, FZ>,
{
    /// Create a `Tableau` augmented with artificial variables.
    ///
    /// The tableau is then in a basic feasible solution having only the artificial variables in the
    /// basis.
    ///
    /// # Arguments
    ///
    /// * `provider`: Provides the problem to find a basic feasible solution for.
    ///
    /// # Return value
    ///
    /// The tableau.
    pub(crate) fn new(provider: &'provider MP) -> Self {
        let m = provider.nr_rows();

        // (row index, column index) coordinate tuples of suitable pivots in a slack column.
        let real = provider.positive_slack_indices();
        debug_assert!(real.is_sorted_by_key(|&(row, _column)| row));
        // Amount of slack variables that can be used for the initial artificial basis.
        let nr_real = real.len();
        // Amount of artificial variables that are still needed. Could be zero.
        let nr_artificial = m - nr_real;

        // Collect the rows that are not in the `real` value. Simple complement of the `real` set.
        let mut artificial = Vec::with_capacity(nr_artificial);
        let mut i = 0;
        for ith_artificial in 0..nr_artificial {
            while i < nr_real && ith_artificial + i == real[i].0 {
                i += 1;
            }
            artificial.push(ith_artificial + i);
            debug_assert!(!real.iter().any(|&(row, _)| row == ith_artificial + i));
        }
        debug_assert!(artificial.iter().all(|&row| row < m));

        // Create the map from the row to column indices where the initial basis elements are.
        //
        // Neither the artificial variables nor the real initial basis variables are necessarily in
        // a uninterrupted range, which is why this block is needed.
        debug_assert!(artificial.is_sorted() && real.is_sorted_by_key(|&(row, _column)| row));
        let mut artificial_counter = 0;
        let basis_indices = (0..m).map(|i| {
            let artificial_column = artificial_counter;
            let real_column = || nr_artificial + real[i - artificial_counter].1;

            let can_take_from_artificial = artificial_counter < nr_artificial;
            let can_take_from_real = i - artificial_counter < nr_real;

            match (can_take_from_artificial, can_take_from_real) {
                (true, true) => {
                    // Which index row is lower?
                    if artificial[artificial_column] < real[i - artificial_counter].0 {
                        artificial_counter += 1;
                        artificial_column
                    } else {
                        real_column()
                    }
                },
                (true, false) => {
                    artificial_counter += 1;
                    artificial_column
                },
                (false, true) => {
                    real_column()
                },
                (false, false) => {
                    unreachable!("If both `ac == nr_a` and `i - ac == nr_r`, loop should have ended")
                },
            }
        }).collect::<Vec<_>>();
        let basis_columns = basis_indices.iter().copied().collect();

        let carry = CarryMatrix::create_for_artificial(
            &artificial,
            &real,
            provider.constraint_values(),
        );

        Tableau {
            carry,
            basis_indices,
            basis_columns,

            kind: Artificial {
                column_to_row: artificial.clone(),

                ONE: F::one(),
                ZERO: F::zero(),

                provider,
                phantom_zero: PhantomData,
            },
        }
    }

    /// Creates a Simplex tableau with a specific basis.
    ///
    /// Currently only used for testing.
    ///
    /// # Arguments
    ///
    /// * `provider`: Provides the original problem for which the other arguments describe a basis.
    /// * `carry`: `CarryMatrix` with the basis transformation. Corresponds to `basis_indices`.
    /// * `basis_indices`: Maps each row to a column, describing a basis. Corresponds to `carry`.
    ///
    /// # Return value
    ///
    /// `Tableau` with for the provided problem with the provided basis.
    pub(crate) fn new_with_basis(
        provider: &'provider MP,
        carry: CarryMatrix<F, FZ>,
        basis_indices: Vec<usize>,
        basis_columns: HashSet<usize>,
        column_to_row_artificials: Vec<usize>,
    ) -> Self {
        debug_assert!(column_to_row_artificials.iter().all(|i| basis_indices.contains(&i)));

        Tableau {
            carry,
            basis_indices,
            basis_columns,

            kind: Artificial {
                column_to_row: column_to_row_artificials,

                ONE: F::one(),
                ZERO: F::zero(),

                provider,

                phantom_zero: PhantomData
            },
        }
    }

    /// Number of artificial variables in this tableau.
    pub fn nr_artificial_variables(&self) -> usize {
        self.kind.nr_artificial_variables()
    }

    /// Whether there are any artificial variables in the basis.
    pub fn has_artificial_in_basis(&self) -> bool {
        self.basis_columns.iter().any(|&c| c < self.nr_artificial_variables())
    }

    /// Get the indices of the artificial variables that are still in the basis.
    pub fn artificial_basis_columns(&self) -> HashSet<usize> {
        self.basis_columns
            .iter()
            .filter(|&&v| v < self.nr_artificial_variables())
            .copied()
            .collect()
    }

    pub fn pivot_row_from_artificial(&self, artificial_index: usize) -> usize {
        debug_assert!(artificial_index < self.nr_artificial_variables());
        // Only used to remove variables from basis
        debug_assert!(self.is_in_basis(&artificial_index));

        self.kind.column_to_row[artificial_index]
    }
}

/// The `TableauType` in case the `Tableau` does not contain any artificial variables.
///
/// This `Tableau` variant should only be constructed with a known feasible basis.
#[derive(Eq, PartialEq, Debug)]
pub struct NonArtificial<'a, F: Field, FZ: SparseElementZero<F>, MP: MatrixProvider<F, FZ>> {
    /// Supplies data about the problem.
    ///
    /// This data doesn't change throughout the lifetime of this `Tableau`, and it is independent of
    /// the current basis as described by the `carry` and `basis_columns` attributes.
    provider: &'a MP,

    phantom: PhantomData<(F, FZ)>,
}
impl<'a, F, FZ, MP> Kind<F, FZ> for NonArtificial<'a, F, FZ, MP>
    where
        F: Field,
        FZ: SparseElementZero<F>,
        MP: MatrixProvider<F, FZ>,
{
    /// Coefficient of variable `j` in the objective function.
    ///
    /// # Arguments
    ///
    /// * `tableau`: Tableau to retrieve the cost value from.
    /// * `j`: Column index of the variable, in range 0 until self.nr_columns().
    ///
    /// # Return value
    ///
    /// The cost of variable `j`.
    fn initial_cost_value(&self, j: usize) -> &F {
        debug_assert!(j < self.provider.nr_columns());

        self.provider.cost_value(j)
    }

    /// Retrieve an original column.
    ///
    /// # Arguments
    ///
    /// * `tableau`: Tableau to retrieve the column from.
    /// * `j`: Column index.
    ///
    /// # Return value
    ///
    /// The generated column, relative to the basis represented in the `Tableau`.
    fn original_column(&self, j: usize) -> Column<&F, FZ, F> {
        debug_assert!(j < self.provider.nr_columns());

        self.provider.column(j)
    }

    fn nr_artificial_variables(&self) -> usize {
        0
    }

    fn nr_rows(&self) -> usize {
        self.provider.nr_rows()
    }

    fn nr_columns(&self) -> usize {
        self.provider.nr_columns()
    }
}

impl<'provider, F, FZ, MP> Tableau<F, FZ, NonArtificial<'provider, F, FZ, MP>>
    where
        F: Field,
        for<'r> &'r F: FieldRef<F>,
        FZ: SparseElementZero<F>,
        MP: MatrixProvider<F, FZ>,
{
    /// Creates a Simplex tableau with a specific basis.
    ///
    /// Currently only used for testing.
    ///
    /// # Arguments
    ///
    /// * `provider`: Provides the original problem for which the other arguments describe a basis.
    /// * `carry`: `CarryMatrix` with the basis transformation. Corresponds to `basis_indices`.
    /// * `basis_indices`: Maps each row to a column, describing a basis. Corresponds to `carry`.
    ///
    /// # Return value
    ///
    /// `Tableau` with for the provided problem with the provided basis.
    pub(crate) fn new_with_basis(
        provider: &'provider MP,
        carry: CarryMatrix<F, FZ>,
        basis_indices: Vec<usize>,
        basis_columns: HashSet<usize>,
    ) -> Self {
        Tableau {
            carry,
            basis_indices,
            basis_columns,

            kind: NonArtificial {
                provider,

                phantom: PhantomData,
            },
        }
    }

    /// Create a `Tableau` from an artificial tableau.
    ///
    /// # Arguments
    ///
    /// * `artificial_tableau`: `Tableau` instance created with artificial variables.
    ///
    /// # Return value
    ///
    /// `Tableau` with the same basis, but non-artificial cost row.
    pub fn from_artificial(
        mut artificial_tableau: Tableau<F, FZ, Artificial<'provider, F, FZ, MP>>,
    ) -> Self {
        debug_assert!(artificial_tableau.artificial_basis_columns().is_empty());

        // Shift the basis column indices back
        let nr_artificial = artificial_tableau.nr_artificial_variables();
        artificial_tableau.basis_indices.iter_mut()
            .for_each(|column| *column -= nr_artificial);
        let basis_columns = artificial_tableau.basis_columns.into_iter()
            .map(|column| column - nr_artificial)
            .collect();

        let carry = CarryMatrix::from_artificial(
            artificial_tableau.carry,
            artificial_tableau.kind.provider,
            &artificial_tableau.basis_indices,
        );

        Tableau {
            carry,
            basis_indices: artificial_tableau.basis_indices,
            basis_columns,

            kind: NonArtificial {
                provider: artificial_tableau.kind.provider,

                phantom: PhantomData,
            },
        }
    }

    /// Create a `Tableau` from an artificial tableau while removing some rows.
    ///
    /// # Arguments
    ///
    /// * `artificial_tableau`: `Tableau` instance created with artificial variables.
    /// * `rows_removed`: `RemoveRows` instance containing all the **sorted** rows that should be
    /// removed from e.g. the basis inverse matrix.
    ///
    /// # Return value
    ///
    /// `Tableau` with the same basis, but non-artificial cost row.
    pub fn from_artificial_removing_rows<'b: 'provider>(
        artificial_tableau: Tableau<F, FZ, Artificial<'provider, F, FZ, MP>>,
        rows_removed: &'b RemoveRows<'provider, F, FZ, MP>,
    ) -> Tableau<F, FZ, NonArtificial<'b, F, FZ, RemoveRows<'provider, F, FZ, MP>>> {
        let nr_artificial = artificial_tableau.nr_artificial_variables();
        debug_assert!(
            artificial_tableau.basis_indices.iter()
                .all(|&v| v >= nr_artificial || rows_removed.rows_to_skip.contains(&v))
        );

        let mut basis_columns = artificial_tableau.basis_columns;
        for &row in &rows_removed.rows_to_skip {
            let was_there = basis_columns.remove(&artificial_tableau.basis_indices[row]);
            debug_assert!(was_there);
        }
        let basis_columns = basis_columns.into_iter().map(|j| j - nr_artificial).collect();

        let mut basis_indices = artificial_tableau.basis_indices;
        remove_indices(&mut basis_indices, &rows_removed.rows_to_skip);
        basis_indices.iter_mut().for_each(|index| *index -= nr_artificial);

        // Remove same row and column from carry matrix
        let carry = CarryMatrix::from_artificial_remove_rows(
            artificial_tableau.carry,
            rows_removed,
            &basis_indices,
        );

        Tableau {
            carry,
            basis_indices,
            basis_columns,

            kind: NonArtificial {
                provider: rows_removed,

                phantom: PhantomData,
            },
        }
    }
}
