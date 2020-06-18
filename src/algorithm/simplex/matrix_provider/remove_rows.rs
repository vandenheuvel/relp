//! # Removing rows from a matrix provider without modifying it
//!
//! When a matrix is represented with an unusual backend, like a network, it might be practical to
//! remove rows from the matrix it represents without having to adapt the underlying implementation.
//! This module provides a wrapper around any matrix provider that removes rows from it.
use std::collections::HashSet;
use std::fmt::{Display, Formatter};
use std::fmt;
use std::marker::PhantomData;

use itertools::repeat_n;

use crate::algorithm::simplex::matrix_provider::{Column, MatrixProvider};
use crate::algorithm::simplex::matrix_provider::variable::FeasibilityLogic;
use crate::algorithm::utilities::remove_sparse_indices;
use crate::data::linear_algebra::traits::SparseElementZero;
use crate::data::linear_algebra::vector::{Dense as DenseVector, Sparse as SparseVector, Vector};
use crate::data::linear_program::elements::BoundDirection;
use crate::data::number_types::traits::Field;

/// Wraps a `MatrixProvider` deleting some of its constraint rows, variable bounds should not be
/// deleted.
///
/// Used for deleting duplicate constraints after finding primal feasibility.
#[derive(PartialEq, Debug)]
pub struct RemoveRows<'a, F: Field, FZ: SparseElementZero<F>, MP: MatrixProvider<F, FZ>> {
    provider: &'a MP,
    /// List of rows that this method removes.
    ///
    /// Sorted at all times.
    pub rows_to_skip: Vec<usize>,
    // TODO(OPTIMIZATION): Consider using a `HashMap`.

    phantom_number_type: PhantomData<F>,
    phantom_number_type_zero: PhantomData<FZ>,
}

impl<'a, F, FZ, MP> RemoveRows<'a, F, FZ, MP>
    where
        F: Field + 'a,
        FZ: SparseElementZero<F>,
        MP: MatrixProvider<F, FZ> + 'a,
{
    /// Create a new `RemoveRows` instance.
    ///
    /// # Arguments
    ///
    /// * `provider`: Reference to an instance implementing the `MatrixProvider` trait. Rows from
    /// this provider will be removed.
    /// * `rows_to_skip`: A **sorted** list of rows that are skipped.
    ///
    /// # Return value
    ///
    /// A new `RemoveRows` instance.
    pub fn new(provider: &'a MP, rows_to_skip: Vec<usize>) -> Self {
        debug_assert!(rows_to_skip.is_sorted());
        debug_assert_eq!(rows_to_skip.iter().collect::<HashSet<_>>().len(), rows_to_skip.len());

        RemoveRows {
            provider,
            rows_to_skip,

            phantom_number_type: PhantomData,
            phantom_number_type_zero: PhantomData,
        }
    }

    /// Get the index of the same row in the original `MatrixProvider`.
    ///
    /// # Arguments
    ///
    /// * `i`: Index of row in the version of the problem from which rows were removed (this
    /// struct).
    ///
    /// # Return value
    ///
    /// Index of row in the original problem.
    pub fn get_underlying_row_index(&self, i: usize) -> usize {
        debug_assert!(i < self.provider.nr_rows() - self.rows_to_skip.len());

        Self::get_underlying_index(&self.rows_to_skip, i)
    }

    /// Delete a row
    ///
    /// # Arguments
    ///
    /// * `i`: Index of row in the version of the problem from which rows were removed (this
    /// struct).
    pub fn delete_row(&mut self, i: usize) {
        debug_assert!(i < self.provider.nr_rows() - self.rows_to_skip.len());

        let in_original_problem = self.get_underlying_row_index(i);
        debug_assert!(self.rows_to_skip.contains(&in_original_problem));
        let insertion_index = match self.rows_to_skip.binary_search(&in_original_problem) {
            Ok(_) => panic!("Deleting a row that already was deleted!"),
            Err(nr) => nr,
        };
        self.rows_to_skip.insert(insertion_index, in_original_problem);
    }

    /// How many constraints were removed.
    pub fn nr_constraints_deleted(&self) -> usize {
        self.rows_to_skip.len()
    }
}

impl<'a, F: 'a, FZ, MP> RemoveRows<'a, F, FZ, MP>
    where
        F: Field,
        FZ: SparseElementZero<F>,
        MP: 'a + MatrixProvider<F, FZ>,
{
    /// Method abstracting over the row and column getter methods.
    ///
    /// # Arguments
    ///
    /// * `i`: Index in the reduced version of the problem.
    ///
    /// # Return value
    ///
    /// Index in the original problem.
    fn get_underlying_index(skip_indices_array: &Vec<usize>, i: usize) -> usize {
        if skip_indices_array.len() == 0 {
            // If no indices have been deleted, it's just the original value
            i
        } else if skip_indices_array.len() == 1 {
            // If one index has been deleted, see if that was before of after the value tested
            if i < skip_indices_array[0] {
                i
            } else {
                i + skip_indices_array.len()
            }
        } else {
            // skip_indices_array >= 2
            if i < skip_indices_array[0] {
                i
            } else {
                // Binary search with invariants:
                //   1. skip_indices_array[lower_bound] - lower_bound <= i
                //   2. skip_indices_array[upper_bound] - upper_bound > i
                let (mut lower_bound, mut upper_bound) = (0, skip_indices_array.len());
                while upper_bound - lower_bound != 1 {
                    let middle = (lower_bound + upper_bound) / 2;
                    if skip_indices_array[middle] - middle <= i {
                        lower_bound = middle
                    } else {
                        upper_bound = middle
                    }
                }

                i + upper_bound
            }
        }
    }

    /// Method abstracting over the row and column deletion methods.
    ///
    /// # Arguments
    ///
    /// * `i`: Index in the reduced version of the problem to be deleted from the original problem.
    fn delete_index(skip_indices_array: &mut Vec<usize>, i: usize) {
        let in_original_problem = Self::get_underlying_index(skip_indices_array, i);
        debug_assert!(skip_indices_array.contains(&in_original_problem));

        let insertion_index = match skip_indices_array.binary_search(&in_original_problem) {
            Ok(_) => panic!("Deleting an index that already was deleted."),
            Err(nr) => nr,
        };
        skip_indices_array.insert(insertion_index, in_original_problem);
    }
}


impl<'a, F, FZ, MP> MatrixProvider<F, FZ> for RemoveRows<'a, F, FZ, MP>
    where
        F: Field + 'a,
        FZ: SparseElementZero<F>,
        MP: MatrixProvider<F, FZ>,
{
    fn column(&self, j: usize) -> Column<&F, FZ, F> {
        match self.provider.column(j) {
            Column::Sparse(mut vector) => {
                vector.remove_indices(&self.rows_to_skip);
                Column::Sparse(vector)
            },
            Column::Slack(index, value) => {
                match self.rows_to_skip.binary_search(&index) {
                    Ok(_) => {
                        let new_length = self.provider.nr_rows() - self.rows_to_skip.len();
                        Column::Sparse(SparseVector::new(Vec::with_capacity(0), new_length))
                    },
                    Err(skipped_before) => {
                        Column::Slack(index - skipped_before, value)
                    },
                }
            },
        }
    }

    fn cost_value(&self, j: usize) -> &F {
        self.provider.cost_value(j)
    }

    fn constraint_values(&self) -> DenseVector<F> {
        let mut all = self.provider.constraint_values();
        all.remove_indices(&self.rows_to_skip);
        all
    }

    fn bound_row_index(&self, j: usize, bound_type: BoundDirection) -> Option<usize> {
        self.provider.bound_row_index(j, bound_type).map(|nr| nr - self.nr_constraints_deleted())
    }

    fn bounds(&self, j: usize) -> (&F, &Option<F>) {
        self.provider.bounds(j)
    }

    fn positive_slack_indices(&self) -> Vec<(usize, usize)> {
        let mut from_parent = self.provider.positive_slack_indices();
        remove_sparse_indices(&mut from_parent, &self.rows_to_skip);
        from_parent
    }

    fn nr_positive_slacks(&self) -> usize {
        // Requires introduction of a counter, but this code should never be run anyways (this is
        // never part of a first phase search for a feasible value, when this is relevant).
        unimplemented!();
    }

    /// This implementation assumes that only constraint rows are removed from the `MatrixProvider`.
    fn nr_constraints(&self) -> usize {
        self.provider.nr_constraints() - self.nr_constraints_deleted()
    }

    /// This implementation assumes that only constraint rows are removed from the `MatrixProvider`.
    fn nr_bounds(&self) -> usize {
        self.provider.nr_bounds()
    }

    fn nr_rows(&self) -> usize {
        self.provider.nr_rows() - self.nr_constraints_deleted()
    }

    fn nr_columns(&self) -> usize {
        self.provider.nr_columns()
    }

    fn reconstruct_solution<FZ2: SparseElementZero<F>>(
        &self,
        column_values: SparseVector<F, FZ2, F>,
    ) -> SparseVector<F, FZ2, F> {
        self.provider.reconstruct_solution(column_values)
    }
}

impl<'a, F, FZ, MP> FeasibilityLogic<'a, F, FZ> for RemoveRows<'a, F, FZ, MP>
    where
        F: Field + 'a,
        FZ: SparseElementZero<F>,
        MP: MatrixProvider<F, FZ> + FeasibilityLogic<'a, F, FZ>,
{
    fn is_feasible(&self, j: usize, value: F) -> bool {
        self.provider.is_feasible(j, value)
    }

    fn closest_feasible(&self, j: usize, value: F) -> (Option<F>, Option<F>) {
        self.provider.closest_feasible(j, value)
    }
}

impl<'a, F, FZ, MP> Display for RemoveRows<'a, F, FZ, MP>
    where
        F: Field + 'a,
        FZ: SparseElementZero<F>,
        MP: MatrixProvider<F, FZ>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let width = 8;

        write!(f, "{}", repeat_n(" ", width).collect::<Vec<_>>().concat())?;
        for column in 0..self.nr_columns() {
            write!(f, "{:^width$}", column, width = width)?;
        }
        writeln!(f)?;
        writeln!(f, "{}", repeat_n("-",(1 + self.nr_columns()) * width).collect::<String>())?;

        for row in 0..self.nr_rows() {
            write!(f, "{:>width$}", format!("{} |", row), width = width)?;
            for column in 0..self.nr_columns() {
                let x = self.column(column);
                let value = format!("{}", match &x {
                    Column::Sparse(ref vector) => vector[row].clone(),
                    &Column::Slack(index, direction) => if index == row {
                        direction.into()
                    } else {
                        F::zero()
                    },
                });
                write!(f, "{:^width$}", value, width = width)?;
            }
            writeln!(f)?;
        }
        writeln!(f)
    }
}

#[cfg(test)]
mod test {
    use num::rational::Ratio;

    use crate::algorithm::simplex::matrix_provider::matrix_data::MatrixData;
    use crate::algorithm::simplex::matrix_provider::remove_rows::RemoveRows;

    #[test]
    fn get_underlying_index() {
        type T = Ratio<i32>;

        for (deleted, size) in vec![
            (vec![2, 5, 7, 9, 12, 15, 16, 19, 20, 21], 25),
            (vec![2], 5),
            (vec![2, 3], 6),
        ] {
            let left_from_original = (0..size).filter(|i| !deleted.contains(i)).collect::<Vec<_>>();
            for (in_reduced, in_original) in (0..left_from_original.len()).zip(left_from_original.into_iter()) {
                assert_eq!(RemoveRows::<T, T, MatrixData<T, T>>::get_underlying_index(&deleted, in_reduced), in_original);
            }
        }
    }
}
