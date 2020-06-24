//! # Carry matrix
//!
//! A sparse row major basis inverse with a dense representation of the latest b and -pi (see
//! Papadimitriou's Combinatorial Optimization).
use std::borrow::Borrow;
use std::collections::HashSet;
use std::fmt::{Display, Formatter};
use std::fmt;

use itertools::repeat_n;

use crate::algorithm::two_phase::matrix_provider::{Column, MatrixProvider};
use crate::algorithm::two_phase::matrix_provider::remove_rows::RemoveRows;
use crate::algorithm::two_phase::tableau::inverse_maintenance::InverseMaintenance;
use crate::algorithm::utilities::remove_indices;
use crate::data::linear_algebra::traits::{SparseComparator, SparseElement, SparseElementZero};
use crate::data::linear_algebra::vector::{Dense as DenseVector, Dense, Sparse as SparseVector, Sparse, Vector};
use crate::data::number_types::traits::{Field, FieldRef};

/// The carry matrix looks like:
///
///   obj  ||     -pi
/// ----------------------
///    |   ||
///    |   ||    basis
///    b   ||   inverse
///    |   ||     B^-1
///    |   ||
///
/// The `b` (and `minus_pi`) in this struct change continuously; they don't correspond to the `b`
/// (and `minus_pi`) of the original problem.
///
/// Used in simplex method, inside the loop, in the methods.
///
/// TODO: Write better docstring
#[derive(Eq, PartialEq, Clone, Debug)]
pub struct CarryMatrix<F: SparseElement<F> + SparseComparator, FZ: SparseElementZero<F>> {
    /// Negative of the objective function value.
    ///
    /// Is non-negative (objective value is non-positive) for artificial tableaus.
    minus_objective: F,
    /// Used to compute updated versions of relative cost of columns.
    minus_pi: DenseVector<F>,
    /// Latest version of the constraints.
    pub(super) b: DenseVector<F>,
    /// Basis inverse matrix, row major.
    ///
    /// TODO(ENHANCEMENT): Implement a faster basis inverse maintenance algorithm.
    basis_inverse_rows: Vec<SparseVector<F, FZ, F>>,
}

impl<F, FZ> CarryMatrix<F, FZ>
where
    F: Field,
    for <'r> &'r F: FieldRef<F>,
    FZ: SparseElementZero<F>,
{
    /// Create a `CarryMatrix` for a tableau with a known basis inverse.
    ///
    /// # Arguments
    ///
    /// * `b`: Constraint values of the original problem with respect to the provided basis.
    /// * `basis_inverse_rows`: Rows of the matrix B^-1, where B is the matrix of columns of the
    /// basis.
    ///
    /// # Return value
    ///
    /// `CarryMatrix` with the provided values and a `minus_pi` zero vector.
    pub fn new(
        minus_objective: F,
        minus_pi: DenseVector<F>,
        b: DenseVector<F>,
        basis_inverse_rows: Vec<SparseVector<F, FZ, F>>,
    ) -> Self {
        let m = minus_pi.len();
        debug_assert_eq!(minus_pi.len(), m);
        debug_assert_eq!(b.len(), m);
        debug_assert!(basis_inverse_rows.iter().all(|r| r.len() == m));

        CarryMatrix {
            minus_objective,
            minus_pi,
            b,
            basis_inverse_rows,
        }
    }


    /// Create a `CarryMatrix` from a carry matrix with an artificial cost row.
    ///
    /// # Arguments
    ///
    /// * `artificial_carry_matrix`: A `CarryMatrix` which has an artificial cost row.
    /// * `provider`: Provider of the problem, used for original cost in `minus_pi` restoration.
    /// * `basis`: Vector with, ordered by pivot row, the basis indices of the new basis.
    ///
    /// # Return value
    ///
    /// `CarryMatrix` with a restored `minus_pi` vector and objective value

    /// Create a `CarryMatrix` from a carry matrix with an artificial cost row while removing
    /// several row and column indices.
    ///
    /// # Arguments
    ///
    /// * `artificial_carry_matrix`: A `CarryMatrix` which has an artificial cost row.
    /// * `provider`: Provider of the reduced problem.
    /// * `basis`: Vector with, ordered by pivot row, the basis indices of the new basis.
    ///
    /// # Return value
    ///
    /// `CarryMatrix` of reduced dimension with a restored `minus_pi` vector and objective value.

    /// Create the `minus_pi` field from an existing basis.
    ///
    /// # Arguments
    ///
    /// * `basis_inverse_rows`: A basis inverse that represents a basic feasible solution.
    /// * `provider`: Matrix provider.
    /// * `basis`: Indices of the basis elements.
    fn create_minus_pi_from_artificial(
        basis_inverse_rows: &Vec<SparseVector<F, FZ, F>>,
        provider: &impl MatrixProvider<F, FZ>,
        basis: &Vec<usize>,
    ) -> DenseVector<F> {
        let m = basis_inverse_rows.len();
        debug_assert_eq!(provider.nr_rows(), m);
        debug_assert_eq!(basis.len(), m);

        let mut pi = repeat_n(F::zero(), m).collect::<Vec<_>>();
        for row in 0..m {
            for (column, value) in basis_inverse_rows[row].iter_values() {
                pi[*column] += provider.cost_value(basis[row]) * value;
            }
        }

        let data = pi.into_iter().map(|v| -v).collect::<Vec<_>>();
        let len = data.len();
        DenseVector::new(data, len)
    }

    /// Create the -pi value from an existing basis.
    ///
    /// # Arguments
    ///
    /// * `provider`: Matrix provider.
    /// * `basis`: Basis indices (elements are already shifted, no compensation for the artificial
    /// variables is needed).
    /// * `b`: Constraint values with respect to this basis.
    fn create_minus_obj_from_artificial<MP: MatrixProvider<F, FZ>>(
        provider: &MP,
        basis: &Vec<usize>,
        b: &DenseVector<F>,
    ) -> F {
        let mut objective = F::zero();
        for row in 0..provider.nr_rows() {
            objective += provider.cost_value(basis[row]) * &b[row];
        }
        -objective
    }

    /// Normalize the pivot row.
    ///
    /// That is, the pivot value will be set to `1`.
    ///
    /// # Arguments
    ///
    /// * `pivot_row_index`: Index of the pivot row.
    /// * `column`: Column relative to the current basis to be entered into that basis.
    fn normalize_pivot_row(
        &mut self,
        pivot_row_index: usize,
        column: &SparseVector<F, FZ, F>,
    ) {
        let pivot_value = &column[pivot_row_index];
        debug_assert_ne!(pivot_value, &F::zero());

        self.basis_inverse_rows[pivot_row_index].element_wise_divide(pivot_value);
        self.b[pivot_row_index] /= pivot_value;
    }

    /// Normalize the pivot row and row reduce the other basis inverse rows.
    ///
    /// # Arguments
    ///
    /// * `pivot_row_index`: Index of the pivot row.
    /// * `column`: Column relative to the current basis to be entered into that basis.
    ///
    /// # Note
    ///
    /// This method requires a normalized pivot element.
    fn row_reduce_update_basis_inverse_and_b(
        &mut self,
        pivot_row_index: usize,
        column: &SparseVector<F, FZ, F>,
    ) {
        debug_assert!(pivot_row_index < self.m());
        debug_assert_eq!(column.len(), self.m());

        // TODO(OPTIMIZATION): Improve the below algorithm; when does SIMD kick in?
        let (b_left, b_right) = self.b.data.split_at_mut(pivot_row_index);
        let (b_middle, b_right) = b_right.split_first_mut().unwrap();
        let (rows_left, rows_right) = self.basis_inverse_rows.split_at_mut(pivot_row_index);
        let (rows_middle, rows_right) = rows_right.split_first_mut().unwrap();

        for (edit_row_index, column_value) in column.iter_values() {
            if *edit_row_index != pivot_row_index {
                if *edit_row_index < pivot_row_index {
                    b_left[*edit_row_index] -= column_value * &*b_middle;
                    rows_left[*edit_row_index].add_multiple_of_row(-column_value, &rows_middle);
                } else if *edit_row_index == pivot_row_index {
                    continue;
                } else {
                    b_right[*edit_row_index - (pivot_row_index + 1)] -= column_value * &*b_middle;
                    rows_right[*edit_row_index - (pivot_row_index + 1)].add_multiple_of_row(-column_value, &rows_middle);
                }
            }
        }
    }

    /// Update `self.minus_pi` and the objective function value by performing a row reduction
    /// operation.
    ///
    /// # Arguments
    ///
    /// * `pivot_row_index`: Index of the pivot row.
    /// * `column_value`: Relative cost value for the pivot column.
    ///
    /// # Note
    ///
    /// This method requires a normalized pivot element.
    fn row_reduce_update_minus_pi_and_obj(&mut self, pivot_row_index: usize, column_value: F) {
        self.minus_objective -= &column_value * &self.b[pivot_row_index];

        for (column_index, value) in self.basis_inverse_rows[pivot_row_index].iter_values() {
            self.minus_pi[*column_index] -= &column_value * value;
        }
    }

    /// A property of the dimensions of this matrix.
    ///
    /// A `CarryMatrix` is always square matrix of size `m + 1` times `m + 1`. Here, `m` is the
    /// length of the constraint column vector and the width of the `minus_pi` row vector. Also, the
    /// basis inverse submatrix B^-1 has dimension `m` times `m`.
    fn m(&self) -> usize {
        self.b.len() // == self.minus_pi.len()
        // == self.basis_inverse_rows.len()
        // == ...
        // == ...
        // == ...
        // == self.basis_inverse_rows[self.m() - 1].len()
    }
}

impl<F, FZ> InverseMaintenance<F, FZ> for CarryMatrix<F, FZ>
where
    F: Field,
    for <'r> &'r F: FieldRef<F>,
    FZ: SparseElementZero<F>,
{
    fn create_for_fully_artificial(
        b: DenseVector<F>
    ) -> Self {
        let m = b.len();

        let mut b_sum = F::zero();
        for v in b.iter_values() {
            b_sum += v;
        }

        Self {
            minus_objective: -b_sum,
            minus_pi: DenseVector::constant(-F::one(), m),
            b,
            // Identity matrix
            basis_inverse_rows: (0..m)
                .map(|i| SparseVector::new(vec![(i, F::one())], m))
                .collect(),
        }
    }

    fn create_for_partially_artificial(
        artificial_rows: &Vec<usize>,
        free_basis_values: &Vec<(usize, usize)>,
        b: DenseVector<F>,
    ) -> Self {
        let m = b.len();
        debug_assert_eq!(artificial_rows.len() + free_basis_values.len(), m);  // Correct sizes
        let merged = artificial_rows.iter().copied()
            .chain(free_basis_values.iter().map(|&(i, _)| i)).collect::<HashSet<_>>();
        debug_assert!(merged.iter().all(|&i| i < m));  // Correct range
        debug_assert_eq!(merged.len(), m);  // Uniqueness

        // Initial value of zero is the value that the objective value has when a feasible solution
        // is reached.
        let mut objective = F::zero();
        for &index in artificial_rows {
            // One because we minimize a simple sum of non-negative artificial variables in the
            // basis.
            objective += F::one() * &b[index];
        }

        // Only the artificial columns "had a cost to them" before they were added to the basis.
        // Putting elements in the basis also influences the minus_pi field.
        let mut counter = 0;
        let minus_pi_values = (0..m).map(|row| {
            if counter < artificial_rows.len() && artificial_rows[counter] == row {
                counter += 1;
                -F::one()
            } else {
                F::zero()
            }
        }).collect();

        Self {
            minus_objective: -objective,
            minus_pi: DenseVector::new(minus_pi_values, m),
            b,
            // Identity matrix
            basis_inverse_rows: (0..m)
                .map(|i| SparseVector::new(vec![(i, F::one())], m))
                .collect(),
        }
    }

    fn from_basis(basis: &Vec<usize>, provider: &impl MatrixProvider<F, FZ>) -> Self {
        // TODO: Implement matrix inversion
        unimplemented!("Would involve matrix inversion")
    }

    fn from_basis_pivots(basis_columns: &Vec<(usize, usize)>, provider: &impl MatrixProvider<F, FZ>) -> Self {
        // TODO: Implement matrix inversion
        unimplemented!("Would involve matrix inversion")
    }

    fn from_artificial<MP: MatrixProvider<F, FZ>>(
        artificial_carry_matrix: Self,
        provider: &MP,
        basis: &Vec<usize>,
    ) -> Self {
        let minus_pi = CarryMatrix::create_minus_pi_from_artificial(
            &artificial_carry_matrix.basis_inverse_rows,
            provider,
            basis,
        );
        let minus_obj = CarryMatrix::create_minus_obj_from_artificial(
            provider,
            basis,
            &artificial_carry_matrix.b,
        );

        Self::new(
            minus_obj,
            minus_pi,
            artificial_carry_matrix.b,
            artificial_carry_matrix.basis_inverse_rows,
        )
    }

    fn from_artificial_remove_rows<MP: MatrixProvider<F, FZ>>(
        artificial: Self,
        rows_removed: &RemoveRows<F, FZ, MP>,
        basis_indices: &Vec<usize>,
    ) -> Self {
        debug_assert_eq!(basis_indices.len(), rows_removed.nr_rows());

        // Remove the rows
        let mut basis_inverse_rows = artificial.basis_inverse_rows;
        remove_indices(&mut basis_inverse_rows, &rows_removed.rows_to_skip);
        // Remove the columns
        for element in &mut basis_inverse_rows {
            element.remove_indices(&rows_removed.rows_to_skip);
        }

        let minus_pi = CarryMatrix::create_minus_pi_from_artificial(
            &basis_inverse_rows,
            rows_removed,
            basis_indices,
        );

        let mut b = artificial.b;
        b.remove_indices(&rows_removed.rows_to_skip);

        let minus_obj = CarryMatrix::create_minus_obj_from_artificial(
            rows_removed,
            basis_indices,
            &b,
        );

        Self::new(
            minus_obj,
            minus_pi,
            b,
            basis_inverse_rows,
        )
    }

    fn change_basis(&mut self, pivot_row_index: usize, column: &Sparse<F, FZ, F>, cost: F) {
        debug_assert!(pivot_row_index < self.m());
        debug_assert_eq!(column.len(), self.m());

        // The order of these calls matters: the first of the two normalizes the pivot row
        self.normalize_pivot_row(pivot_row_index, column);
        self.row_reduce_update_basis_inverse_and_b(pivot_row_index, column);
        self.row_reduce_update_minus_pi_and_obj(pivot_row_index, cost);
    }

    fn cost_difference(&self, original_column: Column<&F, FZ, F>) -> F {
        match &original_column {
            Column::Sparse(vector) => {
                debug_assert_eq!(vector.len(), self.m());
            },
            Column::Slack(index, _) => debug_assert!(*index < self.m()),
        }

        match original_column {
            Column::Sparse(vector) => {
                vector.inner_product_with_dense(&self.minus_pi)
            },
            Column::Slack(index, direction) => {
                &self.minus_pi[index] * direction.into::<F>()
            },
        }
    }

    fn generate_column(&self, original_column: Column<&F, FZ, F>) -> Sparse<F, FZ, F> {
        if let Column::Sparse(vector) = &original_column {
            debug_assert_eq!(vector.len(), self.m());
        }

        let tuples = (0..self.m())
            .map(|i| {
                match original_column {
                    Column::Sparse(ref vector) => {
                        self.basis_inverse_rows[i].inner_product(vector)
                    },
                    Column::Slack(index, direction) => {
                        &self.basis_inverse_rows[i][index] * direction.into::<F>()
                    },
                }
            })
            .enumerate()
            .filter(|(_, v)| v.borrow() != FZ::zero().borrow())
            .collect();

        SparseVector::new(tuples, self.m())
    }

    fn generate_element(&self, i: usize, original_column: Column<&F, FZ, F>) -> F {
        debug_assert!(i < self.m());
        if let Column::Sparse(vector) = &original_column {
            debug_assert_eq!(vector.len(), self.m());
        }

        match original_column {
            Column::Sparse(vector) => {
                self.basis_inverse_rows[i].inner_product(&vector)
            },
            Column::Slack(index, direction) => {
                &self.basis_inverse_rows[i][index] * direction.into::<F>()
            },
        }
    }

    fn b(&self) -> Dense<F> {
        // TODO: Rewrite this method, perhaps also in trait
        self.b.clone()
    }

    fn get_objective_function_value(&self) -> F {
        -&self.minus_objective
    }

    fn get_constraint_value(&self, i: usize) -> &F {
        &self.b[i]
    }
}

impl<F, G: SparseElementZero<F>> Display for CarryMatrix<F, G>
    where
        F: Field,
        for<'r> &'r F: FieldRef<F>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "CarryMatrix:\n============")?;
        writeln!(f, "Objective function value: {}", self.get_objective_function_value())?;
        writeln!(f, "Minus PI:")?;
        <DenseVector<F> as Display>::fmt(&self.minus_pi, f)?;
        writeln!(f, "b:")?;
        <DenseVector<F> as Display>::fmt(&self.b, f)?;
        writeln!(f, "B^-1:")?;
        let width = 8;

        write!(f, "{}", repeat_n(" ", width / 2).collect::<Vec<_>>().concat())?;
        for column in 0..self.m() {
            write!(f, "{:^width$}", column, width = width)?;
        }
        writeln!(f)?;
        writeln!(f, "{}", repeat_n("-",(1 + self.m()) * width).collect::<String>())?;

        for row in 0..self.m() {
            write!(f, "{:>width$}", format!("{} |", row), width = width / 2)?;
            for column in 0..self.m() {
                write!(f, "{:^width$}", format!("{}", self.basis_inverse_rows[row][column]), width = width)?;
            }
            writeln!(f)?;
        }
        writeln!(f)
    }
}
