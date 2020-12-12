//! # Carry matrix
//!
//! A sparse row major basis inverse with a dense representation of the latest b and -pi (see
//! Papadimitriou's Combinatorial Optimization).
use std::collections::HashSet;
use std::fmt::{Display, Formatter};
use std::fmt;

use itertools::repeat_n;

use crate::algorithm::two_phase::matrix_provider::{Column, MatrixProvider, OrderedColumn};
use crate::algorithm::two_phase::matrix_provider::filter::Filtered;
use crate::algorithm::two_phase::tableau::inverse_maintenance::{ColumnOps, CostOps, InternalOps, InternalOpsHR};
use crate::algorithm::two_phase::tableau::inverse_maintenance::InverseMaintenance;
use crate::algorithm::utilities::remove_indices;
use crate::data::linear_algebra::SparseTuple;
use crate::data::linear_algebra::traits::Element;
use crate::data::linear_algebra::vector::{Dense as DenseVector, Dense, Sparse as SparseVector, Sparse, Vector};
use std::cmp::Ordering;

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
pub struct Carry<F> {
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
    basis_inverse_rows: Vec<SparseVector<F, F>>,
}

impl<F> Carry<F>
where
    F: InternalOps + InternalOpsHR,
{
    /// Create a `Carry` for a tableau with a known basis inverse.
    ///
    /// # Arguments
    ///
    /// * `b`: Constraint values of the original problem with respect to the provided basis.
    /// * `basis_inverse_rows`: Rows of the matrix B^-1, where B is the matrix of columns of the
    /// basis.
    ///
    /// # Return value
    ///
    /// `Carry` with the provided values and a `minus_pi` zero vector.
    pub fn new(
        minus_objective: F,
        minus_pi: DenseVector<F>,
        b: DenseVector<F>,
        basis_inverse_rows: Vec<SparseVector<F, F>>,
    ) -> Self {
        let m = minus_pi.len();
        debug_assert_eq!(minus_pi.len(), m);
        debug_assert_eq!(b.len(), m);
        debug_assert!(basis_inverse_rows.iter().all(|r| r.len() == m));

        Carry {
            minus_objective,
            minus_pi,
            b,
            basis_inverse_rows,
        }
    }

    /// Create the `minus_pi` field from an existing basis.
    ///
    /// # Arguments
    ///
    /// * `basis_inverse_rows`: A basis inverse that represents a basic feasible solution.
    /// * `provider`: Matrix provider.
    /// * `basis`: Indices of the basis elements.
    fn create_minus_pi_from_artificial<'a, MP: MatrixProvider>(
        basis_inverse_rows: &[SparseVector<F, F>],
        provider: &'a MP,
        basis: &[usize],
    ) -> DenseVector<F>
    where
        F: ColumnOps<<MP::Column as Column>::F> + CostOps<MP::Cost<'a>>,
    {
        let m = basis_inverse_rows.len();
        debug_assert_eq!(provider.nr_rows(), m);
        debug_assert_eq!(basis.len(), m);

        let mut pi = repeat_n(F::zero(), m).collect::<Vec<_>>();
        for row in 0..m {
            for (column, value) in basis_inverse_rows[row].iter_values() {
                pi[*column] += value * provider.cost_value(basis[row]);
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
    fn create_minus_obj_from_artificial<'a, MP: MatrixProvider>(
        provider: &'a MP,
        basis: &[usize],
        b: &DenseVector<F>,
    ) -> F
    where
        F: ColumnOps<<MP::Column as Column>::F> + CostOps<MP::Cost<'a>>,
    {
        let mut objective = F::zero();
        for row in 0..provider.nr_rows() {
            objective += &b[row] * provider.cost_value(basis[row]);
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
        column: &SparseVector<F, F>,
    ) {
        let pivot_value = column.get(pivot_row_index)
            .expect("Pivot value can't be zero.");

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
        column: &SparseVector<F, F>,
    ) {
        debug_assert!(pivot_row_index < self.m());
        debug_assert_eq!(column.len(), self.m());

        // TODO(OPTIMIZATION): Improve the below algorithm; when does SIMD kick in?
        let (b_left, b_right) = self.b.data.split_at_mut(pivot_row_index);
        let (b_middle, b_right) = b_right.split_first_mut().unwrap();
        let (rows_left, rows_right) = self.basis_inverse_rows.split_at_mut(pivot_row_index);
        let (rows_middle, rows_right) = rows_right.split_first_mut().unwrap();

        for (edit_row_index, column_value) in column.iter_values() {
            match edit_row_index.cmp(&pivot_row_index) {
                Ordering::Less => {
                    b_left[*edit_row_index] -= column_value * &*b_middle;
                    rows_left[*edit_row_index].add_multiple_of_row(-column_value, &rows_middle);
                },
                Ordering::Equal => {},
                Ordering::Greater => {
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
    /// A `Carry` is always square matrix of size `m + 1` times `m + 1`. Here, `m` is the
    /// length of the constraint column vector and the width of the `minus_pi` row vector. Also, the
    /// basis inverse submatrix B^-1 has dimension `m` times `m`.
    fn m(&self) -> usize {
        self.b.len()
        //           == self.minus_pi.len()
        //           == self.basis_inverse_rows.len()
        //           == self.basis_inverse_rows[0].len()
        //           == ...
        //           == self.basis_inverse_rows[self.m() - 1].len()
    }
}

impl<F> InverseMaintenance for Carry<F>
where
    F: InternalOps + InternalOpsHR,
{
    type F = F;

    fn create_for_fully_artificial<G: Element>(
        b: DenseVector<G>
    ) -> Self
    where
        Self::F: ColumnOps<G>,
    {
        let m = b.len();

        let mut b_sum = Self::F::zero();
        for v in b.iter_values() {
            b_sum += v;
        }

        Self {
            minus_objective: -b_sum,
            minus_pi: DenseVector::constant(-Self::F::one(), m),
            b: DenseVector::new(b.data.into_iter().map(|v| v.into()).collect(), m),
            // Identity matrix
            basis_inverse_rows: (0..m)
                .map(|i| SparseVector::new(vec![(i, Self::F::one())], m))
                .collect(),
        }
    }

    fn create_for_partially_artificial<G: Element>(
        artificial_rows: &[usize],
        free_basis_values: &[(usize, usize)],
        b: DenseVector<G>,
    ) -> Self
    where
        Self::F: ColumnOps<G>,
    {
        let m = b.len();
        debug_assert_eq!(artificial_rows.len() + free_basis_values.len(), m);  // Correct sizes
        let merged = artificial_rows.iter().copied()
            .chain(free_basis_values.iter().map(|&(i, _)| i)).collect::<HashSet<_>>();
        debug_assert!(merged.iter().all(|&i| i < m));  // Correct range
        debug_assert_eq!(merged.len(), m);  // Uniqueness

        // Initial value of zero is the value that the objective value has when a feasible solution
        // is reached.
        let mut objective = Self::F::zero();
        for &index in artificial_rows {
            // One because we minimize a simple sum of non-negative artificial variables in the
            // basis.
            objective += &b[index];
        }

        // Only the artificial columns "had a cost to them" before they were added to the basis.
        // Putting elements in the basis also influences the minus_pi field.
        let mut counter = 0;
        let minus_pi_values = (0..m).map(|row| {
            if counter < artificial_rows.len() && artificial_rows[counter] == row {
                counter += 1;
                -Self::F::one()
            } else {
                Self::F::zero()
            }
        }).collect();

        Self {
            minus_objective: -objective,
            minus_pi: DenseVector::new(minus_pi_values, m),
            b: DenseVector::new(b.data.into_iter().map(|v| v.into()).collect(), m),
            // Identity matrix
            basis_inverse_rows: (0..m)
                .map(|i| SparseVector::new(vec![(i, Self::F::one())], m))
                .collect(),
        }
    }

    #[allow(unused_variables)]
    fn from_basis(basis: &[usize], provider: &impl MatrixProvider) -> Self {
        // TODO: Implement matrix inversion
        unimplemented!()
    }

    #[allow(unused_variables)]
    fn from_basis_pivots(
        basis_columns: &[(usize, usize)],
        provider: &impl MatrixProvider,
    ) -> Self {
        // TODO: Implement matrix inversion
        unimplemented!()
    }

    fn from_artificial<'provider, MP: MatrixProvider>(
        artificial: Self,
        provider: &'provider MP,
        basis: &[usize],
    ) -> Self
    where
        F: InternalOpsHR + ColumnOps<<MP::Column as Column>::F> + CostOps<MP::Cost<'provider>>,
    {
        let minus_pi = Carry::create_minus_pi_from_artificial(
            &artificial.basis_inverse_rows,
            provider,
            basis,
        );
        let minus_obj = Carry::create_minus_obj_from_artificial(
            provider,
            basis,
            &artificial.b,
        );

        Self::new(
            minus_obj,
            minus_pi,
            artificial.b,
            artificial.basis_inverse_rows,
        )
    }

    fn from_artificial_remove_rows<'a, MP: Filtered>(
        artificial: Self,
        rows_removed: &'a MP,
        basis_indices: &[usize],
    ) -> Self
    where
        Self::F: ColumnOps<<<MP as MatrixProvider>::Column as Column>::F> + CostOps<MP::Cost<'a>>,
    {
        debug_assert_eq!(basis_indices.len(), rows_removed.nr_rows());

        // Remove the rows
        let mut basis_inverse_rows = artificial.basis_inverse_rows;
        remove_indices(&mut basis_inverse_rows, rows_removed.filtered_rows());
        // Remove the columns
        for element in &mut basis_inverse_rows {
            element.remove_indices(rows_removed.filtered_rows());
        }

        let minus_pi = Carry::create_minus_pi_from_artificial(
            &basis_inverse_rows,
            rows_removed,
            basis_indices,
        );

        let mut b = artificial.b;
        b.remove_indices(rows_removed.filtered_rows());

        let minus_obj = Carry::create_minus_obj_from_artificial(
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

    fn change_basis(&mut self, pivot_row_index: usize, column: &Sparse<Self::F, Self::F>, cost: Self::F) {
        debug_assert!(pivot_row_index < self.m());
        debug_assert_eq!(column.len(), self.m());

        // The order of these calls matters: the first of the two normalizes the pivot row
        self.normalize_pivot_row(pivot_row_index, column);
        self.row_reduce_update_basis_inverse_and_b(pivot_row_index, column);
        self.row_reduce_update_minus_pi_and_obj(pivot_row_index, cost);
    }

    fn cost_difference<G, C: Column<F=G> + OrderedColumn>(&self, original_column: &C) -> Self::F
    where
        Self::F: ColumnOps<G>,
    {
        self.minus_pi.sparse_inner_product(original_column.iter())
    }

    fn generate_column<G, C: Column<F=G> + OrderedColumn>(&self, original_column: C) -> SparseVector<Self::F, Self::F>
    where
        Self::F: ColumnOps<G>,
    {
        let column_iter = original_column.iter();

        let tuples = (0..self.m())
            .map(|i| self.generate_element(i, column_iter.clone()))
            .enumerate()
            .filter(|(_, v)| !v.is_zero())
            .collect();

        SparseVector::new(tuples, self.m())
    }

    fn generate_element<'a, G: 'a, I: Iterator<Item=&'a SparseTuple<G>>>(&self, i: usize, original_column: I) -> Self::F
    where
        Self::F: ColumnOps<G>,
    {
        debug_assert!(i < self.m());

        self.basis_inverse_rows[i].sparse_inner_product(original_column)
    }

    fn b(&self) -> Dense<Self::F> {
        // TODO: Rewrite this method, perhaps also in trait
        self.b.clone()
    }

    fn get_objective_function_value(&self) -> Self::F {
        -self.minus_objective.clone()
    }

    fn get_constraint_value(&self, i: usize) -> &Self::F {
        &self.b[i]
    }
}

impl<F> Display for Carry<F>
where
    F: InternalOps + InternalOpsHR,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "Carry:\n============")?;
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
                let value = match self.basis_inverse_rows[row].get(column) {
                    Some(value) => value.to_string(),
                    None => "0".to_string(),
                };
                write!(f, "{:^width$}", value, width = width)?;
            }
            writeln!(f)?;
        }
        writeln!(f)
    }
}
