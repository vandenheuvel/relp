//! # Data structures for Simplex
//!
//! Contains the simplex tableau and logic for elementary operations which can be perfomed upon it.
//! The tableau is extended with supplementary data structures for efficiency.
use std::borrow::Borrow;
use std::collections::HashSet;
use std::fmt::{Debug, Display, Formatter, Result as FormatResult};
use std::iter::repeat;

use itertools::repeat_n;

use crate::algorithm::simplex::tableau::inverse_maintenance::CarryMatrix;
use crate::algorithm::simplex::tableau::kind::Kind;
use crate::data::linear_algebra::traits::SparseElementZero;
use crate::data::linear_algebra::vector::{Sparse as SparseVector, Vector};
use crate::data::number_types::traits::{Field, FieldRef, OrderedField};

pub mod inverse_maintenance;
pub mod kind;

/// The most high-level data structure that is used by the Simplex algorithm: the Simplex tableau.
///
/// It holds only a reference to the (immutable) problem it solves, but owns the data structures
/// that describe the current solution basis.
#[allow(non_snake_case)]
#[derive(Eq, PartialEq, Debug)]
pub struct Tableau<F, FZ, K>
where
    F: Field,
    FZ: SparseElementZero<F>,
    K: Kind<F, FZ>,
{
    /// Matrix of size (m + 1) x (m + 1).
    ///
    /// This attribute changes with a basis change.
    carry: CarryMatrix<F, FZ>,

    /// Maps the rows to the column containing its pivot.
    ///
    /// The rows are indexed 0 through self.nr_rows(), while the columns are indexed 0 through
    /// self.nr_columns().
    ///
    /// This attribute changes with a basis change.
    basis_indices: Vec<usize>,
    /// All columns currently in the basis.
    ///
    /// Could also be derived from `basis_indices`, but is here for faster reading and writing.
    basis_columns: HashSet<usize>,

    /// Whether this tableau has artificial variables (and is in the first phase of the two-phase
    /// algorithm) or not. See the `Kind` trait for more information.
    pub kind: K,
}

impl<'a, F, FZ, K> Tableau<F, FZ, K>
    where
        F: Field + 'a,
        for <'r> &'r F: FieldRef<F>,
        FZ: SparseElementZero<F>,
        K: Kind<F, FZ>,
{
    /// Brings a column into the basis by updating the `self.carry` matrix and updating the
    /// data structures holding the collection of basis columns.
    pub fn bring_into_basis(
        &mut self,
        pivot_column_index: usize,
        pivot_row_index: usize,
        column: &SparseVector<F, FZ, F>,
        cost: F
    ) {
        debug_assert!(pivot_column_index < self.kind.nr_columns());
        debug_assert!(pivot_row_index < self.nr_rows());

        self.carry.row_reduce_for_basis_change(pivot_row_index, column, cost);
        self.update_basis_indices(pivot_row_index, pivot_column_index);
    }

    /// Update the basis index.
    ///
    /// Removes the index of the variable leaving the basis from the `basis_column_map` attribute,
    /// while inserting the entering variable index.
    ///
    /// # Arguments
    ///
    /// * `pivot_row`: Row index of the pivot, in range 0 until self.nr_rows().
    /// * `pivot_column`: Column index of the pivot, in range 0 until self.nr_columns(). Is not yet
    /// in the basis.
    fn update_basis_indices(&mut self, pivot_row: usize, pivot_column: usize) {
        debug_assert!(pivot_row < self.nr_rows());
        debug_assert!(pivot_column < self.nr_columns());

        let leaving_column = self.basis_indices[pivot_row];
        self.basis_columns.remove(&leaving_column);
        self.basis_indices[pivot_row] = pivot_column;
        self.basis_columns.insert(pivot_column);
    }

    /// Calculates the relative cost of a column.
    ///
    /// # Arguments
    ///
    /// * `j`: Index of column to calculate the relative cost for, in range `0` through
    /// `self.nr_variables()`.
    ///
    /// # Return value
    ///
    /// The relative cost.
    ///
    /// # Note
    ///
    /// That column will typically not be a basis column. Although the method could be valid for
    /// those inputs as well, this should never be calculated, as the relative cost always equals
    /// zero in that situation.
    pub fn relative_cost(&self, j: usize) -> F {
        debug_assert!(j < self.nr_columns());

        self.carry.cost_difference(self.kind.original_column(j)) + self.kind.initial_cost_value(j)
    }

    /// Column of original problem with respect to the current basis.
    ///
    /// Generate a column of the tableau as it would look like with the current basis by matrix
    /// multiplying the original column and the carry matrix.
    ///
    /// # Arguments
    ///
    /// * `j`: Column index of the variable, in range 0 until self.nr_columns().
    ///
    /// # Return value
    ///
    /// `SparseVector<T>` of size `m`.
    pub fn generate_column(&self, j: usize) -> SparseVector<F, FZ, F> {
        debug_assert!(j < self.nr_columns());

        self.carry.generate_column(self.kind.original_column(j))
    }

    /// Single element with respect to the current basis.
    pub fn generate_element(&self, i: usize, j: usize) -> F {
        debug_assert!(i < self.nr_rows());
        debug_assert!(j < self.nr_columns());

        self.carry.generate_element(i, self.kind.original_column(j))
    }

    /// Whether a column is in the basis.
    ///
    /// # Return value
    ///
    /// `bool` with value true if the column is in the basis.
    ///
    /// # Note
    ///
    /// This method may not be accurate when there are artificial variables.
    pub fn is_in_basis(&self, column: &usize) -> bool {
        debug_assert!(*column < self.nr_columns());

        self.basis_columns.contains(column)
    }

    /// Get the current basic feasible solution.
    ///
    /// # Return value
    ///
    /// `SparseVector<T>` of the solution.
    pub fn current_bfs(&self) -> SparseVector<F, FZ, F> {
        let mut tuples = self.carry.b.iter_values()
            .enumerate()
            .map(|(i, v)| (self.basis_indices[i], v))
            .filter(|(_, v)| v.borrow() != FZ::zero().borrow())
            .collect::<Vec<_>>();
        tuples.sort_by_key(|&(i, _)| i);
        SparseVector::new(
            tuples.into_iter().map(|(i, v)| (i, v.clone())).collect(),
            self.nr_columns(),
        )
    }

    /// Get the cost of the current solution.
    ///
    /// # Return value
    ///
    /// The current value of the objective function.
    ///
    /// # Note
    ///
    /// This function works for both artificial and non-artificial tableau's.
    pub fn objective_function_value(&self) -> F {
        self.carry.get_objective_function_value()
    }

    /// Number of rows in the tableau.
    ///
    /// # Return value
    ///
    /// The number of rows.
    pub fn nr_rows(&self) -> usize {
        self.kind.nr_rows()
    }

    /// Number of variables in the problem.
    ///
    /// # Return value
    ///
    /// The number of variables.
    ///
    /// # Note
    ///
    /// This number might be extremely large, depending on the `AdaptedTableauProvider`.
    pub fn nr_columns(&self) -> usize {
        self.kind.nr_columns()
    }
}

impl<'a, OF, OFZ, K> Tableau<OF, OFZ, K>
where
        OF: OrderedField + 'a,
        for<'r> &'r OF: FieldRef<OF>,
        OFZ: SparseElementZero<OF>,
        K: Kind<OF, OFZ>,
{
    /// Determine the row to pivot on.
    ///
    /// Determine the row to pivot on, given the column. This is the row with the positive but
    /// minimal ratio between the current constraint vector and the column.
    ///
    /// When there are multiple choices for the pivot row, Bland's anti cycling algorithm
    /// is used to avoid cycles.
    ///
    /// TODO: Reconsider the below
    /// Because this method allows for less strategy and heuristics, it is not included in the
    /// `PivotRule` trait.
    ///
    /// # Arguments
    ///
    /// * `column`: Problem column with respect to the current basis with length `m`.
    ///
    /// # Return value
    ///
    /// Index of the row to pivot on. If not found, the problem is optimal.
    pub fn select_primal_pivot_row(&self, column: &SparseVector<OF, OFZ, OF>) -> Option<usize> {
        debug_assert_eq!(column.len(), self.nr_rows());

        // (chosen index, minimum ratio, corresponding leaving_column (for Bland's algorithm))
        let mut min_values: Option<(usize, OF, usize)> = None;
        for (row, xij) in column.iter_values() {
            if xij > &OF::zero() {
                let ratio = self.carry.get_constraint_value(*row) / xij;
                // Bland's anti cycling algorithm
                let leaving_column = self.basis_indices[*row];
                if let Some((min_index, min_ratio, min_leaving_column)) = &mut min_values {
                    if &ratio == min_ratio && leaving_column < *min_leaving_column {
                        *min_index = *row;
                        *min_leaving_column = leaving_column;
                    } else if &ratio < min_ratio {
                        *min_index = *row;
                        *min_ratio = ratio;
                        *min_leaving_column = leaving_column;
                    }
                } else {
                    min_values = Some((*row, ratio, leaving_column))
                }
            }
        }

        min_values.map(|(min_index, _, _)| min_index)
    }
}



/// Check whether the tableau currently has a valid basic feasible solution.
///
/// Only used for debug purposes.
#[allow(clippy::nonminimal_bool)]
pub fn is_in_basic_feasible_solution_state<'a, OF, OFZ, K>(
    tableau: &Tableau<OF, OFZ, K>,
) -> bool
where
    OF: OrderedField + 'a,
    for<'r> &'r OF: FieldRef<OF>,
    OFZ: SparseElementZero<OF>,
    K: Kind<OF, OFZ>,
{
    // Checking basis_columns
    // Correct number of basis columns (uniqueness is implied because it's a set)
    let nr_basis_columns = tableau.basis_columns.len() == tableau.nr_rows();

    // Checking basis_indices
    // Correct number of basis columns
    let nr_basis_indices = tableau.basis_indices.len() == tableau.nr_rows();
    let as_set = tableau.basis_indices.iter().map(|&v| v).collect::<HashSet<_>>();
    // Uniqueness of the basis columns
    let uniqueness = as_set.len() == tableau.nr_rows();
    // Same columns as in `basis_indices`
    let same = as_set.difference(&tableau.basis_columns).count() == 0;

    // Checking carry matrix
    let carry = {
        // `basis_inverse_rows` are a proper inverse by regenerating basis columns
        let mut basis = true;
        for (i, &basis_column) in tableau.basis_indices.iter().enumerate() {
            if !(
                    tableau.generate_column(basis_column)
                        == SparseVector::new(vec![(i, OF::one())], tableau.nr_rows())
            ) {
                basis = false;
                break;
            }
        }
        // `minus_pi` get to relative zero cost for basis columns
        let mut minus_pi = true;
        for &basis_column in tableau.basis_indices.iter() {
            if !(tableau.relative_cost(basis_column) == OF::zero()) {
                minus_pi = false;
                break;
            }
        }
        // `b` >= 0
        let mut b_ok = true;
        let b = &tableau.carry.b;
        for row in 0..tableau.nr_rows() {
            if !(b[row] >= OF::zero()) {
                b_ok = false;
                break;
            }
        }

        basis && minus_pi && b_ok
    };

    true
        && nr_basis_columns
        && nr_basis_indices
        && uniqueness
        && same
        && carry
}

impl<'a, F, FZ, K> Display for Tableau<F, FZ, K>
    where
        F: Field + 'a,
        for<'r> &'r F: FieldRef<F>,
        FZ: SparseElementZero<F>,
        K: Kind<F, FZ>,
{
    fn fmt(&self, f: &mut Formatter) -> FormatResult {
        writeln!(f, "Tableau:")?;

        writeln!(f, "=== Current State ===")?;
        let column_width = 10;
        let counter_width = 8;
        // Column counter
        write!(f, "{0:width$}", "", width = counter_width)?;
        write!(f, "{0:^width$}", "b", width = column_width)?;
        write!(f, "|")?;
        for column_index in 0..self.nr_columns() {
            write!(f, "{0:^width$}", column_index, width = column_width)?;
        }
        writeln!(f)?;

        // Separator
        writeln!(f, "{}", repeat_n("-", counter_width + (1 + self.nr_columns()) * column_width)
            .collect::<String>())?;

        // Cost row
        write!(f, "{0:>width$}", format!("{}  |", "cost"), width = counter_width)?;
        let value = format!("{}", -self.carry.get_objective_function_value());
        write!(f, "{0:^width$}", value, width = column_width)?;
        write!(f, "|")?;
        for column_index in 0..self.nr_columns() {
            let number = format!("{}", self.relative_cost(column_index));
            write!(f, "{0:^width$.5}", number, width = column_width)?;
        }
        writeln!(f)?;

        // Separator
        writeln!(f, "{}", repeat("-")
            .take(counter_width + (1 + self.nr_columns()) * column_width)
            .collect::<String>())?;

        // Row counter and row data
        for row_index in 0..self.nr_rows() {
            write!(f, "{:>width$}", format!("{}  |", row_index), width = counter_width)?;
            write!(f, "{0:^width$}", format!("{}", self.carry.b[row_index]), width = column_width)?;
            write!(f, "|")?;
            for column_index in 0..self.nr_columns() {
                let number = format!("{}", self.generate_column(column_index)[row_index]);
                write!(f, "{:^width$}", number, width = column_width)?;
            }
            writeln!(f)?;
        }
        writeln!(f)?;

        writeln!(f, "=== Basis Columns ===")?;
        let mut basis = self.basis_indices.iter()
            .enumerate()
            .map(|(i, &j)| (i, j))
            .collect::<Vec<_>>();
        basis.sort_by_key(|&(i, _)| i);
        writeln!(f, "{:?}", basis)?;

        writeln!(f, "=== Basis Inverse ===")?;
        <CarryMatrix<F, FZ> as Display>::fmt(&self.carry, f)
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use num::FromPrimitive;
    use num::rational::Ratio;

    use crate::algorithm::simplex::matrix_provider::matrix_data::MatrixData;
    use crate::algorithm::simplex::strategy::pivot_rule::{FirstProfitable, PivotRule};
    use crate::algorithm::simplex::tableau::inverse_maintenance::CarryMatrix;
    use crate::algorithm::simplex::tableau::kind::NonArtificial;
    use crate::algorithm::simplex::tableau::Tableau;
    use crate::data::linear_algebra::traits::SparseElementZero;
    use crate::data::linear_algebra::vector::{Dense, Sparse as SparseVector};
    use crate::data::linear_algebra::vector::test::TestVector;
    use crate::data::number_types::traits::{Field, FieldRef};
    use crate::F;
    use crate::tests::problem_2::{artificial_tableau_form, create_matrix_data_data, matrix_data_form};

    fn tableau<'a, F, FZ>(
        data: &'a MatrixData<'a, F, FZ>,
    ) -> Tableau<F, FZ, NonArtificial<'a, F, FZ, MatrixData<'a, F, FZ>>>
    where
        F: Field + FromPrimitive + 'a,
        for<'r> &'r F: FieldRef<F>,
        FZ: SparseElementZero<F>,
    {
        let carry = {
            let minus_objective = F!(-6);
            let minus_pi = Dense::from_test_data(vec![1, -1, -1]);
            let b = Dense::from_test_data(vec![1, 2, 3]);
            let basis_inverse_rows = vec![
                SparseVector::from_test_data(vec![1, 0, 0]),
                SparseVector::from_test_data(vec![-1, 1, 0]),
                SparseVector::from_test_data(vec![-1, 0, 1]),
            ];
            CarryMatrix::<F, FZ>::create(minus_objective, minus_pi, b, basis_inverse_rows)
        };
        let basis_indices = vec![2, 3, 4];
        let mut basis_columns = HashSet::new();
        basis_columns.insert(2);
        basis_columns.insert(3);
        basis_columns.insert(4);

        Tableau::<_, _, NonArtificial<_, _, _>>::new_with_basis(
            data,
            carry,
            basis_indices,
            basis_columns,
        )
    }

    #[test]
    fn cost() {
        let (constraints, b) = create_matrix_data_data();
        let matrix_data_form = matrix_data_form(&constraints, &b);
        let artificial_tableau = artificial_tableau_form(&matrix_data_form);
        assert_eq!(artificial_tableau.objective_function_value(), Ratio::from_i32(8).unwrap());

        let tableau = tableau(&matrix_data_form);
        assert_eq!(tableau.objective_function_value(), Ratio::from_i32(6).unwrap());
    }

    #[test]
    fn relative_cost() {
        let (constraints, b) = create_matrix_data_data();
        let matrix_data_form = matrix_data_form(&constraints, &b);
        let artificial_tableau = artificial_tableau_form(&matrix_data_form);
        assert_eq!(artificial_tableau.relative_cost(0), Ratio::from_i32(0).unwrap());

        assert_eq!(
            artificial_tableau.relative_cost(artificial_tableau.nr_artificial_variables() + 0),
            Ratio::from_i32(-10).unwrap(),
        );

        let tableau = tableau(&matrix_data_form);
        assert_eq!(tableau.relative_cost(0), Ratio::from_i32(-3).unwrap());
        assert_eq!(tableau.relative_cost(1), Ratio::from_i32(-3).unwrap());
        assert_eq!(tableau.relative_cost(2), Ratio::from_i32(0).unwrap());
    }

    #[test]
    fn generate_column() {
        let (constraints, b) = create_matrix_data_data();
        let matrix_data_form = matrix_data_form(&constraints, &b);
        let artificial_tableau = artificial_tableau_form(&matrix_data_form);

        let index_to_test = artificial_tableau.nr_artificial_variables() + 0;
        let column = artificial_tableau.generate_column(index_to_test);
        let expected = SparseVector::from_test_data(vec![3, 5, 2]);
        assert_eq!(column, expected);
        let result = artificial_tableau.relative_cost(index_to_test);
        assert_eq!(result, Ratio::<i32>::from_i32(-10).unwrap());

        let tableau = tableau(&matrix_data_form);
        let index_to_test = 0;
        let column = tableau.generate_column(index_to_test);
        let expected = SparseVector::from_test_data(vec![3, 2, -1]);
        assert_eq!(column, expected);
        let result = tableau.relative_cost(index_to_test);
        assert_eq!(result, Ratio::<i32>::from_i32(-3).unwrap());
    }

    #[test]
    fn bring_into_basis() {
        let (constraints, b) = create_matrix_data_data();
        let matrix_data_form = matrix_data_form(&constraints, &b);
        let mut artificial_tableau = artificial_tableau_form(&matrix_data_form);
        let column = artificial_tableau.nr_artificial_variables() + 0;
        let column_data = artificial_tableau.generate_column(column);
        let row = artificial_tableau.select_primal_pivot_row(&column_data).unwrap();
        let cost = artificial_tableau.relative_cost(column);
        artificial_tableau.bring_into_basis(column, row, &column_data, cost);

        assert!(artificial_tableau.is_in_basis(&column));
        assert!(!artificial_tableau.is_in_basis(&0));
        assert_eq!(artificial_tableau.objective_function_value(), Ratio::<i32>::new(14, 3));

        let mut tableau = tableau(&matrix_data_form);
        let column = 1;
        let column_data = tableau.generate_column(column);
        let row = tableau.select_primal_pivot_row(&column_data).unwrap();
        let cost = tableau.relative_cost(column);
        tableau.bring_into_basis(column, row, &column_data, cost);

        assert!(tableau.is_in_basis(&column));
        assert_eq!(tableau.objective_function_value(), Ratio::<i32>::new(9, 2));
    }

    fn bfs_tableau<'a, F, FZ>(
        data: &'a MatrixData<'a, F, FZ>,
    ) -> Tableau<F, FZ, NonArtificial<'a, F, FZ, MatrixData<'a, F, FZ>>>
    where
        F: Field + FromPrimitive,
        for<'r> &'r F: FieldRef<F>,
        FZ: SparseElementZero<F>,
    {
        let m = 3;
        let carry = {
            let minus_objective = F::from_i32(0).unwrap();
            let minus_pi = Dense::from_test_data(vec![1, 1, 1]);
            let b = Dense::from_test_data(vec![1, 2, 3]);
            let basis_inverse_rows = vec![
                SparseVector::from_test_data(vec![1, 0, 0]),
                SparseVector::from_test_data(vec![-1, 1, 0]),
                SparseVector::from_test_data(vec![-1, 0, 1]),
            ];
            CarryMatrix::<F, FZ>::create(minus_objective, minus_pi, b, basis_inverse_rows)
        };
        let basis_indices = vec![m + 2, m + 3, m + 4];
        let mut basis_columns = HashSet::new();
        basis_columns.insert(m + 2);
        basis_columns.insert(m + 3);
        basis_columns.insert(m + 4);

        Tableau::<_, _, NonArtificial<_, _, _>>::new_with_basis(
            data,
            carry,
            basis_indices,
            basis_columns,
        )
    }

    #[test]
    fn create_tableau() {
        type T = Ratio<i32>;

        let (constraints, b) = create_matrix_data_data();
        let matrix_data_form = matrix_data_form(&constraints, &b);
        let bfs_tableau = bfs_tableau(&matrix_data_form);
        let mut rule = <FirstProfitable as PivotRule>::new();
        assert!(rule.select_primal_pivot_column(&bfs_tableau).is_none());
    }
}
