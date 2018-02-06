//! # Data structures for Simplex
//!
//! Contains the simplex tableau and logic for elementary operations which can be perfomed upon it.
//! The tableau is extended with supplementary data structures for efficiency.
use std::collections::{HashMap, HashSet};
use std::f64;

use data::linear_program::canonical_form::CanonicalForm;
use data::linear_program::elements::VariableType;
use data::linear_algebra::matrix::{DenseMatrix, Matrix, SparseMatrix};
use data::linear_algebra::vector::{DenseVector, SparseVector, Vector};


pub const ARTIFICIAL: &str = "ARTIFICIAL";
pub const ACTUAL_COST: usize = 0;
pub const ARTIFICIAL_COST: usize = 1;

/// Holds all information necessary to execute the Simplex algorithm.
///
/// The original tableau, as can be extracted from a linear program in `CanonicalForm`, is not
/// being mutated during the lifetime of the tableau. The same holds from the cost rows. The carry
/// matrix is used to represent the different changes in basis, while the current basis columns are
/// maintained in a map and set.
///
///
///
#[derive(Debug, PartialEq)]
pub struct Tableau {
    /// Immutable coefficient data of dimension m x n.
    original_matrix: SparseMatrix,
    /// Immutable cost vector of dimension n.
    real_c: SparseVector,
    /// Immutable artificial cost vector of dimension n. Is always the same, but added to allow for
    /// a uniform treatment of the tableau during the two phases.
    artificial_c: SparseVector,

    /// Matrix of size (m + 1) x (m + 1), changed every basis change
    carry: DenseMatrix,
    /// Maps the rows to the column containing it's pivot.
    basis_columns_map: HashMap<usize, usize>,
    /// Set containing all basis columns
    basis_columns_set: HashSet<usize>,
    /// m
    nr_rows: usize,
    /// n
    nr_columns: usize,

    /// Contains the names of all variables. Used only for outputting the result of a computation.
    column_info: Vec<String>,
}

impl Tableau {
    /// Create a new `Tableau` instance.
    ///
    /// This method should be used when no basis is known.
    pub fn new(data: SparseMatrix,
               real_c: SparseVector,
               carry: DenseMatrix,
               basis_columns: HashMap<usize, usize>,
               column_info: Vec<String>,
               ) -> Tableau {
        let basis_columns_set = basis_columns.values()
            .map(|&v| v)
            .collect::<HashSet<usize>>();

        let nr_rows = data.nr_rows();
        let nr_columns = data.nr_columns();

        debug_assert_eq!(data.nr_rows(), nr_rows);
        debug_assert_eq!(carry.nr_rows(), 1 + 1 + nr_rows);
        debug_assert_eq!(carry.nr_columns(), 1 + nr_rows);
        debug_assert_eq!(basis_columns.len(), nr_rows);

        debug_assert_eq!(data.nr_columns(), nr_columns);
        debug_assert_eq!(real_c.len(), nr_columns);
        debug_assert_eq!(column_info.len(), nr_columns);

        for i in 0..nr_rows {
            debug_assert!(carry.get_value(1 + 1 + i, 0) > 0f64);
        }

        // The artificial cost vector is always here, regardless of whether it is used
        let mut artificial_c = SparseVector::zeros(nr_columns);
        for i in 0..nr_rows {
            artificial_c.set_value(i, 1f64);
        }

        Tableau {
            original_matrix: data,
            real_c,
            artificial_c,

            carry,
            basis_columns_map: basis_columns,
            basis_columns_set,
            nr_rows,
            nr_columns,

            column_info,
        }
    }
    /// Create a `Tableau` instance using a known basic feasible solution.
    pub fn create_from(canonical: &CanonicalForm, carry: DenseMatrix, basis: HashMap<usize, usize>) -> Tableau {
        let m = canonical.nr_constraints();

        let data = canonical.data();

        let cost = canonical.cost();

        let mut basis = basis;
        for val in basis.values_mut() {
            *val -= m;
        }

        Tableau::new(data, cost, carry, basis, canonical.variable_info())
    }
    /// Brings a column into the basis by updating the `self.carry` matrix and updating the
    /// data structures holding the collection of basis columns.
    pub fn bring_into_basis(&mut self, pivot_column: usize) {
        debug_assert!(pivot_column < self.nr_columns);

        let column = self.generate_column(pivot_column);
        let pivot_row = self.find_pivot_row(&column);

        self.carry.multiply_row(1 + 1 + pivot_row, 1f64 / column.get_value(1 + 1 + pivot_row));
        for edit_row in 0..self.carry.nr_rows() {
            if edit_row != 1 + 1 + pivot_row {
                if column.get_value(edit_row).abs() > 1e-10 {
                    let factor = column.get_value(edit_row);
                    self.carry.mul_add_rows(1 + 1 + pivot_row, edit_row, -factor);
                }
            }
        }

        self.update_basis_indices(pivot_row, pivot_column);
    }
    /// Removes the index of the variable leaving the basis from the `basis_column_map` and
    /// `basis_column_set` data structures, while inserting the entering variable index.
    fn update_basis_indices(&mut self, pivot_row: usize, pivot_column: usize) {
        debug_assert!(self.basis_columns_map.contains_key(&pivot_row));

        let leaving_column = *self.basis_columns_map.get(&pivot_row).unwrap();

        debug_assert!(self.basis_columns_set.contains(&leaving_column));

        self.basis_columns_set.remove(&leaving_column);
        self.basis_columns_set.insert(pivot_column);
        self.basis_columns_map.insert(pivot_row, pivot_column);
    }
    /// Generate a column of the tableau as it would look like with the current basis by matrix
    /// multiplying the original column and the carry matrix.
    pub fn generate_column(&self, column: usize) -> SparseVector {
        debug_assert!(column < self.nr_columns);

        let mut generated = Vec::with_capacity(1 + 1 + self.nr_rows);
        generated.push((ACTUAL_COST, self.relative_cost(ACTUAL_COST, column)));
        generated.push((ARTIFICIAL_COST, self.relative_cost(ARTIFICIAL_COST, column)));
        for row in 0..self.nr_rows() {
            let value = self.original_matrix.column(column)
                .map(|&(j, v)| v * self.carry.get_value(1 + 1 + row, 1 + j))
                .sum::<f64>();
            if value != 0f64 {
                generated.push((1 + 1 + row, value));
            }
        }

        SparseVector::from_tuples(generated, 1 + 1 + self.nr_rows())
    }
    /// Determine the row to pivot on, given the column. This is the row with the positive but
    /// minimal ratio between the current constraint vector and the column.
    pub fn find_pivot_row(&self, column: &SparseVector) -> usize {
        debug_assert_eq!(column.len(), 1 + 1 + self.nr_rows);

        let mut min_index = usize::max_value();
        let mut min_ratio = f64::INFINITY;

        for (row, xij) in column.values() {
            if *row == ACTUAL_COST || *row == ARTIFICIAL_COST {
                continue;
            }
            if *xij > 0f64 {
                let ratio = self.carry.get_value(*row, 0) / xij;
                if ratio < min_ratio {
                    min_index = *row;
                    min_ratio = ratio;
                }
            }
        }

        debug_assert_ne!(min_index, usize::max_value());

        min_index - 1 - 1
    }
    /// Find a profitable column.
    pub fn profitable_column(&self, cost_row: usize) -> Option<usize> {
        (0..self.nr_columns())
            .filter(|column| !self.basis_columns_set.contains(&column))
            .find(|column| self.relative_cost(cost_row, *column) < 0f64)
    }
    /// Calculates the relative cost of a non-basis column to pivot on.
    pub fn relative_cost(&self, cost_row: usize, column: usize) -> f64 {
        debug_assert!(column < self.nr_columns);

        let mut total;
        if cost_row == ACTUAL_COST {
            total = self.real_c.get_value(column);
        } else {
            total = self.artificial_c.get_value(column);
        }
        for (column, value) in self.original_matrix.column(column) {
            total += *value * self.carry.get_value(cost_row, 1 + column);
        }
        total
    }
    /// Gets a copy of the carry matrix.
    pub fn carry(&self) -> DenseMatrix {
        self.carry.clone()
    }
    /// Checks whether a column is in the basis.
    pub fn in_basis(&self, column: usize) -> bool {
        debug_assert!(column < self.nr_columns);

        self.basis_columns_set.contains(&column)
    }
    /// Gets the current basis columns as a set.
    pub fn basis_columns(&self) -> HashSet<usize> {
        self.basis_columns_set.clone()
    }
    /// Gets a map from the rows to the column containing a pivot of that row.
    pub fn basis_columns_map(&self) -> HashMap<usize, usize> {
        self.basis_columns_map.clone()
    }
    /// Return the current values for all variables in the current basic feasible solution.
    pub fn current_bfs(&self) -> SparseVector {
        let mut data = Vec::new();
        for row in self.basis_columns_map.keys() {
            let column = self.basis_columns_map.get(row).unwrap();
            data.push((*column, self.carry.get_value(1 + 1 + *row, 0)));
        }
        data.sort_by_key(|&(i, _)| i);
        SparseVector::from_tuples(data, self.nr_columns)
    }
    /// Get the cost of the current solution.
    pub fn cost(&self, cost_row: usize) -> f64 {
        -self.carry.get_value(cost_row, 0)
    }
    /// Get the number of variables
    pub fn nr_columns(&self) -> usize {
        self.nr_columns
    }
    /// Get the number of non-cost rows
    pub fn nr_rows(&self) -> usize {
        self.nr_rows
    }
}

/// Creates a Simplex tableau augmented with artificial variables in a basic feasible solution
/// having only the artificial variables in the basis.
pub fn create_artificial_tableau(canonical: &CanonicalForm) -> Tableau {
    let m = canonical.nr_constraints();
    let n = m + canonical.nr_variables();

    let data = SparseMatrix::identity(m).hcat(canonical.data());

    let mut cost = SparseVector::zeros(n);
    let c = canonical.cost();
    for i in 0..c.len() {
        cost.set_value(m + i, c.get_value(i));
    }

    let b = canonical.b();
    let basis_inverse = DenseMatrix::identity(m);
    let carry = create_carry_matrix(b, basis_inverse);

    let mut basis = HashMap::with_capacity(m);
    for i in 0..m {
        basis.insert(i, i);
    }

    let mut column_info = Vec::new();
    for _ in 0..m {
        column_info.push(String::from(ARTIFICIAL));
    }
    column_info.extend(canonical.variable_info());

    Tableau::new(data, cost, carry, basis, column_info)
}

/// Create a carry matrix for a tableau with not yet having had any operations applied to it.
fn create_carry_matrix(b: DenseVector, basis_inverse: DenseMatrix) -> DenseMatrix {
    let m = b.len();
    debug_assert_eq!(basis_inverse.nr_rows(), m);
    debug_assert_eq!(basis_inverse.nr_columns(), m);

    let cost_row = DenseVector::zeros(1 + m);
    let mut artificial_cost_row = DenseVector::zeros(1 + m);
    for column in 0..m {
        artificial_cost_row.set_value(1 + column, -1f64);
    }
    let minus_cost = - b.iter().sum::<f64>();
    artificial_cost_row.set_value(0, minus_cost);
    cost_row.vcat(artificial_cost_row.vcat(b.hcat(basis_inverse)))
}


#[cfg(test)]
mod test {

    use super::*;

    fn canonical_large() -> CanonicalForm {
        let data = vec![vec![1f64, 1f64, 0f64, 1f64, 0f64, 0f64, 0f64],
                        vec![1f64, 0f64, 1f64, 0f64, -1f64, 0f64, 0f64],
                        vec![0f64, -1f64, 1f64, 0f64, 0f64, 0f64, 0f64],
                        vec![1f64, 0f64, 0f64, 0f64, 0f64, 1f64, 0f64],
                        vec![0f64, 1f64, 0f64, 0f64, 0f64, 0f64, 1f64]];
        let data = SparseMatrix::from_data(data);

        let b = DenseVector::from_data(vec![5f64,
                                            10f64,
                                            7f64,
                                            4f64,
                                            1f64]);

        let cost = SparseVector::from_data(vec![1f64, 4f64, 9f64, 0f64, 0f64, 0f64, 0f64]);

        let column_info = vec![(String::from("XONE"), VariableType::Continuous),
                               (String::from("YTWO"), VariableType::Continuous),
                               (String::from("ZTHREE"), VariableType::Continuous),
                               (String::from("SLACK0"), VariableType::Continuous),
                               (String::from("SLACK1"), VariableType::Continuous),
                               (String::from("SLACK2"), VariableType::Continuous),
                               (String::from("SLACK3"), VariableType::Continuous)];

        CanonicalForm::new(data, b, cost, column_info)
    }

    #[test]
    fn test_create_artificial_tableau_large() {
        let result = create_artificial_tableau(&canonical_large());

        let data = SparseMatrix::from_data(vec![vec![1f64, 0f64, 0f64, 0f64, 0f64, 1f64, 1f64, 0f64, 1f64, 0f64, 0f64, 0f64],
                                                vec![0f64, 1f64, 0f64, 0f64, 0f64, 1f64, 0f64, 1f64, 0f64, -1f64, 0f64, 0f64],
                                                vec![0f64, 0f64, 1f64, 0f64, 0f64, 0f64, -1f64, 1f64, 0f64, 0f64, 0f64, 0f64],
                                                vec![0f64, 0f64, 0f64, 1f64, 0f64, 1f64, 0f64, 0f64, 0f64, 0f64, 1f64, 0f64],
                                                vec![0f64, 0f64, 0f64, 0f64, 1f64, 0f64, 1f64, 0f64, 0f64, 0f64, 0f64, 1f64]]);
        let c = SparseVector::from_data(vec![0f64, 0f64, 0f64, 0f64, 0f64, 1f64, 4f64, 9f64, 0f64, 0f64, 0f64, 0f64]);
        let carry = DenseMatrix::from_data(vec![vec![0f64, 0f64, 0f64, 0f64, 0f64, 0f64],
                                                vec![-27f64, -1f64, -1f64, -1f64, -1f64, -1f64],
                                                vec![5f64, 1f64, 0f64, 0f64, 0f64, 0f64],
                                                vec![10f64, 0f64, 1f64, 0f64, 0f64, 0f64],
                                                vec![7f64, 0f64, 0f64, 1f64, 0f64, 0f64],
                                                vec![4f64, 0f64, 0f64, 0f64, 1f64, 0f64],
                                                vec![1f64, 0f64, 0f64, 0f64, 0f64, 1f64]]);
        let mut basis_columns = HashMap::new();
        for i in 0..5 {
            basis_columns.insert(i, i);
        }
        let column_info = vec![String::from(ARTIFICIAL),
                               String::from(ARTIFICIAL),
                               String::from(ARTIFICIAL),
                               String::from(ARTIFICIAL),
                               String::from(ARTIFICIAL),
                               String::from("XONE"),
                               String::from("YTWO"),
                               String::from("ZTHREE"),
                               String::from("SLACK0"),
                               String::from("SLACK1"),
                               String::from("SLACK2"),
                               String::from("SLACK3")];
        let expected = Tableau::new(data, c, carry, basis_columns, column_info);

        assert_eq!(result, expected);
    }

    fn canonical_small() -> CanonicalForm {
        let data = SparseMatrix::from_data(vec![vec![3f64, 2f64, 1f64, 0f64, 0f64],
                                                vec![5f64, 1f64, 1f64, 1f64, 0f64],
                                                vec![2f64, 5f64, 1f64, 0f64, 1f64]]);
        let b = DenseVector::from_data(vec![1f64, 3f64, 4f64]);
        let cost = SparseVector::from_data(vec![1f64, 1f64, 1f64, 1f64, 1f64]);
        let column_info = vec![(String::from("X1"), VariableType::Continuous),
                               (String::from("X2"), VariableType::Continuous),
                               (String::from("X3"), VariableType::Continuous),
                               (String::from("X4"), VariableType::Continuous),
                               (String::from("X5"), VariableType::Continuous)];

        CanonicalForm::new(data, b, cost, column_info)
    }

    fn artificial_tableau() -> Tableau {
        let data = SparseMatrix::from_data(vec![vec![1f64, 0f64, 0f64, 3f64, 2f64, 1f64, 0f64, 0f64],
                                                vec![0f64, 1f64, 0f64, 5f64, 1f64, 1f64, 1f64, 0f64],
                                                vec![0f64, 0f64, 1f64, 2f64, 5f64, 1f64, 0f64, 1f64]]);
        let real_c = SparseVector::from_data(vec![0f64, 0f64, 0f64, 1f64, 1f64, 1f64, 1f64, 1f64]);
        let carry = DenseMatrix::from_data(vec![vec![0f64, 0f64, 0f64, 0f64],
                                                vec![-8f64, -1f64, -1f64, -1f64],
                                                vec![1f64, 1f64, 0f64, 0f64],
                                                vec![3f64, 0f64, 1f64, 0f64],
                                                vec![4f64, 0f64, 0f64, 1f64]]);
        let mut basis_columns = HashMap::new();
        basis_columns.insert(0, 0);
        basis_columns.insert(1, 1);
        basis_columns.insert(2, 2);
        let column_info = vec![String::from(ARTIFICIAL),
                               String::from(ARTIFICIAL),
                               String::from(ARTIFICIAL),
                               String::from("X1"),
                               String::from("X2"),
                               String::from("X3"),
                               String::from("X4"),
                               String::from("X5")];

        Tableau::new(data, real_c, carry, basis_columns, column_info)
    }

    fn tableau() -> Tableau {
        let data = SparseMatrix::from_data(vec![vec![3f64, 2f64, 1f64, 0f64, 0f64],
                                                vec![5f64, 1f64, 1f64, 1f64, 0f64],
                                                vec![2f64, 5f64, 1f64, 0f64, 1f64]]);
        let original_c = SparseVector::from_data(vec![1f64, 1f64, 1f64, 1f64, 1f64]);
        let carry = DenseMatrix::from_data(vec![vec![-6f64, 1f64, -1f64, -1f64],
                                                vec![0f64, 1f64, 1f64, 1f64],
                                                vec![1f64, 1f64, 0f64, 0f64],
                                                vec![2f64, -1f64, 1f64, 0f64],
                                                vec![3f64, -1f64, 0f64, 1f64]]);
        let mut basis_columns = HashMap::new();
        basis_columns.insert(0, 2);
        basis_columns.insert(1, 3);
        basis_columns.insert(2, 4);
        let column_info = vec![String::from("X1"),
                               String::from("X2"),
                               String::from("X3"),
                               String::from("X4"),
                               String::from("X5")];

        Tableau::new(data, original_c, carry, basis_columns, column_info)
    }

    #[test]
    fn test_create_artificial_tableau() {
        let result = create_artificial_tableau(&canonical_small());
        let expected = artificial_tableau();

        assert_eq!(result, expected);
    }


    #[test]
    fn test_cost() {
        let artificial_tableau = artificial_tableau();
        assert_approx_eq!(artificial_tableau.cost(ARTIFICIAL_COST), 8f64);
        assert_approx_eq!(artificial_tableau.cost(ACTUAL_COST), 0f64);

        let tableau = tableau();
        assert_approx_eq!(tableau.cost(ACTUAL_COST), 6f64);
    }

    #[test]
    fn test_relative_cost() {
        let artificial_tableau = artificial_tableau();
        assert_approx_eq!(artificial_tableau.relative_cost(ARTIFICIAL_COST, 0), 0f64);
        assert_approx_eq!(artificial_tableau.relative_cost(ARTIFICIAL_COST, 3), -10f64);

        assert_approx_eq!(artificial_tableau.relative_cost(ACTUAL_COST, 0), 0f64);
        assert_approx_eq!(artificial_tableau.relative_cost(ACTUAL_COST, 3), 1f64);

        let tableau = tableau();
        assert_approx_eq!(tableau.relative_cost(ACTUAL_COST, 0), -3f64);
        assert_approx_eq!(tableau.relative_cost(ACTUAL_COST, 1), -3f64);
        assert_approx_eq!(tableau.relative_cost(ACTUAL_COST, 2), 0f64);
    }

    #[test]
    fn test_generate_column() {
        let artificial_tableau = artificial_tableau();
        let column = artificial_tableau.generate_column(3);
        let expected = SparseVector::from_data(vec![1f64, -10f64, 3f64, 5f64, 2f64]);

        for i in 0..column.len() {
            assert_approx_eq!(column.get_value(i), expected.get_value(i));
        }

        let tableau = tableau();
        let column = tableau.generate_column(0);
        let expected = SparseVector::from_data(vec![-3f64, f64::NAN, 3f64, 2f64, -1f64]);

        for row in 0..column.len() {
            if row != ARTIFICIAL_COST {
                assert_approx_eq!(column.get_value(row), expected.get_value(row));
            }
        }
    }

    #[test]
    fn test_find_profitable_column() {
        let artificial_tableau = artificial_tableau();
        let expected = 3;
        if let Some(column) = artificial_tableau.profitable_column(ARTIFICIAL_COST) {
            assert_eq!(column, expected);
        } else {
            assert!(false, format!("Column {} should be profitable", expected));
        }

        let tableau = tableau();
        let expected = 0;
        if let Some(column) = tableau.profitable_column(ACTUAL_COST) {
            assert_eq!(column, expected);
        } else {
            assert!(false, format!("Column {} should be profitable", expected));
        }
    }

    #[test]
    fn test_find_pivot_row() {
        let artificial_tableau = artificial_tableau();
        let column = SparseVector::from_data(vec![1f64, -10f64, 3f64, 5f64, 2f64]);
        assert_eq!(artificial_tableau.find_pivot_row(&column), 0);

        let column = SparseVector::from_data(vec![1f64, -8f64, 2f64, 1f64, 5f64]);
        assert_eq!(artificial_tableau.find_pivot_row(&column), 0);

        let tableau = tableau();
        let column = SparseVector::from_data(vec![-3f64, f64::NAN, 3f64, 2f64, -1f64]);
        assert_eq!(tableau.find_pivot_row(&column), 0);

        let column = SparseVector::from_data(vec![-3f64, f64::NAN, 2f64, -1f64, 3f64]);
        assert_eq!(tableau.find_pivot_row(&column), 0);
    }

    #[test]
    fn test_bring_into_basis() {
        let mut artificial_tableau = artificial_tableau();
        let column = 3;
        artificial_tableau.bring_into_basis(3);

        assert!(artificial_tableau.in_basis(column));
        assert_approx_eq!(artificial_tableau.cost(ARTIFICIAL_COST), 14f64 / 3f64);
        assert_approx_eq!(artificial_tableau.cost(ACTUAL_COST), 1f64 / 3f64);

        let mut tableau = tableau();
        let column = 1;
        tableau.bring_into_basis(column);

        assert!(tableau.in_basis(column));
        assert_approx_eq!(tableau.cost(ACTUAL_COST), 9f64 / 2f64);
    }

    fn bfs_tableau() -> Tableau {
        let data = SparseMatrix::from_data(vec![vec![1f64, 0f64, 0f64, 3f64, 2f64, 1f64, 0f64, 0f64],
                                                vec![0f64, 1f64, 0f64, 5f64, 1f64, 1f64, 1f64, 0f64],
                                                vec![0f64, 0f64, 1f64, 2f64, 5f64, 1f64, 0f64, 1f64]]);
        let real_c = SparseVector::from_data(vec![0f64, 0f64, 0f64, 1f64, 1f64, 1f64, 1f64, 1f64]);
        let carry = DenseMatrix::from_data(vec![vec![-6f64, 1f64, -1f64, -1f64],
                                                vec![0f64, 1f64, 1f64, 1f64],
                                                vec![1f64, 1f64, 0f64, 0f64],
                                                vec![2f64, -1f64, 1f64, 0f64],
                                                vec![3f64, -1f64, 0f64, 1f64]]);
        let mut basis_columns = HashMap::new();
        basis_columns.insert(0, 5);
        basis_columns.insert(1, 6);
        basis_columns.insert(2, 7);
        let column_info = vec![String::from(ARTIFICIAL),
                               String::from(ARTIFICIAL),
                               String::from(ARTIFICIAL),
                               String::from("X1"),
                               String::from("X2"),
                               String::from("X3"),
                               String::from("X4"),
                               String::from("X5")];

        Tableau::new(data, real_c, carry, basis_columns, column_info)
    }

    #[test]
    fn test_create_tableau() {
        let bfs_tableau = bfs_tableau();
        assert!(bfs_tableau.profitable_column(ARTIFICIAL_COST).is_none());

        let without_artificials = Tableau::create_from(&canonical_small(),
        bfs_tableau.carry(),
        bfs_tableau.basis_columns_map());
        let expected = tableau();

        assert_eq!(without_artificials, expected);
    }
}
