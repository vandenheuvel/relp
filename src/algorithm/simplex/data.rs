//! # Data structures for Simplex
//!
//! Contains the simplex tableau and logic for elementary operations which can be perfomed upon it.
//! The tableau is extended with supplementary data structures for efficiency.
use std::collections::{HashMap, HashSet};
use std::f64;
use std::slice::Iter;

use algorithm::simplex::EPSILON;
use algorithm::simplex::tableau_provider::matrix_data::MatrixData;
use algorithm::simplex::tableau_provider::TableauProvider;

use data::linear_program::canonical_form::CanonicalForm;
use data::linear_program::elements::{Variable, VariableType};
use data::linear_algebra::matrix::{DenseMatrix, Matrix, SparseMatrix};
use data::linear_algebra::vector::{DenseVector, SparseVector, Vector};

pub const ARTIFICIAL: &str = "ARTIFICIAL";

#[derive(Clone, Copy, Debug)]
pub enum CostRow {
    Actual = 0,
    Artificial = 1,
}

/// Holds all information necessary to execute the Simplex algorithm.
///
/// The original tableau, as can be extracted from a linear program in `CanonicalForm`, is not
/// being mutated during the lifetime of the tableau. The same holds from the cost rows. The carry
/// matrix is used to represent the different changes in basis, while the current basis columns are
/// maintained in a map and set.
#[derive(Debug, PartialEq)]
pub struct Tableau<T> where T: TableauProvider {
    /// Supplies data about the original tableau
    original_matrix_data: T,

    /// Matrix of size (m + 1) x (m + 1), changed every basis change
    carry: DenseMatrix,
    /// Maps the rows to the column containing it's pivot.
    basis_columns: HashMap<usize, usize>,
    /// m
    nr_rows: usize,
    /// n
    nr_columns: usize,

    /// Contains the names of all variables. Used only for outputting the result of a computation.
    column_info: Vec<String>,
}

impl<T> Tableau<T> where T: TableauProvider {
    /// Create a new `Tableau` instance.
    ///
    /// This method should be used when no basis is known.
    pub fn new(original_matrix_data: T,
               carry: DenseMatrix,
               basis_columns: HashMap<usize, usize>,
               column_info: Vec<String>,
               ) -> Tableau<T> where T: TableauProvider {

        let nr_rows = original_matrix_data.nr_rows();
        let nr_columns = original_matrix_data.nr_columns();

        debug_assert_eq!(original_matrix_data.nr_rows(), nr_rows);
        debug_assert_eq!(carry.nr_rows(), 1 + 1 + nr_rows);
        debug_assert_eq!(carry.nr_columns(), 1 + nr_rows);
        debug_assert_eq!(basis_columns.len(), nr_rows);

        debug_assert_eq!(original_matrix_data.nr_columns(), nr_columns);
        debug_assert_eq!(column_info.len(), nr_columns);

        Tableau {
            original_matrix_data,

            carry,
            basis_columns,
            nr_rows,
            nr_columns,

            column_info,
        }
    }
    /// Brings a column into the basis by updating the `self.carry` matrix and updating the
    /// data structures holding the collection of basis columns.
    pub fn bring_into_basis(&mut self, pivot_row_nr: usize, pivot_column_nr: usize, column: &SparseVector) {
        debug_assert!(pivot_column_nr < self.nr_columns);
        debug_assert!(pivot_row_nr < self.nr_rows());

        self.carry.multiply_row(1 + 1 + pivot_row_nr, 1f64 / column.get_value(1 + 1 + pivot_row_nr));
        for edit_row in 0..self.carry.nr_rows() {
            if edit_row != 1 + 1 + pivot_row_nr {
                if column.get_value(edit_row).abs() > 1e-10 {
                    let factor = column.get_value(edit_row);
                    self.carry.mul_add_rows(1 + 1 + pivot_row_nr, edit_row, -factor);
                }
            }
        }

        self.update_basis_indices(pivot_row_nr, pivot_column_nr);
    }
    /// Removes the index of the variable leaving the basis from the `basis_column_map` and
    /// `basis_column_set` data structures, while inserting the entering variable index.
    fn update_basis_indices(&mut self, pivot_row: usize, pivot_column: usize) {
        debug_assert!(self.basis_columns.contains_key(&pivot_row));

        let leaving_column = *self.basis_columns.get(&pivot_row).unwrap();
        self.basis_columns.insert(pivot_row, pivot_column);
    }
    /// Generate a column of the tableau as it would look like with the current basis by matrix
    /// multiplying the original column and the carry matrix.
    pub fn generate_column(&self, column: usize) -> SparseVector {
        debug_assert!(column < self.nr_columns);

        let mut generated = Vec::with_capacity(1 + 1 + self.nr_rows);
        generated.push((CostRow::Actual as usize, self.relative_cost(CostRow::Actual, column)));
        generated.push((CostRow::Artificial as usize, self.relative_cost(CostRow::Artificial, column)));
        for row in 0..self.nr_rows() {
            let value = self.original_matrix_data.column(column)
                .map(|&(j, v)| v * self.carry.get_value(1 + 1 + row, 1 + j))
                .sum::<f64>();
            if value.abs() > EPSILON {
                generated.push((1 + 1 + row, value));
            }
        }

        SparseVector::from_tuples(generated, 1 + 1 + self.nr_rows())
    }
    pub fn generate_artificial_column(&self, j: usize) -> SparseVector {
        debug_assert!(j < self.nr_rows());

        SparseVector::from_data(self.carry.column(j))
    }
    /// Determine the row to pivot on, given the column. This is the row with the positive but
    /// minimal ratio between the current constraint vector and the column.
    pub fn find_pivot_row(&self, column: &SparseVector) -> usize {
        debug_assert_eq!(column.len(), 1 + 1 + self.nr_rows);

        let mut min_index = usize::max_value();
        let mut min_ratio = f64::INFINITY;
        let mut min_leaving_column = usize::max_value();

        for &(row, xij) in column.values() {
            if row == CostRow::Actual as usize || row == CostRow::Artificial as usize {
                continue;
            }
            if xij > EPSILON {
                let ratio = self.carry.get_value(row, 0) / xij;
                let leaving_column = *self.basis_columns.get(&(row - (1 + 1))).unwrap();
                if (ratio - min_ratio).abs() < EPSILON && leaving_column < min_leaving_column {
                    min_index = row;
                    min_leaving_column = leaving_column;
                } else if ratio < min_ratio {
                    min_index = row;
                    min_ratio = ratio;
                    min_leaving_column = leaving_column;
                }
                }
        }

        debug_assert_ne!(min_index, usize::max_value());
        debug_assert_ne!(min_leaving_column, usize::max_value());

        min_index - 1 - 1
    }
    /// Find a profitable column.
    pub fn profitable_column(&self, cost_row: CostRow) -> Option<usize> {
        (0..self.nr_columns())
            .filter(|column| !self.is_in_basis(column))
            .find(|column| self.relative_cost(cost_row, *column) < -EPSILON)
    }
    /// Calculates the relative cost of a non-basis column to pivot on.
    pub fn relative_cost(&self, cost_row: CostRow, column: usize) -> f64 {
        debug_assert!(column < self.nr_columns);

        let mut total = match cost_row {
            CostRow::Actual => self.original_matrix_data.get_actual_cost_value(column),
            CostRow::Artificial => self.original_matrix_data.get_artificial_cost_value(column),
        };

        for (column, value) in self.original_matrix_data.column(column) {
            total += *value * self.carry.get_value(cost_row as usize, 1 + column);
        }
        total
    }
    /// Gets a copy of the carry matrix.
    pub fn carry(&self) -> DenseMatrix {
        self.carry.clone()
    }
    /// Checks whether a column is in the basis.
    ///
    /// Note: This method may not be accurate when there are artificial variables.
    pub fn is_in_basis(&self, column: &usize) -> bool {
        debug_assert!(*column < self.nr_columns);

        self.basis_columns.values().find(|&j| j == column).is_some()
    }
    /// Gets a map from the rows to the column containing a pivot of that row.
    pub fn basis_columns(&self) -> HashMap<usize, usize> {
        self.basis_columns.clone()
    }
    pub fn get_basis_column(&self, i: &usize) -> Option<&usize> {
        self.basis_columns.get(i)
    }
    /// Return the current values for all variables in the current basic feasible solution.
    pub fn current_bfs(&self) -> SparseVector {
        let mut data = Vec::new();
        for row in self.basis_columns.keys() {
            let column = self.basis_columns.get(row).unwrap();
            data.push((*column, self.carry.get_value(1 + 1 + *row, 0)));
        }
        data.sort_by_key(|&(i, _)| i);
        SparseVector::from_tuples(data, self.nr_columns)
    }
    /// Get the cost of the current solution.
    pub fn cost(&self, cost_row: CostRow) -> f64 {
        -self.carry.get_value(cost_row as usize, 0)
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

impl Tableau<MatrixData> {
    /// Create a `Tableau` instance using a known basic feasible solution.
    pub fn create_from(canonical: &CanonicalForm,
                       carry: DenseMatrix, basis: HashMap<usize, usize>) -> Tableau<MatrixData> {
        let data = MatrixData::new(canonical.data(), canonical.cost());
        Tableau::new(data, carry, basis, canonical.variable_info())
    }
}

/// Creates a Simplex tableau augmented with artificial variables in a basic feasible solution
/// having only the artificial variables in the basis.
pub fn create_artificial_tableau(canonical: &CanonicalForm) -> Tableau<MatrixData> {
    let m = canonical.nr_constraints();
    let n = m + canonical.nr_variables();

    let data = MatrixData::new(canonical.data(), canonical.cost());
    let carry = create_carry_matrix(canonical.b(), DenseMatrix::identity(m));

    let mut basis = HashMap::with_capacity(m);
    for i in 0..m {
        basis.insert(i, i);
    }

    Tableau::new(data, carry, basis, canonical.variable_info())
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

        let column_info = vec![Variable { name: "XONE".to_string(),   variable_type: VariableType::Continuous, offset: 0f64, },
                               Variable { name: "YTWO".to_string(),   variable_type: VariableType::Continuous, offset: 0f64, },
                               Variable { name: "ZTHREE".to_string(), variable_type: VariableType::Continuous, offset: 0f64, },
                               Variable { name: "SLACK0".to_string(), variable_type: VariableType::Continuous, offset: 0f64, },
                               Variable { name: "SLACK1".to_string(), variable_type: VariableType::Continuous, offset: 0f64, },
                               Variable { name: "SLACK2".to_string(), variable_type: VariableType::Continuous, offset: 0f64, },
                               Variable { name: "SLACK3".to_string(), variable_type: VariableType::Continuous, offset: 0f64, }];

        CanonicalForm::new(data, b, cost, 0f64, column_info, Vec::new())
    }

    #[test]
    fn test_create_artificial_tableau_large() {
        let result = create_artificial_tableau(&canonical_large());

        let data = SparseMatrix::from_data(vec![vec![1f64, 1f64, 0f64, 1f64, 0f64, 0f64, 0f64],
                                                vec![1f64, 0f64, 1f64, 0f64, -1f64, 0f64, 0f64],
                                                vec![0f64, -1f64, 1f64, 0f64, 0f64, 0f64, 0f64],
                                                vec![1f64, 0f64, 0f64, 0f64, 0f64, 1f64, 0f64],
                                                vec![0f64, 1f64, 0f64, 0f64, 0f64, 0f64, 1f64]]);
        let c = SparseVector::from_data(vec![1f64, 4f64, 9f64, 0f64, 0f64, 0f64, 0f64]);
        let data = MatrixData::new(data, c);
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
        let column_info = vec![String::from("XONE"),
                               String::from("YTWO"),
                               String::from("ZTHREE"),
                               String::from("SLACK0"),
                               String::from("SLACK1"),
                               String::from("SLACK2"),
                               String::from("SLACK3")];
        let expected = Tableau::new(data, carry, basis_columns, column_info);

        assert_eq!(result, expected);
    }

    fn canonical_small() -> CanonicalForm {
        let data = SparseMatrix::from_data(vec![vec![3f64, 2f64, 1f64, 0f64, 0f64],
                                                vec![5f64, 1f64, 1f64, 1f64, 0f64],
                                                vec![2f64, 5f64, 1f64, 0f64, 1f64]]);
        let b = DenseVector::from_data(vec![1f64, 3f64, 4f64]);
        let cost = SparseVector::from_data(vec![1f64, 1f64, 1f64, 1f64, 1f64]);
        let column_info = vec![Variable { name: "X1".to_string(), variable_type: VariableType::Continuous, offset: 0f64, },
                               Variable { name: "X2".to_string(), variable_type: VariableType::Continuous, offset: 0f64, },
                               Variable { name: "X3".to_string(), variable_type: VariableType::Continuous, offset: 0f64, },
                               Variable { name: "X4".to_string(), variable_type: VariableType::Continuous, offset: 0f64, },
                               Variable { name: "X5".to_string(), variable_type: VariableType::Continuous, offset: 0f64, }];

        CanonicalForm::new(data, b, cost, 0f64, column_info, Vec::new())
    }

    fn artificial_tableau() -> Tableau<MatrixData> {
        let data = SparseMatrix::from_data(vec![vec![3f64, 2f64, 1f64, 0f64, 0f64],
                                                vec![5f64, 1f64, 1f64, 1f64, 0f64],
                                                vec![2f64, 5f64, 1f64, 0f64, 1f64]]);
        let real_c = SparseVector::from_data(vec![1f64, 1f64, 1f64, 1f64, 1f64]);
        let data = MatrixData::new(data, real_c);
        let carry = DenseMatrix::from_data(vec![vec![0f64, 0f64, 0f64, 0f64],
                                                vec![-8f64, -1f64, -1f64, -1f64],
                                                vec![1f64, 1f64, 0f64, 0f64],
                                                vec![3f64, 0f64, 1f64, 0f64],
                                                vec![4f64, 0f64, 0f64, 1f64]]);
        let mut basis_columns = HashMap::new();
        basis_columns.insert(0, 0);
        basis_columns.insert(1, 1);
        basis_columns.insert(2, 2);
        let column_info = vec![String::from("X1"),
                               String::from("X2"),
                               String::from("X3"),
                               String::from("X4"),
                               String::from("X5")];

        Tableau::new(data, carry, basis_columns, column_info)
    }

    fn tableau() -> Tableau<MatrixData> {
        let data = SparseMatrix::from_data(vec![vec![3f64, 2f64, 1f64, 0f64, 0f64],
                                                vec![5f64, 1f64, 1f64, 1f64, 0f64],
                                                vec![2f64, 5f64, 1f64, 0f64, 1f64]]);
        let original_c = SparseVector::from_data(vec![1f64, 1f64, 1f64, 1f64, 1f64]);
        let data = MatrixData::new(data, original_c);
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

        Tableau::new(data, carry, basis_columns, column_info)
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
        assert_approx_eq!(artificial_tableau.cost(CostRow::Artificial), 8f64);
        assert_approx_eq!(artificial_tableau.cost(CostRow::Actual), 0f64);

        let tableau = tableau();
        assert_approx_eq!(tableau.cost(CostRow::Actual), 6f64);
    }

    #[test]
    fn test_relative_cost() {
        let artificial_tableau = artificial_tableau();
        assert_approx_eq!(artificial_tableau.relative_cost(CostRow::Artificial, 0), -10f64);

        assert_approx_eq!(artificial_tableau.relative_cost(CostRow::Actual, 0), 1f64);

        let tableau = tableau();
        assert_approx_eq!(tableau.relative_cost(CostRow::Actual, 0), -3f64);
        assert_approx_eq!(tableau.relative_cost(CostRow::Actual, 1), -3f64);
        assert_approx_eq!(tableau.relative_cost(CostRow::Actual, 2), 0f64);
    }

    #[test]
    fn test_generate_column() {
        let artificial_tableau = artificial_tableau();
        let column = artificial_tableau.generate_column(0);
        let expected = SparseVector::from_data(vec![1f64, -10f64, 3f64, 5f64, 2f64]);

        for i in 0..column.len() {
            assert_approx_eq!(column.get_value(i), expected.get_value(i));
        }

        let tableau = tableau();
        let column = tableau.generate_column(0);
        let expected = SparseVector::from_data(vec![-3f64, f64::NAN, 3f64, 2f64, -1f64]);

        for row in 0..column.len() {
            if row != CostRow::Artificial as usize {
                assert_approx_eq!(column.get_value(row), expected.get_value(row));
            }
        }
    }

    #[test]
    fn test_find_profitable_column() {
        let artificial_tableau = artificial_tableau();
        let expected = 3;
        if let Some(column) = artificial_tableau.profitable_column(CostRow::Artificial) {
            assert_eq!(column, expected);
        } else {
            assert!(false, format!("Column {} should be profitable", expected));
        }

        let tableau = tableau();
        let expected = 0;
        if let Some(column) = tableau.profitable_column(CostRow::Actual) {
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
        let column = 0;
        let column_data = artificial_tableau.generate_column(column);
        let row = artificial_tableau.find_pivot_row(&column_data);
        artificial_tableau.bring_into_basis(row, column, &column_data);

        assert!(artificial_tableau.is_in_basis(&column));
        assert_approx_eq!(artificial_tableau.cost(CostRow::Artificial), 14f64 / 3f64);
        assert_approx_eq!(artificial_tableau.cost(CostRow::Actual), 1f64 / 3f64);

        let mut tableau = tableau();
        let column = 1;
        let column_data = tableau.generate_column(column);
        let row = tableau.find_pivot_row(&column_data);
        tableau.bring_into_basis(row, column, &column_data);

        assert!(tableau.is_in_basis(&column));
        assert_approx_eq!(tableau.cost(CostRow::Actual), 9f64 / 2f64);
    }

    fn bfs_tableau() -> Tableau<MatrixData> {
        let data = SparseMatrix::from_data(vec![vec![3f64, 2f64, 1f64, 0f64, 0f64],
                                                vec![5f64, 1f64, 1f64, 1f64, 0f64],
                                                vec![2f64, 5f64, 1f64, 0f64, 1f64]]);
        let real_c = SparseVector::from_data(vec![1f64, 1f64, 1f64, 1f64, 1f64]);
        let data = MatrixData::new(data, real_c);
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

        Tableau::new(data, carry, basis_columns, column_info)
    }

    #[test]
    fn test_create_tableau() {
        let bfs_tableau = bfs_tableau();
        assert!(bfs_tableau.profitable_column(CostRow::Artificial).is_none());

        let without_artificials = Tableau::create_from(&canonical_small(),
            bfs_tableau.carry(),
            bfs_tableau.basis_columns());
        let expected = tableau();

        assert_eq!(without_artificials, expected);
    }
}
