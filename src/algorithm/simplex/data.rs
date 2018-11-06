//! # Data structures for Simplex
//!
//! Contains the simplex tableau and logic for elementary operations which can be perfomed upon it.
//! The tableau is extended with supplementary data structures for efficiency.
use std::collections::{HashMap};
use std::f64;
use std::fmt::{Display, Formatter, Result as FormatResult};
use std::iter::repeat;

use algorithm::simplex::EPSILON;
use algorithm::simplex::tableau_provider::TableauProvider;
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
pub struct Tableau<'a, T: 'a> where T: TableauProvider {
    /// Supplies data about the original tableau
    provider: &'a T,

    /// Matrix of size (m + 1) x (m + 1), changed every basis change
    carry: DenseMatrix,
    /// Maps the rows to the column containing it's pivot.
    basis_columns: HashMap<usize, usize>,
    /// m
    nr_rows: usize,
    /// n
    nr_columns: usize,
}

impl<'a, T> Tableau<'a, T> where T: TableauProvider {
    /// Creates a Simplex tableau augmented with artificial variables in a basic feasible solution
    /// having only the artificial variables in the basis.
    pub fn new_artificial(provider: &'a T) -> Tableau<'a, T> {
        let m = provider.nr_rows();
        let carry = Self::create_carry_matrix(provider.get_b().clone(), DenseMatrix::identity(m));
        let basis_columns = (0..m).map(|i| (i, i)).collect();

        Tableau::new(provider, carry, basis_columns)
    }
    /// Creates a Simplex tableau with a specific basis.
    pub fn new(provider: &T,
               carry: DenseMatrix,
               basis_columns: HashMap<usize, usize>) -> Tableau<T> {
        let nr_rows = provider.nr_rows();
        let nr_columns = provider.nr_columns();

        debug_assert_eq!(provider.nr_rows(), nr_rows);
        debug_assert_eq!(carry.nr_rows(), 1 + 1 + nr_rows);
        debug_assert_eq!(carry.nr_columns(), 1 + nr_rows);
        debug_assert_eq!(basis_columns.len(), nr_rows);

        debug_assert_eq!(provider.nr_columns(), nr_columns);

        Tableau {
            provider,
            carry,
            basis_columns,

            nr_rows,
            nr_columns,
        }
    }
    /// Create a carry matrix for a tableau with a known basis inverse.
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

        let _leaving_column = *self.basis_columns.get(&pivot_row).unwrap();
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
            let value = self.provider.column(column)
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

        SparseVector::from_data(self.carry.column(1 + j))
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
            CostRow::Actual => self.provider.get_actual_cost_value(column),
            CostRow::Artificial => self.provider.get_artificial_cost_value(column),
        };

        for (column, value) in self.provider.column(column) {
            total += *value * self.carry.get_value(cost_row as usize, 1 + column);
        }
        total
    }
    fn as_sparse_matrix(&self) -> SparseMatrix {
        let mut m = SparseMatrix::zeros(1 + 1 + self.nr_rows, self.nr_columns);
        for j in 0..self.nr_columns {
            m.set_column(j, self.generate_column(j).values());
        }
        m
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

impl<'a, T> Display for Tableau<'a, T> where T: TableauProvider {
    fn fmt(&self, f: &mut Formatter) -> FormatResult {
        writeln!(f, "Tableau:")?;
        writeln!(f, "=== Current Tableau ===")?;
        let column_width = 10;
        let counter_width = 5;
        // Column counter
        write!(f, "{0:width$}", "", width = counter_width)?;
        for column_index in 0..self.nr_columns {
            write!(f, "{0:>width$}", column_index, width = column_width)?;
        }
        writeln!(f, "")?;

        // Row counter and row data
        let m = self.as_sparse_matrix();
        for row_index in 0..m.nr_rows() {
            if row_index == 1 + 1 {
                writeln!(f, "{}", repeat("-").take(counter_width + m.nr_columns() * column_width)
                    .collect::<String>())?;
            }
            write!(f, "{0: <width$}", if row_index < 2 { row_index } else { row_index - 1 - 1},
                   width = counter_width)?;
            for column_index in 0..m.nr_columns() {
                write!(f, "{0:>width$.5}", m.get_value(row_index, column_index),
                       width = column_width)?;
            }
            writeln!(f, "")?;
        }
        writeln!(f, "= Artificial Part of Tableau =")?;
        for row_index in 0..(1 + 1 + self.nr_rows) {
            for column_index in 0..self.nr_rows {
                write!(f, "{0:>width$.5}",
                       self.generate_artificial_column(column_index).get_value(row_index),
                       width = column_width)?;
            }
            writeln!(f, "")?;
        }

        writeln!(f, "=== Basis Columns ===")?;
        let mut basis = self.basis_columns.iter()
            .map(|(&i, &j)| (i, j))
            .collect::<Vec<(usize, usize)>>();
        basis.sort_by_key(|&(i, _)| i);
        writeln!(f, "{:?}", basis)?;

        writeln!(f, "=== Basis Inverse ===")?;
        self.carry.fmt(f)?;

        writeln!(f, "=== Data Provider ===")?;
        self.provider.fmt(f)
    }
}


#[cfg(test)]
mod test {

    use super::*;
    use data::linear_program::canonical_form::CanonicalForm;
    use algorithm::simplex::tableau_provider::matrix_data::MatrixData;
    use data::linear_program::elements::Variable;
    use data::linear_program::elements::VariableType;

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
        let matrix_data = MatrixData::from(canonical_large());
        let result = Tableau::new_artificial(&matrix_data);

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
        let expected = Tableau::new(&matrix_data, carry, basis_columns);

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

    fn matrix_data() -> MatrixData {
        let data = SparseMatrix::from_data(vec![vec![3f64, 2f64, 1f64, 0f64, 0f64],
                                                vec![5f64, 1f64, 1f64, 1f64, 0f64],
                                                vec![2f64, 5f64, 1f64, 0f64, 1f64]]);
        let original_c = SparseVector::from_data(vec![1f64, 1f64, 1f64, 1f64, 1f64]);
        let column_info = vec![String::from("X1"),
                               String::from("X2"),
                               String::from("X3"),
                               String::from("X4"),
                               String::from("X5")];
        let b = DenseVector::from_data(vec![1f64, 3f64, 4f64]);

        MatrixData::new(data, original_c, b, column_info)
    }

    fn artificial_tableau(data: &MatrixData) -> Tableau<MatrixData> {
        let carry = DenseMatrix::from_data(vec![vec![0f64, 0f64, 0f64, 0f64],
                                                vec![-8f64, -1f64, -1f64, -1f64],
                                                vec![1f64, 1f64, 0f64, 0f64],
                                                vec![3f64, 0f64, 1f64, 0f64],
                                                vec![4f64, 0f64, 0f64, 1f64]]);
        let basis_columns = (0..3).map(|i| (i, i)).collect();

        Tableau::new(data, carry, basis_columns)
    }

    fn tableau(data: &MatrixData) -> Tableau<MatrixData> {
        let carry = DenseMatrix::from_data(vec![vec![-6f64, 1f64, -1f64, -1f64],
                                                vec![0f64, 1f64, 1f64, 1f64],
                                                vec![1f64, 1f64, 0f64, 0f64],
                                                vec![2f64, -1f64, 1f64, 0f64],
                                                vec![3f64, -1f64, 0f64, 1f64]]);
        let mut basis_columns = HashMap::new();
        basis_columns.insert(0, 2);
        basis_columns.insert(1, 3);
        basis_columns.insert(2, 4);

        Tableau::new(data, carry, basis_columns)
    }

    #[test]
    fn test_create_artificial_tableau() {
        let data = matrix_data();
        let from_canonical = MatrixData::from(canonical_small());
        let result = Tableau::new_artificial(&from_canonical);
        let expected = artificial_tableau(&data);

        assert_eq!(result, expected);
    }


    #[test]
    fn test_cost() {
        let data = matrix_data();
        let artificial_tableau = artificial_tableau(&data);
        assert_abs_diff_eq!(artificial_tableau.cost(CostRow::Artificial), 8f64);
        assert_abs_diff_eq!(artificial_tableau.cost(CostRow::Actual), 0f64);

        let tableau = tableau(&data);
        assert_abs_diff_eq!(tableau.cost(CostRow::Actual), 6f64);
    }

    #[test]
    fn test_relative_cost() {
        let data = matrix_data();
        let artificial_tableau = artificial_tableau(&data);
        assert_abs_diff_eq!(artificial_tableau.relative_cost(CostRow::Artificial, 0), -10f64);

        assert_abs_diff_eq!(artificial_tableau.relative_cost(CostRow::Actual, 0), 1f64);

        let tableau = tableau(&data);
        assert_abs_diff_eq!(tableau.relative_cost(CostRow::Actual, 0), -3f64);
        assert_abs_diff_eq!(tableau.relative_cost(CostRow::Actual, 1), -3f64);
        assert_abs_diff_eq!(tableau.relative_cost(CostRow::Actual, 2), 0f64);
    }

    #[test]
    fn test_generate_column() {
        let data = matrix_data();
        let artificial_tableau = artificial_tableau(&data);
        let column = artificial_tableau.generate_column(0);
        let expected = SparseVector::from_data(vec![1f64, -10f64, 3f64, 5f64, 2f64]);

        for i in 0..column.len() {
            assert_abs_diff_eq!(column.get_value(i), expected.get_value(i));
        }

        let tableau = tableau(&data);
        let column = tableau.generate_column(0);
        let expected = SparseVector::from_data(vec![-3f64, f64::NAN, 3f64, 2f64, -1f64]);

        for row in 0..column.len() {
            if row != CostRow::Artificial as usize {
                assert_abs_diff_eq!(column.get_value(row), expected.get_value(row));
            }
        }
    }

    #[test]
    fn test_find_profitable_column() {
        let data = matrix_data();
        let artificial_tableau = artificial_tableau(&data);
        let expected = 3;
        if let Some(column) = artificial_tableau.profitable_column(CostRow::Artificial) {
            assert_eq!(column, expected);
        } else {
            assert!(false, format!("Column {} should be profitable", expected));
        }

        let tableau = tableau(&data);
        let expected = 0;
        if let Some(column) = tableau.profitable_column(CostRow::Actual) {
            assert_eq!(column, expected);
        } else {
            assert!(false, format!("Column {} should be profitable", expected));
        }
    }

    #[test]
    fn test_find_pivot_row() {
        let data = matrix_data();
        let artificial_tableau = artificial_tableau(&data);
        let column = SparseVector::from_data(vec![1f64, -10f64, 3f64, 5f64, 2f64]);
        assert_eq!(artificial_tableau.find_pivot_row(&column), 0);

        let column = SparseVector::from_data(vec![1f64, -8f64, 2f64, 1f64, 5f64]);
        assert_eq!(artificial_tableau.find_pivot_row(&column), 0);

        let tableau = tableau(&data);
        let column = SparseVector::from_data(vec![-3f64, f64::NAN, 3f64, 2f64, -1f64]);
        assert_eq!(tableau.find_pivot_row(&column), 0);

        let column = SparseVector::from_data(vec![-3f64, f64::NAN, 2f64, -1f64, 3f64]);
        assert_eq!(tableau.find_pivot_row(&column), 0);
    }

    #[test]
    fn test_bring_into_basis() {
        let data = matrix_data();
        let mut artificial_tableau = artificial_tableau(&data);
        let column = 0;
        let column_data = artificial_tableau.generate_column(column);
        let row = artificial_tableau.find_pivot_row(&column_data);
        artificial_tableau.bring_into_basis(row, column, &column_data);

        assert!(artificial_tableau.is_in_basis(&column));
        assert_abs_diff_eq!(artificial_tableau.cost(CostRow::Artificial), 14f64 / 3f64);
        assert_abs_diff_eq!(artificial_tableau.cost(CostRow::Actual), 1f64 / 3f64);

        let mut tableau = tableau(&data);
        let column = 1;
        let column_data = tableau.generate_column(column);
        let row = tableau.find_pivot_row(&column_data);
        tableau.bring_into_basis(row, column, &column_data);

        assert!(tableau.is_in_basis(&column));
        assert_abs_diff_eq!(tableau.cost(CostRow::Actual), 9f64 / 2f64);
    }

    fn bfs_tableau(data: &MatrixData) -> Tableau<MatrixData> {
        let carry = DenseMatrix::from_data(vec![vec![-6f64, 1f64, -1f64, -1f64],
                                                vec![0f64, 1f64, 1f64, 1f64],
                                                vec![1f64, 1f64, 0f64, 0f64],
                                                vec![2f64, -1f64, 1f64, 0f64],
                                                vec![3f64, -1f64, 0f64, 1f64]]);
        let mut basis_columns = HashMap::new();
        basis_columns.insert(0, 2);
        basis_columns.insert(1, 3);
        basis_columns.insert(2, 4);

        Tableau::new(data, carry, basis_columns)
    }

    #[test]
    fn test_create_tableau() {
        let data = matrix_data();
        let bfs_tableau = bfs_tableau(&data);
        assert!(bfs_tableau.profitable_column(CostRow::Artificial).is_none());

        let without_artificial = Tableau::new(&data,
            bfs_tableau.carry(),
            bfs_tableau.basis_columns());
        let expected = tableau(&data);

        assert_eq!(without_artificial, expected);
    }
}
