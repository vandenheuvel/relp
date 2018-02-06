//! # Matrix implementations
//!
//! The `Matrix` trait defines a set of operations available for all matrix types defined in this
//! module.

use std::iter::Iterator;
use std::slice::Iter;

use data::linear_algebra::vector::Vector;

/// Defines basic ways to create or change a matrix, regardless of back-end.
pub trait Matrix {
    fn from_data(data: Vec<Vec<f64>>) -> Self;
    fn identity(size: usize) -> Self;
    fn zeros(rows: usize, columns: usize) -> Self;
    fn multiply_row(&mut self, row: usize, factor: f64);
    fn get_value(&self, row: usize, column: usize) -> f64;
    fn set_value(&mut self, row: usize, column: usize, value: f64);
    fn nr_rows(&self) -> usize;
    fn nr_columns(&self) -> usize;
    fn size(&self) -> usize;
}

/// Uses a Vec<Vec<f64>> as underlying data structure. Dimensions are fixed at creation.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseMatrix {
    data: Vec<Vec<f64>>,
    nr_rows: usize,
    nr_columns: usize,
}

impl DenseMatrix {
    /// Get all values in column `j` of this matrix.
    pub fn column(&self, j: usize) -> Vec<f64> {
        debug_assert!(j < self.nr_columns);

        self.data.iter().map(|v| v[j]).collect()
    }
    /// Get all values in row `i` ofo this matrix.
    pub fn row(&self, i: usize) -> Iter<'_, f64> {
        debug_assert!(i < self.nr_rows);

        self.data[i].iter()
    }
    /// Change row `i` to the provided `row_values`.
    pub fn set_row(&mut self, i: usize, row_values: Iter<'_, f64>) {
        debug_assert!(i < self.nr_rows);

        self.data[i] = row_values.map(|&v| v).collect();

        debug_assert_eq!(self.data[i].len(), self.nr_columns);
    }
    /// Add a multiple of row `read_row` to row `write_row`.
    pub fn mul_add_rows(&mut self, read_row: usize, write_row: usize, factor: f64) {
        debug_assert!(read_row < self.nr_rows);
        debug_assert!(write_row < self.nr_rows);

        for j in 0..self.nr_columns {
            self.data[write_row][j] += factor * self.data[read_row][j];
        }
    }
    /// Get the data of this matrix
    pub fn data(self) -> Vec<Vec<f64>> {
        self.data
    }
}

impl Matrix for DenseMatrix {
    /// Create a `DenseMatrix` from the provided data.
    fn from_data(data: Vec<Vec<f64>>) -> DenseMatrix {
        let (nr_rows, nr_columns) = get_data_dimensions(&data);
        DenseMatrix { data, nr_rows, nr_columns }
    }
    /// Create a dense square identity matrix of size `len`.
    fn identity(len: usize) -> DenseMatrix {
        debug_assert!(len > 0);

        let mut data = Vec::new();
        for i in 0..len {
            let mut row = Vec::new();
            for j in 0..len {
                if i == j {
                    row.push(1f64);
                } else {
                    row.push(0f64);
                }
            }
            data.push(row);
        }

        DenseMatrix::from_data(data)
    }
    /// Create a dense matrix of zero's of dimension `rows` x `columns`.
    fn zeros(rows: usize, columns: usize) -> DenseMatrix {
        debug_assert!(rows > 0);
        debug_assert!(columns > 0);

        let mut data = Vec::new();
        for _ in 0..rows {
            let mut row = Vec::new();
            for _ in 0..columns {
                row.push(0f64);
            }
            data.push(row);
        }

        DenseMatrix::from_data(data)
    }
    /// Multiply row `i` with a factor `factor`.
    fn multiply_row(&mut self, i: usize, factor: f64) {
        debug_assert!(i < self.nr_rows);

        for value in self.data[i].iter_mut() {
            *value *= factor;
        }
    }
    /// Get the value at coordinate (`i`, `j`).
    fn get_value(&self, i: usize, j: usize) -> f64 {
        debug_assert!(i < self.nr_rows);
        debug_assert!(j < self.nr_columns);

        self.data[i][j]
    }
    /// Set the value at coordinate (`i`, `j`) to `value`.
    fn set_value(&mut self, i: usize, j: usize, value: f64) {
        debug_assert!(i < self.nr_rows);
        debug_assert!(j < self.nr_columns);

        self.data[i][j] = value;
    }
    /// Get the number of rows in this matrix.
    fn nr_rows(&self) -> usize {
        self.nr_rows
    }
    /// Get the number of columns in this matrix.
    fn nr_columns(&self) -> usize {
        self.nr_columns
    }
    /// Get the number of values in this matrix.
    fn size(&self) -> usize {
        self.nr_rows * self.nr_columns
    }
}

/// Uses a two indices as underlying data structures: a row-major Vec<Vec<f64>> as well as a
/// column-major Vec<Vec<f64>>. Indices start at `0`.
#[derive(Debug, PartialEq)]
pub struct SparseMatrix {
    rows: Vec<Vec<(usize, f64)>>,
    columns: Vec<Vec<(usize, f64)>>,
    nr_rows: usize,
    nr_columns: usize,
}

impl SparseMatrix {
    /// Concatenate another `SparseMatrix` to the "right" (high column indices) of this matrix
    /// "horizontally" (number of rows must be equal).
    pub fn hcat(self, other: SparseMatrix) -> SparseMatrix {
        debug_assert_eq!(other.nr_rows(), self.nr_rows());

        let cols = self.nr_columns();
        let nr_rows = self.nr_rows();
        let nr_columns = cols + other.nr_columns();

        let mut rows = self.rows;
        for i in 0..nr_rows {
            for (column, value) in other.row(i) {
                rows[i].push((cols + column, *value));
            }
        }
        let mut columns = self.columns;
        columns.extend(other.columns);

        SparseMatrix { rows, columns, nr_rows, nr_columns, }
    }
    /// Get all (`row`, `value`) tuples of column `j`.
    pub fn column(&self, j: usize) -> Iter<'_, (usize, f64)> {
        debug_assert!(j < self.nr_columns);
        self.columns[j].iter()
    }
    /// Get all (`column`, `value`) tuples of row `i`.
    pub fn row(&self, i: usize) -> Iter<'_, (usize, f64)> {
        debug_assert!(i < self.nr_rows);

        self.rows[i].iter()
    }
    /// Helper method for setting a value in one of the data structures of the `SparseMatrix`.
    fn set_value_helper(major_vector: &mut Vec<Vec<(usize, f64)>>, major: usize, minor: usize, value: f64) {
        let mut updated = false;
        let minor_vector = &mut major_vector[major];

        for index in 0..minor_vector.len() {
            if minor_vector[index].0 < minor {
                continue;
            } else if minor_vector[index].0 == minor {
                minor_vector[index].1 = value;
            } else if minor_vector[index].0 > minor {
                minor_vector.insert(index, (minor, value));
            }
            updated = true;
            break;
        }
        if !updated {
            minor_vector.push((minor, value));
        }
    }
    /// Change row `i` to the values provided in `new_row`.
    pub fn set_row(&mut self, i: usize, new_row: Iter<'_, (usize, f64)>) {
        debug_assert!(i < self.nr_rows());

        self.rows[i] = new_row.map(|&t| t).collect();

        for (column, value) in self.rows[i].iter() {
            SparseMatrix::set_value_helper(&mut self.columns, *column, i, *value);
        }

        debug_assert!(self.rows[i].len() < self.nr_columns);
        debug_assert!(if let Some(maximum) = self.rows[i].iter().map(|(column, _)| *column).max() {
            maximum < self.nr_columns
        } else { true });
    }
}

impl Matrix for SparseMatrix {
    /// Create a `SparseMatrix` from the provided data.
    fn from_data(data: Vec<Vec<f64>>) -> SparseMatrix {
        let (nr_rows, nr_columns) = get_data_dimensions(&data);

        let rows: Vec<Vec<(usize, f64)>> = data.iter()
            .map(|v| v.into_iter()
                .enumerate()
                .filter(|&(_, &value)| value != 0f64)
                .map(|(column, &value)| (column, value))
                .collect())
            .collect();
        debug_assert_eq!(nr_rows, rows.len());

        let mut columns = Vec::with_capacity(nr_columns);
        for (row, row_data) in data.iter().enumerate() {
            for (column, &value) in row_data.iter().enumerate() {
                if column >= columns.len() {
                    columns.push(Vec::new());
                }

                if value != 0f64 {
                    columns[column].push((row, value));
                }
            }
        }
        debug_assert_eq!(nr_columns, columns.len());

        SparseMatrix { rows, columns, nr_rows, nr_columns, }
    }
    /// Create a dense square identity matrix of size `len`.
    fn identity(len: usize) -> SparseMatrix {
        debug_assert_ne!(len, 0);

        let mut matrix = SparseMatrix::zeros(len, len);
        for i in 0..len {
            matrix.set_value(i, i, 1f64);
        }

        matrix
    }
    /// Create a dense matrix of zero's of dimension `nr_rows` x `nr_columns`.
    fn zeros(nr_rows: usize, nr_columns: usize) -> SparseMatrix {
        debug_assert_ne!(nr_rows, 0);
        debug_assert_ne!(nr_columns, 0);

        let mut rows = Vec::with_capacity(nr_rows);
        for _ in 0..nr_rows {
            rows.push(Vec::new());
        }

        let mut columns = Vec::with_capacity(nr_columns);
        for _ in 0..nr_columns {
            columns.push(Vec::new());
        }

        SparseMatrix { rows, columns, nr_rows, nr_columns, }
    }
    /// Multiply row `i` with a factor `factor`.
    fn multiply_row(&mut self, i: usize, factor: f64) {
        debug_assert!(i < self.nr_rows);
        if factor == 1f64 {
            return;
        }

        for (_, value) in self.rows[i].iter_mut() {
            *value *= factor;
        }

        for (column, _) in self.rows[i].iter() {
            if let Some((_, value)) = self.columns[*column].iter_mut().find(|(row, _)| *row == i) {
                *value *= factor;
            }
        }
    }
    /// Get the value at coordinate (`i`, `j`).
    fn get_value(&self, i: usize, j: usize) -> f64 {
        debug_assert!(i < self.nr_rows);
        debug_assert!(j < self.nr_columns);

        match self.rows[i].iter().find(|&&(column, _)| j == column) {
            Some(&(_, value)) => value,
            None => 0f64
        }
    }
    /// Set the value at coordinate (`i`, `j`) to `value`.
    fn set_value(&mut self, i: usize, j: usize, value: f64) {
        debug_assert!(i < self.nr_rows);
        debug_assert!(j < self.nr_columns);

        SparseMatrix::set_value_helper(&mut self.rows, i, j, value);
        SparseMatrix::set_value_helper(&mut self.columns, j, i, value);
    }
    /// Get the number of rows in this matrix.
    fn nr_rows(&self) -> usize {
        self.nr_rows
    }
    /// Get the number of columns in this matrix.
    fn nr_columns(&self) -> usize {
        self.nr_columns
    }
    /// Get the number of non-zero values in this matrix.
    fn size(&self) -> usize {
        self.rows.iter()
            .map(|vector| vector.len())
            .sum()
    }
}

impl Clone for SparseMatrix {
    fn clone(&self) -> SparseMatrix {
        SparseMatrix {
            rows: self.rows.clone(),
            columns: self.columns.clone(),
            nr_rows: self.nr_rows,
            nr_columns: self.nr_columns,
        }
    }
}

/// If all column sizes agree, return the dimensions of the vector `data`.
fn get_data_dimensions(data: &Vec<Vec<f64>>) -> (usize, usize) {
    let nr_rows = data.len();
    let nr_columns = data[0].len();

    debug_assert!(nr_rows > 0);
    debug_assert!(nr_columns > 0);
    debug_assert!(data.iter()
        .map(|vector| vector.len())
        .filter(|&length| length != nr_columns)
        .count() == 0, format!("Column lengths not equal: first column has length {}", nr_columns));

    (nr_rows, nr_columns)
}

#[cfg(test)]
mod test {

    use super::*;

    macro_rules! test_matrix {
            ( $matrix_type:ident ) => {
                $matrix_type::from_data(vec![vec![1f64, 2f64, 0f64],
                                             vec![0f64, 5f64, 6f64]])
            }
        }

    macro_rules! create {
            ( $matrix_type:ident ) => {
                from_data!($matrix_type);
                zeros!($matrix_type);
                identity!($matrix_type);
            }
        }

    macro_rules! from_data {
            ( $matrix_type:ident ) => {
                {
                    let m = test_matrix!($matrix_type);

                    assert_approx_eq!(m.get_value(0, 0), 1f64);
                    assert_approx_eq!(m.get_value(1, 2), 6f64);
                }
            }
        }

    macro_rules! zeros {
            ( $matrix_type:ident ) => {
                {
                    let (rows, columns) = (299, 482);
                    let m = $matrix_type::zeros(rows, columns);

                    assert_approx_eq!(m.get_value(0, 0), 0f64);
                    assert_approx_eq!(m.get_value(rows - 1, columns - 1), 0f64);
                }
            }
        }

    macro_rules! identity {
            ( $matrix_type:ident ) => {
                {
                    let size = 133;
                    let m = $matrix_type::identity(size);

                    assert_approx_eq!(m.get_value(0, 0), 1f64);
                    assert_approx_eq!(m.get_value(size - 1, size - 1), 1f64);
                    assert_approx_eq!(m.get_value(0, 1), 0f64);
                    assert_approx_eq!(m.get_value(1, 0), 0f64);
                    assert_approx_eq!(m.get_value(0, size - 1), 0f64);
                    assert_approx_eq!(m.get_value(size - 1, size - 1 - 1), 0f64);
                }
            }
        }

    macro_rules! get_set {
            ( $matrix_type:ident ) => {
                {
                    let mut m = test_matrix!($matrix_type);

                    // Getting a zero value
                    assert_approx_eq!(m.get_value(0, 2), 0f64);

                    // Getting a nonzero value
                    assert_approx_eq!(m.get_value(0, 1), 2f64);

                    // Setting to the same value doesn't change
                    let v = m.get_value(0, 1);
                    m.set_value(0, 1, v);
                    assert_approx_eq!(m.get_value(0, 1), v);

                    // Changing a value
                    let v = 3f64;
                    m.set_value(1, 1, v);
                    assert_approx_eq!(m.get_value(1, 1), v);
                }
            }
        }

    macro_rules! out_of_bounds_get {
            ( $matrix_type:ident ) => {
                {
                    let m = test_matrix!($matrix_type);

                    m.get_value(2, 0);
                }
            }
        }

    macro_rules! out_of_bounds_set {
            ( $matrix_type:ident ) => {
                {
                    let mut m = test_matrix!($matrix_type);

                    m.set_value(2, 0, 4f64);
                }
            }
        }

    macro_rules! multiply_row {
            ( $matrix_type:ident ) => {
                {
                    // Multiply by one
                    let mut m = test_matrix!($matrix_type);
                    let m_copy = m.clone();
                    m.multiply_row(0, 1f64);
                    assert_approx_eq!(m.get_value(0, 1), m_copy.get_value(0, 1));

                    // Multiply by zero
                    let mut m = test_matrix!($matrix_type);
                    m.multiply_row(1, 0f64);
                    assert_approx_eq!(m.get_value(1, 2), 0f64);

                    // Multiply by minus one
                    let mut m = test_matrix!($matrix_type);
                    let m_copy = m.clone();
                    m.multiply_row(0, -1f64);
                    assert_approx_eq!(m.get_value(0, 1), -m_copy.get_value(0, 1));

                    let mut m = test_matrix!($matrix_type);
                    let m_copy = m.clone();
                    let factor = 4.56f64;
                    m.multiply_row(0, factor);
                    assert_approx_eq!(m.get_value(0, 2), factor * m_copy.get_value(0, 2));
                }
            }
        }

    #[cfg(test)]
    mod dense_matrix {

        use super::*;

        #[test]
        fn create() {
            create!(DenseMatrix);
        }

        #[test]
        fn get_set() {
            get_set!(DenseMatrix);
        }

        #[test]
        #[should_panic]
        fn out_of_bounds_get() {
            out_of_bounds_get!(DenseMatrix);
        }

        #[test]
        #[should_panic]
        fn out_of_bounds_set() {
            out_of_bounds_set!(DenseMatrix);
        }

        #[test]
        fn row_column() {
            let m = test_matrix!(DenseMatrix);

            assert_approx_eq!(m.column(2)[0], 0f64);
            assert_approx_eq!(m.column(1).iter().sum::<f64>(), 2f64 + 5f64);

            assert_approx_eq!(m.row(0).nth(0).unwrap(), 1f64);
            assert_approx_eq!(m.row(1).sum::<f64>(), 5f64 + 6f64);
        }

        #[test]
        fn test_mul_add_rows() {
            // On arbitrary matrix
            let mut m = test_matrix!(DenseMatrix);

            let read_row = 0;
            let edit_row = 1;
            let test_column = 1;
            let multiple = -7.43f64;
            let test_value = m.get_value(edit_row, test_column);
            m.mul_add_rows(read_row, edit_row, multiple);

            assert_approx_eq!(m.get_value(edit_row, test_column),
                    test_value + multiple * m.get_value(read_row, test_column));


            // On matrix with a 1f64 value, resulting in a 0f64 on the row being changed
            let mut m = test_matrix!(DenseMatrix);

            let pivot_row = 1;
            let pivot_column = 2;
            let test_row = 0;
            let test_column = 1;
            let multiple = m.get_value(test_row, pivot_column);
            let test_value = m.get_value(test_row, test_column);
            m.set_value(pivot_row, pivot_column, 1f64);
            m.mul_add_rows(pivot_row, test_row, multiple);

            assert_approx_eq!(m.get_value(test_row, pivot_column), 0f64);
            assert_approx_eq!(m.get_value(test_row, test_column),
                    test_value - multiple * m.get_value(pivot_row, test_column));
        }
    }

    #[cfg(test)]
    mod sparse_matrix {

        use super::*;

        #[test]
        fn new() {
            create!(SparseMatrix);
        }

        #[test]
        fn get_set() {
            get_set!(SparseMatrix);
        }

        #[test]
        #[should_panic]
        fn out_of_bounds_get() {
            out_of_bounds_get!(SparseMatrix);
        }

        #[test]
        #[should_panic]
        fn out_of_bounds_set() {
            out_of_bounds_set!(SparseMatrix);
        }

        #[test]
        fn row_column() {
            let m = test_matrix!(SparseMatrix);

            assert_eq!(m.column(2).nth(0).unwrap(), &(1 as usize, 6f64));
            assert_approx_eq!(m.column(1).map(|&(_, value)| value).sum::<f64>(), 2f64 + 5f64);

            assert_eq!(m.row(0).nth(0).unwrap(), &(0 as usize, 1f64));
            assert_approx_eq!(m.row(1).map(|&(_, value)| value).sum::<f64>(), 5f64 + 6f64);
        }

        #[test]
        fn multiply_row() {
            multiply_row!(SparseMatrix);
        }
    }
}
