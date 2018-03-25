//! # Matrix implementations
//!
//! The `Matrix` trait defines a set of operations available for all matrix types defined in this
//! module.

use std::iter::Iterator;
use std::slice::Iter;
use data::linear_algebra::EPSILON;
use data::linear_algebra::MAX_DELTA;
use std::fmt::{Debug, Display, Formatter, Result as FormatResult};

/// Defines basic ways to create or change a matrix, regardless of back-end.
pub trait Matrix: Clone + Debug + Eq {
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
#[derive(Clone, Debug)]
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

impl PartialEq for DenseMatrix {
    fn eq(&self, other: &DenseMatrix) -> bool {
        if !(self.nr_columns() == other.nr_columns() && self.nr_rows() == other.nr_rows()) {
            return false;
        }

        for i in 0..self.nr_rows() {
            for j in 0..self.nr_columns() {
                if !((self.get_value(i, j) - other.get_value(i, j)).abs() < MAX_DELTA) {
                    return false;
                }
            }
        }

        true
    }
}

impl Eq for DenseMatrix {}

impl Display for DenseMatrix {
    fn fmt(&self, f: &mut Formatter) -> FormatResult {
        writeln!(f, "Rows: {}\tColumns: {}", self.nr_rows, self.nr_columns)?;

        let column_width = 10;
        let counter_width = 5;
        // Column counter
        write!(f, "{0:width$}", "", width = counter_width)?;
        for column_index in 0..self.nr_columns {
            write!(f, "{0:>width$}", column_index, width = column_width)?;
        }
        writeln!(f, "")?;

        // Row counter and row data
        for (index, row) in self.data.iter().enumerate() {
            write!(f, "{0: <width$}", index, width = counter_width)?;
            for v in row.iter() {
                write!(f, "{0:>width$.5}", v, width = column_width)?;
            }
            writeln!(f, "")?;
        }
        write!(f, "")
    }
}

/// Uses a two indices as underlying data structures: a row-major Vec<Vec<f64>> as well as a
/// column-major Vec<Vec<f64>>. Indices start at `0`.
#[derive(Debug)]
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

        for &(column, value) in self.rows[i].iter() {
            SparseMatrix::set_value_helper(&mut self.columns, column, i, value);
        }

        debug_assert!(self.rows[i].len() <= self.nr_columns);
        debug_assert!(if let Some(maximum) = self.rows[i].iter().map(|(column, _)| *column).max() {
            maximum < self.nr_columns
        } else { true });
    }
    /// Remove row `i`.
    pub fn remove_row(&mut self, i: usize) {
        // TODO: This method is slow, removed rows could possibly be maintained as differences
        // We don't allow empty matrices
        debug_assert!(self.nr_rows() > 1);
        debug_assert!(i < self.nr_rows());

        let mut removed = false;
        for j in 0..self.nr_columns() {
            let mut index = 0;
            while index < self.columns[j].len() {
                if self.columns[j][index].0 == i && !removed {
                    self.columns[j].remove(index);
                    removed = true;
                } else if self.columns[j][index].0 > i {
                    self.columns[j][index].0 -= 1;
                    index += 1;
                } else {
                    index += 1;
                }
            }
        }

        self.rows.remove(i);
        self.nr_rows -= 1;
    }
    /// Change column `j` to the values provided in `new_column`.
    pub fn set_column(&mut self, j: usize, new_column: Iter<'_, (usize, f64)>) {
        debug_assert!(j < self.nr_columns());

        self.columns[j] = new_column.map(|&t| t).collect();

        for &(row, value) in self.columns[j].iter() {
            SparseMatrix::set_value_helper(&mut self.rows, row, j, value);
        }

        debug_assert!(self.columns[j].len() <= self.nr_rows);
        debug_assert!(if let Some(maximum) = self.columns[j].iter().map(|(row, _)| *row).max() {
            maximum < self.nr_rows
        } else { true });
    }
    /// Remove columns `j`.
    pub fn remove_column(&mut self, j: usize) {
        // TODO: This method is slow, removed columns could possibly be maintained as differences
        debug_assert!(self.nr_columns() > 0);
        debug_assert!(j < self.nr_columns());

        for i in 0..self.nr_rows() {
            let mut index = 0;
            let mut removed = false;
            while index < self.rows[i].len() {
                if self.rows[i][index].0 == j && !removed {
                    self.rows[i].remove(index);
                    removed = true;
                } else if self.rows[i][index].0 >= j {
                    self.rows[i][index].0 -= 1;
                    index += 1;
                } else {
                    index += 1;
                }
            }
        }

        self.columns.remove(j);
        self.nr_columns -= 1;
    }
    /// Add a column on the side of high indices
    pub fn push_zero_column(&mut self) {
        self.columns.push(Vec::new());
        self.nr_columns += 1;
    }
    /// Return all values in a (row, column, value) tuple if value is nonzero.
    pub fn values(&self) -> Vec<(usize, usize, f64)> {
        self.rows.iter()
            .enumerate()
            .flat_map(|(row_index, row)| row.iter()
                .map(move |&(column_index, value)| (row_index, column_index, value)))
            .collect()
    }
}

impl Matrix for SparseMatrix {
    /// Create a `SparseMatrix` from the provided data.
    fn from_data(data: Vec<Vec<f64>>) -> SparseMatrix {
        let (nr_rows, nr_columns) = get_data_dimensions(&data);

        let rows: Vec<Vec<(usize, f64)>> = data.iter()
            .map(|v| v.into_iter()
                .enumerate()
                .filter(|&(_, &value)| value.abs() >= EPSILON)
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

                if value.abs() >= EPSILON {
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

impl PartialEq for SparseMatrix {
    fn eq(&self, other: &SparseMatrix) -> bool {
        if !(self.nr_columns() == other.nr_columns() && self.nr_rows() == other.nr_rows()) {
            return false;
        }

        for i in 0..self.nr_rows() {
            for j in 0..self.nr_columns() {
                if !((self.get_value(i, j) - other.get_value(i, j)).abs() < MAX_DELTA) {
                    return false;
                }
            }
        }

        true
    }
}

impl Eq for SparseMatrix {}

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

    fn test_matrix<T>() -> T where T: Matrix {
        T::from_data(vec![vec![1f64, 2f64, 0f64],
                          vec![0f64, 5f64, 6f64]])
    }
    
    fn from_data<T>() where T: Matrix {
        let m = test_matrix::<T>();

        assert_approx_eq!(m.get_value(0, 0), 1f64);
        assert_approx_eq!(m.get_value(1, 2), 6f64);
    }
    
    fn create<T>() where T: Matrix {
        from_data::<T>();
        zeros::<T>();
        identity::<T>();
    }

    fn eq<T>() where T: Matrix {
        let x = 4f64;
        let deviation = 0.5f64 * MAX_DELTA;
        let data = vec![vec![x]];
        let m1 = T::from_data(data);
        let data = vec![vec![x + deviation]];
        let m2 = T::from_data(data);
        assert_eq!(m1, m2);

        let deviation = 3f64 * MAX_DELTA;
        let data = vec![vec![x + deviation]];
        let m2 = T::from_data(data);
        assert_ne!(m1, m2);
    }

    fn zeros<T>() where T: Matrix {
        let (rows, columns) = (299, 482);
        let m = T::zeros(rows, columns);

        assert_approx_eq!(m.get_value(0, 0), 0f64);
        assert_approx_eq!(m.get_value(rows - 1, columns - 1), 0f64);
    }

    fn identity<T>() where T: Matrix {
        let size = 133;
        let m = T::identity(size);

        assert_approx_eq!(m.get_value(0, 0), 1f64);
        assert_approx_eq!(m.get_value(size - 1, size - 1), 1f64);
        assert_approx_eq!(m.get_value(0, 1), 0f64);
        assert_approx_eq!(m.get_value(1, 0), 0f64);
        assert_approx_eq!(m.get_value(0, size - 1), 0f64);
        assert_approx_eq!(m.get_value(size - 1, size - 1 - 1), 0f64);
    }

    fn get_set<T>() where T: Matrix {
        let mut m = test_matrix::<T>();

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

    fn out_of_bounds_get<T>() where T: Matrix {
        let m = test_matrix::<T>();

        m.get_value(2, 0);
    }

    fn out_of_bounds_set<T>() where T: Matrix {
        let mut m = test_matrix::<T>();

        m.set_value(2, 0, 4f64);
    }
    
    fn multiply_row<T>() where T: Matrix {
        // Multiply by one
        let mut m = test_matrix::<T>();
        let m_copy = m.clone();
        m.multiply_row(0, 1f64);
        assert_approx_eq!(m.get_value(0, 1), m_copy.get_value(0, 1));

        // Multiply by zero
        let mut m = test_matrix::<T>();
        m.multiply_row(1, 0f64);
        assert_approx_eq!(m.get_value(1, 2), 0f64);

        // Multiply by minus one
        let mut m = test_matrix::<T>();
        let m_copy = m.clone();
        m.multiply_row(0, -1f64);
        assert_approx_eq!(m.get_value(0, 1), -m_copy.get_value(0, 1));

        let mut m = test_matrix::<T>();
        let m_copy = m.clone();
        let factor = 4.56f64;
        m.multiply_row(0, factor);
        assert_approx_eq!(m.get_value(0, 2), factor * m_copy.get_value(0, 2));
    }

    #[cfg(test)]
    mod dense_matrix {

        use super::*;

        #[test]
        fn test_create() {
            create::<DenseMatrix>();
        }

        #[test]
        fn test_eq() {
            eq::<DenseMatrix>();
        }

        #[test]
        fn test_get_set() {
            get_set::<DenseMatrix>();
        }

        #[test]
        #[should_panic]
        fn test_out_of_bounds_get() {
            out_of_bounds_get::<DenseMatrix>();
        }

        #[test]
        #[should_panic]
        fn test_out_of_bounds_set() {
            out_of_bounds_set::<DenseMatrix>();
        }

        #[test]
        fn test_row_column() {
            let m = test_matrix::<DenseMatrix>();

            assert_approx_eq!(m.column(2)[0], 0f64);
            assert_approx_eq!(m.column(1).iter().sum::<f64>(), 2f64 + 5f64);

            assert_approx_eq!(m.row(0).nth(0).unwrap(), 1f64);
            assert_approx_eq!(m.row(1).sum::<f64>(), 5f64 + 6f64);
        }

        #[test]
        fn test_test_mul_add_rows() {
            // On arbitrary matrix
            let mut m = test_matrix::<DenseMatrix>();

            let read_row = 0;
            let edit_row = 1;
            let test_column = 1;
            let multiple = -7.43f64;
            let test_value = m.get_value(edit_row, test_column);
            m.mul_add_rows(read_row, edit_row, multiple);

            assert_approx_eq!(m.get_value(edit_row, test_column),
                    test_value + multiple * m.get_value(read_row, test_column));


            // On matrix with a 1f64 value, resulting in a 0f64 on the row being changed
            let mut m = test_matrix::<DenseMatrix>();

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
        fn test_new() {
            create::<SparseMatrix>();
        }

        #[test]
        fn test_eq() {
            eq::<SparseMatrix>();
        }

        #[test]
        fn test_get_set() {
            get_set::<SparseMatrix>();
        }

        #[test]
        #[should_panic]
        fn test_out_of_bounds_get() {
            out_of_bounds_get::<SparseMatrix>();
        }

        #[test]
        #[should_panic]
        fn test_out_of_bounds_set() {
            out_of_bounds_set::<SparseMatrix>();
        }

        #[test]
        fn test_row_column() {
            let m = test_matrix::<SparseMatrix>();

            assert_eq!(m.column(2).nth(0).unwrap(), &(1 as usize, 6f64));
            assert_approx_eq!(m.column(1).map(|&(_, value)| value).sum::<f64>(), 2f64 + 5f64);

            assert_eq!(m.row(0).nth(0).unwrap(), &(0 as usize, 1f64));
            assert_approx_eq!(m.row(1).map(|&(_, value)| value).sum::<f64>(), 5f64 + 6f64);
        }

        #[test]
        fn test_multiply_row() {
            multiply_row::<SparseMatrix>();
        }

        #[test]
        fn test_remove_row() {
            fn remove_row(remove_row: usize) {
                let mut data = vec![vec![1f64, 2f64, 3f64, 4f64],
                                    vec![5f64, 6f64, 7f64, 8f64],
                                    vec![9f64, 10f64, 11f64, 12f64]];
                let mut m = SparseMatrix::from_data(data.clone());
                data.remove(remove_row);
                let expected = SparseMatrix::from_data(data);

                m.remove_row(remove_row);
                let result = m;

                assert_eq!(result, expected);
            }

            for i in 0..3 {
                remove_row(i);
            }
        }

        #[test]
        fn test_remove_column() {
            // Remove a middle column
            let data = vec![vec![1f64, 2f64, 3f64, 4f64],
                                vec![5f64, 6f64, 7f64, 8f64],
                                vec![9f64, 10f64, 11f64, 12f64]];
            let mut m = SparseMatrix::from_data(data);
            let data = vec![vec![1f64, 3f64, 4f64],
                                vec![5f64, 7f64, 8f64],
                                vec![9f64, 11f64, 12f64]];
            let expected = SparseMatrix::from_data(data);
            m.remove_column(1);
            let result = m;

            assert_eq!(result, expected);

            // Remove the last column
            let data = vec![vec![1f64, 2f64, 3f64, 4f64],
                            vec![5f64, 6f64, 7f64, 8f64],
                            vec![9f64, 10f64, 11f64, 12f64]];
            let mut m = SparseMatrix::from_data(data);
            let data = vec![vec![2f64, 3f64, 4f64],
                            vec![6f64, 7f64, 8f64],
                            vec![10f64, 11f64, 12f64]];
            let expected = SparseMatrix::from_data(data);
            m.remove_column(0);
            let result = m;

            assert_eq!(result, expected);

            // Remove the first column
            let data = vec![vec![1f64, 2f64, 3f64, 4f64],
                                vec![5f64, 6f64, 7f64, 8f64],
                                vec![9f64, 10f64, 11f64, 12f64]];
            let mut m = SparseMatrix::from_data(data);
            let data = vec![vec![1f64, 2f64, 3f64],
                            vec![5f64, 6f64, 7f64],
                            vec![9f64, 10f64, 11f64]];
            let expected = SparseMatrix::from_data(data);
            m.remove_column(3);
            let result = m;

            assert_eq!(result, expected);
        }
    }
}
