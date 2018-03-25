use std::fmt::{Display, Formatter, Result as FormatResult};
use std::slice::Iter;

use algorithm::simplex::tableau_provider::TableauProvider;
use data::linear_algebra::matrix::{Matrix, SparseMatrix};
use data::linear_algebra::vector::{DenseVector, SparseVector, Vector};
use data::linear_program::canonical_form::CanonicalForm;

#[derive(Debug, PartialEq)]
pub struct MatrixData {
    data: SparseMatrix,
    cost: SparseVector,
    b: DenseVector,

    variable_info: Vec<String>,
}

impl MatrixData {
    pub fn new(data: SparseMatrix,
               cost: SparseVector,
               b: DenseVector,
               variable_info: Vec<String>) -> MatrixData {
        MatrixData { data, cost, b, variable_info }
    }
}

impl From<CanonicalForm> for MatrixData {
    fn from(canonical: CanonicalForm) -> Self {
        MatrixData::new(canonical.data(),
                        canonical.cost(),
                        canonical.b(),
                        canonical.variable_info())
    }
}

impl TableauProvider for MatrixData {
    fn column(&self, j: usize) -> Iter<'_, (usize, f64)> {
        self.data.column(j)
    }
    fn get_actual_cost_value(&self, j: usize) -> f64 {
        self.cost.get_value(j)
    }
    fn get_artificial_cost_value(&self, _j: usize) -> f64 {
        0f64
    }
    fn get_b(&self) -> &DenseVector {
        &self.b
    }
    fn variable_info(&self) -> &Vec<String> {
        &self.variable_info
    }
    fn nr_rows(&self) -> usize {
        self.data.nr_rows()
    }
    fn nr_columns(&self) -> usize {
        self.data.nr_columns()
    }
}

impl Display for MatrixData {
    fn fmt(&self, f: &mut Formatter) -> FormatResult {
        writeln!(f, "Matrix Data")?;
        writeln!(f, "Rows: {}\tColumns: {}", self.nr_rows(), self.nr_columns())?;

        let column_width = 10;
        let counter_width = 5;
        // Column counter
        write!(f, "{0:width$}", "", width = counter_width)?;
        for column_index in 0..self.nr_columns() {
            write!(f, "{0:>width$}", column_index, width = column_width)?;
        }
        writeln!(f, "")?;

        // Row counter and row data
        for row_index in 0..self.nr_rows() {
            write!(f, "{0: <width$}", row_index, width = counter_width)?;
            for column_index in 0..self.nr_columns() {
                write!(f, "{0:>width$.5}", self.data.get_value(row_index, column_index), width = column_width)?;
            }
            writeln!(f, "")?;
        }
        write!(f, "")
    }
}