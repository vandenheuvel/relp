use std::slice::Iter;

use algorithm::simplex::tableau_provider::TableauProvider;
use data::linear_algebra::matrix::{Matrix, SparseMatrix};
use data::linear_algebra::vector::{SparseVector, Vector};

#[derive(Debug, PartialEq, Eq)]
pub struct MatrixData {
    data: SparseMatrix,
    cost: SparseVector,
}

impl MatrixData {
    pub fn new(data: SparseMatrix, cost: SparseVector) -> MatrixData {
        MatrixData { data, cost, }
    }
}

impl TableauProvider for MatrixData {
    fn column(&self, j: usize) -> Iter<'_, (usize, f64)> {
        self.data.column(j)
    }
    fn get_actual_cost_value(&self, j: usize) -> f64 {
        self.cost.get_value(j)
    }
    fn get_artificial_cost_value(&self, j: usize) -> f64 {
        0f64
    }
    fn nr_rows(&self) -> usize {
        // TODO: Check whether this is +2
        self.data.nr_rows()
    }
    fn nr_columns(&self) -> usize {
        self.data.nr_columns()
    }
}