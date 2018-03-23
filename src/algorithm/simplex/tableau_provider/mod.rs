pub mod matrix_data;
pub mod network;

use std::slice::Iter;
use data::linear_algebra::vector::DenseVector;
use data::linear_algebra::vector::{SparseVector, Vector};

pub trait TableauProvider {
    fn column(&self, j: usize) -> Iter<'_, (usize, f64)>;
    fn get_actual_cost_value(&self, j: usize) -> f64;
    fn get_artificial_cost_value(&self, j: usize) -> f64;
    fn get_b(&self) -> &DenseVector;
    fn variable_info(&self) -> &Vec<String>;
    /// Combines the variable names to the values of a basic feasible solution.
    fn human_readable_bfs(&self, bfs: SparseVector) -> Vec<(String, f64)> {
        debug_assert_eq!(bfs.len(), self.variable_info().len());

        self.variable_info().iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), bfs.get_value(i))).collect()
    }
    fn nr_rows(&self) -> usize;
    fn nr_columns(&self) -> usize;
}
