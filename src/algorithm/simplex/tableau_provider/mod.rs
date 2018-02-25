pub mod matrix_data;
pub mod network;

use std::slice::Iter;

pub trait TableauProvider {
    fn column(&self, j: usize) -> Iter<'_, (usize, f64)>;
    fn get_actual_cost_value(&self, j: usize) -> f64;
    fn get_artificial_cost_value(&self, j: usize) -> f64;
    fn nr_rows(&self) -> usize;
    fn nr_columns(&self) -> usize;
}