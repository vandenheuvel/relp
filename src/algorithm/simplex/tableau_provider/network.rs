use data::linear_algebra::matrix::{Matrix, SparseMatrix};
use algorithm::simplex::tableau_provider::TableauProvider;
use std::slice::Iter;
use data::linear_algebra::vector::{DenseVector, Vector};

#[derive(Debug, Clone, PartialEq)]
pub struct ShortestPathNetwork {
    edges: Vec<[(usize, f64); 2]>,
    cost: Vec<f64>,
    b: DenseVector,
    s: usize,
    t: usize,
    variable_info: Vec<String>,
    nr_vertices: usize,
    nr_edges: usize,
}

impl ShortestPathNetwork {
    pub fn new(adjacency_matrix: SparseMatrix, s: usize, t: usize) -> ShortestPathNetwork {
        debug_assert_eq!(adjacency_matrix.nr_columns(), adjacency_matrix.nr_rows());
        debug_assert!(s < adjacency_matrix.nr_columns());
        debug_assert!(t < adjacency_matrix.nr_columns());

        let values = adjacency_matrix.values();
        let edges = values.iter().map(|&(from, to, _)| {
            let from_coef = if from != t { 1f64 } else { -1f64 };
            let to_coef = if to != t { -1f64 } else { 1f64 };
            if from < to {
                [(from, from_coef), (to, to_coef)]
            } else {
                [(to, to_coef), (from, from_coef)]
            }
        }).collect::<Vec<_>>();
        let cost = values.iter().map(|&(_, _, cost)| cost).collect();
        let nr_vertices = adjacency_matrix.nr_columns();
        let nr_edges = edges.len();
        let mut b = DenseVector::zeros(nr_vertices);
        b.set_value(s, 1f64);
        b.set_value(t, 1f64);

        let variable_info = (0..nr_edges).map(|i| format!("EDGE_NR_{}", i)).collect();

        ShortestPathNetwork { edges, cost, b, s, t, variable_info, nr_vertices, nr_edges, }
    }
}

impl TableauProvider for ShortestPathNetwork {
    fn column(&self, j: usize) -> Iter<'_, (usize, f64)> {
        debug_assert!(j < self.edges.len());

        self.edges[j].iter()
    }
    fn get_actual_cost_value(&self, j: usize) -> f64 {
        debug_assert!(j < self.cost.len());

        self.cost[j]
    }
    fn get_artificial_cost_value(&self, j: usize) -> f64 {
        0f64
    }
    fn get_b(&self) -> &DenseVector {
        &self.b
    }
    fn variable_info(&self) -> &Vec<String> {
        &self.variable_info
    }
    fn nr_rows(&self) -> usize {
        self.nr_vertices
    }
    fn nr_columns(&self) -> usize {
        self.nr_edges
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_new() {
        let data = SparseMatrix::from_data(vec![vec![0f64, 1f64, 0f64, 2f64, 0f64],
                                                vec![0f64, 0f64, 1f64, 1f64, 0f64],
                                                vec![0f64, 0f64, 0f64, 1f64, 1f64],
                                                vec![0f64, 0f64, 0f64, 0f64, 2f64],
                                                vec![0f64, 0f64, 0f64, 0f64, 0f64]]);
        let graph = ShortestPathNetwork::new(data, 0, 4);

        let expected_b = DenseVector::from_data(vec![1f64, 0f64, 0f64, 0f64, 1f64]);
        assert_eq!(graph.get_b(), &expected_b);

        let expected_cost = DenseVector::from_data(vec![1f64, 2f64, 1f64, 1f64, 1f64, 1f64, 2f64]);
        for i in 0..expected_cost.len() {
            assert_eq!(graph.get_actual_cost_value(i), expected_cost.get_value(i));
        }

        assert_eq!(graph.nr_rows(), 5);
        assert_eq!(graph.nr_columns(), 7);

        let expected_columns = vec![vec![(0, 1f64), (1, -1f64)],
                                    vec![(0, 1f64), (3, -1f64)],
                                    vec![(1, 1f64), (2, -1f64)],
                                    vec![(1, 1f64), (3, -1f64)],
                                    vec![(2, 1f64), (3, -1f64)],
                                    vec![(2, 1f64), (4, 1f64)],
                                    vec![(3, 1f64), (4, 1f64)]];
        for i in 0..graph.nr_columns() {
            assert_eq!(graph.column(i).map(|&t| t).collect::<Vec<_>>(), expected_columns[i]);
        }
    }
}