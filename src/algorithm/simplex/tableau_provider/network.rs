use std::fmt::{Display, Formatter, Result as FormatResult};
use std::slice::Iter;

use algorithm::simplex::tableau_provider::TableauProvider;
use data::linear_algebra::matrix::{Matrix, SparseMatrix};
use data::linear_algebra::vector::{DenseVector, Vector};

#[derive(Debug, Clone, PartialEq)]
pub struct ShortestPathNetwork {
    edges: Vec<Vec<(usize, f64)>>,
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
        let nr_vertices = adjacency_matrix.nr_columns();
        let edges = values.iter().map(|&(from, to, _)| {
            let from_coef = if from != t { 1f64 } else { -1f64 };
            let to_coef = if to != t { -1f64 } else { 1f64 };
            let v = if from < to {
                vec![(from, from_coef), (to, to_coef)]
            } else {
                vec![(to, to_coef), (from, from_coef)]
            };
            v.into_iter().filter(|(i, _)| *i < nr_vertices - 1).collect::<Vec<_>>()
        }).collect::<Vec<_>>();
        let cost = values.iter().map(|&(_, _, cost)| cost).collect();
        let nr_edges = edges.len();
        let mut b = DenseVector::zeros(nr_vertices - 1);
        if s < nr_vertices - 1 {
            b.set_value(s, 1f64);
        }
        if t < nr_vertices - 1 {
            b.set_value(t, 1f64);
        }

        let variable_info = (0..nr_edges).map(|i| format!("EDGE_NR_{}", i)).collect();

        ShortestPathNetwork { edges, cost, b, s, t, variable_info, nr_vertices, nr_edges, }
    }
    fn as_sparse_matrix(&self) -> SparseMatrix {
        let mut m = SparseMatrix::zeros(self.nr_vertices, self.nr_edges);
        for j in 0..self.nr_edges {
            m.set_column(j, self.column(j));
        }
        m
    }
}

impl TableauProvider for ShortestPathNetwork {
    fn column(&self, j: usize) -> Iter<'_, (usize, f64)> {
        debug_assert!(j < self.nr_edges);

        self.edges[j].iter()
    }
    fn get_actual_cost_value(&self, j: usize) -> f64 {
        debug_assert!(j < self.nr_edges);

        self.cost[j]
    }
    fn get_artificial_cost_value(&self, j: usize) -> f64 {
        debug_assert!(j < self.nr_columns());

        0f64
    }
    fn get_b(&self) -> &DenseVector {
        &self.b
    }
    fn variable_info(&self) -> &Vec<String> {
        &self.variable_info
    }
    fn nr_rows(&self) -> usize {
        // This problem is overdetermined, the last row can removed
        self.nr_vertices - 1
    }
    fn nr_columns(&self) -> usize {
        self.nr_edges
    }
}

impl Display for ShortestPathNetwork {
    fn fmt(&self, f: &mut Formatter) -> FormatResult {
        writeln!(f, "Shortest Path Network")?;
        writeln!(f, "Vertices: {}\tColumns: {}", self.nr_vertices, self.nr_edges)?;

        let column_width = 10;
        let counter_width = 5;
        // Column counter
        write!(f, "{0:width$}", "", width = counter_width)?;
        for column_index in 0..self.nr_edges {
            write!(f, "{0:>width$}", column_index, width = column_width)?;
        }
        writeln!(f, "")?;

        // Row counter and row data
        let m = self.as_sparse_matrix();
        for row_index in 0..self.nr_vertices {
            write!(f, "{0: <width$}", row_index, width = counter_width)?;
            for column_index in 0..self.nr_edges {
                write!(f, "{0:>width$.5}", m.get_value(row_index, column_index), width = column_width)?;
            }
            writeln!(f, "")?;
        }
        write!(f, "")
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

        let expected_b = DenseVector::from_data(vec![1f64, 0f64, 0f64, 0f64]);
        assert_eq!(graph.get_b(), &expected_b);

        let expected_cost = DenseVector::from_data(vec![1f64, 2f64, 1f64, 1f64, 1f64, 1f64, 2f64]);
        for i in 0..expected_cost.len() {
            assert_eq!(graph.get_actual_cost_value(i), expected_cost.get_value(i));
        }

        assert_eq!(graph.nr_rows(), 4);
        assert_eq!(graph.nr_columns(), 7);

        let expected_columns = vec![vec![(0, 1f64), (1, -1f64)],
                                    vec![(0, 1f64), (3, -1f64)],
                                    vec![(1, 1f64), (2, -1f64)],
                                    vec![(1, 1f64), (3, -1f64)],
                                    vec![(2, 1f64), (3, -1f64)],
                                    vec![(2, 1f64)],
                                    vec![(3, 1f64)]];
        for i in 0..graph.nr_columns() {
            assert_eq!(graph.column(i).map(|&t| t).collect::<Vec<_>>(), expected_columns[i]);
        }
    }
}