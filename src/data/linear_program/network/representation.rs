//! # Representation
//!
//! Representing network data.
use crate::data::linear_algebra::matrix::{ColumnMajor, Order, Sparse as SparseMatrix};
use crate::data::linear_algebra::SparseTupleVec;
use crate::data::linear_algebra::traits::SparseElementZero;
use crate::data::linear_algebra::vector::{Dense as DenseVector, Sparse as SparseVector, Vector};
use crate::data::number_types::traits::Field;

#[derive(Debug, Clone, PartialEq)]
pub struct ArcIncidenceMatrix<F, FZ> {
    /// TODO(OPTIMIZATION): Use a simpler type, like a boolean, to represent to plus and minus one.
    pub data: SparseMatrix<F, FZ, F, ColumnMajor>,
    removed: Vec<usize>,
}

impl<F, FZ> ArcIncidenceMatrix<F, FZ>
where
    F: Field,
    FZ: SparseElementZero<F>,
{
    pub fn new(adjacency_matrix: SparseMatrix<F, FZ, F, ColumnMajor>, mut removed: Vec<usize>) -> (Self, DenseVector<F>) {
        let nr_vertices = adjacency_matrix.nr_columns();
        debug_assert_eq!(adjacency_matrix.nr_columns(), adjacency_matrix.nr_rows());
        // No self-arcs
        debug_assert!((0..nr_vertices).all(|j| adjacency_matrix.data[j].iter().all(|&(i, _)| i != j)));
        removed.sort();

        let (edges, values): (Vec<_>, Vec<_>) = adjacency_matrix.data.into_iter().enumerate()
            .flat_map(|(from, outgoing_arcs)| {
                let removed_ref = &removed;

                outgoing_arcs.into_iter().map(move |(to, values)| {
                    // Flow is leaving, so negative
                    let from_coefficient = -F::one();
                    // Flow is arriving, so positive
                    let to_coefficient = F::one();

                    let from_deleted = removed_ref.binary_search(&from);
                    let to_deleted = removed_ref.binary_search(&to);

                    let tuples = match (from_deleted, to_deleted) {
                        (Ok(_), Ok(_)) => {  // Both deleted
                            vec![]
                        },
                        (Ok(_), Err(to_shift)) => {
                            vec![(to - to_shift, to_coefficient)]
                        }
                        (Err(from_shift), Ok(_)) => {
                            vec![(from - from_shift, from_coefficient)]
                        }
                        (Err(from_shift), Err(to_shift)) => {  // Both there
                            let from_shifted = from - from_shift;
                            let to_shifted = to - to_shift;

                            // Correct ordering
                            if from_shifted < to_shifted {
                                vec![(from_shifted, from_coefficient), (to_shifted, to_coefficient)]
                            } else {
                                vec![(to_shifted, to_coefficient), (from_shifted, from_coefficient)]
                            }
                        },
                    };

                    (tuples, values)
                })
            }).unzip();

        let nr_edges = edges.len();

        (
            Self {
                data: ColumnMajor::new(edges, nr_vertices - removed.len(), nr_edges),
                removed,
            },
            DenseVector::new(values, nr_edges),
        )
    }

    pub fn column(&self, j: usize) -> SparseTupleVec<&F> {
        debug_assert!(j < self.nr_edges());

        self.data.iter_column(j).map(|(i, v)| (*i, v)).collect()
    }

    pub fn nr_vertices(&self) -> usize {
        self.data.nr_rows() + self.removed.len()
    }

    pub fn nr_edges(&self) -> usize {
        self.data.nr_columns()
    }
}
