//! # Representation
//!
//! Representing network data.
use std::fmt;

use index_utils::remove_sparse_indices;
use relp_num::fixed::SignedOne;
use relp_num::{Absorb, Big, NonZero};

use crate::algorithm::two_phase::matrix_provider::column::identity::Identity;
use crate::algorithm::two_phase::matrix_provider::column::{Column, SparseSliceIterator};
use crate::algorithm::two_phase::matrix_provider::filter::generic_wrapper::IntoFilteredColumn;
use crate::data::linear_algebra::SparseTuple;
use crate::data::linear_algebra::matrix::{ColumnMajor, MatrixOrder, SparseMatrix};
use crate::data::linear_algebra::traits::{SparseComparator, SparseElement};
use crate::data::linear_algebra::vector::{DenseVector, Vector};

/// An incidence matrix describes a directed graph.
///
/// See <https://en.wikipedia.org/wiki/Incidence_matrix>.
#[derive(Debug, Clone, PartialEq)]
pub struct IncidenceMatrix {
    data: SparseMatrix<ArcDirection, ArcDirection, ColumnMajor>,
    removed: Vec<usize>,
}

impl IncidenceMatrix {
    /// Build an incidence matrix from an adjacency matrix, dropping the given vertices.
    ///
    /// Also returns the values of the arcs in the order of the columns.
    pub fn new<F: SparseElement<F> + SparseComparator>(
        adjacency_matrix: SparseMatrix<F, F, ColumnMajor>,
        mut removed: Vec<usize>,
    ) -> (Self, DenseVector<F>) {
        let nr_vertices = adjacency_matrix.nr_columns();
        debug_assert_eq!(adjacency_matrix.nr_columns(), adjacency_matrix.nr_rows());
        // No self-arcs
        debug_assert!(
            (0..nr_vertices).all(|j| adjacency_matrix.data[j].iter().all(|&(i, _)| i != j))
        );
        removed.sort_unstable();

        let (edges, values): (Vec<_>, Vec<_>) = adjacency_matrix
            .data
            .into_iter()
            .enumerate()
            .flat_map(|(from, outgoing_arcs)| {
                let removed_ref = &removed;

                outgoing_arcs.into_iter().map(move |(to, values)| {
                    let from_deleted = removed_ref.binary_search(&from);
                    let to_deleted = removed_ref.binary_search(&to);

                    let tuples = match (from_deleted, to_deleted) {
                        (Ok(_), Ok(_)) => {
                            // Both deleted
                            vec![]
                        }
                        (Ok(_), Err(to_shift)) => {
                            vec![(to - to_shift, ArcDirection::Incoming)]
                        }
                        (Err(from_shift), Ok(_)) => {
                            vec![(from - from_shift, ArcDirection::Outgoing)]
                        }
                        (Err(from_shift), Err(to_shift)) => {
                            // Both there
                            let from_shifted = from - from_shift;
                            let to_shifted = to - to_shift;

                            // Correct ordering
                            if from_shifted < to_shifted {
                                vec![
                                    (from_shifted, ArcDirection::Outgoing),
                                    (to_shifted, ArcDirection::Incoming),
                                ]
                            } else {
                                vec![
                                    (to_shifted, ArcDirection::Incoming),
                                    (from_shifted, ArcDirection::Outgoing),
                                ]
                            }
                        }
                    };

                    (tuples, values)
                })
            })
            .unzip();

        let nr_edges = edges.len();

        (
            Self {
                data: ColumnMajor::new(edges, nr_vertices - removed.len(), nr_edges),
                removed,
            },
            DenseVector::new(values, nr_edges),
        )
    }

    /// Column of the incidence matrix, describing a single arc.
    pub fn column(&self, j: usize) -> Vec<SparseTuple<ArcDirection>> {
        debug_assert!(j < self.nr_edges());

        // TODO(ENHANCEMENT): Use improved GATs to avoid this clone.
        self.data.iter_column(j).cloned().collect()
    }

    /// Number of vertices in the graph, including removed ones.
    pub fn nr_vertices(&self) -> usize {
        self.data.nr_rows() + self.removed.len()
    }

    /// Number of edges in the graph.
    pub fn nr_edges(&self) -> usize {
        self.data.nr_columns()
    }
}

/// Column of an incidence matrix, describing a single arc.
#[derive(Clone, Debug)]
pub struct ArcIncidenceColumn(pub Vec<SparseTuple<ArcDirection>>);
impl IntoIterator for ArcIncidenceColumn {
    type Item = SparseTuple<ArcDirection>;
    type IntoIter = impl Iterator<Item = Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
impl Column for ArcIncidenceColumn {
    type F = ArcDirection;
    type Iter<'a> = SparseSliceIterator<'a, ArcDirection>;

    fn iter(&self) -> Self::Iter<'_> {
        SparseSliceIterator::new(&self.0)
    }

    fn index_to_string(&self, i: usize) -> String {
        self.0
            .iter()
            .find(|&&(index, _)| index == i)
            .map_or_else(|| "0".to_string(), |(_, v)| v.to_string())
    }
}
impl Identity for ArcIncidenceColumn {
    fn identity(i: usize, _len: usize) -> Self {
        Self(vec![(i, ArcDirection::Incoming)])
    }
}
impl IntoFilteredColumn for ArcIncidenceColumn {
    type Filtered = Self;

    fn into_filtered(mut self, to_remove: &[usize]) -> Self::Filtered {
        remove_sparse_indices(&mut self.0, to_remove);
        self
    }
}

/// Direction of an arc relative to a vertex.
#[derive(Eq, PartialEq, Clone, Debug)]
pub enum ArcDirection {
    /// The arc points towards the vertex, represented by a "1".
    Incoming,
    /// The arc points away from the vertex, represented by a "-1".
    Outgoing,
}

impl ArcDirection {
    /// The value this direction has in the incidence matrix.
    #[inline]
    fn coefficient(&self) -> SignedOne {
        match self {
            ArcDirection::Incoming => SignedOne::PlusOne,
            ArcDirection::Outgoing => SignedOne::MinusOne,
        }
    }
}

impl NonZero for ArcDirection {
    fn is_not_zero(&self) -> bool {
        true
    }
}

impl fmt::Display for ArcDirection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            ArcDirection::Incoming => "1",
            ArcDirection::Outgoing => "-1",
        })
    }
}

/// An arc entering a vertex contributes `1` to that vertex' row, an arc leaving it `-1`.
///
/// That is the whole of the arithmetic an incidence matrix needs: the coefficient is a
/// [`SignedOne`], so the accumulator is never multiplied, only negated or left alone.
///
/// This is written for [`Big`] rather than for every field, because an implementation that saw only
/// an opaque accumulator could do nothing but convert the coefficient and hand it to the general
/// routine, which is the cost this type exists to avoid. Naming the accumulator concretely is what
/// lets each method delegate to the one for [`SignedOne`], which acts on it in place.
impl<const S: usize> Absorb<ArcDirection> for Big<S> {
    #[inline]
    fn from_narrow(narrow: &ArcDirection) -> Self {
        Self::from_narrow(&narrow.coefficient())
    }
    #[inline]
    fn add_narrow(&mut self, narrow: &ArcDirection) {
        self.add_narrow(&narrow.coefficient());
    }
    #[inline]
    fn sub_narrow(&mut self, narrow: &ArcDirection) {
        self.sub_narrow(&narrow.coefficient());
    }
    #[inline]
    fn mul_narrow(&self, narrow: &ArcDirection) -> Self {
        self.mul_narrow(&narrow.coefficient())
    }
    #[inline]
    fn add_mul_narrow(&mut self, narrow: &ArcDirection, other: &Self) {
        self.add_mul_narrow(&narrow.coefficient(), other);
    }
}
