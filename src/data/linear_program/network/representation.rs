//! # Representation
//!
//! Representing network data.
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use index_utils::remove_sparse_indices;
use relp_num::{Field, Negateable, One};
use relp_num::NonZero;
use relp_num::RationalBig;

use crate::algorithm::two_phase::matrix_provider::column::{Column, SparseSliceIterator};
use crate::algorithm::two_phase::matrix_provider::column::identity::Identity;
use crate::algorithm::two_phase::matrix_provider::filter::generic_wrapper::IntoFilteredColumn;
use crate::data::linear_algebra::matrix::{ColumnMajor, MatrixOrder, SparseMatrix};
use crate::data::linear_algebra::SparseTuple;
use crate::data::linear_algebra::traits::{SparseComparator, SparseElement};
use crate::data::linear_algebra::vector::{DenseVector, Vector};

/// An incidence matrix describes a directed graph.
///
/// See https://en.wikipedia.org/wiki/Incidence_matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct IncidenceMatrix {
    data: SparseMatrix<ArcDirection, ArcDirection, ColumnMajor>,
    removed: Vec<usize>,
}

impl IncidenceMatrix {
    pub fn new<F: SparseElement<F> + SparseComparator>(
        adjacency_matrix: SparseMatrix<F, F, ColumnMajor>,
        mut removed: Vec<usize>,
    ) -> (Self, DenseVector<F>) {
        let nr_vertices = adjacency_matrix.nr_columns();
        debug_assert_eq!(adjacency_matrix.nr_columns(), adjacency_matrix.nr_rows());
        // No self-arcs
        debug_assert!((0..nr_vertices).all(|j| adjacency_matrix.data[j].iter().all(|&(i, _)| i != j)));
        removed.sort_unstable();

        let (edges, values): (Vec<_>, Vec<_>) = adjacency_matrix.data.into_iter().enumerate()
            .flat_map(|(from, outgoing_arcs)| {
                let removed_ref = &removed;

                outgoing_arcs.into_iter().map(move |(to, values)| {
                    let from_deleted = removed_ref.binary_search(&from);
                    let to_deleted = removed_ref.binary_search(&to);

                    let tuples = match (from_deleted, to_deleted) {
                        (Ok(_), Ok(_)) => {  // Both deleted
                            vec![]
                        },
                        (Ok(_), Err(to_shift)) => {
                            vec![(to - to_shift, ArcDirection::Incoming)]
                        }
                        (Err(from_shift), Ok(_)) => {
                            vec![(from - from_shift, ArcDirection::Outgoing)]
                        }
                        (Err(from_shift), Err(to_shift)) => {  // Both there
                            let from_shifted = from - from_shift;
                            let to_shifted = to - to_shift;

                            // Correct ordering
                            if from_shifted < to_shifted {
                                vec![(from_shifted, ArcDirection::Outgoing), (to_shifted, ArcDirection::Incoming)]
                            } else {
                                vec![(to_shifted, ArcDirection::Incoming), (from_shifted, ArcDirection::Outgoing)]
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

    pub fn column(&self, j: usize) -> Vec<SparseTuple<ArcDirection>> {
        debug_assert!(j < self.nr_edges());

        // TODO(ENHANCEMENT): Use improved GATs to avoid this clone.
        self.data.iter_column(j).cloned().collect()
    }

    pub fn nr_vertices(&self) -> usize {
        self.data.nr_rows() + self.removed.len()
    }

    pub fn nr_edges(&self) -> usize {
        self.data.nr_columns()
    }
}

#[derive(Clone, Debug)]
pub struct ArcIncidenceColumn(pub Vec<SparseTuple<ArcDirection>>);
impl IntoIterator for ArcIncidenceColumn {
    type Item = SparseTuple<ArcDirection>;
    type IntoIter = impl Iterator<Item=Self::Item>;

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
        self.0.iter()
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

#[derive(Eq, PartialEq, Clone, Debug)]
pub enum ArcDirection {
    Incoming,
    Outgoing,
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

impl From<ArcDirection> for RationalBig {
    #[inline]
    fn from(direction: ArcDirection) -> Self {
        From::from(&direction)
    }
}

impl From<&ArcDirection> for RationalBig {
    #[inline]
    fn from(direction: &ArcDirection) -> Self {
        match direction {
            ArcDirection::Incoming => RationalBig::one(),
            ArcDirection::Outgoing => -RationalBig::one(),
        }
    }
}

impl Add<ArcDirection> for RationalBig {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: ArcDirection) -> Self::Output {
        AddAssign::add_assign(&mut self, rhs);
        self
    }
}

impl Add<&ArcDirection> for RationalBig {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: &ArcDirection) -> Self::Output {
        AddAssign::add_assign(&mut self, rhs);
        self
    }
}

impl AddAssign<ArcDirection> for RationalBig {
    #[inline]
    fn add_assign(&mut self, rhs: ArcDirection) {
        AddAssign::add_assign(self, &rhs);
    }
}

impl AddAssign<&ArcDirection> for RationalBig {
    #[inline]
    fn add_assign(&mut self, rhs: &ArcDirection) {
        match rhs {
            ArcDirection::Incoming => *self += One,
            ArcDirection::Outgoing => *self -= One,
        }
    }
}

impl Sub<ArcDirection> for RationalBig {
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: ArcDirection) -> Self::Output {
        SubAssign::sub_assign(&mut self, rhs);
        self
    }
}

impl Sub<&ArcDirection> for RationalBig {
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: &ArcDirection) -> Self::Output {
        SubAssign::sub_assign(&mut self, rhs);
        self
    }
}

impl SubAssign<ArcDirection> for RationalBig {
    #[inline]
    fn sub_assign(&mut self, rhs: ArcDirection) {
        SubAssign::sub_assign(self, &rhs);
    }
}

impl SubAssign<&ArcDirection> for RationalBig {
    #[inline]
    fn sub_assign(&mut self, rhs: &ArcDirection) {
        match rhs {
            ArcDirection::Incoming => *self -= One,
            ArcDirection::Outgoing => *self += One,
        }
    }
}

impl Mul<ArcDirection> for RationalBig {
    type Output = Self;

    #[inline]
    fn mul(mut self, rhs: ArcDirection) -> Self::Output {
        MulAssign::mul_assign(&mut self, rhs);
        self
    }
}

impl Mul<&ArcDirection> for RationalBig {
    type Output = Self;

    #[inline]
    fn mul(mut self, rhs: &ArcDirection) -> Self::Output {
        MulAssign::mul_assign(&mut self, rhs);
        self
    }
}

impl MulAssign<ArcDirection> for RationalBig {
    #[inline]
    fn mul_assign(&mut self, rhs: ArcDirection) {
        MulAssign::mul_assign(self, &rhs);
    }
}

impl MulAssign<&ArcDirection> for RationalBig {
    #[inline]
    fn mul_assign(&mut self, rhs: &ArcDirection) {
        match rhs {
            ArcDirection::Incoming => {},
            ArcDirection::Outgoing => self.negate(),
        }
    }
}

impl Mul<&ArcDirection> for &RationalBig {
    type Output = RationalBig;

    #[inline]
    fn mul(self, rhs: &ArcDirection) -> Self::Output {
        match rhs {
            ArcDirection::Incoming => self.clone(),
            ArcDirection::Outgoing => -self.clone(),
        }
    }
}

impl Div<ArcDirection> for RationalBig {
    type Output = Self;

    #[inline]
    fn div(mut self, rhs: ArcDirection) -> Self::Output {
        DivAssign::div_assign(&mut self, rhs);
        self
    }
}

impl Div<&ArcDirection> for RationalBig {
    type Output = Self;

    #[inline]
    fn div(mut self, rhs: &ArcDirection) -> Self::Output {
        DivAssign::div_assign(&mut self, rhs);
        self
    }
}

impl DivAssign<ArcDirection> for RationalBig {
    #[inline]
    fn div_assign(&mut self, rhs: ArcDirection) {
        DivAssign::div_assign(self, &rhs);
    }
}

impl DivAssign<&ArcDirection> for RationalBig {
    #[inline]
    fn div_assign(&mut self, rhs: &ArcDirection) {
        match rhs {
            ArcDirection::Incoming => {},
            ArcDirection::Outgoing => self.negate(),
        }
    }
}
