//! # Maximum Flow Problem
#![feature(impl_trait_in_assoc_type)]

use std::iter;
use std::iter::once;
use std::ops::Range;

use index_utils::remove_sparse_indices;
use num_traits::Zero;
use relp_num::NonZero;
use relp_num::fixed::One;
use relp_num::{Rational64, RationalBig};

use relp::algorithm::two_phase::matrix_provider::MatrixProvider;
use relp::algorithm::two_phase::matrix_provider::column::identity::Identity;
use relp::algorithm::two_phase::matrix_provider::column::{
    Column as ColumnTrait, SparseSliceIterator,
};
use relp::algorithm::two_phase::matrix_provider::filter::generic_wrapper::IntoFilteredColumn;
use relp::algorithm::two_phase::phase_one::PartialInitialBasis;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::basis_inverse_rows::BasisInverseRows;
use relp::algorithm::{OptimizationResult, SolveRelaxation};
use relp::data::linear_algebra::SparseTuple;
use relp::data::linear_algebra::matrix::MatrixOrder;
use relp::data::linear_algebra::matrix::{ColumnMajor, SparseMatrix};
use relp::data::linear_algebra::traits::{SparseComparator, SparseElement};
use relp::data::linear_algebra::vector::{DenseVector, SparseVector, Vector};
use relp::data::linear_program::elements::BoundDirection;
use relp::data::linear_program::network::representation::{ArcDirection, IncidenceMatrix};

/// Maximum flow problem.
struct Primal<F> {
    /// For each edge, two values indicating from which value the arc leaves, and where it goes to.
    arc_incidence_matrix: IncidenceMatrix,
    capacity: DenseVector<F>,

    /// Column indices which are arcs coming from the source.
    s_arc_range: Range<usize>,
}

impl<F> Primal<F>
where
    F: SparseElement<F> + SparseComparator,
{
    pub fn new(adjacency_matrix: SparseMatrix<F, F, ColumnMajor>, s: usize, t: usize) -> Self {
        let nr_vertices = adjacency_matrix.nr_columns();
        debug_assert!(s < nr_vertices && t < nr_vertices);

        let arcs_before_s = adjacency_matrix.data[..s].iter().map(Vec::len).sum();
        let arcs_leaving_s = adjacency_matrix.data[s].len();
        let s_arc_range = arcs_before_s..(arcs_before_s + arcs_leaving_s);
        let (arc_incidence_matrix, capacity) = IncidenceMatrix::new(adjacency_matrix, vec![s, t]);

        Self {
            arc_incidence_matrix,
            capacity,

            s_arc_range,
        }
    }

    pub fn nr_vertices(&self) -> usize {
        self.arc_incidence_matrix.nr_vertices()
    }

    pub fn nr_edges(&self) -> usize {
        self.arc_incidence_matrix.nr_edges()
    }
}

#[derive(Debug, Clone)]
struct Column {
    constraint_values: Vec<(usize, ArcDirection)>,
    slack: (usize, ArcDirection),
}

impl ColumnTrait for Column {
    type F = ArcDirection;
    type Iter<'a> = impl Iterator<Item = SparseTuple<&'a Self::F>> + Clone;

    fn iter(&self) -> Self::Iter<'_> {
        SparseSliceIterator::new(&self.constraint_values)
            .chain(iter::once((self.slack.0, &self.slack.1)))
    }

    fn index_to_string(&self, i: usize) -> String {
        let in_constraint_values = self.constraint_values.iter().find(|&&(ii, _)| ii == i);
        if let Some((_, direction)) = in_constraint_values {
            direction.to_string()
        } else if self.slack.0 == i {
            self.slack.1.to_string()
        } else {
            "0".to_string()
        }
    }
}

impl Identity for Column {
    fn identity(i: usize, _len: usize) -> Self {
        Self {
            constraint_values: Vec::with_capacity(0),
            slack: (i, ArcDirection::Incoming),
        }
    }
}

impl IntoFilteredColumn for Column {
    type Filtered = Self;

    fn into_filtered(mut self, to_remove: &[usize]) -> Self::Filtered {
        remove_sparse_indices(&mut self.constraint_values, to_remove);
        // Slack columns are never removed.
        self
    }
}

impl IntoIterator for Column {
    type Item = SparseTuple<ArcDirection>;
    type IntoIter = impl Iterator<Item = Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.constraint_values.into_iter().chain(once(self.slack))
    }
}

impl<F> MatrixProvider for Primal<F>
where
    F: SparseElement<F> + Zero + Eq + NonZero,
{
    type Column = Column;
    /// Only the capacity slacks of the arcs leaving the source have a cost; the absent value is
    /// the zero.
    ///
    /// A maximum flow is a maximum, while this problem is solved as a minimization. The two meet
    /// in the capacity bound of an arc leaving the source: the flow over that arc and the slack of
    /// its bound sum to the capacity, which is a constant, so minimizing the unused capacity of
    /// the source arcs maximizes the flow leaving the source. What remains is a cost row that is
    /// nothing but ones, which is what `One` is for: it stores the coefficient in no space at all.
    type Cost<'a>
        = Option<One>
    where
        Self: 'a;
    type Rhs = F;

    fn column(&self, j: usize) -> Self::Column {
        debug_assert!(j < self.nr_columns());

        if j < self.nr_edges() {
            Column {
                constraint_values: self.arc_incidence_matrix.column(j),
                slack: (self.nr_constraints() + j, ArcDirection::Incoming),
            }
        } else {
            Column {
                constraint_values: Vec::with_capacity(0),
                slack: (
                    self.nr_constraints() + j - self.nr_edges(),
                    ArcDirection::Incoming,
                ),
            }
        }
    }

    fn cost_value(&self, j: usize) -> Self::Cost<'_> {
        debug_assert!(j < self.nr_columns());

        // The columns after the arcs are the slacks of the capacity bounds, in the same order.
        let is_source_arc_slack = j
            .checked_sub(self.nr_edges())
            .is_some_and(|arc| self.s_arc_range.contains(&arc));

        if is_source_arc_slack { Some(One) } else { None }
    }

    fn right_hand_side(&self) -> DenseVector<F> {
        let mut b = DenseVector::constant(F::zero(), self.nr_constraints());
        b.extend_with_values(self.capacity.iter().cloned().collect());
        b
    }

    fn bound_row_index(&self, j: usize, bound_type: BoundDirection) -> Option<usize> {
        debug_assert!(j < self.nr_columns());

        match bound_type {
            BoundDirection::Lower => None,
            BoundDirection::Upper => {
                if j < self.nr_edges() {
                    Some(self.nr_constraints() + j)
                } else {
                    None
                }
            }
        }
    }

    fn nr_constraints(&self) -> usize {
        self.nr_vertices() - 2
    }

    fn nr_variable_bounds(&self) -> usize {
        self.nr_edges()
    }

    fn nr_columns(&self) -> usize {
        // All edges and their slacks
        self.nr_edges() + self.nr_edges()
    }

    fn reconstruct_solution<G>(&self, _column_values: SparseVector<G, G>) -> SparseVector<G, G> {
        unimplemented!()
    }
}

impl<F> PartialInitialBasis for Primal<F>
where
    F: SparseElement<F> + Zero + Eq + NonZero,
{
    fn pivot_element_indices(&self) -> Vec<(usize, usize)> {
        (0..self.nr_edges())
            .map(|j| (j + self.nr_constraints(), self.nr_edges() + j))
            .collect()
    }

    fn nr_initial_elements(&self) -> usize {
        self.nr_edges()
    }
}

fn main() {
    type T = Rational64;
    type S = RationalBig;

    // Example from Papadimitriou's Combinatorial Optimization.
    let data = ColumnMajor::from_test_data::<T, T, _>(
        &[
            // Directed; from is top, to is on the right
            //   s  a  b  t
            vec![0, 0, 0, 0], // s
            vec![2, 0, 0, 0], // a
            vec![1, 1, 0, 0], // b
            vec![0, 1, 2, 0], // t
        ],
        4,
    );
    let problem = Primal::new(data, 0, 3);

    SolveRelaxation::solve_relaxation::<Carry<S, BasisInverseRows<S>>>(&problem);

    assert_eq!(
        problem.solve_relaxation::<Carry<S, BasisInverseRows<S>>>(),
        OptimizationResult::FiniteOptimum(
            [2, 1, 1, 1, 2, 0, 0, 0, 0, 0]
                .iter()
                .map(|&v| RationalBig::from(v))
                .collect()
        ),
    );
}
