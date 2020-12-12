//! # Maximum Flow Problem
use std::ops::{Add, Mul, Range};

use crate::algorithm::two_phase::matrix_provider::{matrix_data, MatrixProvider};
use crate::algorithm::two_phase::phase_one::PartialInitialBasis;
use crate::data::linear_algebra::matrix::{ColumnMajor, Sparse as SparseMatrix};
use crate::data::linear_algebra::traits::{SparseComparator, SparseElement};
use crate::data::linear_algebra::vector::{Dense as DenseVector, Dense, Sparse as SparseVector};
use crate::data::linear_program::elements::BoundDirection;
use crate::data::linear_program::network::representation::ArcIncidenceMatrix;
use crate::data::number_types::rational::RationalBig;
use crate::data::number_types::traits::{Field, FieldRef};

/// Maximum flow problem.
///
/// TODO(OPTIMIZATION): Use simpler number types for the matrix.
struct Primal<F> {
    /// For each edge, two values indicating from which value the arc leaves, and where it goes to.
    ///
    /// TODO(OPTIMIZATION): Use a simpler type, like a boolean, to represent to plus and minus one.
    arc_incidence_matrix: ArcIncidenceMatrix<F>,
    capacity: DenseVector<F>,

    s: usize,
    t: usize,
    s_arc_range: Range<usize>,
}

enum Cost {
    Zero,
    MinusOne,
}

impl<F> Primal<F>
where
    F: Field,
{
    pub fn new(
        adjacency_matrix: SparseMatrix<F, F, ColumnMajor>,
        s: usize,
        t: usize,
    ) -> Self {
        let nr_vertices = adjacency_matrix.nr_columns();
        debug_assert!(s < nr_vertices && t < nr_vertices);

        let arcs_before_s = adjacency_matrix.data[..s].iter().map(Vec::len).sum();
        let arcs_leaving_s = adjacency_matrix.data[s].len();
        let s_arc_range = arcs_before_s..(arcs_before_s + arcs_leaving_s);
        let (arc_incidence_matrix, capacity) = ArcIncidenceMatrix::new(adjacency_matrix, vec![s, t]);

        Self {
            arc_incidence_matrix,
            capacity,

            s,
            t,
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

impl<F: 'static> MatrixProvider for Primal<F>
where
    F: Field + SparseElement<F> + SparseComparator,
    for <'r> &'r F: FieldRef<F>,
{
    type Column = matrix_data::Column<F>;
    type Cost<'a> = Cost;

    fn column(&self, j: usize) -> Self::Column {
        debug_assert!(j < self.nr_columns());

        if j < self.nr_edges() {
            Self::Column::Sparse {
                constraint_values: self.arc_incidence_matrix.column(j),
                // TODO(ENHANCEMENT): Avoid this `F::one()` constant.
                slack: Some((self.nr_constraints() + j, F::one())),
            }
        } else {
            Self::Column::Slack((self.nr_constraints() + j - self.nr_edges(), F::one()), [])
        }
    }

    fn cost_value(&self, j: usize) -> Self::Cost<'_> {
        debug_assert!(j < self.nr_columns());

        if self.s_arc_range.contains(&j) {
            Cost::MinusOne
        } else {
            Cost::Zero
        }
    }

    fn constraint_values(&self) -> Dense<F> {
        let mut b = DenseVector::constant(F::zero(), self.nr_constraints());
        b.extend_with_values(self.capacity.data.clone());
        b
    }

    fn bound_row_index(&self, j: usize, bound_type: BoundDirection) -> Option<usize> {
        debug_assert!(j < self.nr_columns());

        match bound_type {
            BoundDirection::Lower => None,
            BoundDirection::Upper => if j < self.nr_edges() {
                Some(self.nr_constraints() + j)
            } else {
                None
            },
        }
    }

    fn nr_constraints(&self) -> usize {
        self.nr_vertices() - 2
    }

    fn nr_bounds(&self) -> usize {
        self.nr_edges()
    }

    fn nr_columns(&self) -> usize {
        // All edges and their slacks
        self.nr_edges() + self.nr_edges()
    }

    fn reconstruct_solution<G>(&self, column_values: SparseVector<G, G>) -> SparseVector<G, G> {
        unimplemented!()
    }
}

impl<F: 'static> PartialInitialBasis for Primal<F>
where
    F: Field,
    for<'r> &'r F: FieldRef<F>,
{
    fn pivot_element_indices(&self) -> Vec<(usize, usize)> {
        (0..self.nr_edges()).map(|j| (j + self.nr_constraints(), self.nr_edges() + j)).collect()
    }

    fn nr_initial_elements(&self) -> usize {
        self.nr_edges()
    }
}

impl Add<Cost> for RationalBig {
    type Output = Self;

    fn add(self, rhs: Cost) -> Self::Output {
        match rhs {
            Cost::Zero => self,
            Cost::MinusOne => {
                let (numer, denom): (num::BigInt, num::BigInt) = self.0.into();
                Self(num::BigRational::new(numer - &denom, denom))
            }
        }
    }
}

impl Mul<Cost> for &RationalBig {
    type Output = RationalBig;

    fn mul(self, rhs: Cost) -> Self::Output {
        match rhs {
            Cost::Zero => Self::Output::zero(),
            Cost::MinusOne => -self,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::algorithm::{OptimizationResult, SolveRelaxation};
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
    use crate::data::linear_algebra::matrix::{ColumnMajor, Order};
    use crate::data::linear_algebra::vector::Sparse as SparseVector;
    use crate::data::linear_algebra::vector::test::TestVector;
    use crate::data::linear_program::network::max_flow::Primal;
    use crate::data::number_types::rational::{Rational64, RationalBig};

    #[test]
    fn test_1() {
        type T = Rational64;
        type S = RationalBig;

        // Example from Papadimitriou's Combinatorial Optimization.
        let data = ColumnMajor::from_test_data::<T, T, _>(&vec![
            // Directed; from is top, to is on the right
            //   s  a  b  t
            vec![0, 0, 0, 0], // s
            vec![2, 0, 0, 0], // a
            vec![1, 1, 0, 0], // b
            vec![0, 1, 2, 0], // t
        ], 4);
        let problem = Primal::new(data, 0, 3);
        debug_assert_eq!(
            problem.solve_relaxation::<Carry<S>>(),
            OptimizationResult::FiniteOptimum(SparseVector::from_test_data(
                vec![2, 1, 1, 1, 2, 0, 0, 0, 0, 0]
            )),
        );
    }
}
