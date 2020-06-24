//! # Maximum Flow Problem
use std::ops::Range;

use crate::algorithm::two_phase::matrix_provider::{Column, MatrixProvider};
use crate::algorithm::two_phase::PartialInitialBasis;
use crate::data::linear_algebra::matrix::{ColumnMajor, Sparse as SparseMatrix};
use crate::data::linear_algebra::traits::{SparseComparator, SparseElement, SparseElementZero};
use crate::data::linear_algebra::vector::{Dense as DenseVector, Dense, Sparse as SparseVector, Vector};
use crate::data::linear_program::elements::BoundDirection;
use crate::data::linear_program::network::representation::ArcIncidenceMatrix;
use crate::data::number_types::traits::{Field, FieldRef};

/// Maximum flow problem.
///
/// TODO(OPTIMIZATION): Use simpler number types for the matrix.
struct Primal<F, FZ> {
    /// For each edge, two values indicating from which value the arc leaves, and where it goes to.
    ///
    /// TODO(OPTIMIZATION): Use a simpler type, like a boolean, to represent to plus and minus one.
    arc_incidence_matrix: ArcIncidenceMatrix<F, FZ>,
    capacity: DenseVector<F>,

    s: usize,
    t: usize,
    s_arc_range: Range<usize>,

    ONE: F,
    ZERO: F,
    MINUS_ONE: F,
}

impl<F, FZ> Primal<F, FZ>
where
    F: Field,
    FZ: SparseElementZero<F>,
{
    pub fn new(
        adjacency_matrix: SparseMatrix<F, FZ, F, ColumnMajor>,
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

            ONE: F::one(),
            ZERO: F::zero(),
            MINUS_ONE: -F::one(),
        }
    }

    pub fn nr_vertices(&self) -> usize {
        self.arc_incidence_matrix.nr_vertices()
    }

    pub fn nr_edges(&self) -> usize {
        self.arc_incidence_matrix.nr_edges()
    }
}

impl<F, FZ> MatrixProvider<F, FZ> for Primal<F, FZ>
where
    F: Field + SparseElement<F> + SparseComparator,
    for <'r> &'r F: FieldRef<F>,
    FZ: SparseElementZero<F>,
{
    fn column(&self, j: usize) -> Column<&F, FZ, F> {
        debug_assert!(j < self.nr_columns());

        if j < self.nr_edges() {
            Column::Sparse({
                let mut tuples = self.arc_incidence_matrix.column(j);
                tuples.push((self.nr_constraints() + j, &self.ONE));
                SparseVector::new(tuples, self.nr_rows())
            })
        } else {
            Column::Slack(self.nr_constraints() + j - self.nr_edges(), BoundDirection::Upper)
        }
    }

    fn cost_value(&self, j: usize) -> &F {
        debug_assert!(j < self.nr_columns());

        if self.s_arc_range.contains(&j) {
            &self.MINUS_ONE
        } else {
            &self.ZERO
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

    fn bounds(&self, j: usize) -> (&F, Option<&F>) {
        debug_assert!(j < self.nr_columns());

        (&self.ZERO, Some(&self.capacity[j]))
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

    fn reconstruct_solution<FZ2: SparseElementZero<F>>(
        &self,
        column_values: SparseVector<F, FZ2, F>,
    ) -> SparseVector<F, FZ2, F> {
        unimplemented!()
    }
}

impl<F, FZ> PartialInitialBasis for Primal<F, FZ>
where
    F: Field,
    for<'r> &'r F: FieldRef<F>,
    FZ: SparseElementZero<F>,
{
    fn pivot_element_indices(&self) -> Vec<(usize, usize)> {
        (0..self.nr_edges()).map(|j| (j + self.nr_constraints(), self.nr_edges() + j)).collect()
    }

    fn nr_initial_elements(&self) -> usize {
        self.nr_edges()
    }
}

#[cfg(test)]
mod test {
    use num::rational::Ratio;

    use crate::algorithm::{OptimizationResult, SolveRelaxation};
    use crate::data::linear_algebra::matrix::{ColumnMajor, Order};
    use crate::data::linear_algebra::vector::Sparse as SparseVector;
    use crate::data::linear_algebra::vector::test::TestVector;
    use crate::data::linear_program::network::max_flow::Primal;

    type T = Ratio<i32>;

    #[test]
    fn test_1() {
        // Example from Papadimitriou's Combinatorial Optimization.
        let data = ColumnMajor::from_test_data::<T, T, T, _>(&vec![
            // Directed; from is top, to is on the right
            //   s  a  b  t
            vec![0, 0, 0, 0], // s
            vec![2, 0, 0, 0], // a
            vec![1, 1, 0, 0], // b
            vec![0, 1, 2, 0], // t
        ], 4);
        let problem = Primal::new(data, 0, 3);
        debug_assert_eq!(
            problem.solve_relaxation(),
            OptimizationResult::FiniteOptimum(SparseVector::from_test_data(
                vec![2, 1, 1, 1, 2, 0, 0, 0, 0, 0]
            )),
        );
    }
}
