//! # Shortest path problem
use std::fmt::{Display, Formatter, Result as FormatResult};

use crate::algorithm::two_phase::matrix_provider::{Column, MatrixProvider};
use crate::data::linear_algebra::matrix::{ColumnMajor, Sparse as SparseMatrix};
use crate::data::linear_algebra::vector::{Dense as DenseVector, Sparse as SparseVector};
use crate::data::linear_program::elements::BoundDirection;
use crate::data::linear_program::network::representation::ArcIncidenceColumn;
use crate::data::linear_program::network::representation::ArcIncidenceMatrix;
use crate::data::number_types::traits::Field;

/// Solving a shortest path problem as a linear program.
#[derive(Debug, Clone, PartialEq)]
pub struct Primal<F> {
    /// For each edge, two values indicating from which value the arc leaves, and where it goes to.
    arc_incidence_matrix: ArcIncidenceMatrix<F>,
    /// Length of the arc.
    cost: DenseVector<F>,
    /// Source node index.
    s: usize,
    /// Sink node index.
    t: usize,

    ZERO: F,
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

        // Remove one redundant row; the one with the negative constant, such that b is nonnegative
        let (arc_incidence_matrix, cost) = ArcIncidenceMatrix::new(
            adjacency_matrix, vec![s],
        );

        Self {
            arc_incidence_matrix,
            cost,
            s,
            t,
            ZERO: F::zero(),
        }
    }

    fn nr_vertices(&self) -> usize {
        self.arc_incidence_matrix.nr_vertices()
    }

    fn nr_edges(&self) -> usize {
        self.arc_incidence_matrix.nr_edges()
    }
}

impl<F: 'static> MatrixProvider for Primal<F>
where
    F: Field,
{
    type Column = ArcIncidenceColumn<F>;

    fn column(&self, j: usize) -> Self::Column {
        debug_assert!(j < self.nr_edges());

        ArcIncidenceColumn(self.arc_incidence_matrix.column(j))
    }

    fn cost_value(&self, j: usize) -> &F {
        debug_assert!(j < self.nr_edges());

        &self.cost[j]
    }

    fn constraint_values(&self) -> DenseVector<F> {
        let mut b = DenseVector::constant(F::zero(), self.nr_rows());
        let t_index = if self.t < self.s { self.t } else { self.t - 1 };
        b[t_index] = F::one();

        b
    }

    #[allow(clippy::used_underscore_binding)]
    fn bound_row_index(&self, _j: usize, _bound_type: BoundDirection) -> Option<usize> {
        debug_assert!(_j < self.nr_edges());

        None
    }

    fn nr_constraints(&self) -> usize {
        // This problem is overdetermined, the last row was removed
        self.nr_vertices() - 1
    }

    fn nr_bounds(&self) -> usize {
        0
    }

    fn nr_columns(&self) -> usize {
        self.nr_edges()
    }

    fn reconstruct_solution<G>(&self, column_values: SparseVector<G, G>) -> SparseVector<G, G> {
        unimplemented!()
    }
}

impl<F: 'static> Display for Primal<F>
where
    F: Field,
{
    fn fmt(&self, f: &mut Formatter) -> FormatResult {
        writeln!(f, "Shortest Path Network")?;
        writeln!(f, "Vertices: {}\tEdges: {}", self.nr_vertices(), self.nr_edges())?;
        writeln!(f, "Source: {}\tSink: {}", self.s, self.t)?;

        let width = 10;
        let counter_width = 5;
        // Column counter
        write!(f, "{0:width$}", "", width = counter_width)?;
        for column_index in 0..self.nr_columns() {
            write!(f, "{0:>width$}", column_index, width = width)?;
        }
        writeln!(f)?;

        // Row counter and row data
        for row in 0..self.nr_rows() {
            write!(f, "{0: <width$}", row, width = counter_width)?;
            for column in 0..self.nr_columns() {
                write!(f, "{0:>width$.5}", self.column(column).index_to_string(row), width = width)?;
            }
            writeln!(f)?;
        }
        write!(f, "")
    }
}

#[cfg(test)]
mod test {
    use crate::algorithm::{OptimizationResult, SolveRelaxation};
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
    use crate::data::linear_algebra::matrix::{ColumnMajor, Order};
    use crate::data::linear_algebra::vector::Sparse as SparseVector;
    use crate::data::linear_algebra::vector::test::TestVector;
    use crate::data::linear_program::network::shortest_path::Primal;
    use crate::data::number_types::rational::Rational32;

    #[test]
    fn test_1() {
        type T = Rational32;
        // TODO(ENHANCEMENT): Decouple these two types
        type S = T;

        // Example from Papadimitriou's Combinatorial Optimization.
        let data = ColumnMajor::from_test_data::<T, T, _>(&vec![
            // Directed; from is top, to is on the right
            //   s  a  b  t
            vec![0, 0, 0, 0], // s
            vec![1, 0, 0, 0], // a
            vec![2, 2, 0, 0], // b
            vec![0, 3, 1, 0], // t
        ], 4);
        let problem = Primal::new(data, 0, 3);
        debug_assert_eq!(
            problem.solve_relaxation::<Carry<S>>(),
            OptimizationResult::FiniteOptimum(SparseVector::from_test_data(
                vec![0, 1, 0, 0, 1]
            )),
        );
    }
}
