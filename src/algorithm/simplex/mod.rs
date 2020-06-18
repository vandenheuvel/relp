//! # The Simplex algorithm
//!
//! This module contains all data structures and logic specific to the simplex algorithm. The
//! algorithm is implemented as described in chapters 2 and 4 of Combinatorial Optimization, a book
//! by Christos H. Papadimitriou and Kenneth Steiglitz.
use crate::algorithm::simplex::logic::{artificial_primal, FeasibilityResult, primal, Rank};
use crate::algorithm::simplex::matrix_provider::MatrixProvider;
use crate::algorithm::simplex::matrix_provider::remove_rows::RemoveRows;
use crate::algorithm::simplex::strategy::pivot_rule::PivotRule;
use crate::algorithm::simplex::tableau::kind::{Artificial, NonArtificial};
use crate::algorithm::simplex::tableau::Tableau;
use crate::data::linear_algebra::traits::{SparseElementZero, SparseElement, SparseComparator};
use crate::data::number_types::traits::{OrderedField, OrderedFieldRef};
use crate::data::linear_algebra::vector::Sparse;

pub mod tableau;
pub mod logic;
pub mod matrix_provider;
pub mod strategy;


/// Solve a linear program relaxation
///
/// First finds a basic feasible solution, and then optimizes it.
pub fn solve_relaxation<OF, OFZ, MP, ArtificialPR, NonArtificialPR> (
    provider: &MP,
) -> OptimizationResult<OF, OFZ>
where
    OF: OrderedField,
    for<'r> &'r OF: OrderedFieldRef<OF>,
    OFZ: SparseElementZero<OF>,
    MP: MatrixProvider<OF, OFZ>,
    ArtificialPR: PivotRule,
    NonArtificialPR: PivotRule,
{
    let mut artificial_tableau = Tableau::<_, _, Artificial<_, _, _>>::new(provider);
    match artificial_primal::<_, _, _, ArtificialPR>(&mut artificial_tableau) {
        FeasibilityResult::Infeasible => OptimizationResult::Infeasible,
        FeasibilityResult::Feasible(rank) => match rank {
            Rank::Deficient(rows_to_remove) if !rows_to_remove.is_empty() => {
                let rows_removed = RemoveRows::new(provider, rows_to_remove);
                let mut non_artificial_tableau = Tableau::<_, _, NonArtificial<_, _, _>>::from_artificial_removing_rows(artificial_tableau, &rows_removed);
                primal::<_, _, _, NonArtificialPR>(&mut non_artificial_tableau)
            },
            _ => {
                let mut non_artificial_tableau = Tableau::<_, _, NonArtificial<_, _, _>>::from_artificial(artificial_tableau);
                primal::<_, _, _, NonArtificialPR>(&mut non_artificial_tableau)
            },
        },
    }
}

/// A linear program is either infeasible, unbounded or has a finite optimum.
///
/// This is determined as the result of an algorithm
#[allow(missing_docs)]
#[derive(Eq, PartialEq, Debug)]
pub enum OptimizationResult<F: SparseElement<F> + SparseComparator, FZ: SparseElementZero<F>> {
    Infeasible,
    FiniteOptimum(Sparse<F, FZ, F>),
    Unbounded,
}

#[cfg(test)]
mod test {
    use num::FromPrimitive;
    use num::rational::Ratio;

    use crate::algorithm::simplex::matrix_provider::matrix_data::{MatrixData, Variable};
    use crate::algorithm::simplex::{solve_relaxation, OptimizationResult};
    use crate::algorithm::simplex::strategy::pivot_rule::FirstProfitable;
    use crate::data::linear_algebra::matrix::{ColumnMajor, Order};
    use crate::data::linear_algebra::vector::{Dense, Sparse as SparseVector};
    use crate::data::linear_algebra::vector::test::TestVector;
    use crate::data::linear_program::elements::VariableType;
    use crate::R32;

    type T = Ratio<i32>;

    #[test]
    fn solve_relaxation_1() {
        let constraints = ColumnMajor::from_test_data(&vec![
            vec![1, 0],
            vec![1, 1],
        ], 2);
        let b = Dense::from_test_data(vec![
            3f64 / 2f64,
            5f64 / 2f64,
        ]);
        let variables = vec![
            Variable {
                cost: R32!(-2),
                upper_bound: None,
                variable_type: VariableType::Integer,
            },
            Variable {
                cost: R32!(-1),
                upper_bound: None,
                variable_type: VariableType::Integer,
            },
        ];

        let data = MatrixData::new(&constraints, &b, 0, 2, 0, variables);

        let result = solve_relaxation::<T, T, _, FirstProfitable, FirstProfitable>(&data);
        assert_eq!(result, OptimizationResult::FiniteOptimum(SparseVector::from_test_tuples(vec![
            (0, 3f64 / 2f64),
            (1, 1f64),
        ], 4)));
    }
}
