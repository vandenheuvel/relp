//! # The Simplex algorithm
//!
//! This module contains all data structures and logic specific to the simplex algorithm. The
//! algorithm is implemented as described in chapters 2 and 4 of Combinatorial Optimization, a book
//! by Christos H. Papadimitriou and Kenneth Steiglitz.
use crate::algorithm::simplex::data::{Artificial, NonArtificial, RemoveRows, Tableau};
use crate::algorithm::simplex::logic::{artificial_primal, FeasibilityResult, primal, Rank};
use crate::algorithm::simplex::matrix_provider::{MatrixProvider};
use crate::algorithm::simplex::strategy::pivot_rule::PivotRule;
use crate::data::linear_program::elements::LinearProgramType;
use crate::data::number_types::traits::OrderedField;

pub mod data;
pub mod logic;
pub mod matrix_provider;
pub mod strategy;


/// Solve a linear program relaxation
///
/// First finds a basic feasible solution, and then optimizes it.
pub fn solve_relaxation<OF: OrderedField, MP, ArtificialPR, NonArtificialPR> (
    provider: &MP,
) -> LinearProgramType<OF>
    where
        MP: MatrixProvider<OF>,
        ArtificialPR: PivotRule<Artificial>,
        NonArtificialPR: PivotRule<NonArtificial>,
{
    let mut artificial_tableau = Tableau::<OF, Artificial, _>::new(provider);
    match artificial_primal::<_, _, ArtificialPR>(&mut artificial_tableau) {
        FeasibilityResult::Infeasible => return LinearProgramType::Infeasible,
        FeasibilityResult::Feasible(Rank::Deficient(rows_to_remove)) => {
            let rows_removed = RemoveRows::new(provider, rows_to_remove);
            let mut non_artificial_tableau = Tableau::<OF, NonArtificial, MP>::from_artificial_removing_rows(artificial_tableau, &rows_removed);
            primal::<_, _, NonArtificialPR>(&mut non_artificial_tableau)
        },
        FeasibilityResult::Feasible(Rank::Full) => {
            let mut non_artificial_tableau = Tableau::<OF, NonArtificial, MP>::from_artificial(artificial_tableau);
            primal::<_, _, NonArtificialPR>(&mut non_artificial_tableau)
        },
    }.into()
}

#[cfg(test)]
mod test {
    use num::rational::Ratio;
    use num::traits::FromPrimitive;

    use crate::algorithm::simplex::solve_relaxation;
    use crate::algorithm::simplex::matrix_provider::matrix_data::{MatrixData, Variable};
    use crate::algorithm::simplex::strategy::pivot_rule::FirstProfitable;
    use crate::data::linear_algebra::vector::DenseVector;
    use crate::data::linear_algebra::vector::test::TestVector;
    use crate::data::linear_program::elements::{LinearProgramType, VariableType};
    use crate::data::number_types::traits::RealField;
    use crate::R64;
    use crate::RF;

    fn matrix_data_1<RF: RealField>() -> MatrixData<RF> {
        let equal = vec![];
        let upper_bounded = vec![
            vec![(0, RF!(1))],
            vec![(0, RF!(1)), (1, RF!(1))],
        ];
        let lower_bounded = vec![];
        let variables = vec![Variable {
            cost: RF!(-1),
            upper_bound: None,
            variable_type: VariableType::Integer,
        }; 2 ];
        let b = DenseVector::from_test_data(vec![
            3f64 / 2f64,
            5f64 / 2f64,
        ]);

        MatrixData::new(
            equal,
            upper_bounded,
            lower_bounded,
            b,
            variables,
            Vec::with_capacity(0),
        )
    }

    #[test]
    fn test_solve_relaxation_1() {
        let result = solve_relaxation::<Ratio<i64>, _, FirstProfitable, FirstProfitable>(&matrix_data_1());
        assert_eq!(result, LinearProgramType::FiniteOptimum(-R64!(5f64 / 2f64)));
    }
}
