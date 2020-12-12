use std::collections::HashSet;

use num::FromPrimitive;

use crate::algorithm::OptimizationResult;
use crate::algorithm::two_phase::{phase_one, phase_two};
use crate::algorithm::two_phase::matrix_provider::matrix_data::{MatrixData, Variable};
use crate::algorithm::two_phase::phase_one::{Rank, RankedFeasibilityResult};
use crate::algorithm::two_phase::strategy::pivot_rule::FirstProfitable;
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
use crate::algorithm::two_phase::tableau::kind::artificial::partially::Partially;
use crate::algorithm::two_phase::tableau::kind::non_artificial::NonArtificial;
use crate::algorithm::two_phase::tableau::Tableau;
use crate::data::linear_algebra::matrix::{ColumnMajor, Order, Sparse};
use crate::data::linear_algebra::vector::{Dense, Sparse as SparseVector};
use crate::data::linear_algebra::vector::test::TestVector;
use crate::data::linear_program::elements::VariableType;
use crate::data::number_types::rational::{Rational64, RationalBig};
use crate::data::number_types::traits::{Field};
use crate::R64;
use crate::RB;

type T = Rational64;
type S = RationalBig;

#[test]
fn conversion_pipeline() {
    let (constraints, b) = create_matrix_data_data::<T>();
    let matrix_data_form = matrix_data_form(&constraints, &b);

    // Artificial tableau form
    let artificial_tableau_form_computed = Tableau::<_, Partially<_>>::new(&matrix_data_form);
    assert_eq!(artificial_tableau_form_computed, artificial_tableau_form(&matrix_data_form));

    // Get to a basic feasible solution
    let feasibility_result = phase_one::primal::<_, _, MatrixData<T>, FirstProfitable>(artificial_tableau_form_computed);
    let mut tableau_form_computed = match feasibility_result {
        RankedFeasibilityResult::Feasible {
            rank,
            nr_artificial_variables,
            inverse_maintainer,
            basis,
        } => {
            assert_eq!(rank, Rank::Full);
            Tableau::<_, NonArtificial<_>>::from_artificial(
                inverse_maintainer,
                nr_artificial_variables,
                basis,
                &matrix_data_form,
            )
        },
        _ => panic!(),
    };

    // Non-artificial tableau form
    assert_eq!(tableau_form_computed, tableau_form(&matrix_data_form));

    // Get to a basic feasible solution
    let result = phase_two::primal::<_, _, FirstProfitable>(&mut tableau_form_computed);
    assert_eq!(result, OptimizationResult::FiniteOptimum(SparseVector::from_test_tuples(vec![
        (1, 0.5f64),
        (3, 2.5f64),
        (4, 1.5f64),
    ], 5)));
}

pub fn create_matrix_data_data<T: Field + FromPrimitive>() -> (Sparse<T, T, ColumnMajor>, Dense<T>) {
    let constraints = ColumnMajor::from_test_data(
        &vec![
            vec![3, 2, 1, 0, 0],
            vec![5, 1, 1, 1, 0],
            vec![2, 5, 1, 0, 1],
        ],
        5,
    );
    
    let b = Dense::from_test_data(vec![
        1,
        3,
        4,
    ]);

    (constraints, b)
}

pub fn matrix_data_form<'a>(
    constraints: &'a Sparse<T, T, ColumnMajor>,
    b: &'a Dense<T>,
) -> MatrixData<'a, T> {
    let variables = vec![
        Variable {
            cost: R64!(1),
            upper_bound: None,
            variable_type: VariableType::Continuous
        }; 5
    ];

    MatrixData::new(
        &constraints,
        &b,
        3,
        0,
        0,
        variables,
    )
}

pub fn artificial_tableau_form<'a>(
    data: &'a MatrixData<'a, T>,
) -> Tableau<Carry<S>, Partially<MatrixData<'a, T>>> {
    let m = 3;
    let carry = {
        let minus_objective = RB!(-8);
        let minus_pi = Dense::from_test_data(vec![-1; 3]);
        let b = Dense::from_test_data(vec![1, 3, 4]);
        let basis_inverse_rows = (0..m)
            .map(|i| SparseVector::standard_basis_vector(i, m))
            .collect();
        Carry::new(minus_objective, minus_pi, b, basis_inverse_rows)
    };
    let artificials = (0..3).collect::<Vec<_>>();
    let basis_indices = artificials.clone();
    let basis_columns = basis_indices.iter().copied().collect();

    Tableau::<_, Partially<_>>::new_with_basis(
        data,
        carry,
        basis_indices,
        basis_columns,
        artificials,
    )
}

pub fn tableau_form<'a>(
    data: &'a MatrixData<'a, T>,
) -> Tableau<Carry<S>, NonArtificial<MatrixData<'a, T>>> {
    let carry = {
        let minus_objective = RB!(-9f64 / 2f64);
        let minus_pi = Dense::from_test_data(vec![2.5f64, -1f64, -1f64]);
        let b = Dense::from_test_data(vec![
            0.5f64,
            2.5f64,
            1.5f64,
        ]);
        let basis_inverse_rows = vec![
            SparseVector::from_test_data(vec![0.5f64, 0f64, 0f64]),
            SparseVector::from_test_data(vec![-0.5f64, 1f64, 0f64]),
            SparseVector::from_test_data(vec![-2.5f64, 0f64, 1f64]),
        ];

        Carry::new(minus_objective, minus_pi, b, basis_inverse_rows)
    };
    let basis_indices = vec![1, 3, 4];
    let basis_columns = {
        let mut basis_columns = HashSet::new();
        basis_columns.insert(1);
        basis_columns.insert(3);
        basis_columns.insert(4);

        basis_columns
    };

    Tableau::<_, NonArtificial<_>>::new_with_inverse_maintainer(
        data,
        carry,
        basis_indices,
        basis_columns,
    )
}
