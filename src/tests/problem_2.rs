use std::collections::HashSet;

use num::FromPrimitive;
use num::rational::Ratio;

use crate::{R32, RF};
use crate::algorithm::simplex::data::{Artificial, CarryMatrix, NonArtificial, Tableau};
use crate::algorithm::simplex::logic::{artificial_primal, FeasibilityResult, OptimizationResult, primal, Rank};
use crate::algorithm::simplex::matrix_provider::matrix_data::{MatrixData, Variable};
use crate::algorithm::simplex::strategy::pivot_rule::FirstProfitable;
use crate::data::linear_algebra::vector::{DenseVector, SparseVector};
use crate::data::linear_algebra::vector::test::TestVector;
use crate::data::linear_program::elements::VariableType;
use crate::data::number_types::traits::RealField;

#[test]
fn test_conversion_pipeline() {
    let matrix_data_form = matrix_data_form();

    // Artificial tableau form
    let mut artificial_tableau_form_computed = Tableau::<_, Artificial, _>::new(&matrix_data_form);
    assert_eq!(artificial_tableau_form_computed, artificial_tableau_form(&matrix_data_form));

    // Get to a basic feasible solution
    let feasibility_result = artificial_primal::<_, _, FirstProfitable>(&mut artificial_tableau_form_computed);
    assert_eq!(feasibility_result, FeasibilityResult::Feasible(Rank::Full));

    // Non-artificial tableau form
    let mut tableau_form_computed = Tableau::<_, NonArtificial, _>::from_artificial(artificial_tableau_form_computed);
    assert_eq!(tableau_form_computed, tableau_form(&matrix_data_form));

    // Get to a basic feasible solution
    let result = primal::<_, _, FirstProfitable>(&mut tableau_form_computed);
    assert_eq!(result, OptimizationResult::FiniteOptimum(R32!(4.5)));
}

pub fn matrix_data_form<RF: RealField>() -> MatrixData<RF> {
    let equal = vec![
        vec![(0, RF!(3)), (1, RF!(2)), (2, RF!(1))],
        vec![(0, RF!(5)), (1, RF!(1)), (2, RF!(1)), (3, RF!(1))],
        vec![(0, RF!(2)), (1, RF!(5)), (2, RF!(1)), (4, RF!(1))],
    ];
    let upper_bounded = vec![];
    let lower_bounded = vec![];
    let b = DenseVector::from_test_data(vec![
        1f64,
        3f64,
        4f64,
    ]);
    let variables = vec![Variable { cost: RF!(1), upper_bound: None, variable_type: VariableType::Continuous }; 5];

    MatrixData::new(
        equal,
        upper_bounded,
        lower_bounded,
        b,
        variables,
        Vec::with_capacity(0),
    )
}

pub fn artificial_tableau_form<RF: RealField>(data: &MatrixData<RF>) -> Tableau<RF, Artificial, MatrixData<RF>> {
    let m = 3;
    let carry = {
        let minus_objective = RF!(-8);
        let minus_pi = DenseVector::from_test_data(vec![-1f64, -1f64, -1f64]);
        let b = DenseVector::from_test_data(vec![1f64, 3f64, 4f64]);
        let basis_inverse_rows = (0..m)
            .map(|i| SparseVector::standard_basis_vector(i, m))
            .collect();
        CarryMatrix::create(minus_objective, minus_pi, b, basis_inverse_rows)
    };
    let basis_indices = (0..m).collect();
    let basis_columns = (0..m).collect();

    Tableau::new_with_basis(
        data,
        carry,
        basis_indices,
        basis_columns,
    )
}

pub fn tableau_form<F: RealField>(data: &MatrixData<F>) -> Tableau<F, NonArtificial, MatrixData<F>> {
    let carry = {
        let minus_objective = F::from_f64(-9f64 / 2f64).unwrap();
        let minus_pi = DenseVector::from_test_data(vec![2.5f64, -1f64, -1f64]);
        let b = DenseVector::from_test_data(vec![
            0.5f64,
            2.5f64,
            1.5f64,
        ]);
        let basis_inverse_rows = vec![
            SparseVector::from_test_data(vec![0.5f64, 0f64, 0f64]),
            SparseVector::from_test_data(vec![-0.5f64, 1f64, 0f64]),
            SparseVector::from_test_data(vec![-2.5f64, 0f64, 1f64]),
        ];

        CarryMatrix::create(minus_objective, minus_pi, b, basis_inverse_rows)
    };
    let basis_indices = vec![1, 3, 4];
    let basis_columns = {
        let mut basis_columns = HashSet::new();
        basis_columns.insert(1);
        basis_columns.insert(3);
        basis_columns.insert(4);

        basis_columns
    };

    Tableau::<F, NonArtificial, MatrixData<F>>::new_with_basis(data, carry, basis_indices, basis_columns)
}
