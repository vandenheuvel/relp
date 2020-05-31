use std::collections::HashSet;

use num::FromPrimitive;
use num::rational::Ratio;

use crate::{R32, RF};
use crate::algorithm::simplex::data::{Artificial, CarryMatrix, NonArtificial, Tableau};
use crate::algorithm::simplex::logic::{artificial_primal, FeasibilityResult, OptimizationResult, primal, Rank};
use crate::algorithm::simplex::matrix_provider::matrix_data::{MatrixData, Variable};
use crate::algorithm::simplex::strategy::pivot_rule::FirstProfitable;
use crate::data::linear_algebra::matrix::{ColumnMajor, Order, Sparse};
use crate::data::linear_algebra::vector::{DenseVector, SparseVector};
use crate::data::linear_algebra::vector::test::TestVector;
use crate::data::linear_program::elements::VariableType;
use crate::data::number_types::traits::RealField;

type T = Ratio<i32>;

#[test]
fn test_conversion_pipeline() {
    let (constraints, b) = create_matrix_data_data();
    let matrix_data_form = matrix_data_form(&constraints, &b);

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
    let result = primal::<T, _, FirstProfitable>(&mut tableau_form_computed);
    assert_eq!(result, OptimizationResult::FiniteOptimum(SparseVector::from_test_tuples(vec![
        (1, 0.5f64),
        (3, 2.5f64),
        (4, 1.5f64),
    ], 5)));
}

pub fn create_matrix_data_data() -> (Sparse<T, ColumnMajor>, DenseVector<T>) {
    let constraints = ColumnMajor::from_test_data(
        &vec![
            vec![3, 2, 1, 0, 0],
            vec![5, 1, 1, 1, 0],
            vec![2, 5, 1, 0, 1],
        ],
        5,
    );
    
    let b = DenseVector::from_test_data(vec![
        1,
        3,
        4,
    ]);

    (constraints, b)
}

pub fn matrix_data_form<'a>(
    constraints: &'a Sparse<T, ColumnMajor>,
    b: &'a DenseVector<T>,
) -> MatrixData<'a, T> {
    let variables = vec![
        Variable {
            cost: R32!(1),
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
        Vec::with_capacity(0),
    )
}

pub fn artificial_tableau_form<'a, RF: RealField>(
    data: &'a MatrixData<RF>,
) -> Tableau<'a, RF, Artificial, MatrixData<'a, RF>> {
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

pub fn tableau_form<'a, RF: RealField>(
    data: &'a MatrixData<RF>,
) -> Tableau<'a, RF, NonArtificial, MatrixData<'a, RF>> {
    let carry = {
        let minus_objective = RF!(-9f64 / 2f64);
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

    Tableau::<RF, NonArtificial, MatrixData<RF>>::new_with_basis(data, carry, basis_indices, basis_columns)
}
