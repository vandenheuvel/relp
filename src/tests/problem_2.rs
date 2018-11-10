use std::collections::HashSet;

use num::FromPrimitive;
use num::rational::Ratio;

use crate::{F, R32};
use crate::algorithm::simplex::data::{Artificial, CarryMatrix, NonArtificial, Tableau};
use crate::algorithm::simplex::logic::{artificial_primal, FeasibilityResult, OptimizationResult, primal, Rank};
use crate::algorithm::simplex::matrix_provider::matrix_data::{MatrixData, Variable};
use crate::algorithm::simplex::strategy::pivot_rule::FirstProfitable;
use crate::data::linear_algebra::matrix::{ColumnMajor, Order, Sparse};
use crate::data::linear_algebra::traits::SparseElementZero;
use crate::data::linear_algebra::vector::{Dense, Sparse as SparseVector};
use crate::data::linear_algebra::vector::test::TestVector;
use crate::data::linear_program::elements::VariableType;
use crate::data::number_types::traits::{Field, FieldRef};

type T = Ratio<i32>;

#[test]
fn conversion_pipeline() {
    let (constraints, b) = create_matrix_data_data();
    let matrix_data_form = matrix_data_form(&constraints, &b);

    // Artificial tableau form
    let mut artificial_tableau_form_computed = Tableau::<_, _, Artificial, _>::new(&matrix_data_form);
    assert_eq!(artificial_tableau_form_computed, artificial_tableau_form(&matrix_data_form));

    // Get to a basic feasible solution
    let feasibility_result = artificial_primal::<_, _, _, FirstProfitable>(&mut artificial_tableau_form_computed);
    assert_eq!(feasibility_result, FeasibilityResult::Feasible(Rank::Full));

    // Non-artificial tableau form
    let mut tableau_form_computed = Tableau::<_, _, NonArtificial, _>::from_artificial(artificial_tableau_form_computed);
    assert_eq!(tableau_form_computed, tableau_form(&matrix_data_form));

    // Get to a basic feasible solution
    let result = primal::<T, T, _, FirstProfitable>(&mut tableau_form_computed);
    assert_eq!(result, OptimizationResult::FiniteOptimum(SparseVector::from_test_tuples(vec![
        (1, 0.5f64),
        (3, 2.5f64),
        (4, 1.5f64),
    ], 5)));
}

pub fn create_matrix_data_data() -> (Sparse<T, T, T, ColumnMajor>, Dense<T>) {
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
    constraints: &'a Sparse<T, T, T, ColumnMajor>,
    b: &'a Dense<T>,
) -> MatrixData<'a, T, T> {
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
    )
}

pub fn artificial_tableau_form<'a, F, FZ>(
    data: &'a MatrixData<'a, F, FZ>,
) -> Tableau<'a, F, FZ, Artificial, MatrixData<'a, F, FZ>>
where
    F: Field + FromPrimitive,
    for<'r> &'r F: FieldRef<F>,
    FZ: SparseElementZero<F>,
{
    let m = 3;
    let carry = {
        let minus_objective = F!(-8);
        let minus_pi = Dense::from_test_data(vec![-1, -1, -1]);
        let b = Dense::from_test_data(vec![1, 3, 4]);
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

pub fn tableau_form<'a, F, FZ>(
    data: &'a MatrixData<'a, F, FZ>
) -> Tableau<'a, F, FZ, NonArtificial, MatrixData<'a, F, FZ>>
    where
        F: Field + FromPrimitive + 'a,
        for<'r> &'r F: FieldRef<F>,
        FZ: SparseElementZero<F>,
{
    let carry = {
        let minus_objective = F!(-9f64 / 2f64);
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

    Tableau::<_, _, NonArtificial, _>::new_with_basis(data, carry, basis_indices, basis_columns)
}
