use relp_num::{R64, RB};
use relp_num::{Rational64, RationalBig};

use crate::algorithm::{OptimizationResult, SolveRelaxation};
use crate::algorithm::two_phase::matrix_provider::matrix_data::MatrixData;
use crate::algorithm::two_phase::phase_two;
use crate::algorithm::two_phase::strategy::pivot_rule::FirstProfitable;
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::basis_inverse_rows::BasisInverseRows;
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::LUDecomposition;
use crate::data::linear_algebra::matrix::{ColumnMajor, MatrixOrder};
use crate::data::linear_algebra::vector::{DenseVector, SparseVector, Vector};
use crate::data::linear_algebra::vector::test::TestVector;
use crate::data::linear_program::elements::VariableType;
use crate::data::linear_program::general_form::Variable;
use crate::tests::problem_2::{create_matrix_data_data, matrix_data_form, tableau_form};

#[test]
fn simplex() {
    type T = Rational64;

    let (constraints, b, variables) = create_matrix_data_data::<T>();
    let matrix_data_form = matrix_data_form(&constraints, &b, &variables);
    let mut tableau = tableau_form(&matrix_data_form);
    let result = phase_two::primal::<_, _, FirstProfitable>(&mut tableau);
    assert!(matches!(result, OptimizationResult::FiniteOptimum(_)));
    assert_eq!(tableau.objective_function_value(), RB!(9, 2));
}

#[test]
fn solve_matrix() {
    type S = RationalBig;

    let (constraints, b, variables) = create_matrix_data_data();
    let matrix_data_form = matrix_data_form(&constraints, &b, &variables);

    let result = SolveRelaxation::solve_relaxation::<Carry<S, LUDecomposition<S>>>(&matrix_data_form);
    //  Optimal value: R64!(4.5)
    assert_eq!(result, OptimizationResult::FiniteOptimum(SparseVector::from_test_tuples(vec![
        (1, 0.5f64),
        (3, 2.5f64),
        (4, 1.5f64),
    ], 5)));
}

#[test]
fn solve_relaxation_1() {
    type T = Rational64;
    type S = RationalBig;

    let constraints = ColumnMajor::from_test_data::<T, _, _>(&[
        vec![1, 0],
        vec![1, 1],
    ], 2);
    let b = DenseVector::from_test_data(vec![
        (3, 2),
        (5, 2),
    ]);
    let variables = vec![
        Variable {
            cost: R64!(-2),
            lower_bound: Some(R64!(0)),
            upper_bound: None,
            shift: R64!(0),
            variable_type: VariableType::Integer,
            flipped: false,
        },
        Variable {
            cost: R64!(-1),
            lower_bound: Some(R64!(0)),
            upper_bound: None,
            shift: R64!(0),
            variable_type: VariableType::Integer,
            flipped: false,
        },
    ];

    let data = MatrixData::new(
        &constraints,
        &b,
        vec![],
        0,
        0,
        2,
        0,
        &variables,
    );

    let result = data.solve_relaxation::<Carry<S, LUDecomposition<S>>>();
    assert_eq!(result, OptimizationResult::FiniteOptimum(SparseVector::from_test_tuples(vec![
        (0, 3f64 / 2f64),
        (1, 1f64),
    ], 4)));
}

#[test]
fn redundant_row() {
    let constraints = ColumnMajor::from_test_data(&[
        vec![1, 1],
        vec![1, 1],
        vec![1, 1],
    ], 2);
    let b = DenseVector::new(vec![RB!(1), RB!(1), RB!(1)], 3);
    let variables = vec![
        Variable {
            variable_type: VariableType::Continuous,
            cost: RB!(-2),
            lower_bound: Some(RB!(0)), // Implicitly assumed
            upper_bound: Some(RB!(3, 4)),
            shift: RB!(0),
            flipped: false,
        },
        Variable {
            variable_type: VariableType::Continuous,
            cost: RB!(-1),
            lower_bound: Some(RB!(0)), // Implicitly assumed
            upper_bound: None,
            shift: RB!(0),
            flipped: false,
        },
    ];

    let matrix = MatrixData::new(
        &constraints,
        &b,
        vec![],
        3, 0, 0, 0,
        &variables
    );
    let result = matrix.solve_relaxation::<Carry<RationalBig, BasisInverseRows<_>>>();
    assert_eq!(result, OptimizationResult::FiniteOptimum(
        [RB!(3, 4), RB!(1, 4), RB!(0)].iter().cloned().collect()
    ));
}

#[test]
fn empty_row_at_eq() {
    let constraints = ColumnMajor::from_test_data(&[
        vec![1, 1],
        vec![0, 0],
    ], 2);
    let b = DenseVector::new(vec![RB!(1), RB!(0)], 2);
    let variables = vec![
        Variable {
            variable_type: VariableType::Continuous,
            cost: RB!(-2),
            lower_bound: Some(RB!(0)), // Implicitly assumed
            upper_bound: Some(RB!(3, 4)),
            shift: RB!(0),
            flipped: false,
        },
        Variable {
            variable_type: VariableType::Continuous,
            cost: RB!(-1),
            lower_bound: Some(RB!(0)), // Implicitly assumed
            upper_bound: None,
            shift: RB!(0),
            flipped: false,
        },
    ];

    let matrix = MatrixData::new(
        &constraints,
        &b,
        vec![],
        2, 0, 0, 0,
        &variables
    );
    let result = matrix.solve_relaxation::<Carry<RationalBig, BasisInverseRows<_>>>();
    assert_eq!(result, OptimizationResult::FiniteOptimum(
        [RB!(3, 4), RB!(1, 4), RB!(0)].iter().cloned().collect()
    ));
}

#[test]
fn empty_row_at_ineq() {
    let constraints = ColumnMajor::from_test_data(&[
        vec![1, 1],
        vec![0, 0],
    ], 2);
    let b = DenseVector::new(vec![RB!(1), RB!(1)], 2);
    let variables = vec![
        Variable {
            variable_type: VariableType::Continuous,
            cost: RB!(-2),
            lower_bound: Some(RB!(0)), // Implicitly assumed
            upper_bound: Some(RB!(3, 4)),
            shift: RB!(0),
            flipped: false,
        },
        Variable {
            variable_type: VariableType::Continuous,
            cost: RB!(-1),
            lower_bound: Some(RB!(0)), // Implicitly assumed
            upper_bound: None,
            shift: RB!(0),
            flipped: false,
        },
    ];

    let matrix = MatrixData::new(
        &constraints,
        &b,
        vec![],
        1, 0, 1, 0,
        &variables
    );
    let result = matrix.solve_relaxation::<Carry<RationalBig, BasisInverseRows<_>>>();
    assert_eq!(result, OptimizationResult::FiniteOptimum(
        [RB!(3, 4), RB!(1, 4), RB!(1), RB!(0)].iter().cloned().collect()
    ));
}
