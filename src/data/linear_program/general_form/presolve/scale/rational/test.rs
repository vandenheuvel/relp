use relp_num::{R16, R8, Rational16, Rational8, RationalBig, RB};

use crate::data::linear_algebra::matrix::{ColumnMajor, MatrixOrder};
use crate::data::linear_algebra::traits::SparseComparator;
use crate::data::linear_algebra::vector::DenseVector;
use crate::data::linear_algebra::vector::test::TestVector;
use crate::data::linear_program::elements::{Objective, RangedConstraintRelation, VariableType};
use crate::data::linear_program::general_form::{GeneralForm, Scalable, Scaling, Variable};
use crate::data::linear_program::general_form::presolve::scale::rational::GeneralFormFactorization;

/// All unique prime numbers, any scaling would at best keep things the same.
#[test]
fn test_scale_nothing() {
    let mut general_form: GeneralForm<Rational8> = GeneralForm::new(
        Objective::Minimize,
        ColumnMajor::from_test_data::<_, _, u8>(&[
            vec![1, 2],
        ], 2),
        vec![
            RangedConstraintRelation::Equal,
        ],
        DenseVector::from_test_data::<u8>(vec![3]),
        vec![
            Variable {
                variable_type: VariableType::Continuous,
                cost: R8!(19),
                lower_bound: Some(R8!(5)),
                upper_bound: Some(R8!(7)),
                shift: R8!(0),
                flipped: false,
            },
            Variable {
                variable_type: VariableType::Continuous,
                cost: R8!(17),
                lower_bound: Some(R8!(11)),
                upper_bound: Some(R8!(13)),
                shift: R8!(0),
                flipped: false,
            },
        ],
        vec!["x".to_string(), "y".to_string()],
        R8!(0),
    );
    let scaling = general_form.scale();
    let expected_scaling = Scaling {
        cost_factor: R8!(1),
        constraint_row_factors: vec![R8!(1)],
        constraint_column_factors: vec![R8!(1), R8!(1)],
    };
    assert_eq!(scaling, expected_scaling);
}

/// Scale only the cost coefficients, because there is a duplicate factor there (only).
#[test]
fn test_scale_cost() {
    let mut general_form: GeneralForm<Rational16> = GeneralForm::new(
        Objective::Minimize,
        ColumnMajor::from_test_data::<_, _, u16>(&[
            vec![1, 2],
        ], 2),
        vec![
            RangedConstraintRelation::Equal,
        ],
        DenseVector::from_test_data::<u16>(vec![3]),
        vec![
            Variable {
                variable_type: VariableType::Continuous,
                cost: R16!(17 * 19),
                lower_bound: Some(R16!(5)),
                upper_bound: Some(R16!(7)),
                shift: R16!(0),
                flipped: false,
            },
            Variable {
                variable_type: VariableType::Continuous,
                cost: R16!(19),
                lower_bound: Some(R16!(11)),
                upper_bound: Some(R16!(13)),
                shift: R16!(0),
                flipped: false,
            },
        ],
        vec!["x".to_string(), "y".to_string()],
        R16!(16),
    );
    let scaling = general_form.scale();
    let expected_scaling = Scaling {
        cost_factor: R16!(1, 19),
        constraint_row_factors: vec![R16!(1)],
        constraint_column_factors: vec![R16!(1), R16!(1)],
    };
    assert_eq!(scaling, expected_scaling);
}

/// Scale only the coefficients of a single constraint as there is a duplicate factor there (only).
#[test]
fn test_scale_constraint() {
    let mut general_form: GeneralForm<Rational16> = GeneralForm::new(
        Objective::Minimize,
        ColumnMajor::from_test_data::<_, _, u16>(&[
            vec![2 * 1, 2],
        ], 2),
        vec![
            RangedConstraintRelation::Equal,
        ],
        DenseVector::from_test_data::<u16>(vec![2 * 3]),
        vec![
            Variable {
                variable_type: VariableType::Continuous,
                cost: R16!(17),
                lower_bound: Some(R16!(5)),
                upper_bound: Some(R16!(7)),
                shift: R16!(0),
                flipped: false,
            },
            Variable {
                variable_type: VariableType::Continuous,
                cost: R16!(19),
                lower_bound: Some(R16!(11)),
                upper_bound: Some(R16!(13)),
                shift: R16!(0),
                flipped: false,
            },
        ],
        vec!["x".to_string(), "y".to_string()],
        R16!(16),
    );
    let scaling = general_form.scale();
    let expected_scaling = Scaling {
        cost_factor: R16!(1),
        constraint_row_factors: vec![R16!(1, 2)],
        constraint_column_factors: vec![R16!(1), R16!(1)],
    };
    assert_eq!(scaling, expected_scaling);
}

/// Scale only the coefficients of a single variable as there is a duplicate factor there (only).
#[test]
fn test_scale_variable() {
    let mut general_form: GeneralForm<Rational16> = GeneralForm::new(
        Objective::Minimize,
        ColumnMajor::from_test_data::<_, _, u16>(&[
            vec![1 * 19, 2],
        ], 2),
        vec![
            RangedConstraintRelation::Equal,
        ],
        DenseVector::from_test_data::<u16>(vec![3]),
        vec![
            Variable {
                variable_type: VariableType::Continuous,
                cost: R16!(17 * 19),
                lower_bound: Some(R16!(5, 19)),
                upper_bound: Some(R16!(7, 19)),
                shift: R16!(0),
                flipped: false,
            },
            Variable {
                variable_type: VariableType::Continuous,
                cost: R16!(19),
                lower_bound: Some(R16!(11)),
                upper_bound: Some(R16!(13)),
                shift: R16!(0),
                flipped: false,
            },
        ],
        vec!["x".to_string(), "y".to_string()],
        R16!(16),
    );
    let scaling = general_form.scale();
    let expected_scaling = Scaling {
        cost_factor: R16!(1),
        constraint_row_factors: vec![R16!(1)],
        constraint_column_factors: vec![R16!(19), R16!(1)],
    };
    assert_eq!(scaling, expected_scaling);
}

/// Don't scale, because the factors in the bound count stronger.
#[test]
fn test_scale_variable_bound_vs_constraint() {
    let mut general_form: GeneralForm<Rational16> = GeneralForm::new(
        Objective::Minimize,
        ColumnMajor::from_test_data::<_, _, u16>(&[
            vec![1 * 19, 2],
        ], 2),
        vec![
            RangedConstraintRelation::Equal,
        ],
        DenseVector::from_test_data::<u16>(vec![3]),
        vec![
            Variable {
                variable_type: VariableType::Continuous,
                cost: R16!(17),
                lower_bound: Some(R16!(5)),
                upper_bound: Some(R16!(7)),
                shift: R16!(0),
                flipped: false,
            },
            Variable {
                variable_type: VariableType::Continuous,
                cost: R16!(19),
                lower_bound: Some(R16!(11)),
                upper_bound: Some(R16!(13)),
                shift: R16!(0),
                flipped: false,
            },
        ],
        vec!["x".to_string(), "y".to_string()],
        R16!(16),
    );
    let scaling = general_form.scale();
    let expected_scaling = Scaling {
        cost_factor: R16!(1),
        constraint_row_factors: vec![R16!(1)],
        constraint_column_factors: vec![R16!(1), R16!(1)],
    };
    assert_eq!(scaling, expected_scaling);
}

#[test]
fn test_scale() {
    let mut general_form: GeneralForm<Rational8> = GeneralForm::new(
        Objective::Minimize,
        ColumnMajor::from_test_data::<_, _, u8>(&[
            vec![11, 2],
            vec![4, 6],
            vec![7, 14],
            vec![0, 11],
        ], 2),
        vec![
            RangedConstraintRelation::Equal,
            RangedConstraintRelation::Less,
            RangedConstraintRelation::Greater,
            RangedConstraintRelation::Equal,
        ],
        DenseVector::from_test_data::<u8>(vec![3, 0, 21, 11]),
        vec![
            Variable {
                variable_type: VariableType::Continuous,
                cost: R8!(4),
                lower_bound: Some(R8!(0)),
                upper_bound: Some(R8!(6)),
                shift: R8!(0),
                flipped: false,
            },
            Variable {
                variable_type: VariableType::Continuous,
                cost: R8!(11),
                lower_bound: Some(R8!(1)),
                upper_bound: Some(R8!(2)),
                shift: R8!(0),
                flipped: false,
            },
        ],
        vec!["x".to_string(), "y".to_string()],
        R8!(16),
    );
    let original = general_form.clone();

    let mut factorization = general_form.factorize();

    let expected = GeneralFormFactorization {
        factors: vec![2, 3, 7, 11],
        b: vec![Some(vec![(3, 1)]), None, Some(vec![(3, 1), (7, 1)]), Some(vec![(11, 1)])],
        c: vec![Some(vec![(2, 2)]), Some(vec![(11, 1)])],
        bounds: vec![(None, Some(vec![(2, 1), (3, 1)])), (Some(vec![]), Some(vec![(2, 1)]))],
        constraints: vec![
            vec![(0, vec![(11, 1)]), (1, vec![(2, 2)]), (2, vec![(7, 1)])],
            vec![(0, vec![(2, 1)]), (1, vec![(2, 1), (3, 1)]), (2, vec![(2, 1), (7, 1)]), (3, vec![(11, 1)])],
        ],
    };
    assert_eq!(factorization, expected);

    let factor = factorization.remove_factor_info();
    let expected_factor = 11;
    assert_eq!(factor, expected_factor);
    let expected_factorization = GeneralFormFactorization {
        factors: vec![2, 3, 7],
        b: vec![Some(vec![(3, 1)]), None, Some(vec![(3, 1), (7, 1)]), Some(vec![])],
        c: vec![Some(vec![(2, 2)]), Some(vec![])],
        bounds: vec![(None, Some(vec![(2, 1), (3, 1)])), (Some(vec![]), Some(vec![(2, 1)]))],
        constraints: vec![
            vec![(0, vec![]), (1, vec![(2, 2)]), (2, vec![(7, 1)])],
            vec![(0, vec![(2, 1)]), (1, vec![(2, 1), (3, 1)]), (2, vec![(2, 1), (7, 1)]), (3, vec![])],
        ],
    };
    assert_eq!(factorization, expected_factorization);

    let scaling = general_form.scale();
    let expected_scaling = Scaling {
        cost_factor: R8!(1),
        constraint_row_factors: vec![R8!(1), R8!(1, 2), R8!(1, 7), R8!(1, 11)],
        constraint_column_factors: vec![R8!(1), R8!(1)],
    };
    assert_eq!(scaling, expected_scaling);
    let expected_general_form = GeneralForm::new(
        Objective::Minimize,
        ColumnMajor::from_test_data::<_, _, u8>(&[
            vec![11, 2],
            vec![2, 3],
            vec![1, 2],
            vec![0, 1],
        ], 2),
        vec![
            RangedConstraintRelation::Equal,
            RangedConstraintRelation::Less,
            RangedConstraintRelation::Greater,
            RangedConstraintRelation::Equal,
        ],
        DenseVector::from_test_data::<u8>(vec![3, 0, 3, 1]),
        vec![
            Variable {
                variable_type: VariableType::Continuous,
                cost: R8!(4),
                lower_bound: Some(R8!(0)),
                upper_bound: Some(R8!(6)),
                shift: R8!(0),
                flipped: false,
            },
            Variable {
                variable_type: VariableType::Continuous,
                cost: R8!(11),
                lower_bound: Some(R8!(1)),
                upper_bound: Some(R8!(2)),
                shift: R8!(0),
                flipped: false,
            },
        ],
        vec!["x".to_string(), "y".to_string()],
        R8!(16),
    );
    assert_eq!(general_form, expected_general_form);

    general_form.scale_back(scaling);
    assert_eq!(general_form, original);
}

#[test]
fn test_solve_single_without_b() {
    let factorization = GeneralFormFactorization::<Rational16> {
        factors: vec![2, 3, 7, 11],
        b: vec![None, None, None, None],
        c: vec![Some(vec![(11, 1)]), Some(vec![(2, 2)])],
        bounds: vec![(None, None), (None, None)],
        constraints: vec![
            vec![(0, vec![]), (1, vec![(2, 2)]), (2, vec![(7, 1)])],
            vec![(0, vec![(2, 1)]), (1, vec![(2, 1), (3, 1)]), (2, vec![(2, 1), (7, 1)]), (3, vec![(11, 1)])],
        ],
    };

    let solution = factorization.solve();
    let expected_solution = vec![
        (11, ((0, vec![0, 0, 0, -1]), vec![0, 0])),
        (7, ((0, vec![0, 0, -1, 0]), vec![0, 0])),
        (3, ((0, vec![0, 0, 0, 0]), vec![0, 0])),
        (2, ((0, vec![0, -1, 0, 1]), vec![0, -1])),
    ];
    assert_eq!(solution, expected_solution);
}

#[test]
fn test_range() {
    let mut general_form: GeneralForm<Rational8> = GeneralForm::new(
        Objective::Minimize,
        ColumnMajor::from_test_data::<_, _, i8>(&[
            vec![7, -7],
            vec![1, 1],
        ], 2),
        vec![
            RangedConstraintRelation::Range(R8!(2 * 7)),
            RangedConstraintRelation::Less,
        ],
        DenseVector::from_test_data::<u8>(vec![7, 1]),
        vec![
            Variable {
                variable_type: VariableType::Continuous,
                cost: R8!(2),
                lower_bound: Some(R8!(0)),
                upper_bound: Some(R8!(1, 2)),
                shift: R8!(0),
                flipped: false,
            },
            Variable {
                variable_type: VariableType::Continuous,
                cost: R8!(1),
                lower_bound: Some(R8!(1, 2)),
                upper_bound: Some(R8!(1)),
                shift: R8!(0),
                flipped: false,
            },
        ],
        vec!["x".to_string(), "y".to_string()],
        R8!(16),
    );
    let original = general_form.clone();
    let scaling = general_form.scale();
    let expected: GeneralForm<Rational8> = GeneralForm::new(
        Objective::Minimize,
        ColumnMajor::from_test_data::<_, _, i8>(&[
            vec![1, -1],
            vec![1, 1],
        ], 2),
        vec![
            RangedConstraintRelation::Range(R8!(2)),
            RangedConstraintRelation::Less,
        ],
        DenseVector::from_test_data::<u8>(vec![1, 1]),
        vec![
            Variable {
                variable_type: VariableType::Continuous,
                cost: R8!(2),
                lower_bound: Some(R8!(0)),
                upper_bound: Some(R8!(1, 2)),
                shift: R8!(0),
                flipped: false,
            },
            Variable {
                variable_type: VariableType::Continuous,
                cost: R8!(1),
                lower_bound: Some(R8!(1, 2)),
                upper_bound: Some(R8!(1)),
                shift: R8!(0),
                flipped: false,
            },
        ],
        vec!["x".to_string(), "y".to_string()],
        R8!(16),
    );
    assert_eq!(general_form, expected);

    general_form.scale_back(scaling);
    assert_eq!(general_form, original);
}

#[test]
fn test_big() {
    let mut general_form: GeneralForm<RationalBig> = GeneralForm::new(
        Objective::Minimize,
        ColumnMajor::from_test_data::<_, _, i8>(&[
            vec![7, -7],
            vec![1, 1],
        ], 2),
        vec![
            RangedConstraintRelation::Range(RB!(2 * 7)),
            RangedConstraintRelation::Less,
        ],
        DenseVector::from_test_data::<u8>(vec![7, 1]),
        vec![
            Variable {
                variable_type: VariableType::Continuous,
                cost: RB!(2),
                lower_bound: Some(RB!(0)),
                upper_bound: Some(RB!(1, 2)),
                shift: RB!(0),
                flipped: false,
            },
            Variable {
                variable_type: VariableType::Continuous,
                cost: RB!(1),
                lower_bound: Some(RB!(1, 2)),
                upper_bound: Some(RB!(1)),
                shift: RB!(0),
                flipped: false,
            },
        ],
        vec!["x".to_string(), "y".to_string()],
        RB!(16),
    );
    let scaling = general_form.scale();
    let expected = Scaling {
        cost_factor: RB!(1),
        constraint_row_factors: vec![RB!(1, 7), RB!(1)],
        constraint_column_factors: vec![RB!(1), RB!(1)],
    };
    assert_eq!(scaling, expected);
}
