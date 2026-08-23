//! # Tests for presolving
//!
//! [`findings`] holds one test per defect that this code has had, [`rules`] covers what each rule
//! is supposed to do, and [`driver`] covers how they are scheduled.
use relp_num::R32;
use relp_num::Rational32;

use crate::data::linear_algebra::matrix::{ColumnMajor, MatrixOrder};
use crate::data::linear_algebra::vector::DenseVector;
use crate::data::linear_algebra::vector::test::TestVector;
use crate::data::linear_program::elements::{
    Objective, RangedConstraintRelation, VariableType,
};
use crate::data::linear_program::general_form::{GeneralForm, Variable};

mod driver;
mod findings;
mod rules;

type T = Rational32;

/// A continuous variable.
fn variable(cost: i32, lower: Option<i32>, upper: Option<i32>) -> Variable<T> {
    Variable {
        variable_type: VariableType::Continuous,
        cost: R32!(cost),
        lower_bound: lower.map(|value| R32!(value)),
        upper_bound: upper.map(|value| R32!(value)),
        shift: R32!(0),
        flipped: false,
    }
}

/// An integer variable.
fn integer(cost: i32, lower: Option<i32>, upper: Option<i32>) -> Variable<T> {
    Variable {
        variable_type: VariableType::Integer,
        ..variable(cost, lower, upper)
    }
}

/// `<=`, `==`, `>=` and a range, for brevity at the call sites.
fn less() -> RangedConstraintRelation<T> {
    RangedConstraintRelation::Less
}
fn equal() -> RangedConstraintRelation<T> {
    RangedConstraintRelation::Equal
}
fn greater() -> RangedConstraintRelation<T> {
    RangedConstraintRelation::Greater
}
fn range(width: i32) -> RangedConstraintRelation<T> {
    RangedConstraintRelation::Range(R32!(width))
}

/// Build a problem to presolve.
///
/// `rows` is given row by row, including the zeros.
fn problem(
    objective: Objective,
    rows: &[Vec<i32>],
    relations: Vec<RangedConstraintRelation<T>>,
    right_hand_sides: Vec<i32>,
    variables: Vec<Variable<T>>,
) -> GeneralForm<T> {
    let nr_variables = variables.len();
    let names = (0..nr_variables).map(|j| format!("x{}", j)).collect();

    GeneralForm::new(
        objective,
        ColumnMajor::from_test_data::<T, T, _>(rows, nr_variables),
        relations,
        DenseVector::from_test_data(right_hand_sides),
        variables,
        names,
        R32!(0),
    )
}
