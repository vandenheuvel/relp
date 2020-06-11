use std::convert::TryInto;

use num::{BigInt, FromPrimitive, ToPrimitive};
use num::rational::Ratio;

use rust_lp::algorithm::simplex::matrix_provider::MatrixProvider;
use rust_lp::algorithm::simplex::{solve_relaxation, OptimizationResult};
use rust_lp::algorithm::simplex::strategy::pivot_rule::FirstProfitable;
use rust_lp::data::linear_program::elements::LinearProgramType;
use rust_lp::data::linear_program::general_form::GeneralForm;
use rust_lp::data::number_types::traits::OrderedField;
use rust_lp::io::import;

use super::get_test_file_path;

/// Testing a problem, comparing only the objective value.
fn test(file_name: &str, objective: f64) {
    type T = Ratio<BigInt>;

    let path = get_test_file_path(file_name);
    let result = import::<T, T>(&path).unwrap();

    let mut general = result.try_into().ok().unwrap();
    let data = general.derive_matrix_data().ok().unwrap();
    let result = solve_relaxation::<T, T, _, FirstProfitable, FirstProfitable>(&data);

    match result {
        OptimizationResult::FiniteOptimum(vector) => {
            let reconstructed = data.reconstruct_solution(vector);
            let solution = general.compute_full_solution_with_reduced_solution(reconstructed);
            assert_eq!(solution.objective_value.floor().to_integer().to_i64().unwrap(), objective.floor() as i64);
        },
        _ => assert!(false),
    }
}

#[test]
#[ignore = "Too computationally intensive"]
fn test_50v() {
    test("50v-10", 2879.065687f64);
}

#[test]
#[ignore = "Too computationally intensive"]
fn test_30n() {
    test("30n20b8", 43.33557298f64);
}

#[test]
#[ignore = "Too computationally intensive"]
fn test_acc() {
    test("acc-tight4", 0f64);
}
