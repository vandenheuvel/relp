use std::convert::TryInto;

use num::FromPrimitive;

use rust_lp::algorithm::{OptimizationResult, SolveRelaxation};
use rust_lp::algorithm::two_phase::matrix_provider::MatrixProvider;
use rust_lp::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
use rust_lp::data::linear_program::general_form::GeneralForm;
use rust_lp::data::linear_program::solution::Solution;
use rust_lp::data::number_types::rational::{Rational64, RationalBig};
use rust_lp::data::number_types::traits::Abs;
use rust_lp::io::import;
use rust_lp::RB;

use super::get_test_file_path;

type T = RationalBig;

fn solve(file_name: &str) -> Solution<T> {
    let path = get_test_file_path(file_name);
    let mps = import(&path).unwrap();

    let mut general: GeneralForm<Rational64> = mps.try_into().ok().unwrap();
    let data = general.derive_matrix_data().ok().unwrap();
    let result = data.solve_relaxation::<Carry<T>>();

    match result {
        OptimizationResult::FiniteOptimum(vector) => {
            let reconstructed = data.reconstruct_solution(vector);
            general.compute_full_solution_with_reduced_solution(reconstructed)
        },
        _ => panic!(),
    }
}

#[test]
#[ignore = "Has a row name consisting of only numbers, the parser doesn't support that."]
fn test_25FV47() {
    let result = solve("25FV47");
    assert!((result.objective_value - RB!(5.5018459e+03)).abs() < RB!(1e-5)); // Gurobi
}

#[test]
#[ignore = "Too computationally intensive"]
fn test_80BAU3B() {
    let result = solve("80BAU3B");
    assert!((result.objective_value - RB!(9.872241924e+05)).abs() < RB!(1e-5)); // Gurobi
}

#[test]
#[ignore = "Too computationally expensive."]
fn test_ADLITTLE() {
    let result = solve("ADLITTLE");
    assert!((result.objective_value - RB!(2.254949632e+05)).abs() < RB!(1e-5)); // Gurobi
}
