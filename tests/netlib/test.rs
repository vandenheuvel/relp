use std::convert::TryInto;

use num::{BigInt, FromPrimitive};
use num::rational::Ratio;

use rust_lp::algorithm::simplex::{OptimizationResult, solve_relaxation};
use rust_lp::algorithm::simplex::matrix_provider::MatrixProvider;
use rust_lp::algorithm::simplex::strategy::pivot_rule::FirstProfitable;
use rust_lp::data::linear_algebra::traits::SparseElementZero;
use rust_lp::data::linear_program::solution::Solution;
use rust_lp::data::number_types::traits::{OrderedField, OrderedFieldRef};
use rust_lp::io::import;
use rust_lp::{R128, BR};

use super::get_test_file_path;

fn solve<T: OrderedField + FromPrimitive, TZ: SparseElementZero<T>>(
    file_name: &str,
) -> Solution<T>
where
    for<'r> &'r T: OrderedFieldRef<T>,
{
    let path = get_test_file_path(file_name);
    let mps = import(&path).unwrap();

    let mut general = mps.try_into().ok().unwrap();
    let data = general.derive_matrix_data().ok().unwrap();
    let result = solve_relaxation::<T, TZ, _, FirstProfitable, FirstProfitable>(&data);

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
    type T = Ratio<i128>;

    let result = solve::<T, T>("25FV47");
    assert_eq!(result.objective_value, R128!(5.5018459e+03)); // Gurobi
}

#[test]
#[ignore = "Too computationally expensive."]
fn test_80BAU3B() {
    type T = Ratio<BigInt>;

    let result = solve::<T, T>("80BAU3B");
    assert_eq!(result.objective_value, BR!(9.872241924e+05)); // Gurobi
}

#[test]
#[ignore = "Too computationally expensive."]
fn test_ADLITTLE() {
    type T = Ratio<BigInt>;

    let result = solve::<T, T>("ADLITTLE");
    assert_eq!(result.objective_value, BR!(2.254949632e+05)); // Gurobi
}
