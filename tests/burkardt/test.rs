use std::convert::TryInto;

use num::FromPrimitive;
use num::rational::Ratio;

use rust_lp::algorithm::simplex::logic::OptimizationResult;
use rust_lp::algorithm::simplex::solve_relaxation;
use rust_lp::algorithm::simplex::strategy::pivot_rule::FirstProfitable;
use rust_lp::data::linear_algebra::vector::{SparseVector, Vector};
use rust_lp::data::linear_program::elements::LinearProgramType;
use rust_lp::data::linear_program::general_form::GeneralForm;
use rust_lp::data::number_types::float::FiniteErrorControlledFloat;
use rust_lp::data::number_types::traits::OrderedField;
use rust_lp::io::import;
use rust_lp::R128;

use super::get_test_file_path;
use rust_lp::algorithm::simplex::matrix_provider::MatrixProvider;
use rust_lp::data::linear_program::solution::Solution;

type T = Ratio<i32>;

fn to_general_form(file_name: &str) -> GeneralForm<T> {
    let path = get_test_file_path(file_name);
    let result = import(&path).unwrap();

    result.try_into().ok().unwrap()
}

#[test]
fn maros() {
    let mut general = to_general_form("maros");
    let data = general.derive_matrix_data().ok().unwrap();
    let result = solve_relaxation::<T, _, FirstProfitable, FirstProfitable>(&data);

    match result {
        OptimizationResult::FiniteOptimum(vector) => {
            let reconstructed = data.reconstruct_solution(vector);
            let solution = general.compute_full_solution_with_reduced_solution(reconstructed);
            assert_eq!(solution, Solution::new(
                T::new(385, 3),
                vec![
                    ("VOL1".to_string(), T::new(10, 3)),
                    ("VOL2".to_string(), T::new(40, 3)),
                    ("VOL3".to_string(), T::new(20, 1)),
                    ("VOL4".to_string(), T::new(0, 1)),
                ],
            ));
        },
        _ => assert!(false),
    }
}

#[test]
fn nazareth() {
    let mut general = to_general_form("nazareth");
    let data = general.derive_matrix_data().ok().unwrap();
    let result = solve_relaxation::<T, _, FirstProfitable, FirstProfitable>(&data);
    assert_eq!(result, OptimizationResult::Unbounded);
}

#[test]
fn testprob() {
    let mut general = to_general_form("testprob");
    let data = general.derive_matrix_data().ok().unwrap();
    let result = solve_relaxation::<T, _, FirstProfitable, FirstProfitable>(&data);

    match result {
        OptimizationResult::FiniteOptimum(vector) => {
            let reconstructed = data.reconstruct_solution(vector);
            let solution = general.compute_full_solution_with_reduced_solution(reconstructed);
            assert_eq!(solution, Solution::new(
                T::new(54, 1),
                vec![
                    ("X1".to_string(), T::new(4, 1)),
                    ("X2".to_string(), T::new(-1, 1)),
                    ("X3".to_string(), T::new(6, 1)),
                ],
            ));
        },
        _ => assert!(false),
    }
}
