use std::convert::TryInto;

use num::{FromPrimitive, Zero, BigInt, One};
use num::rational::Ratio;

use rust_lp::algorithm::simplex::logic::OptimizationResult;
use rust_lp::algorithm::simplex::solve_relaxation;
use rust_lp::algorithm::simplex::strategy::pivot_rule::FirstProfitable;
use rust_lp::data::linear_algebra::vector::{SparseVector, Vector};
use rust_lp::data::linear_program::elements::LinearProgramType;
use rust_lp::data::linear_program::general_form::GeneralForm;
use rust_lp::data::number_types::float::FiniteErrorControlledFloat;
use rust_lp::data::number_types::traits::{OrderedField, Field};
use rust_lp::io::import;
use rust_lp::R128;

use super::get_test_file_path;
use rust_lp::algorithm::simplex::matrix_provider::MatrixProvider;
use rust_lp::data::linear_program::solution::Solution;

fn to_general_form<T: OrderedField + FromPrimitive>(file_name: &str) -> GeneralForm<T> {
    let path = get_test_file_path(file_name);
    let result = import(&path).unwrap();

    result.try_into().ok().unwrap()
}

#[test]
#[ignore = "Big Integer types are perhaps needed for this one."]
fn adlittle() {
    type T = Ratio<i128>;

    let mut general = to_general_form("adlittle");
    let data = general.derive_matrix_data().ok().unwrap();
    let result = solve_relaxation::<T, _, FirstProfitable, FirstProfitable>(&data);

    match result {
        OptimizationResult::FiniteOptimum(vector) => {
            let reconstructed = data.reconstruct_solution(vector);
            let solution = general.compute_full_solution_with_reduced_solution(reconstructed);
            let expected = Solution::new(
                T::new(999999999999999999, 0),
                vec![
                    ("...100".to_string(), T::new(0, 1)),
                ],
            );

            assert!(expected.is_probably_equal_to(&solution, 0.1_f64));
        },
        _ => assert!(false),
    }
}

#[test]
fn afiro() {
    type T = Ratio<i128>;

    let mut general = to_general_form("afiro");
    let data = general.derive_matrix_data().ok().unwrap();
    let result = solve_relaxation::<T, _, FirstProfitable, FirstProfitable>(&data);

    match result {
        OptimizationResult::FiniteOptimum(vector) => {
            let reconstructed = data.reconstruct_solution(vector);
            let solution = general.compute_full_solution_with_reduced_solution(reconstructed);
            let expected = Solution::new(
                T::new(-406659, 875),
                vec![
                    ("X01".to_string(), T::new(80, 1)),
                    ("X02".to_string(), T::new(51, 2)),
                    ("X03".to_string(), T::new(109, 2)),
                    ("X04".to_string(), T::new(424, 5)),
                    ("X06".to_string(), T::new(255, 14)),
                    ("X07".to_string(), T::zero()),
                    ("X08".to_string(), T::zero()),
                    ("X09".to_string(), T::zero()),
                    ("X10".to_string(), T::zero()),
                    ("X11".to_string(), T::zero()),
                    ("X12".to_string(), T::zero()),
                    ("X13".to_string(), T::zero()),
                    ("X14".to_string(), T::new(255, 14)),
                    ("X15".to_string(), T::zero()),
                    ("X16".to_string(), T::new(9999999999999999, 1)),
                    ("X22".to_string(), T::new(500, 1)),
                    ("X23".to_string(), T::new(11898, 25)),
                    ("X24".to_string(), T::new(602, 25)),
                    ("X25".to_string(), T::zero()),
                    ("X26".to_string(), T::new(215, 1)),
                    ("X28".to_string(), T::zero()),
                    ("X29".to_string(), T::zero()),
                    ("X30".to_string(), T::zero()),
                    ("X31".to_string(), T::zero()),
                    ("X32".to_string(), T::zero()),
                    ("X33".to_string(), T::zero()),
                    ("X34".to_string(), T::zero()),
                    ("X35".to_string(), T::zero()),
                    ("X36".to_string(), T::new(11898, 35)),
                    ("X37".to_string(), T::new(11898, 35)),
                    ("X38".to_string(), T::zero()),
                    ("X39".to_string(), T::zero()),
                ],
            );

            assert!(expected.is_probably_equal_to(&solution, 0.1_f64));
        },
        _ => assert!(false),
    }
}

#[test]
#[ignore = "Not yet implemented: The same range value occurring twice for a single row while being equal should be accepted."]
fn empstest() {
    type T = Ratio<i32>;

    let path = get_test_file_path("empstest");
    import::<T>(&path).unwrap();
}

#[test]
fn maros() {
    type T = Ratio<i32>;

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
    type T = Ratio<i32>;

    let mut general = to_general_form("nazareth");
    let data = general.derive_matrix_data().ok().unwrap();
    let result = solve_relaxation::<T, _, FirstProfitable, FirstProfitable>(&data);
    assert_eq!(result, OptimizationResult::Unbounded);
}

#[test]
fn testprob() {
    type T = Ratio<i32>;

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
