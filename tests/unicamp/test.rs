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
use rust_lp::R32;

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
fn model_data_1() {
    type T = Ratio<i32>;

    let result = solve::<T, T>("model_data_1");
    assert!(result.is_probably_equal_to(&Solution::new(
        R32!(123, 38),  // GLPK
        vec![
            ("COL01".to_string(), R32!(5, 2)),
            ("COL02".to_string(), R32!(0)),
            ("COL03".to_string(), R32!(0)),
            ("COL04".to_string(), R32!(9, 14)),
            ("COL05".to_string(), R32!(1, 2)),
            ("COL06".to_string(), R32!(4)),
            ("COL07".to_string(), R32!(0)),
            ("COL08".to_string(), R32!(5, 19)),
        ],
    ), 0.5));
}

#[test]
#[ignore = "In this implementation, at least one RHS is needed."]
fn model_data_2() {
    type T = Ratio<i32>;

    let result = solve::<T, T>("model_data_2");
    assert_eq!(result, Solution::new(  // GLPK
        R32!(0),
        vec![
            ("DCOL1".to_string(), R32!(0)),
        ],
    ));
}

#[test]
fn model_data_3() {
    type T = Ratio<i32>;

    let result = solve::<T, T>("model_data_3");
    assert_eq!(result, Solution::new(  // GLPK
        R32!(70),
        vec![
            ("SUP1".to_string(), R32!(200, 3)),
            ("SUP2".to_string(), R32!(100, 3)),
            ("SUP3".to_string(), R32!(100)),
        ],
    ));
}

#[test]
fn model_data_4() {
    type T = Ratio<i32>;

    let result = solve::<T, T>("model_data_4");
    assert_eq!(result, Solution::new(  // GLPK
        R32!(7, 1),
        vec![
            ("COL01".to_string(), R32!(1, 1)),
            ("COL02".to_string(), R32!(2, 1)),
            ("COL03".to_string(), R32!(2, 1)),
        ],
    ));
}

#[test]
#[ignore = "This problem type is not supported."]
fn model_data_5() {
    type T = Ratio<i32>;

    let result = solve::<T, T>("model_data_5");
    assert_eq!(result, Solution::new(  // GLPK
        R32!(332, 1),
        vec![
            ("COL01".to_string(), R32!(1, 1)),
            ("COL02".to_string(), R32!(2, 1)),
            ("COL03".to_string(), R32!(2, 1)),
        ],
    ));
}

#[test]
fn model_data_6() {
    type T = Ratio<i32>;

    let result = solve::<T, T>("model_data_6");
    assert!(result.is_probably_equal_to(&Solution::new(  // GLPK
        R32!(28, 1),
        vec![
            ("X00".to_string(), R32!(0)),
            ("X01".to_string(), R32!(1)),
            ("X02".to_string(), R32!(1)),
            ("X03".to_string(), R32!(0)),
            ("X04".to_string(), R32!(0)),
            ("X05".to_string(), R32!(0)),
            ("X06".to_string(), R32!(0)),
            ("X07".to_string(), R32!(0)),
            ("X10".to_string(), R32!(1)),
            ("X11".to_string(), R32!(0)),
            ("X12".to_string(), R32!(0)),
            ("X13".to_string(), R32!(2)),
            ("X14".to_string(), R32!(0)),
            ("X15".to_string(), R32!(0)),
            ("X16".to_string(), R32!(0)),
            ("X17".to_string(), R32!(0)),
            ("X20".to_string(), R32!(1)),
            ("X21".to_string(), R32!(0)),
            ("X22".to_string(), R32!(0)),
            ("X23".to_string(), R32!(3)),
            ("X24".to_string(), R32!(0)),
            ("X25".to_string(), R32!(0)),
            ("X26".to_string(), R32!(0)),
            ("X27".to_string(), R32!(0)),
        ],
    ), 0.5));
}

#[test]
#[ignore = "Identical to model_data_1."]
fn model_data_7() {
    type T = Ratio<i32>;

    let _result = solve::<T, T>("model_data_7");
}

#[test]
#[ignore = "Unsupported modification of to model_data_7."]
fn model_data_8() {
    type T = Ratio<i32>;

    let _result = solve::<T, T>("model_data_8");
}

#[test]
#[ignore = "Unnamed problem files are not supported."]
fn model_data_9() {
    type T = Ratio<i32>;

    let result = solve::<T, T>("model_data_9");
    let expected = Solution::new(  // GLPK
        R32!(-100, 1),
        vec![
            ("C0000001".to_string(), R32!(0)),
            ("C0000002".to_string(), R32!(1)),
            ("C0000003".to_string(), R32!(1)),
            ("C0000004".to_string(), R32!(0)),
        ],
    );
    assert!(result.is_probably_equal_to(&expected, 0.5));
}
