//! # Generation and execution
//!
//! Functions to integration-test are generated using macro's.
use std::convert::TryInto;

use num::FromPrimitive;
use num::rational::Ratio;

use rust_lp::algorithm::simplex::solve_relaxation;
use rust_lp::algorithm::simplex::strategy::pivot_rule::FirstProfitable;
use rust_lp::data::linear_program::elements::LinearProgramType;
use rust_lp::data::linear_program::general_form::GeneralForm;
use rust_lp::data::number_types::float::FiniteErrorControlledFloat;
use rust_lp::data::number_types::traits::OrderedField;
use rust_lp::R128;
use rust_lp::io::import;

use super::get_test_file_path;

fn test(file_name: String, objective: f64, epsilon: f64) {
    type T = Ratio<i128>;

    let name = String::from(file_name);
    let path = get_test_file_path(&name);
    let result = import(&path).unwrap();

    let mut general: GeneralForm<T> = result.try_into().ok().unwrap();
    let data = general.derive_matrix_data().ok().unwrap();

    let result = solve_relaxation::<T, _, FirstProfitable, FirstProfitable>(&data);

    assert!(match result {
        LinearProgramType::FiniteOptimum(v) => (v - R128!(objective)).abs() < R128!(epsilon),
        _ => false,
    })
}

#[test]
fn test_50v() {
    test(String::from("50v-10"), 2879.065687f64, 2e-6f64);
}

#[test]
#[ignore]
fn test_30n() {
    test(String::from("30n20b8"), 43.33557298f64, 1e-7f64);
}

#[test]
#[ignore]
fn test_acc() {
    test(String::from("acc-tight4"), 0f64, 1e-7f64);
}
