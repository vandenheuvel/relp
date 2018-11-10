//! # Generation and execution
//!
//! Functions to integration-test are generated using macro's.
use std::convert::TryInto;

use rust_lp::algorithm::simplex::matrix_provider::matrix_data::MatrixData;
use rust_lp::algorithm::simplex::solve_relaxation;
use rust_lp::algorithm::simplex::strategy::pivot_rule::FirstProfitable;
use rust_lp::data::linear_program::elements::LinearProgramType;
use rust_lp::data::linear_program::general_form::GeneralForm;
use rust_lp::io::import;
use rust_lp::R64;
use super::get_test_file_path;
use num::rational::Ratio;
use num::FromPrimitive;
use rust_lp::data::number_types::traits::OrderedField;

macro_rules! generate_test {
    ($name:ident, $file:expr, $obj:expr, $epsilon:expr) => {
        /// Testing problem $name
        #[test]
        fn $name() {
            type T = Ratio<i64>;

            let name = String::from($file);
            let path = get_test_file_path(&name);
            let result = import(&path).unwrap();

            let mut general: GeneralForm<T> = result.try_into().ok().unwrap();
            let data = general.derive_matrix_data().ok().unwrap();

            let result = solve_relaxation::<T, _, FirstProfitable, FirstProfitable>(&data);

            assert!(match result {
                LinearProgramType::FiniteOptimum(v) => (v - R64!($obj)).abs() < R64!($epsilon),
                _ => false,
            })
        }
    };
}

generate_test!(test_50v, "50v-10", 2879.065687f64, 2e-6f64);
// generate_test!(test_30n, "30n20b8", 43.33557298f64, 1e-7f64);
// generate_test!(test_acc, "acc-tight4", 0f64, 1e-7f64);
