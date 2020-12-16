use std::convert::TryInto;

use num::FromPrimitive;

use rust_lp::algorithm::{OptimizationResult, SolveRelaxation};
use rust_lp::algorithm::two_phase::matrix_provider::MatrixProvider;
use rust_lp::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
use rust_lp::data::linear_program::general_form::GeneralForm;
use rust_lp::data::number_types::rational::{Rational64, RationalBig};
use rust_lp::data::number_types::traits::Abs;
use rust_lp::io::import;
use rust_lp::RB;

use crate::cook::get_test_file_path;

/// Note that the OBJNAME section was removed from the original problem.
#[test]
fn small_example() {
    let path = get_test_file_path("small_example");
    let result = import(&path).unwrap();

    let mut general: GeneralForm<Rational64> = result.try_into().ok().unwrap();

    let data = general.derive_matrix_data().ok().unwrap();
    let result = data.solve_relaxation::<Carry<RationalBig>>();

    match result {
        OptimizationResult::FiniteOptimum(solution) => {
            let reconstructed = data.reconstruct_solution(solution);
            let solution = general.compute_full_solution_with_reduced_solution(reconstructed);
            assert!((solution.objective_value - RB!(-243, 4)).abs() < RB!(1e-5));
        }
        _ => assert!(false),
    }
}