use std::convert::TryInto;

use relp_num::{Rational64, RationalBig};
use relp_num::Abs;
use relp_num::RB;

use relp::algorithm::{OptimizationResult, SolveRelaxation};
use relp::algorithm::two_phase::matrix_provider::MatrixProvider;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::LUDecomposition;
use relp::data::linear_program::general_form::GeneralForm;
use relp::io::import;

use crate::cook::get_test_file_path;

/// Note that the OBJNAME section was removed from the original problem.
#[test]
fn small_example() {
    let path = get_test_file_path("small_example");
    let result = import(&path).unwrap();

    let mut general: GeneralForm<Rational64> = result.try_into().ok().unwrap();

    general.presolve().unwrap();
    let constraint_type_counts = general.standardize();
    let data = general.derive_matrix_data(constraint_type_counts);
    let result = data.solve_relaxation::<Carry<RationalBig, LUDecomposition<_>>>();

    match result {
        OptimizationResult::FiniteOptimum(solution) => {
            let reconstructed = data.reconstruct_solution(solution);
            let solution = general.compute_full_solution_with_reduced_solution(reconstructed);
            assert!((solution.objective_value - RB!(-143, 2)).abs() < RB!(1e-5)); // glpk
        }
        _ => assert!(false),
    }
}
