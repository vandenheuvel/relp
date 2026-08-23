use std::convert::TryInto;

use relp_num::RB;
use relp_num::{Rational64, RationalBig};

use relp::algorithm::two_phase::matrix_provider::MatrixProvider;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::LUDecomposition;
use relp::algorithm::{OptimizationResult, SolveRelaxation};
use relp::data::linear_program::general_form::{GeneralForm, Presolved};
use relp::io::import;

use crate::cook::get_test_file_path;

/// Note that the OBJNAME section was removed from the original problem.
#[test]
fn small_example() {
    let path = get_test_file_path("small_example");
    let result = import(&path).unwrap();

    let mut general: GeneralForm<Rational64> = result.try_into().ok().unwrap();
    let original = general.clone();

    assert_eq!(
        general.presolve::<false>(),
        Presolved::Reduced,
        "this instance is not expected to be settled by presolving alone",
    );
    let constraint_type_counts = general.standardize();
    let data = general.derive_matrix_data(constraint_type_counts);
    let result = data.solve_relaxation::<Carry<RationalBig, LUDecomposition<_>>>();

    match result {
        OptimizationResult::FiniteOptimum(optimum) => {
            let reconstructed = data.reconstruct_solution(optimum.primal);
            let solution = general.compute_full_solution_with_reduced_solution(reconstructed);
            original
                .verify_solution(&solution)
                .expect("the solution has to solve the problem that was read");

            // `max x - 2.3 y + 0.5 z` with all variables non negative, `x - y + s <= 10.75` and
            // `2x - z - s >= -100`. Raising `y` or `s` only costs, so both are zero, which leaves
            // `x <= 10.75`; then `z <= 2x + 100 = 121.5`, for `10.75 + 60.75`.
            //
            // This used to be recorded as the negation of that. Maximization problems had their
            // costs negated on the way in without being negated back on the way out, so every
            // answer to one came out with the wrong sign.
            assert_eq!(solution.objective_value, RB!(143, 2));
            assert_eq!(
                solution.solution_values,
                vec![
                    ("x".to_string(), RB!(43, 4)),
                    ("y".to_string(), RB!(0)),
                    ("z".to_string(), RB!(243, 2)),
                    ("s".to_string(), RB!(0)),
                ],
            );
        }
        _ => panic!("Expected a finite optimum."),
    }
}
