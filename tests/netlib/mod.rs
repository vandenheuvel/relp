//! # Netlib
//!
//! Hosted [here](http://www.numerical.rl.ac.uk/cute/netlib.html).
use std::convert::TryInto;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use relp_num::{Rational64, RationalBig};

use relp::algorithm::two_phase::matrix_provider::MatrixProvider;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::LUDecomposition;
use relp::algorithm::{OptimizationResult, SolveRelaxation};
use relp::data::linear_program::general_form::{GeneralForm, Limits, Presolved};
use relp::data::linear_program::solution::Solution;
use relp::io::error::Import;
use relp::io::mps::parse_fixed;

/// # Generation and execution
#[allow(missing_docs)]
mod test;

/// Relative path of the folder where the mps files are stored.
///
/// The path is relative to the project root folder.
fn problem_file_directory() -> PathBuf {
    Path::new(file!()).parent().unwrap().join("problem_files")
}

/// Compute the path of the problem file, based on the problem name.
///
/// # Arguments
///
/// * `name`: Problem name without extension.
///
/// # Return value
///
/// File path relative to the project root folder.
fn get_test_file_path(name: &str) -> PathBuf {
    problem_file_directory().join(name).with_extension("SIF")
}

type T = RationalBig;
type S = RationalBig;

fn solve(file_name: &str) -> Solution<S> {
    let file_path = get_test_file_path(file_name);

    let mut program = String::new();
    File::open(&file_path)
        .map_err(Import::IO)
        .unwrap()
        .read_to_string(&mut program)
        .map_err(Import::IO)
        .unwrap();
    let mps = parse_fixed::<Rational64>(&program).unwrap();

    let mut general: GeneralForm<T> = mps.try_into().unwrap();
    let original = general.clone();

    assert_eq!(
        general.presolve::<false>(),
        Presolved::Reduced,
        "this instance is not expected to be settled by presolving alone",
    );
    let constraint_type_counts = general.standardize();

    let solution = {
        let data = general.derive_matrix_data(constraint_type_counts);
        let result = data.solve_relaxation::<Carry<S, LUDecomposition<_>>>();

        match result {
            OptimizationResult::FiniteOptimum(optimum) => {
                // The multipliers cost nothing extra: the simplex has them at termination either
                // way. Asking for them turns the check below from "this solution is allowed" into
                // "this solution is the best one", on 32 of the 38 instances in this suite.
                let duals = data.constraint_duals(&optimum.dual);
                let reconstructed = data.reconstruct_solution(optimum.primal);
                match general.compute_full_solution_with_duals(reconstructed.clone(), duals) {
                    Some(solution) => solution,
                    None => {
                        // The one reason there can be no certificate is a reduction that keeps an
                        // optimal point without keeping every feasible one. Saying so here is what
                        // stops the check quietly weakening for some other reason.
                        assert!(
                            general.postsolve().applied_dual_reduction,
                            "the multipliers were lost for a reason other than a dual reduction",
                        );
                        general.compute_full_solution_with_reduced_solution(reconstructed)
                    }
                }
            }
            other => panic!("this instance has a finite optimum, got {:?}", other),
        }
    };

    // Every instance is checked against the problem that was read, exactly. This says more than
    // the objective comparisons the tests make afterwards, which are against values from other
    // solvers and so can only be made to a tolerance. Where the multipliers came back, the check
    // is a full optimality certificate rather than a feasibility check.
    original
        .verify_solution(&solution)
        .expect("the solution has to solve the problem that was read");

    solution
}

/// Solve, recovering the multipliers, and check the certificate.
///
/// Presolving runs with only the rules whose effect on the multipliers postsolving undoes, which
/// is every rule but the one that fixes a dominated column.
fn certify(file_name: &str) {
    let file_path = get_test_file_path(file_name);

    let mut program = String::new();
    File::open(&file_path)
        .map_err(Import::IO)
        .unwrap()
        .read_to_string(&mut program)
        .map_err(Import::IO)
        .unwrap();
    let mps = parse_fixed::<Rational64>(&program).unwrap();

    let mut general: GeneralForm<T> = mps.try_into().unwrap();
    let original = general.clone();

    // Presolving with only the rules whose effect on the multipliers is undone.
    assert_eq!(
        general.presolve_with::<false>(&Limits::preserving_duals(), false),
        Presolved::Reduced,
    );

    let constraint_type_counts = general.standardize();
    let (reduced, duals) = {
        let data = general.derive_matrix_data(constraint_type_counts);
        match data.solve_relaxation::<Carry<S, LUDecomposition<_>>>() {
            OptimizationResult::FiniteOptimum(optimum) => (
                data.reconstruct_solution(optimum.primal),
                data.constraint_duals(&optimum.dual),
            ),
            other => panic!("this instance has a finite optimum, got {:?}", other),
        }
    };

    let solution = general
        .compute_full_solution_with_duals(reduced, duals)
        .expect("only dual preserving rules ran, so the multipliers carry back");

    original
        .verify_solution(&solution)
        .expect("the multipliers have to certify that the solution is optimal");
}
