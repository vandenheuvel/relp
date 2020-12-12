//! # Netlib
//!
//! Hosted [here](http://www.numerical.rl.ac.uk/cute/netlib.html).
use std::convert::TryInto;
use std::path::{Path, PathBuf};

use rust_lp::algorithm::{OptimizationResult, SolveRelaxation};
use rust_lp::algorithm::two_phase::matrix_provider::MatrixProvider;
use rust_lp::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
use rust_lp::data::linear_program::general_form::GeneralForm;
use rust_lp::data::linear_program::solution::Solution;
use rust_lp::data::number_types::rational::{RationalBig};
use rust_lp::io::import;

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
    let path = get_test_file_path(file_name);
    let mps = import(&path).unwrap();

    let mut general: GeneralForm<T> = mps.try_into().ok().unwrap();
    let data = general.derive_matrix_data().ok().unwrap();
    let result = data.solve_relaxation::<Carry<S>>();

    match result {
        OptimizationResult::FiniteOptimum(vector) => {
            let reconstructed = data.reconstruct_solution(vector);
            general.compute_full_solution_with_reduced_solution(reconstructed)
        },
        _ => panic!(),
    }
}
