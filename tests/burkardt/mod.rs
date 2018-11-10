//! # John Burkardt's test files
//!
//! Retrieved from http://people.math.sc.edu/Burkardt/datasets/mps/mps.html.
use std::path::{Path, PathBuf};

use rust_lp::data::linear_program::general_form::GeneralForm;
use rust_lp::io::import;

/// # Generation and execution
#[allow(missing_docs)]
mod test;

/// Relative path of the folder where the mps files are stored.
///
/// The path is relative to the project root folder.
fn problem_file_directory() -> PathBuf {
    Path::new(file!()).parent().unwrap().to_path_buf()
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
    problem_file_directory().join(name).with_extension("mps")
}
