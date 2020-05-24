//! # The Mixed Integer Programming Library tests
//!
//! This module integration-tests the project on a number of problem from the
//! [MIPLIB](http://miplib.zib.de/) project.
//!
//! ## Note
//!
//! The tests in this module are only ran when the `miplib_tests` feature is enabled as these tests
//! take a long time (minutes to hours) to run.
use std::path::{Path, PathBuf};

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
