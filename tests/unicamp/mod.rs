//! # Problems from Unicamp
//!
//! Hosted [here](https://www.cenapad.unicamp.br/parque/manuais/OSL/oslweb/features/feat24DT.htm).
use std::convert::TryInto;
use std::ops::{Add, AddAssign};
use std::path::{Path, PathBuf};

use num::{One, Zero};

use rust_lp::algorithm::{OptimizationResult, SolveRelaxation};
use rust_lp::algorithm::two_phase::matrix_provider::MatrixProvider;
use rust_lp::algorithm::two_phase::tableau::inverse_maintenance::{ColumnOps, CostOps, InternalOps, InternalOpsHR};
use rust_lp::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
use rust_lp::data::linear_algebra::traits::Element;
use rust_lp::data::linear_program::solution::Solution;
use rust_lp::data::number_types::rational::Rational64;
use rust_lp::data::number_types::traits::{OrderedField, OrderedFieldRef};
use rust_lp::io::import;
use rust_lp::algorithm::two_phase::tableau::kind::artificial::Cost;

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
    problem_file_directory().join(name).with_extension("mps")
}

fn solve<
    IMT: InternalOps + InternalOpsHR + ColumnOps<GFT> + AddAssign<GFT> + PartialEq<GFT> + Ord,
    GFT: 'static + From<Rational64> + Zero + One + Ord + Element + OrderedField,
>(file_name: &str) -> Solution<IMT>
where
    for<'r> &'r GFT: OrderedFieldRef<GFT>,
    for<'r> &'r IMT: Add<&'r GFT, Output=IMT>,
    for<'r> IMT: CostOps<Option<&'r GFT>>,
    for<'r> IMT: CostOps<Cost>,
{
    let path = get_test_file_path(file_name);
    let mps = import::<GFT>(&path).unwrap();

    let mut general = mps.try_into().ok().unwrap();
    let data = general.derive_matrix_data().ok().unwrap();
    let result = data.solve_relaxation::<Carry<IMT>>();

    match result {
        OptimizationResult::FiniteOptimum(vector) => {
            let reconstructed = data.reconstruct_solution(vector);
            general.compute_full_solution_with_reduced_solution::<IMT>(reconstructed)
        },
        _ => panic!(),
    }
}
