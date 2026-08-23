//! # Problems from Unicamp
//!
//! Hosted [here](https://www.cenapad.unicamp.br/parque/manuais/OSL/oslweb/features/feat24DT.htm).
use std::convert::TryInto;
use std::ops::{Add, AddAssign, Mul, Neg, Sub};
use std::path::{Path, PathBuf};

use num_traits::{One, Zero};
use relp_num::Rational64;
use relp_num::{Abs, OrderedField, OrderedFieldRef, Signed, Absorb};

use relp::algorithm::two_phase::matrix_provider::MatrixProvider;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::LUDecomposition;
use relp::algorithm::two_phase::tableau::inverse_maintenance::ops as im_ops;
use relp::algorithm::two_phase::tableau::kind::artificial::Cost;
use relp::algorithm::{OptimizationResult, SolveRelaxation};
use relp::data::linear_algebra::traits::Element;
use relp::data::linear_program::solution::Solution;
use relp::data::linear_program::general_form::{
    Narrowable, Presolved, ReadsNarrowed, Solved,
};
use relp::io::import;

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

fn solve<IMT, GFT>(file_name: &str) -> Solution<IMT>
where
    IMT: im_ops::Field
        + im_ops::FieldHR
        + AddAssign<GFT>
        + PartialEq<GFT>
        + Ord
        + Signed
        + From<GFT>,
    GFT: 'static + From<Rational64> + Zero + One + Ord + Abs + Element + OrderedField,
    IMT: Absorb<GFT>,
    // The matrix data hands out its cost row as optional references, and the lifts for those are
    // written per accumulator rather than once for every one, so a generic accumulator states them.
    for<'r> IMT: Absorb<Option<&'r GFT>>,
    for<'r> &'r GFT: OrderedFieldRef<GFT>,
    // Recovering the solution evaluates the record presolve left behind, in the accumulator's
    // number type rather than the problem's.
    IMT: OrderedField,
    for<'r> &'r IMT: OrderedFieldRef<IMT> + Neg<Output = IMT>,
    for<'r> IMT: Add<&'r GFT, Output = IMT>,
    for<'r> &'r IMT:
        Add<&'r GFT, Output = IMT> + Mul<&'r GFT, Output = IMT> + Sub<&'r IMT, Output = IMT>,
    IMT: Absorb<Cost>,
    // Storing the problem in the narrowest type that holds it, and reading it back out of
    // whichever that turned out to be.
    GFT: Narrowable,
    IMT: ReadsNarrowed<GFT>,
{
    let path = get_test_file_path(file_name);
    let mps = import::<GFT>(&path).unwrap();

    let mut general = mps.try_into().unwrap();
    match general.presolve::<false>() {
        Presolved::Reduced => {}
        Presolved::Solved(Solution {
            objective_value,
            solution_values,
            ..
        }) => {
            return Solution::new(
                objective_value.into(),
                solution_values
                    .into_iter()
                    .map(|(name, value)| (name, value.into()))
                    .collect(),
            );
        }
        other => panic!("this instance has a finite optimum, got {:?}", other),
    }
    let constraint_type_counts = general.standardize();
    let narrowed = general.narrow(constraint_type_counts);
    let result = narrowed.solve_relaxation::<Carry<IMT, LUDecomposition<_>>>();

    match result {
        Solved::Optimum(optimum) => {
            general.compute_full_solution_with_reduced_solution::<IMT>(optimum.primal)
        }
        _ => panic!(),
    }
}
