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
use relp::data::linear_program::general_form::{
    GeneralForm, Limits, Presolved, ProblemSize, RuleId, Solved,
};
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
        // Narrowed rather than read where it stands: the algorithm only reads these numbers, so
        // it is handed them in the smallest type they fit in.
        let narrowed = general.narrow(constraint_type_counts);
        match narrowed.solve_relaxation::<Carry<S, LUDecomposition<_>>>() {
            Solved::Optimum(optimum) => {
                // The multipliers cost nothing extra: the simplex has them at termination either
                // way. Asking for them turns the check below from "this solution is allowed" into
                // "this solution is the best one", on 32 of the 38 instances in this suite.
                match general
                    .compute_full_solution_with_duals(optimum.primal.clone(), optimum.duals)
                {
                    Some(solution) => solution,
                    None => {
                        // The one reason there can be no certificate is a reduction that keeps an
                        // optimal point without keeping every feasible one. Saying so here is what
                        // stops the check quietly weakening for some other reason.
                        assert!(
                            general.postsolve().applied_dual_reduction,
                            "the multipliers were lost for a reason other than a dual reduction",
                        );
                        general.compute_full_solution_with_reduced_solution(optimum.primal)
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
        let narrowed = general.narrow(constraint_type_counts);
        match narrowed.solve_relaxation::<Carry<S, LUDecomposition<_>>>() {
            Solved::Optimum(optimum) => (optimum.primal, optimum.duals),
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

/// What presolving and narrowing do to the size of each instance.
///
/// Not an assertion: whether a reduction earns its place in exact arithmetic is a question about
/// numbers, and this is where they come from. Run it with
/// `cargo test --release --features netlib -- --ignored --nocapture report_sizes`.
#[test]
#[ignore = "Reports measurements, does not assert anything."]
fn report_sizes() {
    let instances = [
        "ADLITTLE", "AFIRO", "AGG", "AGG2", "AGG3", "BANDM", "BEACONFD", "BLEND", "BOEING1",
        "BOEING2", "BORE3D", "BRANDY", "CAPRI", "CZPROB", "E226", "FINNIS", "GROW7", "ISRAEL",
        "KB2", "LOTFI", "RECIPELP", "SC105", "SC205", "SC50A", "SC50B", "SCAGR25", "SCAGR7",
        "SCORPION", "SCRS8", "SCSD1", "SCSD6", "SCTAP1", "SHARE1B", "SHARE2B", "STANDATA",
        "STANDMPS", "STOCFOR1", "VTP-BASE",
    ];

    println!(
        "{:<10} {:>26} {:>16} {:>16} {:>6} {:>13} {:>13}",
        "instance", "before", "unscaled", "scaled", "rows", "width", "width",
    );
    for name in instances {
        let file_path = get_test_file_path(name);
        let mut program = String::new();
        File::open(&file_path)
            .map_err(Import::IO)
            .unwrap()
            .read_to_string(&mut program)
            .map_err(Import::IO)
            .unwrap();
        let mps = parse_fixed::<Rational64>(&program).unwrap();
        let mut general: GeneralForm<T> = mps.try_into().unwrap();

        // The same instance twice, once with the rule that clears fractions and once without, so
        // that what it costs and what it saves are separated from what the other rules do.
        let mut without = general.clone();
        assert_eq!(without.presolve::<false>(), Presolved::Reduced);
        let unscaled = without.postsolve().statistics.after.clone();
        let counts = without.standardize();
        let unscaled_width = without.narrow(counts).width();

        assert_eq!(
            general.presolve_with::<false>(&Limits::with(RuleId::RowScaling), true),
            Presolved::Reduced,
        );
        let statistics = general.postsolve().statistics.clone();
        let counts = general.standardize();
        let width = general.narrow(counts).width();

        let bits = |size: &ProblemSize| format!("{} bits", size.encoding_bits.unwrap_or(0));
        let size = |size: &ProblemSize| {
            format!(
                "{}x{} {} nz {} bits",
                size.rows,
                size.columns,
                size.nonzeros,
                size.encoding_bits.unwrap_or(0),
            )
        };
        println!(
            "{:<10} {:>26} {:>16} {:>16} {:>6} {:>13} {:>13}",
            name,
            size(&statistics.before),
            bits(&unscaled),
            bits(&statistics.after),
            statistics.per_rule[RuleId::RowScaling].rows_scaled,
            unscaled_width.name(),
            width.name(),
        );
    }
}

/// What the two of them cost, against not doing either.
///
/// Three configurations of the same instances: the pipeline as it was, with the rule that clears
/// fractions added, and with the problem narrowed on its way into the algorithm as well. Timed
/// from the start of presolving to a recovered solution, so that what the rule costs is counted
/// along with what it saves.
///
/// Run it with `cargo test --release --features netlib -- --ignored --nocapture report_timings`.
#[test]
#[ignore = "Reports measurements, does not assert anything."]
fn report_timings() {
    let instances = [
        "ADLITTLE", "AGG", "AGG2", "AGG3", "BANDM", "BEACONFD", "BOEING1", "BOEING2", "BORE3D",
        "BRANDY", "CAPRI", "CZPROB", "E226", "FINNIS", "GROW7", "ISRAEL", "LOTFI", "SC205",
        "SCAGR25", "SCORPION", "SCRS8", "SCSD1", "SCSD6", "SCTAP1", "SHARE1B", "STANDMPS",
        "STOCFOR1", "VTP-BASE",
    ];

    let mut totals = [std::time::Duration::ZERO; 4];
    println!(
        "{:<10} {:>12} {:>12} {:>12} {:>12}",
        "instance", "as it was", "clearing", "narrowed", "both",
    );
    for name in instances {
        let file_path = get_test_file_path(name);
        let mut program = String::new();
        File::open(&file_path)
            .map_err(Import::IO)
            .unwrap()
            .read_to_string(&mut program)
            .map_err(Import::IO)
            .unwrap();
        let mps = parse_fixed::<Rational64>(&program).unwrap();
        let general: GeneralForm<T> = mps.try_into().unwrap();

        let mut taken = [std::time::Duration::ZERO; 4];
        for (index, (clearing, narrowed)) in
            [(false, false), (true, false), (false, true), (true, true)]
                .into_iter()
                .enumerate()
        {
            let mut problem = general.clone();
            let start = std::time::Instant::now();

            let limits = if clearing {
                Limits::with(RuleId::RowScaling)
            } else {
                Limits::default()
            };
            assert_eq!(problem.presolve_with::<false>(&limits, true), Presolved::Reduced);
            let counts = problem.standardize();

            let reduced = if narrowed {
                match problem.narrow(counts).solve_relaxation::<Carry<S, LUDecomposition<_>>>() {
                    Solved::Optimum(optimum) => optimum.primal,
                    other => panic!("{name} has a finite optimum, got {other:?}"),
                }
            } else {
                let data = problem.derive_matrix_data(counts);
                match data.solve_relaxation::<Carry<S, LUDecomposition<_>>>() {
                    OptimizationResult::FiniteOptimum(optimum) => {
                        data.reconstruct_solution(optimum.primal)
                    }
                    other => panic!("{name} has a finite optimum, got {other:?}"),
                }
            };
            let _ = problem.compute_full_solution_with_reduced_solution(reduced);

            taken[index] = start.elapsed();
            totals[index] += taken[index];
        }

        println!(
            "{:<10} {:>11.0?} {:>11.0?} {:>11.0?} {:>11.0?}",
            name, taken[0], taken[1], taken[2], taken[3],
        );
    }

    println!(
        "{:<10} {:>11.0?} {:>11.0?} {:>11.0?} {:>11.0?}",
        "total", totals[0], totals[1], totals[2], totals[3],
    );
}


