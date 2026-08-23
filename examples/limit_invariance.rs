//! # A limit may change how far presolving gets, never what the answer is
//!
//! Solves every instance under limits that let presolve do more or less work, and checks that the
//! optimum is the same one every time and that every solution stands up to the problem as read.
use std::convert::TryInto;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use relp_num::{Rational64, RationalBig};

use relp::algorithm::two_phase::matrix_provider::MatrixProvider;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::LUDecomposition;
use relp::algorithm::{OptimizationResult, SolveRelaxation};
use relp::data::linear_program::general_form::{GeneralForm, Limits, Presolved};
use relp::io::mps::parse_fixed;

type T = RationalBig;

const INSTANCES: &[&str] = &[
    "ADLITTLE", "AFIRO", "AGG", "AGG2", "AGG3", "BEACONFD", "BLEND", "BOEING1", "BOEING2",
    "BORE3D", "BRANDY", "CAPRI", "CZPROB", "E226", "ETAMACRO", "FINNIS", "GANGES", "GFRD-PNC",
    "ISRAEL", "KB2", "LOTFI", "SC50A", "SC50B", "SC105", "SC205", "SCAGR7", "SCRS8", "SCSD1",
    "SCORPION", "SCTAP1", "STANDATA", "STANDMPS", "STOCFOR1", "SHARE1B", "SHARE2B", "RECIPELP",
    "VTP-BASE",
];

fn read(name: &str) -> GeneralForm<T> {
    let path = PathBuf::from("tests/netlib/problem_files")
        .join(name)
        .with_extension("SIF");
    let mut program = String::new();
    File::open(&path).unwrap().read_to_string(&mut program).unwrap();
    parse_fixed::<Rational64>(&program).unwrap().try_into().unwrap()
}

/// The optimum under these limits, and whether the solution survives being checked.
fn solve(original: &GeneralForm<T>, limits: Option<Limits<T>>) -> Option<(T, bool)> {
    let mut general = original.clone();
    if let Some(limits) = limits {
        match general.presolve_with::<false>(&limits, true) {
            Presolved::Reduced => {}
            Presolved::Solved(solution) => {
                let valid = original.verify_solution(&solution).is_ok();
                return Some((solution.objective_value, valid));
            }
            _ => return None,
        }
    }

    let constraint_type_counts = general.standardize();
    let data = general.derive_matrix_data(constraint_type_counts);
    let OptimizationResult::FiniteOptimum(optimum) = data.solve_relaxation::<Carry<T, LUDecomposition<_>>>()
    else {
        return None;
    };
    let duals = data.constraint_duals(&optimum.dual);
    let reconstructed = data.reconstruct_solution(optimum.primal);
    let solution = match general.compute_full_solution_with_duals(reconstructed.clone(), duals) {
        Some(solution) => solution,
        None => general.compute_full_solution_with_reduced_solution(reconstructed),
    };
    let valid = original.verify_solution(&solution).is_ok();
    Some((solution.objective_value, valid))
}

fn main() {
    let mut no_threshold = Limits::<T>::default();
    no_threshold.min_relative_tightening = T::from(0);
    let mut tiny_budget = Limits::<T>::default();
    tiny_budget.max_bound_tightenings = 1;
    let mut no_fill = Limits::<T>::default();
    no_fill.max_fill_in = 0;
    let mut one_round = Limits::<T>::default();
    one_round.max_rounds = 1;

    let configurations: Vec<(&str, Option<Limits<T>>)> = vec![
        ("default", Some(Limits::default())),
        ("no threshold", Some(no_threshold)),
        ("budget of 1", Some(tiny_budget)),
        ("no fill in", Some(no_fill)),
        ("one round", Some(one_round)),
        ("no presolve", None),
    ];

    let mut disagreements = 0;
    let mut invalid = 0;
    for name in INSTANCES {
        let original = read(name);
        let mut reference: Option<T> = None;
        let mut notes = Vec::new();

        for (label, limits) in &configurations {
            let Some((objective, valid)) = solve(&original, limits.clone()) else {
                notes.push(format!("{label}: no optimum"));
                continue;
            };
            if !valid {
                notes.push(format!("{label}: INVALID"));
                invalid += 1;
            }
            match &reference {
                None => reference = Some(objective),
                Some(expected) if *expected != objective => {
                    notes.push(format!("{label}: {objective} != {expected}"));
                    disagreements += 1;
                }
                Some(_) => {}
            }
        }

        println!(
            "{name:<10} {}",
            if notes.is_empty() {
                "same optimum under every limit, every solution checks out".to_string()
            } else {
                notes.join("; ")
            },
        );
    }

    println!("\n{disagreements} disagreements, {invalid} solutions that do not check out");
}
