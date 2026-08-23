//! # A tightening budget that changes the answer
//!
//! `max_bound_tightenings` refuses bounds that a row still in the problem implies, which should
//! only ever cost a reduction. On AGG a small budget changes the optimum, so some rule is relying
//! on a bound it offered having been taken. This says which one, and at which budget.
use std::convert::TryInto;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use num_traits::ToPrimitive;
use relp_num::{Rational64, RationalBig};

use relp::algorithm::two_phase::matrix_provider::MatrixProvider;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::LUDecomposition;
use relp::algorithm::{OptimizationResult, SolveRelaxation};
use relp::data::linear_program::general_form::{GeneralForm, Limits, Presolved, RuleId};
use relp::io::mps::parse_fixed;

type T = RationalBig;

const RULES: &[RuleId] = &[
    RuleId::EmptyRow, RuleId::EmptyColumn, RuleId::FixedColumn, RuleId::SingletonRow,
    RuleId::SingletonColumn, RuleId::DomainPropagation, RuleId::ImpliedFreeColumnSingleton,
    RuleId::DoubletonEquation, RuleId::DuplicateRow, RuleId::DuplicateColumn,
    RuleId::DominatedColumn,
];

fn read(name: &str) -> GeneralForm<T> {
    let path = PathBuf::from("tests/netlib/problem_files").join(name).with_extension("SIF");
    let mut program = String::new();
    File::open(&path).unwrap().read_to_string(&mut program).unwrap();
    parse_fixed::<Rational64>(&program).unwrap().try_into().unwrap()
}

fn run(original: &GeneralForm<T>, limits: Limits<T>) -> Option<(f64, bool)> {
    let mut general = original.clone();
    match general.presolve_with::<false>(&limits, true) {
        Presolved::Reduced => {}
        Presolved::Solved(solution) => {
            let valid = original.verify_solution(&solution).is_ok();
            return Some((solution.objective_value.to_f64().unwrap(), valid));
        }
        _ => return None,
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
    Some((solution.objective_value.to_f64().unwrap(), valid))
}

fn report(label: &str, outcome: Option<(f64, bool)>) {
    match outcome {
        Some((objective, valid)) => println!(
            "{label:<40} {objective:>18.6}  {}",
            if valid { "valid" } else { "INVALID" },
        ),
        None => println!("{label:<40} no optimum"),
    }
}

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "AGG".to_string());
    let original = read(&name);

    report("default", run(&original, Limits::default()));
    for budget in 1..=8 {
        let mut limits = Limits::<T>::default();
        limits.max_bound_tightenings = budget;
        report(&format!("budget of {budget}"), run(&original, limits));
    }

    println!("\nwith a budget of 1, one rule at a time left out:");
    for &rule in RULES {
        let mut limits = Limits::<T>::without(rule);
        limits.max_bound_tightenings = 1;
        report(&format!("  without {}", rule.name()), run(&original, limits));
    }
}
