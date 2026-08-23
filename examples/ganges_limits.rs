//! # Is it the tightening threshold, or the tightening budget?
//!
//! The bound a doubleton substitution puts on the kept variable is the only thing still enforcing
//! the bounds of the variable it removed. If it can be declined, the reduced problem is a
//! relaxation. Both ways of declining one are turned off here, separately.
use std::convert::TryInto;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

use num_traits::ToPrimitive;
use relp_num::{RationalBig, Rational64};

use relp::algorithm::two_phase::matrix_provider::MatrixProvider;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::LUDecomposition;
use relp::algorithm::{OptimizationResult, SolveRelaxation};
use relp::data::linear_program::general_form::{GeneralForm, Limits, Presolved, RuleId};
use relp::io::mps::parse_fixed;

type T = RationalBig;

fn read(name: &str) -> GeneralForm<T> {
    let path = PathBuf::from("tests/netlib/problem_files")
        .join(name)
        .with_extension("SIF");
    let mut program = String::new();
    File::open(&path).unwrap().read_to_string(&mut program).unwrap();
    parse_fixed::<Rational64>(&program).unwrap().try_into().unwrap()
}

fn check(label: &str, original: &GeneralForm<T>, limits: Limits<T>) {
    let mut general = original.clone();
    let start = Instant::now();
    let Presolved::Reduced = general.presolve_with::<false>(&limits, true) else {
        println!("{label:<40} presolve did not leave a problem");
        return;
    };
    let size = format!("{}x{}", general.nr_constraints(), general.nr_variables());
    let refusals: u64 = general
        .postsolve()
        .statistics
        .per_rule
        .iter()
        .map(|(_, rule)| rule.budget_refusals)
        .sum();
    if std::env::var("RELP_REFUSALS").is_ok() {
        for (id, rule) in general.postsolve().statistics.per_rule.iter() {
            if rule.budget_refusals > 0 {
                println!("      budget refusals in {}: {}", id.name(), rule.budget_refusals);
            }
        }
    }

    let constraint_type_counts = general.standardize();
    let data = general.derive_matrix_data(constraint_type_counts);
    let OptimizationResult::FiniteOptimum(optimum) = data.solve_relaxation::<Carry<T, LUDecomposition<_>>>()
    else {
        println!("{label:<40} no finite optimum");
        return;
    };
    let duals = data.constraint_duals(&optimum.dual);
    let reconstructed = data.reconstruct_solution(optimum.primal);
    let solution = match general.compute_full_solution_with_duals(reconstructed.clone(), duals) {
        Some(solution) => solution,
        None => general.compute_full_solution_with_reduced_solution(reconstructed),
    };

    let verdict = match original.verify_solution(&solution) {
        Ok(()) => format!("valid, exact {}", solution.objective_value),
        Err(invalid) => format!("INVALID: {invalid}"),
    };
    println!(
        "{label:<40} {:>8} {:>16.6} {:<12} {:>4} refusals  {verdict}",
        format!("{:.2?}", start.elapsed()),
        solution.objective_value.to_f64().unwrap(),
        size,
        refusals,
    );
}

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "GANGES".to_string());
    let original = read(&name);
    println!("{name}: {}x{}\n", original.nr_constraints(), original.nr_variables());

    check("default", &original, Limits::default());

    let mut no_threshold = Limits::<T>::default();
    no_threshold.min_relative_tightening = T::from(0);
    check("min_relative_tightening = 0", &original, no_threshold);

    let mut no_budget = Limits::<T>::default();
    no_budget.max_bound_tightenings = u32::MAX;
    check("max_bound_tightenings = unlimited", &original, no_budget);

    check("only doubleton equation", &original, Limits::only(RuleId::DoubletonEquation));

    check("only singleton row", &original, Limits::only(RuleId::SingletonRow));

    let mut only_row_no_threshold = Limits::<T>::only(RuleId::SingletonRow);
    only_row_no_threshold.min_relative_tightening = T::from(0);
    check("only singleton row, threshold 0", &original, only_row_no_threshold);

    let mut only_no_threshold = Limits::<T>::only(RuleId::DoubletonEquation);
    only_no_threshold.min_relative_tightening = T::from(0);
    check("only doubleton, threshold 0", &original, only_no_threshold);

    let mut neither = Limits::<T>::default();
    neither.min_relative_tightening = T::from(0);
    neither.max_bound_tightenings = u32::MAX;
    check("neither limit", &original, neither);
}
