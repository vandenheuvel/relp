//! # Which rule breaks GANGES
//!
//! Solves the instance once per configuration and checks the result against the problem as read,
//! so that a wrong answer can be pinned on the rule that produced it.
use std::convert::TryInto;
use std::fs::File;
use std::io::Read;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::time::Instant;

use num_traits::ToPrimitive;
use relp_num::{Rational64, RationalBig};

use relp::algorithm::two_phase::matrix_provider::MatrixProvider;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::LUDecomposition;
use relp::algorithm::{OptimizationResult, SolveRelaxation};
use relp::data::linear_program::general_form::{GeneralForm, Limits, Presolved, RuleId};
use relp::data::linear_program::solution::Solution;
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

/// Presolve as told, solve, postsolve.
fn run(
    mut general: GeneralForm<T>,
    limits: Option<(Limits<T>, bool)>,
) -> Option<(Solution<T>, String)> {
    let mut note = String::new();
    if let Some((limits, allow_dual)) = limits {
        match general.presolve_with::<false>(&limits, allow_dual) {
            Presolved::Reduced => {}
            Presolved::Solved(solution) => return Some((solution, "settled by presolve".into())),
            other => {
                note = format!("presolve said {other:?}");
                return Some((Solution::new(T::from(0), vec![]), note));
            }
        }
        note = format!("{}x{}", general.nr_constraints(), general.nr_variables());
    } else {
        note = format!("{}x{} (not presolved)", general.nr_constraints(), general.nr_variables());
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
        None => {
            note.push_str(", duals lost");
            general.compute_full_solution_with_reduced_solution(reconstructed)
        }
    };
    Some((solution, note))
}

fn check(label: &str, original: &GeneralForm<T>, limits: Option<(Limits<T>, bool)>) {
    let start = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(|| run(original.clone(), limits)));
    let elapsed = start.elapsed();

    match outcome {
        Ok(Some((solution, note))) => {
            let verdict = match original.verify_solution(&solution) {
                Ok(()) => match solution.dual_values {
                    Some(_) => "valid, certified".to_string(),
                    None => "valid".to_string(),
                },
                Err(invalid) => {
                    let detail = match &invalid {
                        relp::data::linear_program::verify::Invalid::VariableOutOfBounds { variable } => {
                            let (name, value) = &solution.solution_values[*variable];
                            format!(" ({name} = {})", value.to_f64().unwrap())
                        }
                        _ => String::new(),
                    };
                    format!("INVALID: {invalid}{detail}")
                }
            };
            println!(
                "{label:<34} {:>10} {:>16.6} {:<24} {}",
                format!("{:.2?}", elapsed),
                solution.objective_value.to_f64().unwrap(),
                note,
                verdict,
            );
        }
        Ok(None) => println!("{label:<34} {:>10} no finite optimum", format!("{:.2?}", elapsed)),
        Err(_) => println!("{label:<34} {:>10} panicked", format!("{:.2?}", elapsed)),
    }
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));

    let name = std::env::args().nth(1).unwrap_or_else(|| "GANGES".to_string());
    let original = read(&name);
    println!(
        "{name}: {}x{}\n",
        original.nr_constraints(),
        original.nr_variables(),
    );
    println!(
        "{:<34} {:>10} {:>16} {:<24} {}",
        "configuration", "time", "objective", "reduced to", "verdict",
    );

    check("every rule", &original, Some((Limits::default(), true)));
    check("dual preserving rules", &original, Some((Limits::preserving_duals(), false)));

    for rule in [
        RuleId::EmptyRow, RuleId::EmptyColumn, RuleId::FixedColumn, RuleId::SingletonRow,
        RuleId::SingletonColumn, RuleId::DomainPropagation, RuleId::ImpliedFreeColumnSingleton,
        RuleId::DoubletonEquation, RuleId::DuplicateRow, RuleId::DuplicateColumn,
        RuleId::DominatedColumn,
    ] {
        check(&format!("without {}", rule.name()), &original, Some((Limits::without(rule), true)));
    }

    check("no presolve", &original, None);
}
