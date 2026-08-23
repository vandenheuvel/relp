//! # GANGES, end to end
//!
//! Does presolving reduce this instance, and is the optimum that comes out the right one?
use std::convert::TryInto;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

use num_traits::ToPrimitive;
use relp_num::{Abs, RB, Rational64, RationalBig};

use relp::algorithm::two_phase::matrix_provider::MatrixProvider;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::LUDecomposition;
use relp::algorithm::{OptimizationResult, SolveRelaxation};
use relp::data::linear_program::general_form::{GeneralForm, Limits, Presolved};
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

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "GANGES".to_string());

    let start = Instant::now();
    let original = read(&name);
    println!(
        "read {name}: {}x{} in {:.2?}",
        original.nr_constraints(),
        original.nr_variables(),
        start.elapsed(),
    );

    // What the dual preserving catalogue manages on its own, for comparison.
    let mut preserving = original.clone();
    let start = Instant::now();
    let outcome = preserving.presolve_with::<false>(&Limits::preserving_duals(), false);
    println!("\n--- presolve, dual preserving rules only ({:.2?}) ---", start.elapsed());
    print!("{}", preserving.postsolve().statistics);
    println!("outcome: {}", describe(&outcome));

    let mut general = original.clone();
    let start = Instant::now();
    let outcome = general.presolve::<false>();
    let presolve_time = start.elapsed();
    println!("\n--- presolve, every rule ({:.2?}) ---", presolve_time);
    print!("{}", general.postsolve().statistics);
    println!("outcome: {}", describe(&outcome));

    let solution = match outcome {
        Presolved::Reduced => {
            let constraint_type_counts = general.standardize();
            let start = Instant::now();
            let data = general.derive_matrix_data(constraint_type_counts);
            let result = data.solve_relaxation::<Carry<T, LUDecomposition<_>>>();
            let solve_time = start.elapsed();
            let OptimizationResult::FiniteOptimum(optimum) = result else {
                println!("\nsolve gave {result:?} after {solve_time:.2?}");
                return;
            };
            let duals = data.constraint_duals(&optimum.dual);
            let reconstructed = data.reconstruct_solution(optimum.primal);
            println!("\nsolve: finite optimum in {solve_time:.2?}");

            let start = Instant::now();
            let solution = match general.compute_full_solution_with_duals(reconstructed.clone(), duals) {
                Some(solution) => {
                    println!("postsolve: multipliers recovered in {:.2?}", start.elapsed());
                    solution
                }
                None => {
                    println!(
                        "postsolve: multipliers lost, applied_dual_reduction = {}",
                        general.postsolve().applied_dual_reduction,
                    );
                    general.compute_full_solution_with_reduced_solution(reconstructed)
                }
            };
            solution
        }
        Presolved::Solved(solution) => {
            println!("presolve settled the instance on its own");
            solution
        }
        other => {
            println!("nothing to solve: {}", describe(&other));
            return;
        }
    };

    let start = Instant::now();
    let verdict = original.verify_solution(&solution);
    println!(
        "verify against the problem as read ({:.2?}): {}",
        start.elapsed(),
        match &verdict {
            Ok(()) => match solution.dual_values {
                Some(_) => "optimality certificate holds".to_string(),
                None => "feasible, and the objective value matches (no certificate)".to_string(),
            },
            Err(invalid) => format!("INVALID: {invalid}"),
        },
    );

    let objective = &solution.objective_value;
    println!("\nobjective: {}", objective.to_f64().unwrap());
    println!("exact:     {objective}");
    for (source, reference) in [
        ("netlib README", -1.0958636356E+05),
        ("README, solver 1", -1.0958573613E+05),
        ("README, solver 2", -1.0958577038E+05),
    ] {
        let difference = (objective - &RB!(reference)).abs();
        println!(
            "  vs {source:<18} {reference:>18.8e}  difference {:.3e}",
            difference.to_f64().unwrap(),
        );
    }
}

fn describe(outcome: &Presolved<T>) -> String {
    match outcome {
        Presolved::Reduced => "Reduced".to_string(),
        Presolved::Solved(solution) => format!("Solved, objective {}", solution.objective_value),
        Presolved::Infeasible => "Infeasible".to_string(),
        Presolved::Unbounded { assuming_feasible } => {
            format!("Unbounded (assuming_feasible: {assuming_feasible})")
        }
        Presolved::InfeasibleOrUnbounded => "InfeasibleOrUnbounded".to_string(),
    }
}
