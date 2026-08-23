//! # What presolving costs, and what it saves
//!
//! Times each phase of the pipeline over the netlib instances, with presolving on and off, so the
//! two questions can be told apart: how long the reductions take to find, and how much shorter a
//! solve they leave behind.
use std::convert::TryInto;
use std::fs::File;
use std::io::Read;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use relp::algorithm::two_phase::matrix_provider::MatrixProvider;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
use relp::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::LUDecomposition;
use relp::algorithm::{OptimizationResult, SolveRelaxation};
use relp::data::linear_program::general_form::{GeneralForm, Limits, Presolved, RuleId};
use relp::io::mps::parse_fixed;
use relp_num::{Rational64, RationalBig};

type T = RationalBig;

const INSTANCES: &[&str] = &[
    "ADLITTLE", "AFIRO", "AGG", "AGG2", "AGG3", "BEACONFD", "BLEND", "BOEING1", "BOEING2",
    "BORE3D", "BRANDY", "CAPRI", "CZPROB", "E226", "ETAMACRO", "FINNIS", "GFRD-PNC", "ISRAEL",
    "KB2", "LOTFI", "SC50A", "SC50B", "SC105", "SC205", "SCAGR7", "SCRS8", "SCSD1", "SCORPION",
    "SCTAP1", "STANDATA", "STANDMPS", "STOCFOR1", "SHARE1B", "SHARE2B", "RECIPELP", "VTP-BASE",
];

/// How long each phase took, and how big the problem was when the solver got it.
struct Run {
    presolve: Duration,
    solve: Duration,
    postsolve: Duration,
    rows: usize,
    columns: usize,
}

fn read(name: &str) -> GeneralForm<T> {
    let path = PathBuf::from("tests/netlib/problem_files")
        .join(name)
        .with_extension("SIF");
    let mut program = String::new();
    File::open(&path).unwrap().read_to_string(&mut program).unwrap();
    parse_fixed::<Rational64>(&program).unwrap().try_into().unwrap()
}

/// Run the whole pipeline once, presolving first when `limits` says to.
fn run(mut general: GeneralForm<T>, limits: Option<Limits<T>>) -> Option<Run> {
    let start = Instant::now();
    if let Some(limits) = limits
        && general.presolve_with::<false>(&limits, true) != Presolved::Reduced
    {
        return None;
    }
    let presolve = start.elapsed();

    let rows = general.nr_constraints();
    let columns = general.nr_variables();

    let start = Instant::now();
    let constraint_type_counts = general.standardize();
    let data = general.derive_matrix_data(constraint_type_counts);
    let OptimizationResult::FiniteOptimum(optimum) = data.solve_relaxation::<Carry<T, LUDecomposition<_>>>()
    else {
        return None;
    };
    let reconstructed = data.reconstruct_solution(optimum.primal);
    let solve = start.elapsed();

    let start = Instant::now();
    let _ = general.compute_full_solution_with_reduced_solution::<T>(reconstructed);
    let postsolve = start.elapsed();

    Some(Run { presolve, solve, postsolve, rows, columns })
}

/// Which rule is responsible for a slowdown: solve once per rule left out.
///
/// Presolving is cheap enough that a rule costing time is almost never costing it here. It is the
/// problem the rule leaves behind that the solver then has to work through.
fn blame(name: &str) {
    let original = read(name);

    let baseline = run(original.clone(), Some(Limits::default())).expect("the instance solves");
    let all_off = catch_unwind(AssertUnwindSafe(|| run(original.clone(), None)));
    println!("{name}: {:.2?} presolved, {:.2?} not presolved", baseline.solve, match &all_off {
        Ok(Some(off)) => format!("{:.2?}", off.solve),
        _ => "failed".to_string(),
    });

    let mut rules: Vec<(RuleId, Duration)> = Vec::new();
    for rule in [
        RuleId::EmptyRow, RuleId::EmptyColumn, RuleId::FixedColumn, RuleId::SingletonRow,
        RuleId::SingletonColumn, RuleId::DomainPropagation, RuleId::ImpliedFreeColumnSingleton,
        RuleId::DoubletonEquation, RuleId::DuplicateRow, RuleId::DuplicateColumn,
        RuleId::DominatedColumn,
    ] {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            run(original.clone(), Some(Limits::without(rule)))
        }));
        if let Ok(Some(without)) = outcome {
            rules.push((rule, without.solve));
        }
    }

    rules.sort_by_key(|&(_, solve)| solve);
    for (rule, solve) in rules {
        let change = solve.as_secs_f64() / baseline.solve.as_secs_f64();
        println!("  without {:<24} {:>9} {:>7.2}x", rule.name(), format!("{:.2?}", solve), change);
    }
}

/// The limits the "on" column runs under.
///
/// `RELP_WITHOUT` names a rule to leave out, so that the suite can be timed against a catalogue
/// with one reduction missing rather than against no presolving at all.
fn limits() -> Limits<T> {
    match std::env::var("RELP_WITHOUT") {
        Err(_) => Limits::default(),
        Ok(name) => {
            let rule = [
                RuleId::EmptyRow, RuleId::EmptyColumn, RuleId::FixedColumn, RuleId::SingletonRow,
                RuleId::SingletonColumn, RuleId::DomainPropagation,
                RuleId::ImpliedFreeColumnSingleton, RuleId::DoubletonEquation,
                RuleId::DuplicateRow, RuleId::DuplicateColumn, RuleId::DominatedColumn,
            ]
            .into_iter()
            .find(|rule| rule.name() == name)
            .unwrap_or_else(|| panic!("no rule is called {name:?}"));
            Limits::without(rule)
        }
    }
}

fn main() {
    // An instance the solver cannot finish without presolving is a data point, not a crash to
    // report, so the backtrace is kept quiet and the row says so instead.
    std::panic::set_hook(Box::new(|_| {}));

    let mut only: Vec<String> = std::env::args().skip(1).collect();
    if only.first().map(String::as_str) == Some("blame") {
        for name in &only[1..] {
            blame(name);
        }
        return;
    }
    only.retain(|name| !name.is_empty());
    // Any name on the command line, so that instances the suite skips can be timed too.
    let instances: Vec<&str> = if only.is_empty() {
        INSTANCES.to_vec()
    } else {
        only.iter().map(String::as_str).collect()
    };
    // Some instances are only worth timing presolved, because the other path does not finish.
    let skip_off = std::env::var("RELP_SKIP_OFF").is_ok();

    println!(
        "{:<10} {:>9} {:>9} {:>9} {:>9} {:>11} {:>11} {:>7}",
        "instance", "read", "presolve", "solve", "post", "total on", "total off", "speedup",
    );

    let (mut total_on, mut total_off) = (Duration::ZERO, Duration::ZERO);
    for name in instances {
        let start = Instant::now();
        let original = read(name);
        let read_time = start.elapsed();

        let on = catch_unwind(AssertUnwindSafe(|| run(original.clone(), Some(limits()))));
        let off = if skip_off {
            Ok(None)
        } else {
            catch_unwind(AssertUnwindSafe(|| run(original, None)))
        };

        let (Ok(Some(on)), Ok(off)) = (on, off) else {
            println!("{name:<10} {:>9} did not complete", format!("{:.0?}", read_time));
            continue;
        };

        let with = read_time + on.presolve + on.solve + on.postsolve;
        total_on += with;

        let without = match &off {
            Some(off) => {
                let without = read_time + off.solve + off.postsolve;
                total_off += without;
                format!("{:.2?}", without)
            }
            _ => {
                // Charge the on-presolve time so the totals stay comparable, and say so.
                total_off += with;
                "failed".to_string()
            }
        };
        let speedup = match &off {
            Some(off) => format!(
                "{:.2}x",
                (read_time + off.solve + off.postsolve).as_secs_f64() / with.as_secs_f64(),
            ),
            _ => "-".to_string(),
        };

        println!(
            "{name:<10} {:>9} {:>9} {:>9} {:>9} {:>11} {:>11} {:>7}   ({}x{} rows x columns left)",
            format!("{:.0?}", read_time),
            format!("{:.0?}", on.presolve),
            format!("{:.0?}", on.solve),
            format!("{:.0?}", on.postsolve),
            format!("{:.2?}", with),
            without,
            speedup,
            on.rows,
            on.columns,
        );
    }

    println!(
        "\ntotal with presolve {:.2?}, without {:.2?}, overall {:.2}x",
        total_on,
        total_off,
        total_off.as_secs_f64() / total_on.as_secs_f64(),
    );
}
