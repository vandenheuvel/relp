//! # Differential testing of the presolve step
//!
//! Presolve must not change the answer. These tests solve the same randomly generated linear
//! programs twice, once with the presolve step and once without it, and compare the outcomes
//! exactly. Because everything is done in rational arithmetic, "exactly" means what it says: there
//! is no tolerance to hide a discrepancy behind.
use relp_num::RationalBig;

use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::basis_inverse_rows::BasisInverseRows;
use crate::data::linear_algebra::matrix::{ColumnMajor, MatrixOrder};
use crate::data::linear_algebra::vector::{DenseVector, Vector};
use crate::data::linear_program::elements::{
    Objective, RangedConstraintRelation, VariableType,
};
use crate::data::linear_program::general_form::{
    GeneralForm, Limits, Presolved, RuleId, Solved, Variable,
};
use crate::data::linear_program::solution::Solution;

type T = RationalBig;

/// What solving a linear program concluded, without the solution values.
///
/// The values themselves are compared separately where that is meaningful; an optimal solution is
/// not generally unique, but the optimal value is.
#[derive(Clone, Eq, PartialEq, Debug)]
enum Outcome {
    Optimum(T),
    Infeasible,
    Unbounded,
    /// Unbounded if the problem has a feasible point, which was not established.
    ///
    /// Presolve reaches this when it finds a variable nothing holds back but is left with a
    /// problem whose feasibility it has not settled. It is a weaker claim than the other three,
    /// not a different one, so it is compatible with both of the answers it does not distinguish.
    UnboundedIfFeasible,
}

impl Outcome {
    /// Whether two outcomes can both be right about the same problem.
    fn agrees_with(&self, other: &Self) -> bool {
        match (self, other) {
            (Outcome::UnboundedIfFeasible, Outcome::Unbounded | Outcome::Infeasible)
            | (Outcome::Unbounded | Outcome::Infeasible, Outcome::UnboundedIfFeasible) => true,
            (left, right) => left == right,
        }
    }
}

/// Solve a problem all the way to an outcome, optionally presolving it first.
///
/// Any solution produced is checked against the problem as it was given, exactly. An objective
/// value that matches while the values themselves are infeasible is exactly the kind of thing
/// comparing two pipelines against each other cannot see.
fn solve(general: GeneralForm<T>, presolve: bool) -> Outcome {
    let original = general.clone();
    let outcome = solve_with(general, presolve.then(Limits::default));

    if let Outcome::Optimum(_) = &outcome {
        let solution = LAST_SOLUTION.with(|last| last.borrow_mut().take());
        if let Some(solution) = solution {
            original
                .verify_solution(&solution)
                .expect("the solution recovered has to solve the problem that was given");
        }
    }

    outcome
}

thread_local! {
    /// The most recent solution produced, so that it can be checked against the original problem.
    static LAST_SOLUTION: std::cell::RefCell<Option<Solution<T>>> =
        const { std::cell::RefCell::new(None) };
}

/// Solve, presolving first with the given limits when there are any.
fn solve_with(mut general: GeneralForm<T>, limits: Option<Limits<T>>) -> Outcome {
    if let Some(limits) = limits {
        match general.presolve_with::<false>(&limits, true) {
            Presolved::Reduced => {}
            Presolved::Solved(solution) => {
                let objective_value = solution.objective_value.clone();
                LAST_SOLUTION.with(|last| *last.borrow_mut() = Some(solution));
                return Outcome::Optimum(objective_value);
            }
            Presolved::Infeasible => return Outcome::Infeasible,
            Presolved::Unbounded {
                assuming_feasible: false,
            } => return Outcome::Unbounded,
            Presolved::Unbounded {
                assuming_feasible: true,
            }
            | Presolved::InfeasibleOrUnbounded => return Outcome::UnboundedIfFeasible,
        }
    }

    solve_reduced(general)
}

/// Solve a problem that has already been presolved, or that is not being presolved at all.
///
/// Through the narrowed problem rather than the wide one, so that every case this harness
/// generates goes through the width choice and the scaling it does on the way in.
fn solve_reduced(mut general: GeneralForm<T>) -> Outcome {
    let constraint_type_counts = general.standardize();
    let reduced = {
        let narrowed = general.narrow(constraint_type_counts);
        match narrowed.solve_relaxation::<Carry<T, BasisInverseRows<_>>>() {
            Solved::Optimum(solution) => solution.primal,
            Solved::Infeasible => return Outcome::Infeasible,
            Solved::Unbounded => return Outcome::Unbounded,
        }
    };

    let solution = general.compute_full_solution_with_reduced_solution::<T>(reduced);
    let objective_value = solution.objective_value.clone();
    LAST_SOLUTION.with(|last| *last.borrow_mut() = Some(solution));
    Outcome::Optimum(objective_value)
}

/// A small, reproducible pseudo random number generator.
///
/// Deterministic by design: a failing case must be reproducible from its seed alone, without a
/// recorded corpus.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
    /// A value in `lower..=upper`.
    fn between(&mut self, lower: i64, upper: i64) -> i64 {
        lower + self.below((upper - lower + 1) as u64) as i64
    }
    /// A value in `lower..=upper`, a quarter of the time over a small denominator.
    ///
    /// Data that is not integral is what several things here only have to deal with once it turns
    /// up: `standardize` shifts by a variable's bound and puts the denominator into the right hand
    /// side, and clearing the fractions of a row is a reduction of its own.
    fn fraction(&mut self, lower: i64, upper: i64) -> T {
        let numerator = self.between(lower, upper);
        match self.below(4) {
            0 => T::from((numerator, 2_i64)),
            1 => T::from((numerator, 3_i64)),
            _ => T::from(numerator),
        }
    }
}

/// Build a small linear program.
///
/// The generator aims at the case combinations that the rules branch on: all four constraint
/// relations, both signs of coefficient, and every pattern of present and absent variable bounds.
fn random_problem(rng: &mut Rng) -> GeneralForm<T> {
    let nr_constraints = rng.below(3) as usize + 1;
    let nr_variables = rng.below(3) as usize + 1;

    let rows = (0..nr_constraints)
        .map(|_| {
            let mut row = (0..nr_variables)
                .map(|_| rng.fraction(-2, 2))
                .collect::<Vec<_>>();

            // A row of nothing but zeros is not a constraint, and handing one to the solver
            // without presolving first reaches a defect of its own: a constraint matrix of rank
            // zero leaves it with no rows at all, which it does not support. That is worth fixing
            // and is nothing to do with presolving, which drops such a row on sight.
            if row.iter().all(num_traits::Zero::is_zero) {
                row[rng.below(nr_variables as u64) as usize] = T::from(1);
            }

            row
        })
        .collect::<Vec<_>>();
    let constraints = ColumnMajor::from_test_data::<T, T, _>(&rows, nr_variables);

    let constraint_types = (0..nr_constraints)
        .map(|_| match rng.below(4) {
            0 => RangedConstraintRelation::Equal,
            1 => RangedConstraintRelation::Less,
            2 => RangedConstraintRelation::Greater,
            _ => RangedConstraintRelation::Range(rng.fraction(1, 3)),
        })
        .collect();

    let b = DenseVector::new(
        (0..nr_constraints).map(|_| rng.fraction(-4, 4)).collect(),
        nr_constraints,
    );

    let variables = (0..nr_variables)
        .map(|_| {
            // Every combination of present and absent bounds, with the lower never above the
            // upper: a problem that is trivially infeasible on its bounds is rejected before
            // presolve ever sees it, so generating one would only test the reader.
            let (lower_bound, upper_bound) = match rng.below(4) {
                0 => (None, None),
                1 => (Some(rng.fraction(-3, 3)), None),
                2 => (None, Some(rng.fraction(-3, 3))),
                _ => {
                    let lower = rng.fraction(-3, 3);
                    let upper = &lower + &rng.fraction(0, 3);
                    (Some(lower), Some(upper))
                }
            };

            Variable {
                variable_type: VariableType::Continuous,
                cost: rng.fraction(-3, 3),
                lower_bound,
                upper_bound,
                shift: T::from(0),
                flipped: false,
            }
        })
        .collect();

    let variable_names = (0..nr_variables).map(|j| format!("x{}", j)).collect();

    GeneralForm::new(
        if rng.below(2) == 0 {
            Objective::Minimize
        } else {
            Objective::Maximize
        },
        constraints,
        constraint_types,
        b,
        variables,
        variable_names,
        T::from(0),
    )
}

/// Presolving must not change the outcome of any problem.
///
/// This is the strongest check there is on presolve, and the cheapest: the two paths differ only
/// in whether the rules ran, and the comparison is exact.
#[test]
fn presolve_preserves_the_outcome() {
    const NR_PROBLEMS: u64 = 2_000;

    let mut mismatches = Vec::new();
    for seed in 0..NR_PROBLEMS {
        let problem = random_problem(&mut Rng(seed));

        let with = solve(problem.clone(), true);
        let without = solve(problem, false);

        if !with.agrees_with(&without) {
            mismatches.push((seed, with, without));
        }
    }

    assert!(
        mismatches.is_empty(),
        "{} of {} problems changed outcome when presolved. \
         Reproduce one with `SEED=<seed> cargo test show_reductions -- --ignored --nocapture`. \
         First few (seed, presolved, truth): {:?}",
        mismatches.len(),
        NR_PROBLEMS,
        &mismatches[..mismatches.len().min(5)],
    );
}

/// Presolving must not change the outcome when it also has to respect integrality.
///
/// The reductions are a different set, because several of them have to decline a substitution
/// that would give an integer variable a fractional value. What may not differ is the answer for
/// the continuous problem, which is what is being solved here either way.
#[test]
fn presolve_respecting_integrality_preserves_the_outcome() {
    const NR_PROBLEMS: u64 = 500;

    let mut mismatches = Vec::new();
    for seed in 0..NR_PROBLEMS {
        let mut problem = random_problem(&mut Rng(seed));
        let truth = solve(problem.clone(), false);

        // Every bound in a generated problem is integral, so respecting integrality does not
        // change what the continuous problem admits, only which rules are willing to fire.
        let reduced = match problem.presolve::<true>() {
            Presolved::Reduced => solve_reduced(problem),
            Presolved::Solved(solution) => Outcome::Optimum(solution.objective_value),
            Presolved::Infeasible => Outcome::Infeasible,
            Presolved::Unbounded {
                assuming_feasible: false,
            } => Outcome::Unbounded,
            Presolved::Unbounded { .. } | Presolved::InfeasibleOrUnbounded => {
                Outcome::UnboundedIfFeasible
            }
        };

        if !reduced.agrees_with(&truth) {
            mismatches.push((seed, reduced, truth));
        }
    }

    assert!(
        mismatches.is_empty(),
        "{} of {} problems changed outcome. First few: {:?}",
        mismatches.len(),
        NR_PROBLEMS,
        &mismatches[..mismatches.len().min(5)],
    );
}

/// No limit may change the answer, only how much of it presolve got round to working out.
///
/// A limit says that a reduction is not worth what it costs. For that to be a statement about
/// effort rather than about the problem, a rule it stops has to be stopped before it has begun:
/// one cut off partway would leave a row removed and the bound that took its place unwritten,
/// which is a problem that admits points the one that came in does not.
///
/// The settings below are chosen to be awkward rather than sensible. Each turns something down
/// far enough that rules run into it constantly, so that a rule which can be interrupted is
/// interrupted here.
#[test]
fn limits_do_not_change_the_outcome() {
    const NR_PROBLEMS: u64 = 500;

    let settings: Vec<(&str, Limits<T>)> = vec![
        ("default", Limits::default()),
        ("no tightening at all", Limits {
            max_bound_tightenings: 0,
            ..Limits::default()
        }),
        ("one tightening per bound", Limits {
            max_bound_tightenings: 1,
            ..Limits::default()
        }),
        ("every tightening is worth it", Limits {
            min_relative_tightening: T::from(0),
            ..Limits::default()
        }),
        // Nothing short of the whole domain counts as an improvement, so every bound a row implies
        // is declined.
        ("no tightening is worth it", Limits {
            min_relative_tightening: T::from(1),
            ..Limits::default()
        }),
        ("no substitution may create a coefficient", Limits {
            max_fill_in: 0,
            ..Limits::default()
        }),
        ("one round", Limits {
            max_rounds: 1,
            ..Limits::default()
        }),
    ];

    let mut mismatches = Vec::new();
    for seed in 0..NR_PROBLEMS {
        let problem = random_problem(&mut Rng(seed));
        let truth = solve(problem.clone(), false);

        for (name, limits) in &settings {
            let outcome = solve_with(problem.clone(), Some(limits.clone()));

            // An objective value that matches while the values themselves are infeasible is
            // exactly what comparing two answers with each other cannot see.
            if let Outcome::Optimum(_) = &outcome
                && let Some(solution) = LAST_SOLUTION.with(|last| last.borrow_mut().take())
                && let Err(invalid) = problem.verify_solution(&solution)
            {
                panic!("seed {seed}, {name}: the solution recovered does not solve the problem \
                        that was given: {invalid:?}");
            }

            if !outcome.agrees_with(&truth) {
                mismatches.push((seed, *name, outcome, truth.clone()));
            }
        }
    }

    assert!(
        mismatches.is_empty(),
        "{} runs under a limit disagreed with the unpresolved answer. \
         First few (seed, limits, under the limit, truth): {:?}",
        mismatches.len(),
        &mismatches[..mismatches.len().min(5)],
    );
}

/// Eliminating a slack from a ranged row must account for the width of the range.
///
/// `min x0 + x2` subject to `0 <= -2 x0 + 2 x1 + x2 <= 2`, `x0 in [-1, 2]`, `x1 <= -3`,
/// `x2 >= 3`. The optimum is 3, at `x0 = -1`, `x1 = -3`, `x2 = 4`.
///
/// `x1` is a zero-cost column singleton, so the slack rule eliminates it. With coefficient
/// `c = 2 > 0` and only an upper bound `u = -3`, the term `c x1` covers `(-inf, c u]`, so some
/// `x1` satisfies the row exactly when `<a, x> + c u >= b - r`, that is
/// `-2 x0 + x2 >= b - r - c u = 6`. The rule drops the `- r` and leaves `-2 x0 + x2 >= 8`, which
/// cuts the optimum off; the reduced problem then reports 5.
///
/// Found by [`presolve_preserves_the_outcome`].
#[test]
fn eliminating_a_slack_from_a_ranged_row_keeps_the_range_width() {
    let problem = GeneralForm::new(
        Objective::Minimize,
        ColumnMajor::from_test_data::<T, T, _>(&[vec![-2, 2, 1]], 3),
        vec![RangedConstraintRelation::Range(T::from(2))],
        DenseVector::new(vec![T::from(2)], 1),
        vec![
            bounded_variable(1, Some(-1), Some(2)),
            bounded_variable(0, None, Some(-3)),
            bounded_variable(1, Some(3), None),
        ],
        vec!["x0".to_string(), "x1".to_string(), "x2".to_string()],
        T::from(0),
    );

    assert_eq!(solve(problem.clone(), false), Outcome::Optimum(T::from(3)));
    assert_eq!(solve(problem, true), Outcome::Optimum(T::from(3)));
}

/// Every reduction, so that a test can insist it reached all of them.
const ALL_REDUCTIONS: [&str; 14] = [
    "empty column",
    "empty row",
    "fixed column",
    "singleton row",
    "bound tightened",
    "row changed",
    "row scaled",
    "redundant row",
    "column singleton",
    "implied free singleton",
    "doubleton equation",
    "duplicate row",
    "duplicate column",
    "dominated column",
];

/// The name of a reduction's variant, for coverage counting.
fn reduction_name(reduction: &crate::data::linear_program::general_form::Reduction<T>) -> &'static str {
    use crate::data::linear_program::general_form::Reduction::*;
    match reduction {
        EmptyColumn { .. } => "empty column",
        EmptyRow { .. } => "empty row",
        FixedColumn { .. } => "fixed column",
        SingletonRow { .. } => "singleton row",
        BoundTightened { .. } => "bound tightened",
        RowChanged { .. } => "row changed",
        RowScaled { .. } => "row scaled",
        RedundantRow { .. } => "redundant row",
        ColumnSingleton { .. } => "column singleton",
        ImpliedFreeColumnSingleton { .. } => "implied free singleton",
        DoubletonEquation { .. } => "doubleton equation",
        DuplicateRow { .. } => "duplicate row",
        DuplicateColumn { .. } => "duplicate column",
        DominatedColumn { .. } => "dominated column",
    }
}

/// Shorthand for a continuous variable with the given cost and bounds.
fn bounded_variable(cost: i64, lower: Option<i64>, upper: Option<i64>) -> Variable<T> {
    Variable {
        variable_type: VariableType::Continuous,
        cost: T::from(cost),
        lower_bound: lower.map(T::from),
        upper_bound: upper.map(T::from),
        shift: T::from(0),
        flipped: false,
    }
}

/// Every rule, so that one can be blamed at a time.
const ALL_RULES: [RuleId; 10] = [
    RuleId::EmptyRow,
    RuleId::EmptyColumn,
    RuleId::FixedColumn,
    RuleId::SingletonRow,
    RuleId::SingletonColumn,
    RuleId::ImpliedFreeColumnSingleton,
    RuleId::DomainPropagation,
    RuleId::DoubletonEquation,
    RuleId::DuplicateRow,
    RuleId::DuplicateColumn,
];

/// For each rule on its own, how many problems it gets wrong.
///
/// A rule that changes an outcome by itself is the one to look at; running them all together only
/// says that one of them is wrong.
#[test]
#[ignore = "Diagnostic helper, not an assertion."]
fn blame_a_rule() {
    const NR_PROBLEMS: u64 = 2_000;

    for rule in ALL_RULES {
        let mut wrong = Vec::new();
        for seed in 0..NR_PROBLEMS {
            let problem = random_problem(&mut Rng(seed));
            let truth = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                solve_with(problem.clone(), None)
            }));
            let reduced = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                solve_with(problem, Some(Limits::only(rule)))
            }));

            match (truth, reduced) {
                (Ok(truth), Ok(reduced)) if !truth.agrees_with(&reduced) => {
                    wrong.push((seed, reduced, truth));
                }
                (Ok(_), Ok(_)) => {}
                _ => {}
            }
        }
        println!(
            "{:<28} {:>4} wrong   {:?}",
            rule.name(),
            wrong.len(),
            &wrong[..wrong.len().min(3)],
        );
    }
}

/// For the first failing problems, which rule has to be switched off to make them pass.
///
/// Every rule is right on its own, so what is left is rules interacting. Removing one at a time
/// from the full set says which one is involved.
#[test]
#[ignore = "Diagnostic helper, not an assertion."]
fn blame_an_interaction() {
    const NR_PROBLEMS: u64 = 2_000;
    let mut reported = 0;

    for seed in 0..NR_PROBLEMS {
        if reported >= 8 {
            break;
        }
        let problem = random_problem(&mut Rng(seed));

        let run = |limits: Option<Limits<T>>, problem: GeneralForm<T>| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| solve_with(problem, limits)))
        };

        let (Ok(truth), Ok(all)) = (
            run(None, problem.clone()),
            run(Some(Limits::default()), problem.clone()),
        ) else {
            continue;
        };
        if truth.agrees_with(&all) {
            continue;
        }

        let culprits = ALL_RULES
            .iter()
            .filter(|&&rule| {
                matches!(
                    run(Some(Limits::without(rule)), problem.clone()),
                    Ok(outcome) if truth.agrees_with(&outcome)
                )
            })
            .map(|rule| rule.name())
            .collect::<Vec<_>>();

        println!(
            "seed {:<6} presolved {:?} truth {:?}   fixed by removing: {:?}",
            seed, all, truth, culprits,
        );
        reported += 1;
    }
}

/// Print a problem and everything presolve did to it.
#[test]
#[ignore = "Diagnostic helper, not an assertion."]
fn show_reductions() {
    let seed: u64 = std::env::var("SEED").unwrap().parse().unwrap();
    let mut problem = random_problem(&mut Rng(seed));

    println!("rows:");
    for (row, (relation, right_hand_side)) in problem
        .constraint_types_for_test()
        .iter()
        .zip(problem.b_for_test().iter())
        .enumerate()
    {
        let terms = (0..problem.nr_variables())
            .filter_map(|column| {
                problem
                    .coefficient_for_test(row, column)
                    .map(|value| format!("{:+}*x{}", value, column))
            })
            .collect::<Vec<_>>()
            .join(" ");
        println!("  {} {} {}", terms, relation, right_hand_side);
    }
    println!("columns:");
    for column in 0..problem.nr_variables() {
        let variable = problem.variable_for_test(column);
        println!(
            "  x{}: cost {} in [{}, {}]",
            column,
            variable.cost,
            variable
                .lower_bound
                .as_ref()
                .map_or("-inf".to_string(), |v| v.to_string()),
            variable
                .upper_bound
                .as_ref()
                .map_or("+inf".to_string(), |v| v.to_string()),
        );
    }

    let outcome = problem.presolve::<false>();
    println!("outcome: {:?}", outcome);
    for reduction in problem.postsolve().reductions() {
        println!("  {:?}", reduction);
    }
}

/// Print a problem, what presolve did to it, and the certificate that came back.
#[test]
#[ignore = "Diagnostic helper, not an assertion."]
fn show_certificate() {
    let seed: u64 = std::env::var("SEED").unwrap().parse().unwrap();
    let mut problem = random_problem(&mut Rng(seed));
    let original = problem.clone();

    for (row, (relation, right_hand_side)) in problem
        .constraint_types_for_test()
        .iter()
        .zip(problem.b_for_test().iter())
        .enumerate()
    {
        let terms = (0..problem.nr_variables())
            .filter_map(|column| {
                problem
                    .coefficient_for_test(row, column)
                    .map(|value| format!("{:+}*x{}", value, column))
            })
            .collect::<Vec<_>>()
            .join(" ");
        println!("row {}:  {} {} {}", row, terms, relation, right_hand_side);
    }
    for column in 0..problem.nr_variables() {
        let variable = problem.variable_for_test(column);
        println!(
            "x{}: cost {} in [{}, {}]",
            column,
            variable.cost,
            variable
                .lower_bound
                .as_ref()
                .map_or("-inf".to_string(), |v| v.to_string()),
            variable
                .upper_bound
                .as_ref()
                .map_or("+inf".to_string(), |v| v.to_string()),
        );
    }

    println!(
        "presolve: {:?}",
        problem.presolve_with::<false>(&Limits::preserving_duals(), false),
    );
    for reduction in problem.postsolve().reductions() {
        println!("  {:?}", reduction);
    }
    println!("surviving rows {:?}", problem.postsolve().surviving_rows());
    println!(
        "surviving columns {:?}",
        problem.postsolve().surviving_columns(),
    );

    let solution = solve_with_duals(problem).expect("a certificate");
    println!("values {:?}", solution.solution_values);
    println!("duals  {:?}", solution.dual_values);
    println!("verify {:?}", original.verify_solution(&solution));
    println!("no presolve at all: {:?}", solve_with(original.clone(), None));
    println!("default presolve:   {:?}", solve_with(original, Some(Limits::default())));
}

/// A recovered dual solution has to certify that the primal one is optimal.
///
/// Feasibility says a solution is allowed; only the multipliers say it is the best one. Checking
/// them exactly is what turns "the two pipelines agree" into "the answer is right".
#[test]
fn recovered_duals_certify_optimality() {
    const NR_PROBLEMS: u64 = 1_000;

    let mut checked = 0;
    let mut failures = Vec::new();
    for seed in 0..NR_PROBLEMS {
        let problem = random_problem(&mut Rng(seed));
        let original = problem.clone();

        if let Some(solution) = solve_with_duals(problem) {
            checked += 1;
            if let Err(invalid) = original.verify_solution(&solution) {
                failures.push((seed, invalid));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} certificates did not check out. First few: {:?}",
        failures.len(),
        checked,
        &failures[..failures.len().min(5)],
    );
    // Most generated problems turn out to be infeasible or unbounded, and those have no
    // certificate to check. The point of the count is only to catch this test quietly checking
    // nothing at all.
    assert!(
        checked > NR_PROBLEMS / 10,
        "only {} of {} problems produced a certificate, which is too few for this to be saying \
         anything",
        checked,
        NR_PROBLEMS,
    );
}

/// Solve a problem, presolved or not, recovering the multipliers along with the values.
fn solve_with_duals(mut general: GeneralForm<T>) -> Option<Solution<T>> {
    let constraint_type_counts = general.standardize();
    let (reduced, duals) = {
        let narrowed = general.narrow(constraint_type_counts);
        match narrowed.solve_relaxation::<Carry<T, BasisInverseRows<_>>>() {
            Solved::Optimum(solution) => (solution.primal, solution.duals),
            _ => return None,
        }
    };

    general.compute_full_solution_with_duals(reduced, duals)
}

/// Presolving with the dual preserving rules must still leave a certificate that checks out.
///
/// The rules that run here do remove rows and columns, so this is not the same test as
/// [`recovered_duals_certify_optimality`]: it is the one that says the recovery of the multipliers
/// through those reductions is right. Every reduction but one is allowed to run, and the count at
/// the end insists that every one of them was in fact reached — a recovery nothing exercises is
/// not a recovery anyone should believe.
#[test]
fn duals_survive_dual_preserving_presolve() {
    use std::collections::BTreeMap;

    const NR_PROBLEMS: u64 = 5_000;

    let mut checked = 0;
    let mut covered: BTreeMap<&str, usize> = BTreeMap::new();
    let mut failures = Vec::new();

    // Writing the rows over a common denominator is off by default, for what it costs the
    // algorithm rather than for anything to do with the multipliers. Its own is undone like every
    // other, and asking for it here is what keeps that covered.
    let mut limits = Limits::preserving_duals();
    limits.disabled[RuleId::RowScaling] = false;

    for seed in 0..NR_PROBLEMS {
        let mut problem = random_problem(&mut Rng(seed));
        let original = problem.clone();

        if problem.presolve_with::<false>(&limits, false) != Presolved::Reduced {
            continue;
        }
        let kinds = problem
            .postsolve()
            .reductions()
            .iter()
            .map(reduction_name)
            .collect::<Vec<_>>();

        let Some(solution) = solve_with_duals(problem) else {
            continue;
        };
        checked += 1;
        if let Err(invalid) = original.verify_solution(&solution) {
            failures.push((seed, invalid));
        }
        for kind in kinds {
            *covered.entry(kind).or_default() += 1;
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} certificates did not check out. First few: {:?}",
        failures.len(),
        checked,
        &failures[..failures.len().min(5)],
    );
    assert!(checked > NR_PROBLEMS / 10, "only {} certificates", checked);

    // Fixing a dominated column is the one reduction with no dual recovery, and
    // `Limits::preserving_duals` is what keeps it out. Everything else has to have been reached.
    for kind in ALL_REDUCTIONS {
        if kind == "dominated column" {
            assert!(!covered.contains_key(kind), "a dual reduction ran");
            continue;
        }
        assert!(
            covered.contains_key(kind),
            "no certificate went through a {}, so its recovery is untested. Coverage: {:?}",
            kind,
            covered,
        );
    }
}
/// Which seeds panic, and whether presolving is what does it.
///
/// A panic on the path that does not presolve is not presolve's, and this is what says so. It is
/// how the four seeds that reach the solver's handling of a rank zero constraint matrix were
/// separated from the presolve defects this file otherwise records.
#[test]
#[ignore = "Diagnostic helper, not an assertion."]
fn find_panics() {
    const NR_PROBLEMS: u64 = 2_000;

    let mut without = Vec::new();
    let mut with = Vec::new();
    for seed in 0..NR_PROBLEMS {
        let problem = random_problem(&mut Rng(seed));
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            solve_with(problem.clone(), None)
        }))
        .is_err()
        {
            without.push(seed);
        }
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            solve_with(problem, Some(Limits::default()))
        }))
        .is_err()
        {
            with.push(seed);
        }
    }
    println!("no presolve: {} panics {:?}", without.len(), &without[..without.len().min(10)]);
    println!("presolve:    {} panics {:?}", with.len(), &with[..with.len().min(10)]);
}

