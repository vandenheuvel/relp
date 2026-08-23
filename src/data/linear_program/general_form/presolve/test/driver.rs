//! # How the rules are scheduled, and when presolving stops
use relp_num::R32;

use crate::data::linear_program::elements::Objective;
use crate::data::linear_program::general_form::presolve::test::{
    equal, greater, less, problem, variable,
};
use crate::data::linear_program::general_form::{Limits, Presolved, RuleId};

/// A problem with no shape to recognise keeps all of its rows and columns.
///
/// Deriving a tighter bound on the way is not a change of that kind: the problem still poses the
/// same question, and asking it of a smaller box is progress rather than a reduction.
#[test]
fn a_problem_with_nothing_to_reduce_keeps_its_rows_and_columns() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, 1, 1], vec![1, -1, 2], vec![2, 1, -1]],
        vec![equal(), equal(), less()],
        vec![3, 1, 4],
        vec![
            variable(1, Some(0), Some(9)),
            variable(1, Some(0), Some(9)),
            variable(1, Some(0), Some(9)),
        ],
    );

    // Three variables tied together by two equalities: nothing is a singleton, nothing is fixed,
    // and no row is redundant or forcing.
    assert_eq!(
        initial.presolve_with::<false>(&Limits::default(), false),
        Presolved::Reduced,
    );
    assert_eq!(initial.nr_constraints(), 3);
    assert_eq!(initial.nr_variables(), 3);

    let statistics = &initial.postsolve().statistics;
    assert_eq!(statistics.per_rule[RuleId::EmptyRow].applications, 0);
    assert_eq!(statistics.per_rule[RuleId::SingletonRow].applications, 0);
    assert_eq!(statistics.per_rule[RuleId::FixedColumn].applications, 0);
    assert_eq!(statistics.per_rule[RuleId::DoubletonEquation].applications, 0);
}

/// One rule feeding another is what the work lists are for.
///
/// A fixed variable empties a row, which becomes a bound on another variable, which fixes it in
/// turn. None of that is visible when each rule is offered only the problem as it started.
#[test]
fn rules_feed_each_other() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, 1, 0], vec![0, 1, 1]],
        vec![equal(), equal()],
        vec![5, 9],
        vec![
            variable(0, Some(2), Some(2)),
            variable(0, Some(0), Some(10)),
            variable(0, Some(0), Some(10)),
        ],
    );

    let Presolved::Solved(solution) = initial.presolve::<false>() else {
        panic!("the chain determines every variable");
    };
    assert_eq!(solution.solution_values[0].1, R32!(2));
    assert_eq!(solution.solution_values[1].1, R32!(3));
    assert_eq!(solution.solution_values[2].1, R32!(6));
}

/// Presolving stops even when the bounds it derives keep improving without ever arriving.
///
/// Two rows that each halve the other's slack is the standard way to make propagation converge
/// without terminating. In exact arithmetic there is no rounding to bring it to a halt, and the
/// numbers grow at every step, so the limit on how often one bound may move is what ends it.
#[test]
fn bound_tightening_that_converges_slowly_still_stops() {
    let mut initial = problem(
        Objective::Minimize,
        // `2x - y >= 0` and `2y - x >= 0` with both free below: each row lifts the other's
        // variable a little, and neither ever settles.
        &[vec![2, -1], vec![-1, 2]],
        vec![greater(), greater()],
        vec![0, 0],
        vec![variable(1, None, Some(100)), variable(1, None, Some(100))],
    );

    let outcome = initial.presolve_with::<false>(&Limits::default(), false);
    assert!(matches!(
        outcome,
        Presolved::Reduced | Presolved::Solved(_) | Presolved::Infeasible,
    ));
}

/// The work limits are what makes stopping a fact rather than a hope.
#[test]
fn a_bound_may_only_be_tightened_so_often() {
    let limits = Limits {
        max_bound_tightenings: 2,
        ..Limits::default()
    };

    let mut initial = problem(
        Objective::Minimize,
        &[vec![2, -1], vec![-1, 2]],
        vec![greater(), greater()],
        vec![0, 0],
        vec![variable(1, None, Some(100)), variable(1, None, Some(100))],
    );
    initial.presolve_with::<false>(&limits, false);

    let tightenings = initial
        .postsolve()
        .reductions()
        .iter()
        .filter(|reduction| {
            matches!(
                reduction,
                crate::data::linear_program::general_form::Reduction::BoundTightened {
                    old: Some(_),
                    ..
                },
            )
        })
        .count();
    assert!(
        tightenings <= 2 * 2 * 2,
        "at most two moves per bound, per direction, per variable, but there were {}",
        tightenings,
    );
}

/// Presolving records what it did, so that a rule can be judged on what it bought.
#[test]
fn what_was_done_is_counted() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, 1], vec![1, 0]],
        vec![less(), equal()],
        vec![10, 3],
        vec![variable(1, Some(0), Some(9)), variable(1, Some(0), Some(9))],
    );
    initial.presolve::<false>();

    let statistics = &initial.postsolve().statistics;
    assert_eq!(statistics.before.rows, 2);
    assert_eq!(statistics.before.columns, 2);
    assert!(statistics.rounds >= 1);
    assert!(
        statistics
            .per_rule
            .iter()
            .any(|(_, rule)| rule.applications > 0),
        "something was applied, so something has to have been counted",
    );
}

/// A problem presolving answers outright reports the answer, not a problem to solve.
#[test]
fn a_problem_that_is_fully_determined_is_reported_as_solved() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, 0], vec![0, 1]],
        vec![equal(), equal()],
        vec![3, 4],
        vec![variable(2, None, None), variable(5, None, None)],
    );

    let Presolved::Solved(solution) = initial.presolve::<false>() else {
        panic!("both rows fix their variable");
    };
    assert_eq!(solution.objective_value, R32!(2 * 3 + 5 * 4));
}
