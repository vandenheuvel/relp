//! # One test per defect this code has had
//!
//! Kept together rather than beside the rule each concerns, so that the set of things that have
//! gone wrong here can be read in one place.
use relp_num::R32;

use crate::data::linear_program::elements::Objective;
use crate::data::linear_program::general_form::{Limits, Presolved};
use crate::data::linear_program::general_form::presolve::test::{
    equal, greater, less, problem, range, variable,
};


/// A variable that presolve removed still has to respect its own bounds.
///
/// `min x` subject to `x + s >= 0`, `x >= 5`, `s >= 0`.
///
/// `s` is a column singleton with no cost, so it is eliminated. Because the relation is `>=`, the
/// coefficient positive and `s` unbounded above, `s` can absorb any residual and the row goes with
/// it. The value recovered for `s` is then whatever the rest of the row leaves over, which is
/// `-5` — outside its own bounds, and so in need of clamping.
#[test]
fn a_removed_variable_respects_its_own_bounds() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, 1]],
        vec![greater()],
        vec![0],
        vec![variable(1, Some(5), None), variable(0, Some(0), None)],
    );

    let Presolved::Solved(solution) = initial.presolve::<false>() else {
        panic!("presolve determines every variable here");
    };

    assert_eq!(solution.solution_values[0].1, R32!(5));
    assert_eq!(
        solution.solution_values[1].1,
        R32!(0),
        "`s` has lower bound 0, so the residual of -5 has to be clamped to it",
    );
}

/// An infeasible problem is infeasible, even when it also has an unbounded column.
///
/// The single row has no coefficients and demands `0 == 1`, so there is no feasible point at all.
/// The single column is in no row and improves the objective without limit, but unboundedness is
/// a property of a problem that has a feasible point.
#[test]
fn infeasibility_takes_precedence_over_unboundedness() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![0]],
        vec![equal()],
        vec![1],
        vec![variable(1, None, None)],
    );

    assert_eq!(initial.presolve::<false>(), Presolved::Infeasible);
}

/// Eliminating a slack from a ranged row has to account for the width of the range.
///
/// `0 <= -2 x0 + 2 x1 + x2 <= 2` with `x1 <= -3`. Eliminating `x1`, whose contribution `2 x1`
/// covers `(-inf, -6]`, leaves `-2 x0 + x2 >= b - r - c u = 2 - 2 + 6 = 6`. Dropping the range
/// width would leave `>= 8`, which cuts off part of the feasible region.
#[test]
fn eliminating_a_slack_from_a_ranged_row_keeps_the_range_width() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![-2, 2, 1]],
        vec![range(2)],
        vec![2],
        vec![
            variable(1, Some(-1), Some(2)),
            variable(0, None, Some(-3)),
            variable(1, Some(3), None),
        ],
    );

    let Presolved::Solved(solution) = initial.presolve::<false>() else {
        panic!("the rules between them determine every variable here");
    };

    // Dropping the range width would leave `-2 x0 + x2 >= 8` and an answer of 5.
    assert_eq!(solution.objective_value, R32!(3));
    assert_eq!(solution.solution_values[0].1, R32!(-1));
    assert_eq!(solution.solution_values[1].1, R32!(-3));
    assert_eq!(solution.solution_values[2].1, R32!(4));
}

/// A ranged row whose activity can only just reach the far end of the range is forcing.
///
/// `2 <= x + 2 y <= 5` with `x in [1, 2]` and `y in [2, 4]`: the least the row can be is
/// `1 + 4 = 5`, which is the upper end of the range, so both variables are pinned. Reading that
/// combination as impossible, on the grounds that it would need a range of zero, is wrong; the
/// range here is 3.
#[test]
fn a_ranged_row_can_be_forcing_at_the_far_end_of_its_range() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, 2]],
        vec![range(3)],
        vec![5],
        vec![variable(1, Some(1), Some(2)), variable(1, Some(2), Some(4))],
    );

    let Presolved::Solved(solution) = initial.presolve::<false>() else {
        panic!("a forcing row determines every variable in it");
    };
    assert_eq!(solution.solution_values[0].1, R32!(1));
    assert_eq!(solution.solution_values[1].1, R32!(2));
    assert_eq!(solution.objective_value, R32!(3));
}

/// A `<=` row is not redundant just because nothing bounds its activity from below.
///
/// `x + y <= 1` with both variables in `[0, 1]`. The lowest activity is 0, which the row admits,
/// but the row has no lower end for that to be news about. Treating the absent end as satisfied
/// and dropping the row loses the constraint entirely.
#[test]
fn a_row_is_not_redundant_because_of_the_side_it_does_not_have() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, 1]],
        vec![less()],
        vec![1],
        vec![variable(-1, Some(0), Some(1)), variable(-1, Some(0), Some(1))],
    );

    let Presolved::Solved(solution) = initial.presolve::<false>() else {
        panic!("the rules between them determine every variable here");
    };

    // Both variables would go to their upper bound, for -2, if the row were dropped.
    assert_eq!(solution.objective_value, R32!(-1));
}

/// Holding a dominated variable at a bound means substituting that value into its rows.
///
/// `min 3 x0 + 3 x1` subject to `x0 - x1 <= 0` and `x0 >= -1`, with `x1` free. Lowering `x0` only
/// loosens the row and improves the objective, so `x0` is held at `-1`. The row then reads
/// `-x1 <= 1`, giving `x1 >= -1` and an optimum of `-6`. Removing the column without moving the
/// right hand side would leave `-x1 <= 0`, and an answer of `-3`.
#[test]
fn fixing_a_dominated_variable_moves_the_right_hand_side() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, -1]],
        vec![less()],
        vec![0],
        vec![variable(3, Some(-1), None), variable(3, None, None)],
    );

    let Presolved::Solved(solution) = initial.presolve::<false>() else {
        panic!("presolve determines every variable here");
    };
    assert_eq!(solution.objective_value, R32!(-6));
    assert_eq!(solution.solution_values[0].1, R32!(-1));
    assert_eq!(solution.solution_values[1].1, R32!(-1));
}

/// A bound that carries a column that is gone is not weighed against what it costs.
///
/// `min -x` subject to `x - y == 0` and `x + u == 5000`, with `x <= 2048`, `y <= 2047` and
/// `u <= 5000`. The first equality writes `y` as `x`, so `y <= 2047` has to become `x <= 2047`:
/// with the row and the column gone, that bound is the only thing left saying what `y` may be.
/// It is one tighter than the bound `x` already has, on a domain 2048 wide, so a threshold that
/// asks whether a tightening is worth its denominator would decline it — and the answer would
/// become `-2048`, at which `y` is 2048 and outside the bounds it was given. Whether the
/// substitution was worth doing is a question that has already been answered by the time this
/// bound is written.
#[test]
fn a_bound_that_carries_a_removed_column_is_not_weighed() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, -1, 0], vec![1, 0, 1]],
        vec![equal(), equal()],
        vec![0, 5000],
        vec![
            variable(-1, Some(0), Some(2048)),
            variable(0, Some(0), Some(2047)),
            variable(0, Some(0), Some(5000)),
        ],
    );

    let Presolved::Solved(solution) = initial.presolve::<false>() else {
        panic!("presolve determines every variable here");
    };

    assert_eq!(solution.objective_value, R32!(-2047));
    assert_eq!(
        solution.solution_values[1].1,
        R32!(2047),
        "`y` is what the substituted bound was protecting",
    );
}

/// A row that pins every variable it has is not gone until those bounds are there.
///
/// `min x + y` subject to `x + y >= 4` with `x <= 2` and `y <= 2`. The row can reach 4 and no
/// further, so it holds both variables at 2 and has nothing left to say afterwards. What it says
/// has to end up in the bounds, and a rule that removed the row and then let a limit decline them
/// one at a time would leave two variables the objective sends to 0.
#[test]
fn a_forcing_row_is_only_gone_once_it_has_been_written_into_the_bounds() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, 1]],
        vec![greater()],
        vec![4],
        vec![variable(1, Some(0), Some(2)), variable(1, Some(0), Some(2))],
    );

    // No bound a row implies may be tightened at all, which is the sharpest form of the question:
    // whatever the rule decides to do, what is left has to be worth the same.
    let limits = Limits {
        max_bound_tightenings: 0,
        ..Limits::default()
    };

    let Presolved::Solved(solution) = initial.presolve_with::<false>(&limits, true) else {
        panic!("presolve determines every variable here");
    };

    assert_eq!(solution.objective_value, R32!(4));
    assert_eq!(solution.solution_values[0].1, R32!(2));
    assert_eq!(solution.solution_values[1].1, R32!(2));
}

/// Bounds that cross leave the variable with no value, which is infeasibility.
///
/// `x + y == 0` with `x >= 3` and `y >= 3` forces both to be at most `-3`, which no value of
/// either can be. Nothing else looks for two bounds passing each other, so the check has to happen
/// where a bound is written.
#[test]
fn bounds_that_cross_are_infeasible() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, 1]],
        vec![equal()],
        vec![0],
        vec![variable(1, Some(3), None), variable(1, Some(3), None)],
    );

    assert_eq!(initial.presolve::<false>(), Presolved::Infeasible);
}
