//! # What each rule does
//!
//! One or two tests per rule, aimed at the shape it recognises rather than at the driver.
use relp_num::R32;

use crate::data::linear_program::elements::Objective;
use crate::data::linear_program::elements::RangedConstraintRelation;
use crate::data::linear_program::general_form::presolve::test::{
    equal, exact_problem, greater, integer, less, problem, range, variable,
};
use crate::data::linear_program::general_form::{Limits, Presolved, RuleId};

/// A row with no coefficients holds or it does not.
#[test]
fn an_empty_row_is_checked_and_dropped() {
    let satisfied = problem(
        Objective::Minimize,
        &[vec![0]],
        vec![less()],
        vec![1],
        vec![variable(0, Some(0), Some(1))],
    );
    assert!(matches!(
        satisfied.clone().presolve::<false>(),
        Presolved::Solved(_)
    ));

    let violated = problem(
        Objective::Minimize,
        &[vec![0]],
        vec![greater()],
        vec![1],
        vec![variable(0, Some(0), Some(1))],
    );
    assert_eq!(violated.clone().presolve::<false>(), Presolved::Infeasible);
}

/// A column in no row goes to whichever of its bounds the objective prefers.
#[test]
fn an_empty_column_goes_to_its_best_bound() {
    for (objective, cost, expected) in [
        (Objective::Minimize, 2, R32!(-3)),
        (Objective::Minimize, -2, R32!(4)),
        (Objective::Maximize, 2, R32!(4)),
        (Objective::Maximize, -2, R32!(-3)),
    ] {
        let mut initial = problem(
            objective,
            &[vec![0]],
            vec![less()],
            vec![0],
            vec![variable(cost, Some(-3), Some(4))],
        );

        let Presolved::Solved(solution) = initial.presolve::<false>() else {
            panic!("nothing constrains the variable");
        };
        assert_eq!(solution.solution_values[0].1, expected);
    }
}

/// A variable whose bounds leave one value is substituted into the rows it appears in.
#[test]
fn a_fixed_column_is_substituted() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, 1]],
        vec![less()],
        vec![10],
        vec![variable(0, Some(4), Some(4)), variable(1, Some(0), None)],
    );

    let Presolved::Solved(solution) = initial.presolve::<false>() else {
        panic!("both variables are determined");
    };
    assert_eq!(solution.solution_values[0].1, R32!(4));
    // The row became `x1 <= 6`, which does not bind against minimizing `x1` from 0.
    assert_eq!(solution.solution_values[1].1, R32!(0));
}

/// A row with one coefficient becomes a bound on that variable.
#[test]
fn a_singleton_row_becomes_a_bound() {
    // `-2 x >= 4` is `x <= -2`, the relation turning around with the negative coefficient.
    let mut initial = problem(
        Objective::Minimize,
        &[vec![-2, 0], vec![1, 1]],
        vec![greater(), less()],
        vec![4, 100],
        vec![variable(-1, Some(-10), None), variable(0, Some(0), Some(1))],
    );

    let Presolved::Solved(solution) = initial.presolve::<false>() else {
        panic!("everything is determined");
    };
    assert_eq!(solution.solution_values[0].1, R32!(-2));
}

/// A ranged singleton row bounds the variable from both sides at once.
#[test]
fn a_ranged_singleton_row_gives_both_bounds() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![2, 0], vec![1, 1]],
        vec![range(4), less()],
        vec![6, 100],
        vec![variable(-1, None, None), variable(0, Some(0), Some(1))],
    );

    let Presolved::Solved(solution) = initial.presolve::<false>() else {
        panic!("everything is determined");
    };
    // `2 <= 2 x <= 6`, so `x` is in `[1, 3]` and minimizing `-x` puts it at 3.
    assert_eq!(solution.solution_values[0].1, R32!(3));
}

/// A zero cost variable in one row is eliminated, and the row weakened to match.
#[test]
fn a_column_singleton_is_eliminated() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, 1]],
        vec![equal()],
        vec![10],
        vec![variable(1, Some(0), None), variable(0, Some(0), Some(4))],
    );

    let Presolved::Solved(solution) = initial.presolve::<false>() else {
        panic!("everything is determined");
    };
    // `x0 + s == 10` with `s` in `[0, 4]` is `6 <= x0 <= 10`, so minimizing puts `x0` at 6.
    assert_eq!(solution.solution_values[0].1, R32!(6));
    assert_eq!(solution.solution_values[1].1, R32!(4));
}

/// A variable with a cost, in one equality row that already confines it, is substituted out.
#[test]
fn an_implied_free_column_singleton_is_substituted() {
    // `x0 + x1 == 5`, `x0 in [0, 5]`, `x1 in [0, 5]`: the row confines each to its own bounds.
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, 1]],
        vec![equal()],
        vec![5],
        vec![variable(3, Some(0), Some(5)), variable(1, Some(0), Some(5))],
    );

    let Presolved::Solved(solution) = initial.presolve::<false>() else {
        panic!("everything is determined");
    };
    // Minimizing `3 x0 + x1` along `x0 + x1 == 5` puts everything on the cheaper variable.
    assert_eq!(solution.objective_value, R32!(5));
    assert_eq!(solution.solution_values[0].1, R32!(0));
    assert_eq!(solution.solution_values[1].1, R32!(5));
}

/// An equality with two coefficients determines one variable from the other.
#[test]
fn a_doubleton_equation_eliminates_a_column() {
    // `2 x0 + 3 x1 == 12`, with a second row so neither column is a singleton.
    let mut initial = problem(
        Objective::Minimize,
        &[vec![2, 3], vec![1, 1]],
        vec![equal(), less()],
        vec![12, 100],
        vec![variable(1, Some(0), Some(6)), variable(1, Some(0), Some(4))],
    );

    let outcome = initial.presolve_with::<false>(&Limits::only(RuleId::DoubletonEquation), true);
    assert_eq!(outcome, Presolved::Reduced);
    assert_eq!(initial.nr_variables(), 1, "one column is eliminated");
    assert_eq!(initial.nr_constraints(), 1, "the equality is used up");
}

/// An equality with two coefficients is not used on an integer variable it would leave fractional.
#[test]
fn a_doubleton_equation_respects_integrality() {
    let build = || {
        problem(
            Objective::Minimize,
            // `2 x0 + 3 x1 == 12` gives `x0 = 6 - 3/2 x1`, which is not integral for every `x1`.
            &[vec![2, 3], vec![1, 1]],
            vec![equal(), less()],
            vec![12, 100],
            vec![integer(1, Some(0), Some(6)), integer(1, Some(0), Some(4))],
        )
    };

    let mut relaxed = build();
    relaxed.presolve_with::<false>(&Limits::only(RuleId::DoubletonEquation), true);
    assert_eq!(relaxed.nr_variables(), 1, "the relaxation may substitute");

    let mut integral = build();
    integral.presolve_with::<true>(&Limits::only(RuleId::DoubletonEquation), true);
    assert_eq!(
        integral.nr_variables(),
        2,
        "respecting integrality, the substitution has to be declined",
    );
}

/// Two rows that are multiples of each other become one.
#[test]
fn duplicate_rows_are_merged() {
    // The second row is `-2` times the first.
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, 2], vec![-2, -4]],
        vec![less(), greater()],
        vec![10, -30],
        vec![variable(1, Some(0), Some(9)), variable(1, Some(0), Some(9))],
    );

    let outcome = initial.presolve_with::<false>(&Limits::only(RuleId::DuplicateRow), true);
    assert_eq!(outcome, Presolved::Reduced);
    assert_eq!(initial.nr_constraints(), 1);
    // `x + 2y <= 10` and `-2x - 4y >= -30`, that is `x + 2y <= 15`: the first is the tighter.
    assert_eq!(initial.b_for_test()[0], R32!(10));
}

/// Two rows that contradict each other are infeasible.
#[test]
fn contradicting_duplicate_rows_are_infeasible() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, 1], vec![2, 2]],
        vec![less(), greater()],
        vec![1, 10],
        vec![variable(1, None, None), variable(1, None, None)],
    );

    // `x + y <= 1` and `x + y >= 5` cannot both hold.
    let outcome = initial.presolve_with::<false>(&Limits::only(RuleId::DuplicateRow), true);
    assert_eq!(outcome, Presolved::Infeasible);
}

/// Two columns that enter every row and the objective in the same proportion become one.
#[test]
fn duplicate_columns_are_merged() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, 2, 1], vec![3, 6, 1]],
        vec![greater(), greater()],
        vec![4, 5],
        vec![
            variable(1, Some(0), Some(5)),
            // Twice the first column, and twice its cost.
            variable(2, Some(0), Some(5)),
            variable(1, Some(0), Some(5)),
        ],
    );

    let outcome = initial.presolve_with::<false>(&Limits::only(RuleId::DuplicateColumn), true);
    assert_eq!(outcome, Presolved::Reduced);
    assert_eq!(initial.nr_variables(), 2);
}

/// A variable that only ever loosens its rows goes to the bound the objective prefers.
#[test]
fn a_dominated_column_is_fixed() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, 1]],
        vec![less()],
        vec![10],
        vec![variable(1, Some(2), Some(8)), variable(1, Some(3), Some(9))],
    );

    let Presolved::Solved(solution) = initial.presolve::<false>() else {
        panic!("both variables want to be as small as they can");
    };
    assert_eq!(solution.solution_values[0].1, R32!(2));
    assert_eq!(solution.solution_values[1].1, R32!(3));
}

/// Rules that only preserve an optimal point can be switched off.
#[test]
fn dual_reductions_can_be_declined() {
    let build = || {
        problem(
            Objective::Minimize,
            &[vec![1, 1]],
            vec![less()],
            vec![10],
            vec![variable(1, Some(2), Some(8)), variable(1, Some(3), Some(9))],
        )
    };

    let mut allowed = build();
    allowed.presolve_with::<false>(&Limits::default(), true);
    assert!(allowed.postsolve().applied_dual_reduction);

    let mut declined = build();
    declined.presolve_with::<false>(&Limits::default(), false);
    assert!(!declined.postsolve().applied_dual_reduction);
}

/// Integer variables get their bounds pulled in to the integers.
#[test]
fn integer_bounds_are_rounded() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![2, 0], vec![1, 1]],
        vec![less(), less()],
        vec![5, 100],
        vec![integer(-1, Some(0), None), variable(0, Some(0), Some(1))],
    );

    // `2 x <= 5` means `x <= 5/2`, and for an integer variable `x <= 2`.
    let Presolved::Solved(solution) = initial.presolve::<true>() else {
        panic!("everything is determined");
    };
    assert_eq!(solution.solution_values[0].1, R32!(2));
}

/// The slack a row has left over becomes a bound on each variable in it.
#[test]
fn a_row_hands_out_the_slack_it_has_left() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![2, 1]],
        vec![less()],
        vec![10],
        vec![variable(1, Some(0), Some(9)), variable(1, Some(0), Some(9))],
    );

    let outcome = initial.presolve_with::<false>(&Limits::only(RuleId::DomainPropagation), true);
    assert_eq!(outcome, Presolved::Reduced);
    // With `x1` no lower than 0, `2 x0` may use all of the row: `x0 <= 5`.
    assert_eq!(initial.variable_for_test(0).upper_bound, Some(R32!(5)));
    // `x1 <= 10` is what the row allows, which is weaker than the bound it already had.
    assert_eq!(initial.variable_for_test(1).upper_bound, Some(R32!(9)));
}

/// A row the variable bounds already imply is dropped.
#[test]
fn a_row_its_variables_cannot_violate_is_dropped() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, 1]],
        vec![less()],
        vec![10],
        vec![variable(1, Some(0), Some(2)), variable(1, Some(0), Some(3))],
    );

    // The most the row can reach is `5`, so `<= 10` says nothing the bounds did not.
    let outcome = initial.presolve_with::<false>(&Limits::only(RuleId::DomainPropagation), true);
    assert_eq!(outcome, Presolved::Reduced);
    assert_eq!(initial.nr_constraints(), 0);
}

/// A range with one end implied keeps the other, and goes on to bound its variables with it.
#[test]
fn a_range_implied_on_one_side_keeps_the_other() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, 1]],
        vec![range(4)],
        vec![10],
        vec![variable(1, Some(0), Some(4)), variable(1, Some(0), Some(4))],
    );

    // `6 <= x0 + x1 <= 10` with both in `[0, 4]`: the row can reach at most 8, so the upper end
    // is implied, and what is left is `x0 + x1 >= 6`.
    let outcome = initial.presolve_with::<false>(&Limits::only(RuleId::DomainPropagation), true);
    assert_eq!(outcome, Presolved::Reduced);
    assert_eq!(initial.nr_constraints(), 1);
    assert_eq!(initial.constraint_types_for_test()[0], greater());
    assert_eq!(initial.b_for_test()[0], R32!(6));
    // With the other variable no higher than 4, each has to supply the remaining 2 itself.
    assert_eq!(initial.variable_for_test(0).lower_bound, Some(R32!(2)));
    assert_eq!(initial.variable_for_test(1).lower_bound, Some(R32!(2)));
}

/// A row its variables can only just satisfy pins every one of them.
#[test]
fn a_row_that_can_only_just_be_satisfied_is_forcing() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, 1]],
        vec![less()],
        vec![5],
        vec![variable(1, Some(3), Some(4)), variable(1, Some(2), Some(3))],
    );

    // The least the row can reach is `3 + 2 == 5`, which is all it may be: both are pinned.
    let outcome = initial.presolve_with::<false>(&Limits::only(RuleId::DomainPropagation), true);
    assert_eq!(outcome, Presolved::Reduced);
    assert_eq!(initial.nr_constraints(), 0);
    assert_eq!(initial.variable_for_test(0).upper_bound, Some(R32!(3)));
    assert_eq!(initial.variable_for_test(1).upper_bound, Some(R32!(2)));
}

/// A row its variables cannot reach at all is infeasible.
#[test]
fn a_row_its_variables_cannot_reach_is_infeasible() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, 1]],
        vec![less()],
        vec![5],
        vec![variable(1, Some(4), Some(5)), variable(1, Some(3), Some(4))],
    );

    // The least the row can reach is `7`, and it may be at most `5`.
    let outcome = initial.presolve_with::<false>(&Limits::only(RuleId::DomainPropagation), true);
    assert_eq!(outcome, Presolved::Infeasible);
}

/// One variable without a bound gets one from the row, even though the activity is unknown.
///
/// The activity end cannot be computed while a bound is missing, but with exactly one missing
/// there is nothing to share out: whatever the row has left over is all that variable's.
#[test]
fn the_one_variable_without_a_bound_still_gets_one() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![1, 1]],
        vec![less()],
        vec![10],
        vec![variable(1, None, None), variable(1, Some(0), Some(3))],
    );

    let outcome = initial.presolve_with::<false>(&Limits::only(RuleId::DomainPropagation), true);
    assert_eq!(outcome, Presolved::Reduced);
    // `x1` is no lower than 0, so `x0 <= 10`.
    assert_eq!(initial.variable_for_test(0).upper_bound, Some(R32!(10)));
    assert_eq!(initial.variable_for_test(0).lower_bound, None);
}

/// A row of fractions is written over their common denominator.
#[test]
fn a_row_of_fractions_is_cleared() {
    // `1/2 x0 + 1/3 x1 <= 1`  ->  `3 x0 + 2 x1 <= 6`.
    let mut initial = exact_problem(
        Objective::Minimize,
        &[vec![R32!(1, 2), R32!(1, 3)]],
        vec![less()],
        vec![R32!(1)],
        vec![variable(1, Some(0), Some(1)), variable(1, Some(0), Some(1))],
    );

    let outcome = initial.presolve_with::<false>(&Limits::only(RuleId::RowScaling), true);
    assert_eq!(outcome, Presolved::Reduced);
    assert_eq!(initial.coefficient_for_test(0, 0), Some(&R32!(3)));
    assert_eq!(initial.coefficient_for_test(0, 1), Some(&R32!(2)));
    assert_eq!(initial.b_for_test()[0], R32!(6));
}

/// A row of integers that share a factor is divided by it.
#[test]
fn a_row_with_a_common_factor_is_divided_by_it() {
    let mut initial = problem(
        Objective::Minimize,
        &[vec![4, 6]],
        vec![less()],
        vec![10],
        vec![variable(1, Some(0), Some(1)), variable(1, Some(0), Some(1))],
    );

    let outcome = initial.presolve_with::<false>(&Limits::only(RuleId::RowScaling), true);
    assert_eq!(outcome, Presolved::Reduced);
    assert_eq!(initial.coefficient_for_test(0, 0), Some(&R32!(2)));
    assert_eq!(initial.coefficient_for_test(0, 1), Some(&R32!(3)));
    assert_eq!(initial.b_for_test()[0], R32!(5));
}

/// A row that is already written the smallest way is left exactly as it is.
#[test]
fn a_row_that_is_already_smallest_is_left_alone() {
    let start = problem(
        Objective::Minimize,
        &[vec![2, 3]],
        vec![less()],
        vec![5],
        vec![variable(1, Some(0), Some(1)), variable(1, Some(0), Some(1))],
    );

    let mut initial = start.clone();
    let outcome = initial.presolve_with::<false>(&Limits::only(RuleId::RowScaling), true);
    assert_eq!(outcome, Presolved::Reduced);
    assert_eq!(initial, start);
    assert!(initial.postsolve().reductions().is_empty());
}

/// The width of a range is part of the row and moves with it.
#[test]
fn a_range_is_scaled_with_the_row_it_belongs_to() {
    // `1 <= 1/2 x0 + 1/2 x1 <= 2`, written as an upper bound of 2 with a width of 1.
    let mut initial = exact_problem(
        Objective::Minimize,
        &[vec![R32!(1, 2), R32!(1, 2)]],
        vec![RangedConstraintRelation::Range(R32!(1))],
        vec![R32!(2)],
        vec![variable(1, Some(0), Some(4)), variable(1, Some(0), Some(4))],
    );

    let outcome = initial.presolve_with::<false>(&Limits::only(RuleId::RowScaling), true);
    assert_eq!(outcome, Presolved::Reduced);
    assert_eq!(initial.coefficient_for_test(0, 0), Some(&R32!(1)));
    assert_eq!(initial.coefficient_for_test(0, 1), Some(&R32!(1)));
    assert_eq!(initial.b_for_test()[0], R32!(4));
    assert_eq!(
        initial.constraint_types_for_test()[0],
        RangedConstraintRelation::Range(R32!(2)),
    );
}

/// Clearing fractions can make a row larger, and then it is not done.
///
/// Denominators that share nothing all end up in every numerator. Four coefficients over four
/// coprime denominators cost less as fractions than the integers they clear to.
#[test]
fn a_row_that_would_grow_is_left_alone() {
    let start = exact_problem(
        Objective::Minimize,
        &[vec![R32!(1, 2), R32!(1, 3), R32!(1, 5), R32!(1, 7)]],
        vec![less()],
        vec![R32!(1)],
        vec![
            variable(1, Some(0), Some(1)),
            variable(1, Some(0), Some(1)),
            variable(1, Some(0), Some(1)),
            variable(1, Some(0), Some(1)),
        ],
    );

    // As fractions the row takes 16 bits; the integers it clears to take 38, which is more than
    // the default allowance of twice, so it is left as it was.
    let mut refused = start.clone();
    assert_eq!(
        refused.presolve_with::<false>(&Limits::only(RuleId::RowScaling), true),
        Presolved::Reduced,
    );
    assert_eq!(refused, start);
    assert!(refused.postsolve().reductions().is_empty());

    // Told that any growth is acceptable, this is what it would have cost.
    let mut cleared = start;
    let mut limits = Limits::only(RuleId::RowScaling);
    limits.max_scaling_growth = (u32::MAX, 1);
    assert_eq!(cleared.presolve_with::<false>(&limits, true), Presolved::Reduced);
    assert_eq!(cleared.coefficient_for_test(0, 0), Some(&R32!(105)));
    assert_eq!(cleared.coefficient_for_test(0, 3), Some(&R32!(30)));
    assert_eq!(cleared.b_for_test()[0], R32!(210));
}
