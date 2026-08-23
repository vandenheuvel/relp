use relp_num::R32;
use relp_num::Rational32;

use crate::data::linear_program::elements::{BoundDirection, RangedConstraintRelation};
use crate::data::linear_program::general_form::presolve::postsolve::{
    BoundOrigin, Postsolve, Reduction,
};

type T = Rational32;

#[test]
fn an_empty_record_returns_the_reduced_solution() {
    let mut postsolve = Postsolve::<T>::new(0, 2);
    postsolve.set_survivors(vec![], vec![0, 1]);

    assert_eq!(
        postsolve.recover_primal(&[R32!(3), R32!(4)]),
        vec![R32!(3), R32!(4)],
    );
}

#[test]
fn explicit_values_are_restored() {
    let mut postsolve = Postsolve::<T>::new(0, 3);
    postsolve.push(Reduction::EmptyColumn {
        column: 0,
        value: R32!(1),
        cost: R32!(0),
    });
    postsolve.push(Reduction::FixedColumn {
        column: 2,
        value: R32!(5),
        coefficients: vec![],
        cost: R32!(0),
    });
    postsolve.set_survivors(vec![], vec![1]);

    assert_eq!(
        postsolve.recover_primal(&[R32!(9)]),
        vec![R32!(1), R32!(9), R32!(5)],
    );
}

/// A slack whose row was dropped takes up the residual, but never outside its own bounds.
#[test]
fn a_removed_slack_stays_within_its_bounds() {
    let reduction = |lower, upper| Reduction::ColumnSingleton {
        column: 1,
        row: 0,
        coefficient: R32!(1),
        // Row was `x0 + s >= 0`, so `s = 0 - x0`.
        coefficients: vec![(0, R32!(1))],
        right_hand_side: R32!(0),
        lower_bound: lower,
        upper_bound: upper,
    };

    let recover = |reduction| {
        let mut postsolve = Postsolve::<T>::new(1, 2);
        postsolve.push(reduction);
        postsolve.set_survivors(vec![], vec![0]);
        postsolve.recover_primal(&[R32!(5)])
    };

    // The exact residual is `-5`, which respects a lower bound of `-9`.
    assert_eq!(recover(reduction(Some(R32!(-9)), None)), vec![R32!(5), R32!(-5)]);
    // With a lower bound of `0` the residual is out of range and must be clamped.
    assert_eq!(recover(reduction(Some(R32!(0)), None)), vec![R32!(5), R32!(0)]);
}

/// Reductions are undone in reverse, so one may depend on a column another restores.
#[test]
fn reductions_may_depend_on_later_ones() {
    let mut postsolve = Postsolve::<T>::new(2, 3);
    // Recorded first: `x0` is the slack of a row also containing `x1`.
    postsolve.push(Reduction::ColumnSingleton {
        column: 0,
        row: 0,
        coefficient: R32!(1),
        coefficients: vec![(1, R32!(2))],
        right_hand_side: R32!(10),
        lower_bound: None,
        upper_bound: None,
    });
    // Recorded second: `x1` itself is then removed, so it is restored first on the way back.
    postsolve.push(Reduction::FixedColumn {
        column: 1,
        value: R32!(3),
        coefficients: vec![],
        cost: R32!(0),
    });
    postsolve.set_survivors(vec![], vec![2]);

    // `x1 = 3`, then `x0 = 10 - 2 * 3 = 4`.
    assert_eq!(
        postsolve.recover_primal(&[R32!(7)]),
        vec![R32!(4), R32!(3), R32!(7)],
    );
}

#[test]
fn a_doubleton_equation_is_solved_for_the_eliminated_variable() {
    let mut postsolve = Postsolve::<T>::new(1, 2);
    // `2 x0 + 3 x1 == 12`, eliminating `x0`.
    postsolve.push(Reduction::DoubletonEquation {
        row: 0,
        eliminated: 0,
        kept: 1,
        coefficient_eliminated: R32!(2),
        coefficient_kept: R32!(3),
        right_hand_side: R32!(12),
        cost_eliminated: R32!(0),
        occurrences_eliminated: vec![],
        kept_lower_bound: None,
        kept_upper_bound: None,
    });
    postsolve.set_survivors(vec![], vec![1]);

    assert_eq!(
        postsolve.recover_primal(&[R32!(2)]),
        vec![R32!(3), R32!(2)],
    );
}

#[test]
fn a_merged_column_is_split_within_both_sets_of_bounds() {
    let mut postsolve = Postsolve::<T>::new(0, 2);
    // `merged = x_kept + 2 * x_removed`, with `x_kept in [0, 3]` and `x_removed in [0, 4]`.
    postsolve.push(Reduction::DuplicateColumn {
        removed: 1,
        kept: 0,
        factor: R32!(2),
        removed_lower_bound: Some(R32!(0)),
        removed_upper_bound: Some(R32!(4)),
        kept_lower_bound: Some(R32!(0)),
        kept_upper_bound: Some(R32!(3)),
    });
    postsolve.set_survivors(vec![], vec![0]);

    // A merged value of 3 fits entirely in the kept variable.
    assert_eq!(postsolve.recover_primal(&[R32!(3)]), vec![R32!(3), R32!(0)]);

    // A merged value of 7 does not: the kept variable takes 3, and `(7 - 3) / 2 = 2` goes to the
    // removed one.
    assert_eq!(postsolve.recover_primal(&[R32!(7)]), vec![R32!(3), R32!(2)]);
}

/// Recording a dual reduction has to be visible, because it changes what an infeasible reduced
/// problem proves.
#[test]
fn the_kind_of_reduction_applied_is_tracked() {
    let mut postsolve = Postsolve::<T>::new(1, 1);
    assert!(!postsolve.applied_primal_reduction);
    assert!(!postsolve.applied_dual_reduction);

    postsolve.push(Reduction::EmptyRow { row: 0 });
    assert!(postsolve.applied_primal_reduction);
    assert!(!postsolve.applied_dual_reduction);

    postsolve.push_dual(Reduction::DominatedColumn {
        column: 0,
        value: R32!(0),
    });
    assert!(postsolve.applied_dual_reduction);
}

/// A singleton row leaves the variable where it was: the bound records that follow it are what
/// carry both its value and its multiplier.
#[test]
fn a_singleton_row_leaves_the_variable_in_place() {
    let mut postsolve = Postsolve::<T>::new(1, 1);
    postsolve.push(Reduction::SingletonRow {
        row: 0,
        column: 0,
        coefficient: R32!(2),
        relation: RangedConstraintRelation::Greater,
        right_hand_side: R32!(4),
    });
    postsolve.set_survivors(vec![], vec![0]);

    assert_eq!(postsolve.recover_primal(&[R32!(2)]), vec![R32!(2)]);
}

/// A bound presolving derived, holding a reduced cost that belongs on the row that implied it.
///
/// `min -x0` with `x0 + x1 <= 4` and `x1 in [0, 1]`. Propagation reads `x0 <= 4` off the row, and
/// the reduced problem then holds `x0` at that bound with a reduced cost of `-1`. The original
/// problem has no such bound, so the `-1` has to go back onto the row: `y = -1 / 1 = -1`, which is
/// a sign a `<=` row admits, and the row is tight at `4`.
#[test]
fn a_derived_bound_hands_its_reduced_cost_back_to_the_row() {
    let mut postsolve = Postsolve::<T>::new(1, 2);
    postsolve.push(Reduction::BoundTightened {
        column: 0,
        direction: BoundDirection::Upper,
        old: None,
        new: R32!(4),
        origin: BoundOrigin::Activity {
            row: 0,
            coefficient: R32!(1),
            coefficients: vec![(0, R32!(1)), (1, R32!(1))],
        },
    });
    postsolve.set_survivors(vec![0], vec![0, 1]);

    assert_eq!(
        postsolve.recover_dual(true, &[R32!(0)], &[R32!(-1), R32!(0)], &[R32!(4), R32!(0)]),
        Some(vec![R32!(-1)]),
    );
}

/// The same, for a maximum. Every multiplier and every reduced cost changes sign with the
/// objective, and the answer is the negation of the one above.
#[test]
fn the_push_back_follows_the_direction_of_the_objective() {
    let mut postsolve = Postsolve::<T>::new(1, 2);
    postsolve.push(Reduction::BoundTightened {
        column: 0,
        direction: BoundDirection::Upper,
        old: None,
        new: R32!(4),
        origin: BoundOrigin::Activity {
            row: 0,
            coefficient: R32!(1),
            coefficients: vec![(0, R32!(1)), (1, R32!(1))],
        },
    });
    postsolve.set_survivors(vec![0], vec![0, 1]);

    assert_eq!(
        postsolve.recover_dual(false, &[R32!(0)], &[R32!(1), R32!(0)], &[R32!(4), R32!(0)]),
        Some(vec![R32!(1)]),
    );
}

/// A bound the solution is not sitting on, and one it is sitting on but is not being held by, both
/// leave the multipliers alone.
#[test]
fn only_a_bound_that_is_doing_the_holding_takes_a_multiplier() {
    let reduction = |direction| Reduction::BoundTightened {
        column: 0,
        direction,
        old: None,
        new: R32!(4),
        origin: BoundOrigin::Activity {
            row: 0,
            coefficient: R32!(1),
            coefficients: vec![(0, R32!(1)), (1, R32!(1))],
        },
    };
    let recover = |reduction, value, cost| {
        let mut postsolve = Postsolve::<T>::new(1, 2);
        postsolve.push(reduction);
        postsolve.set_survivors(vec![0], vec![0, 1]);
        postsolve.recover_dual(true, &[R32!(0)], &[cost, R32!(0)], &[value, R32!(0)])
    };

    // Sitting somewhere else entirely.
    assert_eq!(
        recover(reduction(BoundDirection::Upper), R32!(1), R32!(0)),
        Some(vec![R32!(0)]),
    );
    // Sitting on it, but with the sign of a variable held from below, which this is not.
    assert_eq!(
        recover(reduction(BoundDirection::Upper), R32!(4), R32!(1)),
        Some(vec![R32!(0)]),
    );
}

/// A doubleton equation whose kept column is held by a bound it already had.
///
/// `x0 + x1 == 10` with both variables non negative and costs `3` and `5`. Eliminating `x0` leaves
/// `x1` costing `5 - 3 = 2`, and the reduced problem puts it at its own lower bound of zero. So
/// the reduced cost stays with `x1`, the eliminated column's goes to zero, and the multiplier
/// follows from `3 - 1 * y = 0`.
#[test]
fn a_doubleton_leaves_the_reduced_cost_where_the_bound_is() {
    let mut postsolve = Postsolve::<T>::new(1, 2);
    postsolve.push(Reduction::DoubletonEquation {
        row: 0,
        eliminated: 0,
        kept: 1,
        coefficient_eliminated: R32!(1),
        coefficient_kept: R32!(1),
        right_hand_side: R32!(10),
        cost_eliminated: R32!(3),
        occurrences_eliminated: vec![],
        kept_lower_bound: Some(R32!(0)),
        kept_upper_bound: None,
    });
    postsolve.set_survivors(vec![], vec![1]);

    assert_eq!(
        postsolve.recover_dual(true, &[], &[R32!(2)], &[R32!(10), R32!(0)]),
        Some(vec![R32!(3)]),
    );
}

/// A doubleton equation whose kept column is held by a bound that came across from the other one.
///
/// `2 x0 + x1 == 10`, both non negative, costing `4` and `1`. Eliminating `x0` leaves `x1` costing
/// `1 - 4/2 = -1` and bounded above by `10`, which is `x0 >= 0` in disguise. The reduced problem
/// puts `x1` there, so the reduced cost belongs to `x0`, which is what is actually at a bound of
/// its own; `x1`'s goes to zero and the multiplier comes out at `1`.
#[test]
fn a_doubleton_hands_the_reduced_cost_to_the_column_that_is_at_a_bound() {
    let mut postsolve = Postsolve::<T>::new(1, 2);
    postsolve.push(Reduction::DoubletonEquation {
        row: 0,
        eliminated: 0,
        kept: 1,
        coefficient_eliminated: R32!(2),
        coefficient_kept: R32!(1),
        right_hand_side: R32!(10),
        cost_eliminated: R32!(4),
        occurrences_eliminated: vec![],
        kept_lower_bound: Some(R32!(0)),
        kept_upper_bound: None,
    });
    postsolve.set_survivors(vec![], vec![1]);

    assert_eq!(
        postsolve.recover_dual(true, &[], &[R32!(-1)], &[R32!(0), R32!(10)]),
        Some(vec![R32!(1)]),
    );
}

/// The multiplier of a merged row goes to whichever of the two rows supplied the active end.
///
/// Row 0 is `<a, x> <= 5` and row 1 is `2 <a, x> >= 4`, so the merged row admits `[2, 5]`. A
/// negative multiplier sits on the upper end, which came from row 0; a positive one sits on the
/// lower end, which came from row 1 and has to be divided by the factor on the way there.
#[test]
fn a_merged_row_gives_its_multiplier_to_the_row_that_supplied_the_active_end() {
    let recover = |multiplier| {
        let mut postsolve = Postsolve::<T>::new(2, 1);
        postsolve.push(Reduction::DuplicateRow {
            removed: 1,
            kept: 0,
            factor: R32!(2),
            removed_relation: RangedConstraintRelation::Greater,
            removed_right_hand_side: R32!(4),
            kept_relation: RangedConstraintRelation::Less,
            kept_right_hand_side: R32!(5),
        });
        postsolve.set_survivors(vec![0], vec![0]);
        postsolve.recover_dual(true, &[multiplier], &[R32!(0)], &[R32!(5)])
    };

    assert_eq!(recover(R32!(-1)), Some(vec![R32!(-1), R32!(0)]));
    assert_eq!(recover(R32!(2)), Some(vec![R32!(0), R32!(1)]));
}

/// Merging two parallel columns moves no multiplier at all.
///
/// The rows and the objective both see only `x_kept + factor * x_removed`, which is what the rule
/// checks before merging, so nothing about the rows changes.
#[test]
fn merging_columns_leaves_the_multipliers_alone() {
    let mut postsolve = Postsolve::<T>::new(1, 2);
    postsolve.push(Reduction::DuplicateColumn {
        removed: 1,
        kept: 0,
        factor: R32!(2),
        removed_lower_bound: Some(R32!(0)),
        removed_upper_bound: Some(R32!(4)),
        kept_lower_bound: Some(R32!(0)),
        kept_upper_bound: Some(R32!(3)),
    });
    postsolve.set_survivors(vec![0], vec![0]);

    assert_eq!(
        postsolve.recover_dual(true, &[R32!(-1)], &[R32!(0)], &[R32!(3), R32!(0)]),
        Some(vec![R32!(-1)]),
    );
}

/// Two things rule a dual solution out entirely, and both say so rather than guessing.
#[test]
fn what_cannot_be_recovered_is_declined() {
    // Fixing a dominated column removes feasible points, so what is left has a different dual.
    let mut dominated = Postsolve::<T>::new(1, 1);
    dominated.push_dual(Reduction::DominatedColumn {
        column: 0,
        value: R32!(0),
    });
    dominated.set_survivors(vec![0], vec![]);
    assert!(!dominated.preserves_duals());
    assert_eq!(dominated.recover_dual(true, &[R32!(0)], &[], &[R32!(0)]), None);

    // Rounding a bound inwards is not something a row implied, so there is nowhere to push a
    // multiplier back to.
    let mut rounded = Postsolve::<T>::new(1, 1);
    rounded.respects_integrality = true;
    rounded.set_survivors(vec![0], vec![0]);
    assert!(!rounded.preserves_duals());
    assert_eq!(
        rounded.recover_dual(true, &[R32!(0)], &[R32!(0)], &[R32!(0)]),
        None,
    );
}
