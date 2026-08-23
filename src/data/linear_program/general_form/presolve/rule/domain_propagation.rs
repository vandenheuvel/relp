//! # Reasoning from row activities
//!
//! The activity of a row is the interval `[L, U]` that `<a, x>` can cover given the variable
//! bounds. The row itself admits an interval `[lo, hi]`. Comparing the two is what this rule does.
//!
//! Reading a relation as an interval is what keeps the case analysis short. An equality admits
//! `[b, b]`, a range `[b - r, b]`, a `<=` row `(-inf, b]` and a `>=` row `[b, +inf)`. Then, for
//! each end independently:
//!
//! * the activity cannot reach the interval at all, and the problem is infeasible;
//! * the activity can only just reach it, so every variable is pinned to the bound that gets it
//!   there, and the row is forcing;
//! * the activity is entirely inside on that side, so that side of the row is implied by the
//!   bounds and can be dropped, taking the whole row with it when both sides go;
//! * or the row genuinely constrains the variables, and the slack between the two intervals gives
//!   each variable in turn a bound.
//!
//! Treating the two ends separately is what makes ranged rows ordinary rather than special. A
//! ranged row can be redundant on one side and forcing on the other, and the lowest activity of a
//! ranged row coinciding with the *upper* end of its range is a perfectly normal forcing row, not
//! a contradiction.
use std::cmp::Ordering;
use std::ops::Neg;

use relp_num::OrderedFieldRef;

use crate::data::linear_program::elements::{BoundDirection, RangedConstraintRelation};
use crate::data::linear_program::general_form::presolve::activity::contributes_to;
use crate::data::linear_program::general_form::presolve::postsolve::Reduction;
use crate::data::linear_program::general_form::presolve::problem::Problem;
use crate::data::linear_program::general_form::presolve::queues::{Queue, Queues};
use crate::data::linear_program::general_form::presolve::reducer::{Origin, Reducer, Worth};
use crate::data::linear_program::general_form::presolve::rule::{
    Effect, LocalRule, RuleField, RuleId, Stop,
};

pub struct DomainPropagation;

/// The end of the interval a relation admits, in one direction.
///
/// `None` means the relation says nothing on that side.
fn admissible_end<F: RuleField>(
    relation: &RangedConstraintRelation<F>,
    right_hand_side: &F,
    direction: BoundDirection,
) -> Option<F>
where
    for<'r> &'r F: OrderedFieldRef<F>,
{
    match (relation, direction) {
        (RangedConstraintRelation::Equal, _) => Some(right_hand_side.clone()),
        (RangedConstraintRelation::Range(range), BoundDirection::Lower) => {
            Some(right_hand_side - range)
        }
        (RangedConstraintRelation::Range(_), BoundDirection::Upper) => {
            Some(right_hand_side.clone())
        }
        (RangedConstraintRelation::Less, BoundDirection::Lower) => None,
        (RangedConstraintRelation::Less, BoundDirection::Upper) => Some(right_hand_side.clone()),
        (RangedConstraintRelation::Greater, BoundDirection::Lower) => {
            Some(right_hand_side.clone())
        }
        (RangedConstraintRelation::Greater, BoundDirection::Upper) => None,
    }
}

impl<F: RuleField, const I: bool> LocalRule<F, I> for DomainPropagation
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    type Trigger = (usize, BoundDirection);
    const ID: RuleId = RuleId::DomainPropagation;

    fn queue(queues: &mut Queues) -> &mut Queue<(usize, BoundDirection)> {
        &mut queues.activity
    }

    fn is_applicable((row, _): (usize, BoundDirection), problem: &Problem<F>) -> bool {
        // A row with a single coefficient is a bound, and the rule that says so is both cheaper
        // and more complete.
        problem.rows[row].is_active && problem.matrix.row_len(row) > 1
    }

    fn apply(
        (row, activity_direction): (usize, BoundDirection),
        reducer: &mut Reducer<'_, F, I>,
    ) -> Result<Effect, Stop> {
        match reducer.activity.missing(row, activity_direction) {
            0 => complete(row, activity_direction, reducer),
            1 => single_missing_bound(row, activity_direction, reducer),
            // The trigger is stale: a bound was lost since it was enqueued.
            _ => Ok(Effect::None),
        }
    }
}

/// Both ends of the row's own interval, and the activity end we know.
fn complete<F: RuleField, const I: bool>(
    row: usize,
    activity_direction: BoundDirection,
    reducer: &mut Reducer<'_, F, I>,
) -> Result<Effect, Stop>
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    let activity = reducer
        .activity
        .value(&reducer.problem, row, activity_direction)
        .clone();

    debug_assert_eq!(
        activity,
        crate::data::linear_program::general_form::presolve::activity::Activity::recompute(
            &reducer.problem,
            row,
            activity_direction,
        ),
        "a cached activity that no longer matches its row is a silent wrong answer",
    );

    let relation = reducer.problem.rows[row].relation.clone();
    let right_hand_side = reducer.problem.rows[row].right_hand_side.clone();

    // The end of the row's interval that this activity end can run into, and the one it runs away
    // from. A lower activity is compared against the upper end of the interval, and the other way
    // round.
    let near = admissible_end(&relation, &right_hand_side, !activity_direction);
    let far = admissible_end(&relation, &right_hand_side, activity_direction);

    if let Some(near) = &near {
        let comparison = match activity_direction {
            // `L` against `hi`: the lowest the row can be, against the highest it may be.
            BoundDirection::Lower => activity.cmp(near),
            // `U` against `lo`.
            BoundDirection::Upper => near.cmp(&activity),
        };

        match comparison {
            // The row cannot be satisfied at all.
            Ordering::Greater => return Err(Stop::Infeasible),
            // It can be satisfied in exactly one way: every variable at the bound that gets the
            // activity here.
            Ordering::Equal => return force(row, activity_direction, reducer),
            Ordering::Less => {}
        }
    }

    // Is the side the activity runs away from already implied by the bounds?
    //
    // A side the relation does not have is not an implied side. A `<=` row says nothing about how
    // small its activity may be, and reading that silence as "implied" would drop the row on the
    // strength of the one thing it never claimed.
    let far_is_implied = match &far {
        None => false,
        Some(far) => match activity_direction {
            BoundDirection::Lower => &activity >= far,
            BoundDirection::Upper => &activity <= far,
        },
    };

    if far_is_implied {
        return drop_far_side(row, activity_direction, &relation, &right_hand_side, near, reducer);
    }

    // The row constrains its variables. Give each of them a bound.
    derive_bounds(row, activity_direction, &activity, near, reducer)
}

/// The activity can only just reach the row: pin every variable to the bound that gets it there.
fn force<F: RuleField, const I: bool>(
    row: usize,
    activity_direction: BoundDirection,
    reducer: &mut Reducer<'_, F, I>,
) -> Result<Effect, Stop>
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    let pinned = reducer
        .problem
        .matrix
        .row(row)
        .map(|(column, coefficient)| {
            let direction = contributes_to(activity_direction, coefficient.non_zero_signum());
            let value = reducer.problem.columns[column]
                .bound(direction)
                .expect("the activity was computed from this bound")
                .clone();
            (column, direction, value)
        })
        .collect::<Vec<_>>();

    for (column, direction, value) in pinned {
        // Bring the other bound in to meet the one the activity used. The row goes at the end of
        // this function, and then these bounds are all that is left of it: what the row said is
        // exactly that every one of its variables sits at the end the activity used. So there is
        // nothing to weigh, and by the time the loop is over there is no longer anything that
        // could be asked.
        reducer.set_bound(column, !direction, value, Origin::Row(row), RuleId::DomainPropagation)?;
    }

    let relation = reducer.problem.rows[row].relation.clone();
    let right_hand_side = reducer.problem.rows[row].right_hand_side.clone();
    let coefficients = reducer.remove_row(row, RuleId::DomainPropagation);
    reducer.postsolve.push(Reduction::RedundantRow {
        row,
        relation,
        right_hand_side,
        coefficients,
    });

    Ok(Effect::Reduced)
}

/// One side of the row is implied by the variable bounds: drop it, and the row with it if nothing
/// is left.
fn drop_far_side<F: RuleField, const I: bool>(
    row: usize,
    activity_direction: BoundDirection,
    relation: &RangedConstraintRelation<F>,
    right_hand_side: &F,
    near: Option<F>,
    reducer: &mut Reducer<'_, F, I>,
) -> Result<Effect, Stop>
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    // What is left once the side the activity ran away from is dropped.
    //
    // Only a range has another side to keep. An inequality has just the one, so having it implied
    // leaves nothing; an equality cannot get here, because an activity reaching one of its two
    // coinciding ends is forcing, and going past it is infeasible.
    let remaining = match (relation, activity_direction) {
        (RangedConstraintRelation::Range(_), BoundDirection::Lower) => {
            Some((RangedConstraintRelation::Less, right_hand_side.clone()))
        }
        (RangedConstraintRelation::Range(range), BoundDirection::Upper) => {
            Some((RangedConstraintRelation::Greater, right_hand_side - range))
        }
        _ => None,
    };

    match remaining {
        Some((relation, right_hand_side)) => {
            reducer.change_row(row, relation, right_hand_side, RuleId::DomainPropagation);
            // The row still constrains its variables from the side that is left.
            let activity = reducer
                .activity
                .value(&reducer.problem, row, activity_direction)
                .clone();
            derive_bounds(row, activity_direction, &activity, near, reducer)?;
            Ok(Effect::Tightened)
        }
        None => {
            let relation = relation.clone();
            let right_hand_side = right_hand_side.clone();
            let coefficients = reducer.remove_row(row, RuleId::DomainPropagation);
            reducer.postsolve.push(Reduction::RedundantRow {
                row,
                relation,
                right_hand_side,
                coefficients,
            });
            Ok(Effect::Reduced)
        }
    }
}

/// Give every variable in the row the bound the row's slack allows it.
///
/// With the activity end `activity` known and the row admitting up to `near` on the other side,
/// each variable may use whatever the others leave: `a_j x_j (rel) near - (activity - a_j b_j)`.
fn derive_bounds<F: RuleField, const I: bool>(
    row: usize,
    activity_direction: BoundDirection,
    activity: &F,
    near: Option<F>,
    reducer: &mut Reducer<'_, F, I>,
) -> Result<Effect, Stop>
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    let Some(near) = near else {
        // The row does not limit the activity on that side, so there is nothing to share out.
        return Ok(Effect::None);
    };

    let candidates = reducer
        .problem
        .matrix
        .row(row)
        .map(|(column, coefficient)| {
            let sign = coefficient.non_zero_signum();
            let used = contributes_to(activity_direction, sign);
            let bound = reducer.problem.columns[column]
                .bound(used)
                .expect("the activity was computed from this bound")
                .clone();
            (column, coefficient.clone(), sign, bound)
        })
        .collect::<Vec<_>>();

    let mut effect = Effect::None;
    for (column, coefficient, sign, used_bound) in candidates {
        let residual = activity - &(&coefficient * &used_bound);
        let value = &(&near - &residual) / &coefficient;
        // Dividing by the coefficient turns the direction around when it is negative.
        let direction = contributes_to(!activity_direction, sign);

        if offer(reducer, column, direction, value, row)? == Effect::Tightened {
            effect = Effect::Tightened;
        }
    }

    Ok(effect)
}

/// Exactly one variable in the row lacks the bound the activity end needs: give it one.
fn single_missing_bound<F: RuleField, const I: bool>(
    row: usize,
    activity_direction: BoundDirection,
    reducer: &mut Reducer<'_, F, I>,
) -> Result<Effect, Stop>
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    let relation = reducer.problem.rows[row].relation.clone();
    let right_hand_side = reducer.problem.rows[row].right_hand_side.clone();
    let Some(near) = admissible_end(&relation, &right_hand_side, !activity_direction) else {
        return Ok(Effect::None);
    };

    // Everything except the one variable without a bound.
    let mut partial = F::zero();
    let mut target = None;
    for (column, coefficient) in reducer.problem.matrix.row(row) {
        let sign = coefficient.non_zero_signum();
        match reducer.problem.columns[column].bound(contributes_to(activity_direction, sign)) {
            Some(bound) => partial += coefficient * bound,
            None => {
                debug_assert!(target.is_none(), "the caller checked there is exactly one");
                target = Some((column, coefficient.clone(), sign));
            }
        }
    }

    let (column, coefficient, sign) = target.expect("the caller checked there is exactly one");
    let value = &(&near - &partial) / &coefficient;
    let direction = contributes_to(!activity_direction, sign);

    offer(reducer, column, direction, value, row)
}

/// Decide whether a bound this row implies is worth what it costs, and write it if it is.
///
/// The row stays in the problem either way. What it says about the variable it goes on saying, so
/// a bound declined here is a reduction not made and nothing more: the problem left behind is the
/// one that came in, with one fewer thing already worked out about it.
fn offer<F: RuleField, const I: bool>(
    reducer: &mut Reducer<'_, F, I>,
    column: usize,
    direction: BoundDirection,
    value: F,
    row: usize,
) -> Result<Effect, Stop>
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    match reducer.weigh_bound(column, direction, &value) {
        Worth::Adopt => {
            let origin = Origin::Row(row);
            reducer.set_bound(column, direction, value, origin, RuleId::DomainPropagation)?;
            Ok(Effect::Tightened)
        }
        Worth::Nothing => Ok(Effect::None),
        Worth::NotEnough => {
            reducer.statistics.per_rule[RuleId::DomainPropagation].budget_refusals += 1;
            Ok(Effect::None)
        }
    }
}
