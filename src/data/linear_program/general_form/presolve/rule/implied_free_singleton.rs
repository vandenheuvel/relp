//! # Variables that appear in one equality row, and in the objective
//!
//! A variable in a single row and with a cost cannot simply be dropped the way a slack can: the
//! objective would lose a term. But when the row already confines the variable to within its own
//! bounds, the bounds carry no information, and the equality determines the variable outright:
//! `s = (b - <a, x>) / c`.
//!
//! Substituting that into the objective moves the cost onto the other variables of the row, and
//! removes a row and a column. Doing the cost substitution at this point, rather than leaving it
//! for postsolve, is what keeps the reduced problem's objective the right one.
use std::ops::Neg;

use relp_num::{NonZeroSign, OrderedFieldRef};

use crate::data::linear_program::elements::{
    BoundDirection, RangedConstraintRelation, VariableType,
};
use crate::data::linear_program::general_form::presolve::postsolve::Reduction;
use crate::data::linear_program::general_form::presolve::problem::Problem;
use crate::data::linear_program::general_form::presolve::queues::{Queue, Queues};
use crate::data::linear_program::general_form::presolve::reducer::Reducer;
use crate::data::linear_program::general_form::presolve::rule::{
    Effect, LocalRule, RuleField, RuleId, Stop,
};

pub struct ImpliedFreeColumnSingleton;

impl ImpliedFreeColumnSingleton {
    const ID: RuleId = RuleId::ImpliedFreeColumnSingleton;
}

impl<F: RuleField, const I: bool> LocalRule<F, I> for ImpliedFreeColumnSingleton
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    type Trigger = usize;
    const ID: RuleId = RuleId::ImpliedFreeColumnSingleton;

    fn queue(queues: &mut Queues) -> &mut Queue<usize> {
        &mut queues.implied_free_singleton
    }

    fn is_applicable(column: usize, problem: &Problem<F>) -> bool {
        let variable = &problem.columns[column];
        variable.is_active
            && problem.matrix.column_len(column) == 1
            && !variable.cost.is_zero()
            && variable.fixed_value().is_none()
    }

    fn apply(column: usize, reducer: &mut Reducer<'_, F, I>) -> Result<Effect, Stop> {
        let (row, coefficient) = {
            let (row, coefficient) = reducer
                .problem
                .matrix
                .column(column)
                .next()
                .expect("checked by `is_applicable`");
            (row, coefficient.clone())
        };

        // Only an equality pins the variable down; an inequality leaves it free to run off in the
        // direction the objective likes.
        if !matches!(
            reducer.problem.rows[row].relation,
            RangedConstraintRelation::Equal
        ) {
            return Ok(Effect::None);
        }

        if I && reducer.problem.columns[column].variable_type == VariableType::Integer {
            // Dropping the bounds of an integer variable also drops the guarantee that the value
            // the row gives it is one it may take.
            return Ok(Effect::None);
        }

        if !bounds_are_implied(reducer, column, row, &coefficient) {
            return Ok(Effect::None);
        }

        let right_hand_side = reducer.problem.rows[row].right_hand_side.clone();
        let coefficients = reducer
            .problem
            .matrix
            .row(row)
            .filter(|&(other, _)| other != column)
            .map(|(other, value)| (other, value.clone()))
            .collect::<Vec<_>>();

        // `cost * s = cost * (b - <a, x>) / c`: a constant, and a change to every other cost.
        let cost = reducer.problem.columns[column].cost.clone();
        let scale = &cost / &coefficient;
        reducer.problem.fixed_cost += &scale * &right_hand_side;
        for (other, value) in &coefficients {
            reducer.problem.columns[*other].cost -= &scale * value;
        }

        reducer.remove_column(column, Self::ID);
        reducer.remove_row(row, Self::ID);
        reducer
            .postsolve
            .push(Reduction::ImpliedFreeColumnSingleton {
                column,
                row,
                coefficient,
                coefficients,
                right_hand_side,
                cost,
            });

        Ok(Effect::Reduced)
    }
}

/// Whether the row already keeps the variable within its own bounds.
///
/// `s = (b - A) / c`, so the interval `s` is confined to follows from the interval `A` covers. A
/// variable with no bounds at all is trivially confined by them.
fn bounds_are_implied<F: RuleField, const I: bool>(
    reducer: &Reducer<'_, F, I>,
    column: usize,
    row: usize,
    coefficient: &F,
) -> bool
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    let variable = &reducer.problem.columns[column];
    if variable.is_free() {
        return true;
    }

    // The interval the rest of the row covers.
    let mut lowest = F::zero();
    let mut highest = F::zero();
    for (other, value) in reducer.problem.matrix.row(row) {
        if other == column {
            continue;
        }
        let (low, high) = match value.non_zero_signum() {
            NonZeroSign::Positive => (
                reducer.problem.columns[other].bound(BoundDirection::Lower),
                reducer.problem.columns[other].bound(BoundDirection::Upper),
            ),
            NonZeroSign::Negative => (
                reducer.problem.columns[other].bound(BoundDirection::Upper),
                reducer.problem.columns[other].bound(BoundDirection::Lower),
            ),
        };
        match (low, high) {
            (Some(low), Some(high)) => {
                lowest += value * low;
                highest += value * high;
            }
            // Without both ends the interval is unbounded, and cannot be inside anything.
            _ => return false,
        }
    }

    let right_hand_side = &reducer.problem.rows[row].right_hand_side;
    let from_highest = &(right_hand_side - &highest) / coefficient;
    let from_lowest = &(right_hand_side - &lowest) / coefficient;
    let (implied_low, implied_high) = if coefficient.is_negative() {
        (from_lowest, from_highest)
    } else {
        (from_highest, from_lowest)
    };

    let low_is_implied = match variable.bound(BoundDirection::Lower) {
        None => true,
        Some(bound) => &implied_low >= bound,
    };
    let high_is_implied = match variable.bound(BoundDirection::Upper) {
        None => true,
        Some(bound) => &implied_high <= bound,
    };

    low_is_implied && high_is_implied
}
