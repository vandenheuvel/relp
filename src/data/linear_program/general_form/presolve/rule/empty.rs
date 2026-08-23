//! # Rows and columns with no coefficients
//!
//! A row with no coefficients says something about nothing, so it is either satisfied or the
//! problem is infeasible. A column with no coefficients is not tied to anything, so it goes
//! straight to whichever of its bounds the objective prefers.
use std::ops::Neg;

use relp_num::OrderedFieldRef;

use crate::data::linear_program::elements::{BoundDirection, RangedConstraintRelation};
use crate::data::linear_program::general_form::presolve::postsolve::Reduction;
use crate::data::linear_program::general_form::presolve::problem::Problem;
use crate::data::linear_program::general_form::presolve::queues::{Queue, Queues};
use crate::data::linear_program::general_form::presolve::reducer::Reducer;
use crate::data::linear_program::general_form::presolve::rule::{
    Effect, LocalRule, RuleField, RuleId, Stop,
};

/// A row with no coefficients left.
pub struct EmptyRow;

impl EmptyRow {
    const ID: RuleId = RuleId::EmptyRow;
}

impl<F: RuleField, const I: bool> LocalRule<F, I> for EmptyRow
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    type Trigger = usize;
    const ID: RuleId = RuleId::EmptyRow;

    fn queue(queues: &mut Queues) -> &mut Queue<usize> {
        &mut queues.empty_row
    }

    fn is_applicable(row: usize, problem: &Problem<F>) -> bool {
        problem.rows[row].is_active && problem.matrix.row_len(row) == 0
    }

    fn apply(row: usize, reducer: &mut Reducer<'_, F, I>) -> Result<Effect, Stop> {
        if !is_satisfied(&reducer.problem.rows[row].relation, &reducer.problem.rows[row].right_hand_side)
        {
            return Err(Stop::Infeasible);
        }

        reducer.remove_row(row, Self::ID);
        reducer.postsolve.push(Reduction::EmptyRow { row });
        Ok(Effect::Reduced)
    }
}

/// Whether `0` stands in the given relation to the right hand side.
fn is_satisfied<F>(relation: &RangedConstraintRelation<F>, right_hand_side: &F) -> bool
where
    F: RuleField,
    for<'r> &'r F: OrderedFieldRef<F>,
{
    let zero = F::zero();
    match relation {
        RangedConstraintRelation::Equal => right_hand_side == &zero,
        RangedConstraintRelation::Less => right_hand_side >= &zero,
        RangedConstraintRelation::Greater => right_hand_side <= &zero,
        // `b - r <= 0 <= b`
        RangedConstraintRelation::Range(range) => {
            right_hand_side >= &zero && right_hand_side - range <= zero
        }
    }
}

/// A column with no coefficients left.
pub struct EmptyColumn;

impl EmptyColumn {
    const ID: RuleId = RuleId::EmptyColumn;
}

impl<F: RuleField, const I: bool> LocalRule<F, I> for EmptyColumn
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    type Trigger = usize;
    const ID: RuleId = RuleId::EmptyColumn;

    fn queue(queues: &mut Queues) -> &mut Queue<usize> {
        &mut queues.empty_column
    }

    fn is_applicable(column: usize, problem: &Problem<F>) -> bool {
        problem.columns[column].is_active && problem.matrix.column_len(column) == 0
    }

    fn apply(column: usize, reducer: &mut Reducer<'_, F, I>) -> Result<Effect, Stop> {
        // Any value within the bounds will do when the objective has no preference; the column is
        // in no row, so nothing else can object to it. An existing bound is preferred over an
        // invented value, so that a variable that was going to be zero anyway stays zero.
        let any_feasible_value = || {
            reducer.problem.columns[column]
                .lower_bound
                .clone()
                .or_else(|| reducer.problem.columns[column].upper_bound.clone())
                .unwrap_or_else(F::zero)
        };

        let value = match reducer.problem.improving_direction(column) {
            None => any_feasible_value(),
            Some(direction) => match reducer.problem.columns[column].bound(direction) {
                Some(bound) => bound.clone(),
                None => {
                    // Nothing stops the variable from improving the objective forever. That
                    // settles the question only if the problem has a feasible point, so the
                    // finding is noted and presolve carries on looking for a reason it has none.
                    // Because the column is in no row, giving it a value cannot change whether the
                    // rest of the problem is satisfiable.
                    reducer.unbounded_if_feasible = Some(column);
                    any_feasible_value()
                }
            },
        };

        let cost = reducer.problem.columns[column].cost.clone();
        reducer.problem.fixed_cost += &cost * &value;
        reducer.remove_column(column, Self::ID);
        reducer
            .postsolve
            .push(Reduction::EmptyColumn {
                column,
                value,
                cost,
            });

        Ok(Effect::Reduced)
    }
}

/// A guard so that the direction constants are used, keeping the intent of `improving_direction`
/// visible at this call site.
const _: [BoundDirection; 2] = [BoundDirection::Lower, BoundDirection::Upper];
