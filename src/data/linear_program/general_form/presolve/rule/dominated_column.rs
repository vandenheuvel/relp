//! # Variables that another variable can always stand in for
//!
//! If moving a variable in one direction improves the objective and loosens every row it appears
//! in, nothing keeps it away from that bound, and it can be fixed there.
//!
//! This is an argument about optimality, not about feasibility: it removes feasible points and
//! keeps only the promise that an optimal one survives. That has two consequences the rest of the
//! code has to respect. The reduced problem is no longer a relaxation of the original, so an
//! infeasible reduced problem no longer proves the original infeasible. And the dual of the
//! reduced problem is not the dual of the original, so this rule has no place in a run that has to
//! produce one.
use std::ops::Neg;

use relp_num::{NonZeroSign, OrderedFieldRef};

use crate::data::linear_program::elements::{BoundDirection, RangedConstraintRelation};
use crate::data::linear_program::general_form::presolve::postsolve::Reduction;
use crate::data::linear_program::general_form::presolve::reducer::Reducer;
use crate::data::linear_program::general_form::presolve::rule::{
    Effect, RuleField, RuleId, Stop, SweepRule,
};

pub struct DominatedColumn;

impl DominatedColumn {
    const ID: RuleId = RuleId::DominatedColumn;
}

impl<F: RuleField, const I: bool> SweepRule<F, I> for DominatedColumn
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    const ID: RuleId = RuleId::DominatedColumn;

    fn sweep(reducer: &mut Reducer<'_, F, I>) -> Result<Effect, Stop> {
        let mut effect = Effect::None;

        for column in 0..reducer.problem.columns.len() {
            if !reducer.problem.columns[column].is_active {
                continue;
            }
            reducer.statistics.per_rule[Self::ID].triggers += 1;

            let Some(direction) = free_direction(reducer, column) else {
                reducer.statistics.per_rule[Self::ID].misses += 1;
                continue;
            };

            let Some(value) = reducer.problem.columns[column].bound(direction).cloned() else {
                // Nothing stops the variable, so the problem is unbounded if it has any feasible
                // point at all.
                return Err(Stop::UnboundedIfFeasible { column });
            };

            // Holding the variable at a value means folding that value into the rows it appears
            // in, exactly as for a variable whose bounds left it only one.
            reducer.substitute_value(column, &value, Self::ID);
            reducer.remove_column(column, Self::ID);
            reducer
                .postsolve
                .push_dual(Reduction::DominatedColumn { column, value });

            reducer.statistics.per_rule[Self::ID].applications += 1;
            effect = Effect::Reduced;
        }

        Ok(effect)
    }
}

/// The direction a variable can be moved in without ever making a row harder to satisfy, if
/// that direction also improves the objective.
fn free_direction<F: RuleField, const I: bool>(
    reducer: &Reducer<'_, F, I>,
    column: usize,
) -> Option<BoundDirection>
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    // Without a cost there is nothing to gain, and fixing the variable would only remove choice.
    let direction = reducer.problem.improving_direction(column)?;

    let is_free = reducer
        .problem
        .matrix
        .column(column)
        .all(|(row, coefficient)| {
            loosens(
                &reducer.problem.rows[row].relation,
                direction,
                coefficient.non_zero_signum(),
            )
        });

    is_free.then_some(direction)
}

/// Whether moving a variable in a direction can only loosen a row.
///
/// Decreasing a variable with a positive coefficient lowers the row's activity, which a `<=` row
/// can only welcome. An equality or a range pins the activity from both sides, so nothing is free.
fn loosens<F>(
    relation: &RangedConstraintRelation<F>,
    direction: BoundDirection,
    sign: NonZeroSign,
) -> bool {
    let activity_falls = matches!(
        (direction, sign),
        (BoundDirection::Lower, NonZeroSign::Positive)
            | (BoundDirection::Upper, NonZeroSign::Negative)
    );

    match relation {
        RangedConstraintRelation::Less => activity_falls,
        RangedConstraintRelation::Greater => !activity_falls,
        RangedConstraintRelation::Equal | RangedConstraintRelation::Range(_) => false,
    }
}
