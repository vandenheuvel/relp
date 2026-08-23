//! # Variables that appear in one row and not in the objective
//!
//! Such a variable exists only to take up slack in its row. Eliminating it removes the column, and
//! weakens the row to whatever remains true once the variable is free to absorb what it can.
//!
//! # What the row becomes
//!
//! Write the row as `A + t (rel) b`, where `A` is the rest of the row and `t = c s` is the
//! contribution of the variable being removed. As `s` ranges over its bounds, `t` ranges over an
//! interval `[t_min, t_max]`, either end of which may be absent; a negative `c` swaps which of the
//! variable's own bounds gives which end.
//!
//! The row is then replaced by the condition on `A` alone that some admissible `t` exists:
//!
//! | relation | condition on `A` |
//! |---|---|
//! | `A + t == b` | `b - t_max <= A <= b - t_min` |
//! | `A + t <= b` | `A <= b - t_min` |
//! | `A + t >= b` | `A >= b - t_max` |
//! | `b - r <= A + t <= b` | `b - r - t_max <= A <= b - t_min` |
//!
//! Each is read with an absent end of the interval dropping the condition it appears in; when both
//! conditions fall away the row is satisfiable whatever the rest of the problem does, and goes.
//!
//! Note the `- r` in the lower half of the ranged case. It is the width of the range, and leaving
//! it out tightens the row past what it said, which is the kind of error that produces a confident
//! wrong answer rather than a failure.
use std::ops::Neg;

use relp_num::{NonZeroSign, OrderedFieldRef};

use crate::data::linear_program::elements::RangedConstraintRelation;
use crate::data::linear_program::general_form::presolve::postsolve::Reduction;
use crate::data::linear_program::general_form::presolve::problem::Problem;
use crate::data::linear_program::general_form::presolve::queues::{Queue, Queues};
use crate::data::linear_program::general_form::presolve::reducer::Reducer;
use crate::data::linear_program::general_form::presolve::rule::{
    Effect, LocalRule, RuleField, RuleId, Stop,
};

pub struct SingletonColumn;

impl SingletonColumn {
    const ID: RuleId = RuleId::SingletonColumn;
}

/// What has to happen to the row.
enum Outcome<F> {
    /// The row holds whatever the rest of the problem does.
    RowIsRedundant,
    /// The row becomes this relation with this right hand side.
    Weakened(RangedConstraintRelation<F>, F),
}

impl<F: RuleField, const I: bool> LocalRule<F, I> for SingletonColumn
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    type Trigger = usize;
    const ID: RuleId = RuleId::SingletonColumn;

    fn queue(queues: &mut Queues) -> &mut Queue<usize> {
        &mut queues.singleton_column
    }

    fn is_applicable(column: usize, problem: &Problem<F>) -> bool {
        let variable = &problem.columns[column];
        variable.is_active
            && problem.matrix.column_len(column) == 1
            && variable.cost.is_zero()
            // A variable with only one value left is not a slack, it is a constant. The rule that
            // substitutes it is both cheaper and correct; the reasoning below assumes the two
            // bounds are distinct, and a variable with a single value would give it a range of
            // width zero to work with.
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

        let right_hand_side = reducer.problem.rows[row].right_hand_side.clone();
        let relation = reducer.problem.rows[row].relation.clone();
        let lower_bound = reducer.problem.columns[column].lower_bound.clone();
        let upper_bound = reducer.problem.columns[column].upper_bound.clone();

        // The interval the variable's contribution to the row can cover.
        let scale = |bound: &Option<F>| bound.as_ref().map(|value| &coefficient * value);
        let (smallest, largest) = match coefficient.non_zero_signum() {
            NonZeroSign::Positive => (scale(&lower_bound), scale(&upper_bound)),
            NonZeroSign::Negative => (scale(&upper_bound), scale(&lower_bound)),
        };

        let outcome = match &relation {
            RangedConstraintRelation::Equal => match (&smallest, &largest) {
                (Some(smallest), Some(largest)) => Outcome::Weakened(
                    RangedConstraintRelation::Range(largest - smallest),
                    &right_hand_side - smallest,
                ),
                (Some(smallest), None) => Outcome::Weakened(
                    RangedConstraintRelation::Less,
                    &right_hand_side - smallest,
                ),
                (None, Some(largest)) => Outcome::Weakened(
                    RangedConstraintRelation::Greater,
                    &right_hand_side - largest,
                ),
                (None, None) => Outcome::RowIsRedundant,
            },
            RangedConstraintRelation::Less => match &smallest {
                Some(smallest) => Outcome::Weakened(
                    RangedConstraintRelation::Less,
                    &right_hand_side - smallest,
                ),
                None => Outcome::RowIsRedundant,
            },
            RangedConstraintRelation::Greater => match &largest {
                Some(largest) => Outcome::Weakened(
                    RangedConstraintRelation::Greater,
                    &right_hand_side - largest,
                ),
                None => Outcome::RowIsRedundant,
            },
            RangedConstraintRelation::Range(range) => match (&smallest, &largest) {
                (Some(smallest), Some(largest)) => Outcome::Weakened(
                    RangedConstraintRelation::Range(&(range + largest) - smallest),
                    &right_hand_side - smallest,
                ),
                (Some(smallest), None) => Outcome::Weakened(
                    RangedConstraintRelation::Less,
                    &right_hand_side - smallest,
                ),
                (None, Some(largest)) => Outcome::Weakened(
                    RangedConstraintRelation::Greater,
                    &(&right_hand_side - range) - largest,
                ),
                (None, None) => Outcome::RowIsRedundant,
            },
        };

        // The rest of the row, needed to work out what the variable absorbed.
        let coefficients = reducer
            .problem
            .matrix
            .row(row)
            .filter(|&(other, _)| other != column)
            .map(|(other, value)| (other, value.clone()))
            .collect();

        reducer.remove_column(column, Self::ID);
        reducer.postsolve.push(Reduction::ColumnSingleton {
            column,
            row,
            coefficient,
            coefficients,
            right_hand_side,
            lower_bound,
            upper_bound,
        });

        match outcome {
            Outcome::RowIsRedundant => {
                let coefficients = reducer.remove_row(row, Self::ID);
                reducer.postsolve.push(Reduction::RedundantRow {
                    row,
                    relation,
                    right_hand_side: reducer.problem.rows[row].right_hand_side.clone(),
                    coefficients,
                });
            }
            Outcome::Weakened(relation, right_hand_side) => {
                reducer.change_row(row, relation, right_hand_side, Self::ID);
            }
        }

        Ok(Effect::Reduced)
    }
}
