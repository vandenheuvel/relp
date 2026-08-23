//! # Rows with a single coefficient
//!
//! `a x (<= == >= =r=) b` says nothing about the problem's structure; it is a bound on `x`. Moving
//! it into the variable's own bounds removes a row.
use std::ops::Neg;

use relp_num::{NonZeroSign, OrderedFieldRef};

use crate::data::linear_program::elements::{BoundDirection, RangedConstraintRelation};
use crate::data::linear_program::general_form::presolve::postsolve::{BoundOrigin, Reduction};
use crate::data::linear_program::general_form::presolve::problem::Problem;
use crate::data::linear_program::general_form::presolve::queues::{Queue, Queues};
use crate::data::linear_program::general_form::presolve::reducer::{Origin, Reducer};
use crate::data::linear_program::general_form::presolve::rule::{
    Effect, LocalRule, RuleField, RuleId, Stop,
};

pub struct SingletonRow;

impl SingletonRow {
    const ID: RuleId = RuleId::SingletonRow;
}

impl<F: RuleField, const I: bool> LocalRule<F, I> for SingletonRow
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    type Trigger = usize;
    const ID: RuleId = RuleId::SingletonRow;

    fn queue(queues: &mut Queues) -> &mut Queue<usize> {
        &mut queues.singleton_row
    }

    fn is_applicable(row: usize, problem: &Problem<F>) -> bool {
        problem.rows[row].is_active && problem.matrix.row_len(row) == 1
    }

    fn apply(row: usize, reducer: &mut Reducer<'_, F, I>) -> Result<Effect, Stop> {
        let (column, coefficient) = {
            let (column, coefficient) = reducer
                .problem
                .matrix
                .row(row)
                .next()
                .expect("checked by `is_applicable`");
            (column, coefficient.clone())
        };

        let relation = reducer.problem.rows[row].relation.clone();
        let right_hand_side = reducer.problem.rows[row].right_hand_side.clone();
        let sign = coefficient.non_zero_signum();

        // Dividing by the coefficient turns the relation around when it is negative.
        let upper_end = &right_hand_side / &coefficient;
        let lower_end = match &relation {
            RangedConstraintRelation::Range(range) => {
                Some(&(&right_hand_side - range) / &coefficient)
            }
            _ => None,
        };

        // Which end of the relation bounds the variable from below and which from above.
        let implied = match (&relation, sign) {
            (RangedConstraintRelation::Equal, _) => vec![
                (BoundDirection::Lower, upper_end.clone()),
                (BoundDirection::Upper, upper_end),
            ],
            (RangedConstraintRelation::Less, NonZeroSign::Positive)
            | (RangedConstraintRelation::Greater, NonZeroSign::Negative) => {
                vec![(BoundDirection::Upper, upper_end)]
            }
            (RangedConstraintRelation::Less, NonZeroSign::Negative)
            | (RangedConstraintRelation::Greater, NonZeroSign::Positive) => {
                vec![(BoundDirection::Lower, upper_end)]
            }
            (RangedConstraintRelation::Range(_), NonZeroSign::Positive) => vec![
                (BoundDirection::Lower, lower_end.expect("a range has both ends")),
                (BoundDirection::Upper, upper_end),
            ],
            (RangedConstraintRelation::Range(_), NonZeroSign::Negative) => vec![
                (BoundDirection::Lower, upper_end),
                (BoundDirection::Upper, lower_end.expect("a range has both ends")),
            ],
        };

        // The row goes first, so that the bounds are not derived from a row that still claims to
        // constrain the variable. That leaves nothing for `tighten_bound` to read the row from, but
        // a row with one coefficient is no trouble to write down.
        let origin = BoundOrigin::Activity {
            row,
            coefficient: coefficient.clone(),
            coefficients: vec![(column, coefficient.clone())],
        };

        reducer.remove_row(row, Self::ID);
        reducer.postsolve.push(Reduction::SingletonRow {
            row,
            column,
            coefficient,
            relation,
            right_hand_side,
        });

        for (direction, value) in implied {
            reducer.tighten_bound_from(
                column,
                direction,
                value,
                Origin::Given(origin.clone()),
                Self::ID,
            )?;
        }
        reducer.enqueue_column(column);

        Ok(Effect::Reduced)
    }
}
