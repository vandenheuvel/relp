//! # Writing a row over a common denominator
//!
//! A constraint says the same thing multiplied by any positive number, so of all the ways to write
//! it, presolve may as well leave behind the cheapest. That is the one where the coefficients, the
//! right hand side and the range are integers with no common factor, and it is reached by dividing
//! the row by the greatest common divisor of all of them.
//!
//! # Why this is a rule and not a scaling pass
//!
//! A floating point solver scales for conditioning, once, before it starts. Exact arithmetic has
//! no conditioning to fix, but it has a cost that a floating point solver does not: a denominator
//! in the data is inherited by everything computed from it, and every rule here manufactures
//! denominators — `(b - <a, x>) / c` in the singleton rules, a column multiplied by `a2 / a1` in
//! the doubleton, a cost divided by a coefficient in the implied free singleton. Clearing them
//! where they appear is what stops a chain of reductions compounding, which the limits otherwise
//! have to bound by chain length alone.
//!
//! # Why it is off by default
//!
//! Two separate things go wrong, and the invariance argument only reaches the first of them.
//!
//! **The tableau is not invariant for a row that has a slack.** Multiplying row `i` by `l` and its
//! multiplier by `1 / l` does leave `B^-1 A` and `B^-1 b` exactly as they were, for every basis —
//! of the matrix that is read. The algorithm gives every inequality a slack and every row of the
//! first phase an artificial, and those are unit columns it synthesises rather than reads: they do
//! not scale when a row does. Scaling such a row is scaling those columns the other way, which is
//! a column scaling, and the route through the tableau really does change. GROW7 is 120 equalities
//! and 20 ranges; scaling the twenty takes its second phase from 74 pivots to 240.
//!
//! **A cleared basis is more expensive to factor than a basis of fractions**, even where the
//! tableau it yields is the same. SCSD1 is all equalities, so nothing but the artificials is
//! affected, and clearing its rows leaves the second phase taking *fewer* pivots — 164 against 200
//! — and three times as long, because each one costs 38 milliseconds instead of 14. Clearing
//! denominators makes a problem smaller to hold and its basis matrices larger to work with, and
//! those are not the same quantity.
//!
//! Over 28 netlib instances the two together take 61 seconds of solving to 277; GROW7 alone goes
//! from 10 seconds to 199. Confined to equality rows, where only the second effect is left, the
//! same 28 take 79 seconds. What the rule buys is a smaller problem to hold: a fifth to two fifths
//! off the encoding of AGG, AGG2, AGG3, GROW7, CZPROB, SCSD1 and SCSD6, and a little onto BANDM,
//! BLEND and SHARE1B. Worth having, and worth asking for rather than getting.
//!
//! Removing the first effect would mean scaling the synthesised columns along with the row, which
//! is a change to the matrix provider and to the first phase rather than to presolving. The second
//! would remain.
use std::ops::Neg;

use relp_num::OrderedFieldRef;

use crate::data::linear_program::elements::RangedConstraintRelation;
use crate::data::linear_program::general_form::presolve::problem::Problem;
use crate::data::linear_program::general_form::presolve::queues::{Queue, Queues};
use crate::data::linear_program::general_form::presolve::reducer::Reducer;
use crate::data::linear_program::general_form::presolve::rule::{
    Effect, LocalRule, RuleField, RuleId, Stop,
};
use crate::data::linear_program::general_form::presolve::simplify::worth_clearing;

pub struct RowScaling;

impl RowScaling {
    const ID: RuleId = RuleId::RowScaling;
}

impl<F: RuleField, const I: bool> LocalRule<F, I> for RowScaling
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    type Trigger = usize;

    const ID: RuleId = RuleId::RowScaling;

    fn queue(queues: &mut Queues) -> &mut Queue<usize> {
        &mut queues.row_scaling
    }

    fn is_applicable(row: usize, problem: &Problem<F>) -> bool {
        problem.rows[row].is_active && problem.matrix.row_len(row) > 0
    }

    fn apply(row: usize, reducer: &mut Reducer<'_, F, I>) -> Result<Effect, Stop> {
        let values = values(row, &reducer.problem);
        let Some(factor) = F::simplifying_factor(values.iter().copied()) else {
            return Ok(Effect::None);
        };

        if !worth_clearing(&values, &factor, reducer.limits.max_scaling_growth) {
            reducer.statistics.per_rule[Self::ID].budget_refusals += 1;
            return Ok(Effect::None);
        }

        drop(values);
        reducer.scale_row(row, factor, Self::ID);

        // Nothing left and nothing arrived, so this is not a reduction in the sense the driver
        // means: no rule has work it did not have before.
        Ok(Effect::Tightened)
    }
}

/// Everything about a row that a common denominator has to be found for.
///
/// The right hand side and the range are in here beside the coefficients, and deliberately: a
/// fraction anywhere in `[a | b]` is inherited by every basis inverse computed from the row, so
/// clearing the coefficients alone would leave the expensive part behind. It can make the
/// coefficients larger than clearing them alone would, which is the trade the guard weighs.
fn values<F>(row: usize, problem: &Problem<F>) -> Vec<&F> {
    let constraint = &problem.rows[row];

    let mut values = problem
        .matrix
        .row(row)
        .map(|(_, coefficient)| coefficient)
        .collect::<Vec<_>>();
    values.push(&constraint.right_hand_side);
    if let RangedConstraintRelation::Range(range) = &constraint.relation {
        values.push(range);
    }

    values
}
