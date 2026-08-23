//! # Rows that are multiples of each other
//!
//! Two such rows say the same thing twice, at different scales. Intersecting what they admit
//! leaves one row, and can prove the problem infeasible when the two disagree.
//!
//! In floating point this rule needs a tolerance, and near the margin it has to guess. In exact
//! arithmetic "row `i` is `factor` times row `k`" is decidable, so there is nothing to guess: take
//! the factor from the first coefficient the two share and check the rest.
use std::collections::HashMap;
use std::ops::Neg;

use relp_num::OrderedFieldRef;

use crate::data::linear_program::elements::{BoundDirection, RangedConstraintRelation};
use crate::data::linear_program::general_form::presolve::postsolve::Reduction;
use crate::data::linear_program::general_form::presolve::reducer::Reducer;
use crate::data::linear_program::general_form::presolve::rule::{
    Effect, RuleField, RuleId, Stop, SweepRule,
};

pub struct DuplicateRow;

impl DuplicateRow {
    const ID: RuleId = RuleId::DuplicateRow;
}

impl<F: RuleField, const I: bool> SweepRule<F, I> for DuplicateRow
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    const ID: RuleId = RuleId::DuplicateRow;

    fn sweep(reducer: &mut Reducer<'_, F, I>) -> Result<Effect, Stop> {
        // Coefficients cannot be hashed, but the set of columns a row uses can, and two rows that
        // are multiples of each other use exactly the same columns.
        let mut by_support: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
        for row in 0..reducer.problem.rows.len() {
            if !reducer.problem.rows[row].is_active || reducer.problem.matrix.row_len(row) < 2 {
                continue;
            }
            let support = reducer.problem.matrix.row_indices(row).collect::<Vec<_>>();
            by_support.entry(support).or_default().push(row);
        }

        let mut comparisons = 0;
        let mut effect = Effect::None;

        for candidates in by_support.values() {
            for (position, &kept) in candidates.iter().enumerate() {
                for &removed in &candidates[position + 1..] {
                    if comparisons >= reducer.limits.max_sweep_comparisons {
                        reducer.statistics.per_rule[Self::ID].budget_refusals += 1;
                        return Ok(effect);
                    }
                    comparisons += 1;

                    if !reducer.problem.rows[kept].is_active
                        || !reducer.problem.rows[removed].is_active
                    {
                        continue;
                    }
                    reducer.statistics.per_rule[Self::ID].triggers += 1;

                    match factor_between(reducer, removed, kept) {
                        None => reducer.statistics.per_rule[Self::ID].misses += 1,
                        Some(factor) => {
                            merge(reducer, removed, kept, factor)?;
                            reducer.statistics.per_rule[Self::ID].applications += 1;
                            effect = Effect::Reduced;
                        }
                    }
                }
            }
        }

        Ok(effect)
    }
}

/// `f` such that row `removed` equals `f` times row `kept`, when there is one.
fn factor_between<F: RuleField, const I: bool>(
    reducer: &Reducer<'_, F, I>,
    removed: usize,
    kept: usize,
) -> Option<F>
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    let mut pairs = reducer
        .problem
        .matrix
        .row(removed)
        .zip(reducer.problem.matrix.row(kept));

    let ((_, first_removed), (_, first_kept)) = pairs.next()?;
    let factor = first_removed / first_kept;

    for ((_, from_removed), (_, from_kept)) in pairs {
        if from_removed != &(&factor * from_kept) {
            return None;
        }
    }

    Some(factor)
}

/// Fold what `removed` admits into `kept`, and drop `removed`.
fn merge<F: RuleField, const I: bool>(
    reducer: &mut Reducer<'_, F, I>,
    removed: usize,
    kept: usize,
    factor: F,
) -> Result<(), Stop>
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    // Both rows constrain the same quantity once `removed` is divided through by the factor, so
    // rewrite what `removed` admits in terms of `<a_kept, x>` and intersect.
    let (removed_low, removed_high) = admissible(
        &reducer.problem.rows[removed].relation,
        &reducer.problem.rows[removed].right_hand_side,
    );
    let rescale = |bound: Option<F>| bound.map(|value| &value / &factor);
    let (mut low, mut high) = (rescale(removed_low), rescale(removed_high));
    if factor.is_negative() {
        std::mem::swap(&mut low, &mut high);
    }

    let (kept_low, kept_high) = admissible(
        &reducer.problem.rows[kept].relation,
        &reducer.problem.rows[kept].right_hand_side,
    );
    let low = tightest(low, kept_low, BoundDirection::Lower);
    let high = tightest(high, kept_high, BoundDirection::Upper);

    if let (Some(low), Some(high)) = (&low, &high)
        && low > high
    {
        return Err(Stop::Infeasible);
    }

    let (relation, right_hand_side) = match (low, high) {
        (Some(low), Some(high)) if low == high => (RangedConstraintRelation::Equal, high),
        (Some(low), Some(high)) => (
            RangedConstraintRelation::Range(&high - &low),
            high,
        ),
        (Some(low), None) => (RangedConstraintRelation::Greater, low),
        (None, Some(high)) => (RangedConstraintRelation::Less, high),
        // Two rows that between them admit everything; the surviving one is redundant, and the
        // rule that recognises that will take it.
        (None, None) => (RangedConstraintRelation::Less, F::zero()),
    };

    // Only the one record. A `RedundantRow` alongside it would be undone last on the way back and
    // would overwrite, with a zero, exactly the multiplier the split had just worked out.
    let removed_relation = reducer.problem.rows[removed].relation.clone();
    let removed_right_hand_side = reducer.problem.rows[removed].right_hand_side.clone();
    let kept_relation = reducer.problem.rows[kept].relation.clone();
    let kept_right_hand_side = reducer.problem.rows[kept].right_hand_side.clone();

    reducer.remove_row(removed, RuleId::DuplicateRow);
    reducer.postsolve.push(Reduction::DuplicateRow {
        removed,
        kept,
        factor,
        removed_relation,
        removed_right_hand_side,
        kept_relation,
        kept_right_hand_side,
    });

    reducer.change_row(kept, relation, right_hand_side, RuleId::DuplicateRow);
    Ok(())
}

/// The interval a relation admits for `<a, x>`.
fn admissible<F: RuleField>(
    relation: &RangedConstraintRelation<F>,
    right_hand_side: &F,
) -> (Option<F>, Option<F>)
where
    for<'r> &'r F: OrderedFieldRef<F>,
{
    match relation {
        RangedConstraintRelation::Equal => {
            (Some(right_hand_side.clone()), Some(right_hand_side.clone()))
        }
        RangedConstraintRelation::Range(range) => (
            Some(right_hand_side - range),
            Some(right_hand_side.clone()),
        ),
        RangedConstraintRelation::Less => (None, Some(right_hand_side.clone())),
        RangedConstraintRelation::Greater => (Some(right_hand_side.clone()), None),
    }
}

/// The tighter of two ends, either of which may be absent.
fn tightest<F: RuleField>(left: Option<F>, right: Option<F>, direction: BoundDirection) -> Option<F>
where
    for<'r> &'r F: OrderedFieldRef<F>,
{
    match (left, right) {
        (None, other) | (other, None) => other,
        (Some(left), Some(right)) => Some(match direction {
            BoundDirection::Lower => {
                if left > right { left } else { right }
            }
            BoundDirection::Upper => {
                if left < right { left } else { right }
            }
        }),
    }
}
