//! # Columns that are multiples of each other
//!
//! When column `p` equals `factor` times column `q`, the two variables enter every row only
//! through the combination `x_p + factor * x_q`. If their costs stand in the same ratio, the
//! objective sees only that combination too, and the pair can be replaced by a single variable
//! whose domain is the sum of theirs.
//!
//! As with duplicate rows, the proportionality test needs no tolerance here.
use std::collections::HashMap;
use std::ops::Neg;

use relp_num::OrderedFieldRef;

use crate::data::linear_program::elements::VariableType;
use crate::data::linear_program::general_form::presolve::postsolve::Reduction;
use crate::data::linear_program::general_form::presolve::reducer::Reducer;
use crate::data::linear_program::general_form::presolve::rule::{
    Effect, RuleField, RuleId, Stop, SweepRule,
};

pub struct DuplicateColumn;

impl DuplicateColumn {
    const ID: RuleId = RuleId::DuplicateColumn;
}

impl<F: RuleField, const I: bool> SweepRule<F, I> for DuplicateColumn
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    const ID: RuleId = RuleId::DuplicateColumn;

    fn sweep(reducer: &mut Reducer<'_, F, I>) -> Result<Effect, Stop> {
        let mut by_support: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
        for column in 0..reducer.problem.columns.len() {
            if !reducer.problem.columns[column].is_active
                || reducer.problem.matrix.column_len(column) == 0
            {
                continue;
            }
            let support = reducer
                .problem
                .matrix
                .column_indices(column)
                .collect::<Vec<_>>();
            by_support.entry(support).or_default().push(column);
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

                    if !reducer.problem.columns[kept].is_active
                        || !reducer.problem.columns[removed].is_active
                    {
                        continue;
                    }
                    reducer.statistics.per_rule[Self::ID].triggers += 1;

                    match mergeable::<F, I>(reducer, removed, kept) {
                        None => reducer.statistics.per_rule[Self::ID].misses += 1,
                        Some(factor) => {
                            merge(reducer, removed, kept, factor);
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

/// The factor by which the two columns are parallel, when merging them is sound.
fn mergeable<F: RuleField, const I: bool>(
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
        .column(removed)
        .zip(reducer.problem.matrix.column(kept));

    let ((_, first_removed), (_, first_kept)) = pairs.next()?;
    let factor = first_removed / first_kept;

    for ((_, from_removed), (_, from_kept)) in pairs {
        if from_removed != &(&factor * from_kept) {
            return None;
        }
    }

    // The objective has to see the same combination the rows do. When it does not, one of the two
    // dominates the other, which is a different argument with a different postsolve: it is left to
    // the rule that makes it.
    if reducer.problem.columns[removed].cost
        != &factor * &reducer.problem.columns[kept].cost
    {
        return None;
    }

    if I {
        // The merged domain has to be free of gaps. For integer variables that needs the factor to
        // be one either way, and both variables integer.
        let either_is_integer = reducer.problem.columns[removed].variable_type
            == VariableType::Integer
            || reducer.problem.columns[kept].variable_type == VariableType::Integer;
        if either_is_integer {
            let both_integer = reducer.problem.columns[removed].variable_type
                == VariableType::Integer
                && reducer.problem.columns[kept].variable_type == VariableType::Integer;
            let unit = factor == F::one() || factor == -F::one();
            if !(both_integer && unit) {
                return None;
            }
        }
    }

    // Both halves of the merged domain have to be intervals that add up to one interval, which
    // they always do, but an unbounded piece has to be handled by the split at postsolve, so keep
    // to the case where the removed variable is bounded on both sides.
    reducer.problem.columns[removed].lower_bound.as_ref()?;
    reducer.problem.columns[removed].upper_bound.as_ref()?;

    Some(factor)
}

/// Replace the pair by a single variable standing for `x_kept + factor * x_removed`.
fn merge<F: RuleField, const I: bool>(
    reducer: &mut Reducer<'_, F, I>,
    removed: usize,
    kept: usize,
    factor: F,
) where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    let removed_lower_bound = reducer.problem.columns[removed].lower_bound.clone();
    let removed_upper_bound = reducer.problem.columns[removed].upper_bound.clone();
    let kept_lower_bound = reducer.problem.columns[kept].lower_bound.clone();
    let kept_upper_bound = reducer.problem.columns[kept].upper_bound.clone();

    // The interval `factor * x_removed` covers, with a negative factor turning it around.
    let scaled = |bound: &Option<F>| bound.as_ref().map(|value| &factor * value);
    let (offset_low, offset_high) = if factor.is_negative() {
        (scaled(&removed_upper_bound), scaled(&removed_lower_bound))
    } else {
        (scaled(&removed_lower_bound), scaled(&removed_upper_bound))
    };

    reducer.problem.columns[kept].lower_bound = match (&kept_lower_bound, &offset_low) {
        (Some(bound), Some(offset)) => Some(bound + offset),
        _ => None,
    };
    reducer.problem.columns[kept].upper_bound = match (&kept_upper_bound, &offset_high) {
        (Some(bound), Some(offset)) => Some(bound + offset),
        _ => None,
    };

    // The kept variable's bounds moved, so any activity worked out from the old ones is stale.
    reducer.bounds_replaced(kept);
    reducer.remove_column(removed, RuleId::DuplicateColumn);
    reducer.postsolve.push(Reduction::DuplicateColumn {
        removed,
        kept,
        factor,
        removed_lower_bound,
        removed_upper_bound,
        kept_lower_bound,
        kept_upper_bound,
    });

    reducer.enqueue_column(kept);
}
