//! # Equalities with two coefficients
//!
//! `a_e x_e + a_k x_k == b` determines one variable from the other. Substituting
//! `x_e = (b - a_k x_k) / a_e` everywhere removes both a row and a column.
//!
//! This is sparse Gaussian elimination, and in exact arithmetic its cost is not the fill it
//! creates but the size of the numbers. Every coefficient of the eliminated column is multiplied
//! by `a_k / a_e`, so a chain of these multiplies denominators along its length. Entries after a
//! chain of length `k` are ratios of `k` by `k` minors of the original matrix, which is bounded,
//! but only as long as the chain is. The fill limit is what bounds it.
use std::ops::Neg;

use relp_num::OrderedFieldRef;

use crate::data::linear_program::elements::{
    BoundDirection, RangedConstraintRelation, VariableType,
};
use crate::data::linear_program::general_form::presolve::postsolve::Reduction;
use crate::data::linear_program::general_form::presolve::problem::Problem;
use crate::data::linear_program::general_form::presolve::queues::{Queue, Queues};
use crate::data::linear_program::general_form::presolve::reducer::{Origin, Reducer};
use crate::data::linear_program::general_form::presolve::rule::{
    Effect, LocalRule, RuleField, RuleId, Stop,
};

pub struct DoubletonEquation;

impl DoubletonEquation {
    const ID: RuleId = RuleId::DoubletonEquation;
}

impl<F: RuleField, const I: bool> LocalRule<F, I> for DoubletonEquation
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    type Trigger = usize;
    const ID: RuleId = RuleId::DoubletonEquation;

    fn queue(queues: &mut Queues) -> &mut Queue<usize> {
        &mut queues.doubleton_equation
    }

    fn is_applicable(row: usize, problem: &Problem<F>) -> bool {
        problem.rows[row].is_active
            && problem.matrix.row_len(row) == 2
            && matches!(problem.rows[row].relation, RangedConstraintRelation::Equal)
    }

    fn apply(row: usize, reducer: &mut Reducer<'_, F, I>) -> Result<Effect, Stop> {
        let entries = reducer
            .problem
            .matrix
            .row(row)
            .map(|(column, coefficient)| (column, coefficient.clone()))
            .collect::<Vec<_>>();
        let [(first, first_coefficient), (second, second_coefficient)] = entries.as_slice() else {
            unreachable!("checked by `is_applicable`");
        };

        // Eliminate whichever of the two costs less to substitute: every coefficient of the
        // eliminated column has to be touched, and every row it appears in may gain a coefficient.
        let (eliminated, coefficient_eliminated, kept, coefficient_kept) =
            if reducer.problem.matrix.column_len(*first)
                <= reducer.problem.matrix.column_len(*second)
            {
                (*first, first_coefficient, *second, second_coefficient)
            } else {
                (*second, second_coefficient, *first, first_coefficient)
            };
        let (coefficient_eliminated, coefficient_kept) =
            (coefficient_eliminated.clone(), coefficient_kept.clone());
        let right_hand_side = reducer.problem.rows[row].right_hand_side.clone();

        // `x_e = (b - a_k x_k) / a_e`, so `x_e` is a linear function of `x_k` with this slope.
        let slope = &(-&coefficient_kept) / &coefficient_eliminated;
        let offset = &right_hand_side / &coefficient_eliminated;

        if !is_allowed::<F, I>(reducer, eliminated, kept, &slope, &offset, row) {
            return Ok(Effect::None);
        }

        // The bounds of the eliminated variable become bounds on the kept one. What the kept one
        // had before that is what says, on the way back, which of the two a solution held at a
        // bound is being held by.
        let transplanted = transplant_bounds(reducer, eliminated, &slope, &offset);
        let kept_lower_bound = reducer.problem.columns[kept].lower_bound.clone();
        let kept_upper_bound = reducer.problem.columns[kept].upper_bound.clone();

        // Rewrite every other row the eliminated variable appears in.
        let occurrences = reducer
            .problem
            .matrix
            .column(eliminated)
            .filter(|&(other, _)| other != row)
            .map(|(other, coefficient)| (other, coefficient.clone()))
            .collect::<Vec<_>>();

        // The eliminated column, as it stands. Its reduced cost is what this row's multiplier is
        // solved from, and after the loop below its coefficients are gone.
        let occurrences_eliminated = occurrences.clone();

        for (other, coefficient) in occurrences {
            // `c x_e = c (offset + slope x_k)`: the constant moves to the right hand side and the
            // rest joins the kept column.
            let existing = reducer
                .problem
                .matrix
                .get(other, kept)
                .cloned()
                .unwrap_or_else(F::zero);
            let combined = &existing + &(&coefficient * &slope);

            reducer.set_coefficient(other, kept, combined, Self::ID);
            reducer.shift_right_hand_side(other, -&(&coefficient * &offset), Self::ID);
        }

        // The cost of the eliminated variable moves the same way.
        let cost = reducer.problem.columns[eliminated].cost.clone();
        if !cost.is_zero() {
            reducer.problem.columns[kept].cost += &cost * &slope;
            reducer.problem.fixed_cost += &cost * &offset;
        }

        reducer.remove_column(eliminated, Self::ID);
        reducer.remove_row(row, Self::ID);
        reducer.postsolve.push(Reduction::DoubletonEquation {
            row,
            eliminated,
            kept,
            coefficient_eliminated,
            coefficient_kept,
            right_hand_side,
            cost_eliminated: cost,
            occurrences_eliminated,
            kept_lower_bound,
            kept_upper_bound,
        });

        for (direction, value) in transplanted {
            // Not a bound any row's activity implies: it is this substitution, and the record just
            // pushed is what undoes it.
            reducer.tighten_bound_from(
                kept,
                direction,
                value,
                Origin::Substitution,
                Self::ID,
            )?;
        }
        reducer.enqueue_column(kept);

        Ok(Effect::Reduced)
    }
}

/// Whether the substitution is worth doing, and allowed to be done.
fn is_allowed<F: RuleField, const I: bool>(
    reducer: &Reducer<'_, F, I>,
    eliminated: usize,
    kept: usize,
    slope: &F,
    offset: &F,
    row: usize,
) -> bool
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    // Substituting into a row that does not already mention the kept variable creates a
    // coefficient. Too many of those and the problem gets denser and its numbers larger.
    let fill = reducer
        .problem
        .matrix
        .column(eliminated)
        .filter(|&(other, _)| other != row)
        .filter(|&(other, _)| reducer.problem.matrix.get(other, kept).is_none())
        .count();
    if fill > reducer.limits.max_fill_in {
        return false;
    }

    if I {
        // An integer variable may only be substituted away if every value the kept variable can
        // take sends it to an integer, which needs both the slope and the offset to be integral.
        let eliminated_is_integer =
            reducer.problem.columns[eliminated].variable_type == VariableType::Integer;
        let kept_is_integer = reducer.problem.columns[kept].variable_type == VariableType::Integer;

        if eliminated_is_integer && !(kept_is_integer && slope.is_integer() && offset.is_integer())
        {
            return false;
        }
    }

    true
}

/// Turn the bounds of the eliminated variable into bounds on the kept one.
///
/// `x_e = offset + slope * x_k`, so a bound on `x_e` becomes a bound on `x_k` in the same
/// direction when the slope is positive and the opposite one when it is negative.
fn transplant_bounds<F: RuleField, const I: bool>(
    reducer: &Reducer<'_, F, I>,
    eliminated: usize,
    slope: &F,
    offset: &F,
) -> Vec<(BoundDirection, F)>
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    if slope.is_zero() {
        return Vec::new();
    }
    let flips = slope.is_negative();

    [BoundDirection::Lower, BoundDirection::Upper]
        .into_iter()
        .filter_map(|direction| {
            let bound = reducer.problem.columns[eliminated].bound(direction)?;
            let value = &(bound - offset) / slope;
            Some((if flips { !direction } else { direction }, value))
        })
        .collect()
}
