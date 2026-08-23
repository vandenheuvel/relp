//! # Presolving linear programs
//!
//! Presolving replaces a linear program with a smaller one that has the same answer, and records
//! enough to turn a solution of the smaller problem back into a solution of the original.
//!
//! # How it is put together
//!
//! [`Problem`] is an owned, editable copy of the linear program. [`Reducer`] wraps it together
//! with everything that has to stay in step with it: the record of what has been done, the work
//! lists, the row activities and the counters.
//!
//! A rule is a [`LocalRule`], fired at a row or a column that something just happened to, or a
//! [`SweepRule`], which has to look at the whole problem to find what it is looking for. Local
//! rules are drained to a fixed point, cheapest first; the sweeps run once between rounds.
//!
//! What was done is recorded in a [`Postsolve`], an ordered list of [`Reduction`]s. Recovering a
//! solution is a walk back along it.
//!
//! # Exact arithmetic
//!
//! There is no rounding here, which removes a whole class of unsoundness: whether two rows are
//! multiples of each other, or an activity exactly reaches a right hand side, are decidable rather
//! than matters of tolerance. What it does not remove is cost. Numerators and denominators grow
//! along chains of derivations, and a bound that is tightened over and over converges to a limit
//! that it may never reach while its representation grows without one. [`Limits`] bounds the
//! length of those chains, which is what the growth is proportional to.
use std::ops::Neg;

use relp_num::OrderedFieldRef;

use crate::data::linear_program::general_form::presolve::limits::LimitHit;
use crate::data::linear_program::general_form::presolve::reducer::Reducer;
use crate::data::linear_program::general_form::presolve::rule::dominated_column::DominatedColumn;
use crate::data::linear_program::general_form::presolve::rule::domain_propagation::DomainPropagation;
use crate::data::linear_program::general_form::presolve::rule::doubleton_equation::DoubletonEquation;
use crate::data::linear_program::general_form::presolve::rule::duplicate_column::DuplicateColumn;
use crate::data::linear_program::general_form::presolve::rule::duplicate_row::DuplicateRow;
use crate::data::linear_program::general_form::presolve::rule::empty::{EmptyColumn, EmptyRow};
use crate::data::linear_program::general_form::presolve::rule::fixed_column::FixedColumn;
use crate::data::linear_program::general_form::presolve::rule::implied_free_singleton::ImpliedFreeColumnSingleton;
use crate::data::linear_program::general_form::presolve::rule::singleton_column::SingletonColumn;
use crate::data::linear_program::general_form::presolve::rule::singleton_row::SingletonRow;
use crate::data::linear_program::general_form::presolve::rule::{
    Effect, LocalRule, RuleField, Stop, SweepRule,
};

mod activity;
mod integrality;
mod limits;
mod matrix;
mod postsolve;
mod problem;
mod queues;
mod reducer;
mod rule;
pub(super) mod scale;
mod statistics;

pub use limits::Limits;
pub use postsolve::{BoundOrigin, Postsolve, Reduction};
pub(super) use problem::Problem;
pub use rule::RuleId;
pub use statistics::Statistics;

#[cfg(test)]
mod test;

/// What presolving concluded about a problem.
///
/// Not a `Result`: an infeasible or unbounded problem is an answer, not a failure to produce one.
#[derive(Debug, Eq, PartialEq)]
pub enum Presolved<F> {
    /// A smaller problem remains. Solve it, then recover a solution through the record presolve
    /// left behind.
    Reduced,
    /// Every variable was determined.
    Solved(crate::data::linear_program::solution::Solution<F>),
    /// The problem has no feasible point.
    Infeasible,
    /// The objective can be improved without limit.
    Unbounded {
        /// Whether this rests on the problem having a feasible point at all.
        ///
        /// Presolve finds a variable nothing holds back, which makes the problem unbounded exactly
        /// when it is feasible. Establishing that is not generally something presolve can do.
        assuming_feasible: bool,
    },
    /// One of the two, but which cannot be told apart any more.
    ///
    /// Reached when a rule that only preserves an optimal point ran, and what is left turned out
    /// to have no feasible point: the rule may have removed the ones there were.
    InfeasibleOrUnbounded,
}

/// Run every rule until nothing more can be done, or a limit says to stop.
pub(super) fn run<F: RuleField, const RESPECT_INTEGRALITY: bool>(
    reducer: &mut Reducer<'_, F, RESPECT_INTEGRALITY>,
) -> Result<(), Stop>
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    for round in 0..reducer.limits.max_rounds {
        reducer.statistics.rounds = round + 1;

        local_to_fixpoint(reducer)?;

        // The sweeps are the expensive rules, so they run on the smallest problem the cheap ones
        // could produce, and only once per round.
        let mut changed = false;
        changed |= sweep::<F, RESPECT_INTEGRALITY, DuplicateRow>(reducer)? == Effect::Reduced;
        changed |= sweep::<F, RESPECT_INTEGRALITY, DuplicateColumn>(reducer)? == Effect::Reduced;
        if reducer.allow_dual_reductions {
            changed |=
                sweep::<F, RESPECT_INTEGRALITY, DominatedColumn>(reducer)? == Effect::Reduced;
        }

        if !changed && reducer.queues.are_empty() {
            return Ok(());
        }
    }

    reducer.statistics.stopped_by = Some(LimitHit::Rounds);
    Ok(())
}

/// Drain the work lists until no local rule has anything left to do.
///
/// The order is by how much a rule costs against how likely it is to fire, and a rule that makes
/// the problem smaller sends the scan back to the top: after a row or a column goes, the cheap
/// rules may have work again, and they should get it before the expensive ones run.
fn local_to_fixpoint<F: RuleField, const I: bool>(
    reducer: &mut Reducer<'_, F, I>,
) -> Result<(), Stop>
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    loop {
        // Infeasibility is cheapest to notice on an empty row, so that rule comes before the ones
        // that could instead conclude the problem is unbounded.
        if drain::<F, I, EmptyRow>(reducer)? {
            continue;
        }
        if drain::<F, I, FixedColumn>(reducer)? {
            continue;
        }
        if drain::<F, I, SingletonRow>(reducer)? {
            continue;
        }
        if drain::<F, I, EmptyColumn>(reducer)? {
            continue;
        }
        if drain::<F, I, SingletonColumn>(reducer)? {
            continue;
        }
        if drain::<F, I, ImpliedFreeColumnSingleton>(reducer)? {
            continue;
        }
        if drain::<F, I, DoubletonEquation>(reducer)? {
            continue;
        }
        if drain::<F, I, DomainPropagation>(reducer)? {
            continue;
        }
        return Ok(());
    }
}

/// Offer one rule everything on its work list.
///
/// Returns whether the problem got smaller, in which case the caller should start again from the
/// cheapest rule.
fn drain<F: RuleField, const I: bool, R: LocalRule<F, I>>(
    reducer: &mut Reducer<'_, F, I>,
) -> Result<bool, Stop>
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    if reducer.limits.disabled[R::ID] {
        return Ok(false);
    }

    while let Some(trigger) = R::queue(&mut reducer.queues).pop() {
        reducer.statistics.per_rule[R::ID].triggers += 1;

        if !R::is_applicable(trigger, &reducer.problem) {
            reducer.statistics.per_rule[R::ID].stale_triggers += 1;
            continue;
        }

        match R::apply(trigger, reducer)? {
            Effect::None => reducer.statistics.per_rule[R::ID].misses += 1,
            effect => {
                reducer.statistics.per_rule[R::ID].applications += 1;
                if effect == Effect::Reduced {
                    return Ok(true);
                }
            }
        }
    }

    Ok(false)
}

/// Give one sweep rule a pass over the problem.
fn sweep<F: RuleField, const I: bool, R: SweepRule<F, I>>(
    reducer: &mut Reducer<'_, F, I>,
) -> Result<Effect, Stop>
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    if reducer.limits.disabled[R::ID] {
        return Ok(Effect::None);
    }
    R::sweep(reducer)
}

pub(super) use reducer::Reducer as PresolveReducer;
pub(super) use rule::Stop as PresolveStop;
