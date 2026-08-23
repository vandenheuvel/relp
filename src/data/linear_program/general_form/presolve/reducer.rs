//! # The state a rule works on
//!
//! Rules do not touch the problem directly. They go through the `Reducer`, which keeps the
//! problem, the record of what has been done, the work lists and the activity bookkeeping in step
//! with each other. Getting that bookkeeping right in one place, rather than in each rule, is what
//! keeps the rules readable.
//!
//! Nothing here weighs what a rule asks for against what it costs, with the one exception of
//! [`Reducer::weigh_bound`], which a rule calls to make up its mind and not to be overruled after
//! it has. See the invariant in [`rule`](super::rule).
use std::ops::Neg;

use relp_num::{OrderedField, OrderedFieldRef};

use crate::data::linear_algebra::SparseTupleVec;
use crate::data::linear_algebra::traits::{SparseComparator, SparseElement};
use crate::data::linear_program::elements::{
    BoundDirection, RangedConstraintRelation, VariableType,
};
use crate::data::linear_program::general_form::presolve::activity::{Activity, contributes_to};
use crate::data::linear_program::general_form::presolve::integrality::Roundable;
use crate::data::linear_program::general_form::presolve::limits::Limits;
use crate::data::linear_program::general_form::presolve::postsolve::{
    BoundOrigin, Postsolve, Reduction,
};
use crate::data::linear_program::general_form::presolve::problem::Problem;
use crate::data::linear_program::general_form::presolve::queues::Queues;
use crate::data::linear_program::general_form::presolve::rule::{RuleId, Stop};
use crate::data::linear_program::general_form::presolve::statistics::Statistics;

/// What a bound a rule is considering is worth.
///
/// The answer to the only question a limit is ever asked, and it is asked before the rule has
/// touched anything. See [`Reducer::weigh_bound`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Worth {
    /// Worth writing: the variable has no bound that way yet, or this one is tighter by enough.
    Adopt,
    /// Nothing to write: what is there is at least as tight. Not a refusal; there is no news.
    Nothing,
    /// Tighter, but not by enough to pay for the numbers it would bring.
    NotEnough,
}

/// Where a rule says a bound it is writing came from.
///
/// Distinct from [`BoundOrigin`], which is what ends up in the record: the row variant is resolved
/// into a copy of the row here, and only for a bound that is actually written.
pub enum Origin<F> {
    /// Read off the activity of a row that is still part of the problem.
    Row(usize),
    /// Put there by a substitution, whose own record undoes it.
    Substitution,
    /// Already resolved by the caller, because the row it came from is gone.
    Given(BoundOrigin<F>),
}

/// Everything presolve is working with.
pub struct Reducer<'a, F, const RESPECT_INTEGRALITY: bool> {
    pub problem: Problem<F>,
    pub postsolve: Postsolve<F>,
    pub queues: Queues,
    pub activity: Activity<F>,
    pub limits: &'a Limits<F>,
    /// Whether rules that only preserve an optimal point, rather than the feasible set, may run.
    pub allow_dual_reductions: bool,
    pub statistics: Statistics,

    /// A variable found to improve the objective without limit, if one was.
    ///
    /// Noting it and carrying on, rather than stopping, is what lets presolve go on to discover
    /// that the problem has no feasible point at all — in which case that, and not unboundedness,
    /// is the answer.
    pub unbounded_if_feasible: Option<usize>,

    /// How often each bound has been tightened, so that a slowly converging chain is cut off.
    tightenings: Vec<(u32, u32)>,
}

impl<'a, F, const RESPECT_INTEGRALITY: bool> Reducer<'a, F, RESPECT_INTEGRALITY>
where
    F: OrderedField + SparseElement<F> + SparseComparator + Roundable,
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    /// Set up for a problem, with every rule's work list seeded.
    pub fn new(problem: Problem<F>, limits: &'a Limits<F>, allow_dual_reductions: bool) -> Self {
        let activity = Activity::new(&problem);
        let nr_rows = problem.matrix.nr_rows();
        let nr_columns = problem.matrix.nr_columns();

        let mut reducer = Self {
            postsolve: Postsolve::new(nr_rows, nr_columns),
            queues: Queues::default(),
            activity,
            limits,
            allow_dual_reductions,
            statistics: Statistics {
                before: problem.size(),
                ..Statistics::default()
            },
            unbounded_if_feasible: None,
            tightenings: vec![(0, 0); nr_columns],
            problem,
        };

        // Integrality is a property of the problem, not something a rule derives, so the bounds of
        // integer variables are made integral once here rather than checked everywhere.
        if RESPECT_INTEGRALITY {
            reducer.postsolve.respects_integrality = true;
            for column in 0..nr_columns {
                reducer.round_bounds(column);
            }
        }

        for row in 0..nr_rows {
            reducer.enqueue_row(row);
        }
        for column in 0..nr_columns {
            reducer.enqueue_column(column);
        }

        reducer
    }

    // ---------------------------------------------------------------- work lists

    /// Offer a row to every rule whose trigger it now meets.
    pub fn enqueue_row(&mut self, row: usize) {
        if !self.problem.rows[row].is_active {
            return;
        }

        // Every path that changes a row's numbers ends here, which is what makes this the place to
        // ask whether the row is still written the smallest way. A row that already is costs one
        // pass over it to establish that.
        self.queues.row_scaling.push(row);

        match self.problem.matrix.row_len(row) {
            0 => self.queues.empty_row.push(row),
            1 => self.queues.singleton_row.push(row),
            2 => {
                if matches!(
                    self.problem.rows[row].relation,
                    RangedConstraintRelation::Equal
                ) {
                    self.queues.doubleton_equation.push(row);
                }
            }
            _ => {}
        }

        // Propagation needs at least two coefficients to say anything the singleton row rule would
        // not say better.
        if self.problem.matrix.row_len(row) > 1 {
            for direction in [BoundDirection::Lower, BoundDirection::Upper] {
                if self.activity.missing(row, direction) <= 1 {
                    self.queues.activity.push((row, direction));
                }
            }
        }
    }

    /// Offer a column to every rule whose trigger it now meets.
    pub fn enqueue_column(&mut self, column: usize) {
        if !self.problem.columns[column].is_active {
            return;
        }

        if self.problem.columns[column].fixed_value().is_some() {
            self.queues.fixed_column.push(column);
        }

        match self.problem.matrix.column_len(column) {
            0 => self.queues.empty_column.push(column),
            1 => {
                if self.problem.columns[column].cost.is_zero() {
                    self.queues.singleton_column.push(column);
                } else {
                    self.queues.implied_free_singleton.push(column);
                }
            }
            _ => {}
        }
    }

    // ---------------------------------------------------------------- bounds

    /// Round the bounds of an integer variable inwards.
    fn round_bounds(&mut self, column: usize) {
        if self.problem.columns[column].variable_type != VariableType::Integer {
            return;
        }

        let variable = &mut self.problem.columns[column];
        if let Some(lower) = &variable.lower_bound {
            variable.lower_bound = Some(lower.ceil());
        }
        if let Some(upper) = &variable.upper_bound {
            variable.upper_bound = Some(upper.floor());
        }
    }

    /// Round a bound of an integer variable inwards, when that is what this is.
    ///
    /// `None` when nothing needs rounding, so that the continuous case costs no clone.
    fn integral(&self, column: usize, direction: BoundDirection, value: &F) -> Option<F> {
        if !RESPECT_INTEGRALITY
            || self.problem.columns[column].variable_type != VariableType::Integer
        {
            return None;
        }

        Some(match direction {
            BoundDirection::Lower => value.ceil(),
            BoundDirection::Upper => value.floor(),
        })
    }

    /// What a bound would be worth, asked before a rule commits to anything.
    ///
    /// This is the only place a limit may say no, and the only rule that asks is one whose whole
    /// reduction *is* the bound: the row it read the bound off stays in the problem, so declining
    /// costs a reduction and nothing else. A rule that is removing the row or the column the bound
    /// stands in for does not ask, because for it the bound is not an improvement to weigh but the
    /// place what is being removed now lives. See [`Reducer::set_bound`].
    pub fn weigh_bound(&self, column: usize, direction: BoundDirection, value: &F) -> Worth {
        let rounded = self.integral(column, direction, value);
        let value = rounded.as_ref().unwrap_or(value);

        let Some(existing) = self.problem.columns[column].bound(direction) else {
            // A variable that was unbounded that way is not: news of a kind no threshold has a
            // scale to weigh, and not a link in a chain of ever smaller improvements.
            return Worth::Adopt;
        };

        let is_tighter = match direction {
            BoundDirection::Lower => value > existing,
            BoundDirection::Upper => value < existing,
        };
        if !is_tighter {
            return Worth::Nothing;
        }

        // A chain of ever smaller improvements converges without terminating, and every link makes
        // the numbers bigger. Both limits below cut such a chain off: one by its length, one by
        // how little each link buys.
        let links = match direction {
            BoundDirection::Lower => self.tightenings[column].0,
            BoundDirection::Upper => self.tightenings[column].1,
        };
        if links >= self.limits.max_bound_tightenings {
            return Worth::NotEnough;
        }

        if !self.is_worth_adopting(column, &(value - existing)) {
            return Worth::NotEnough;
        }

        Worth::Adopt
    }

    /// Write a bound, because a rule has decided to.
    ///
    /// This is the single place a variable bound changes, and it weighs nothing: whether the rule
    /// was worth running was settled before it touched the problem, and a rule that has started is
    /// a rule that finishes. What happens here is everything that has to happen alongside a bound
    /// that moves: rounding for integer variables, the check that the two bounds have not crossed,
    /// the record for postsolve, invalidating the activity of every row the variable is in, and
    /// putting those rows back on the work lists.
    ///
    /// A bound no tighter than the one already there is not written. That is not a refusal: what
    /// the rule is saying is already being said, and more strongly, by what is there.
    pub fn set_bound(
        &mut self,
        column: usize,
        direction: BoundDirection,
        value: F,
        origin: Origin<F>,
        rule: RuleId,
    ) -> Result<(), Stop> {
        debug_assert!(self.problem.columns[column].is_active);

        let value = match self.integral(column, direction, &value) {
            Some(rounded) => rounded,
            None => value,
        };

        let old = self.problem.columns[column].bound(direction).cloned();
        let was_added = match &old {
            None => true,
            Some(existing) => {
                let is_tighter = match direction {
                    BoundDirection::Lower => &value > existing,
                    BoundDirection::Upper => &value < existing,
                };
                if !is_tighter {
                    return Ok(());
                }
                false
            }
        };

        // The length of the chain [`Reducer::weigh_bound`] cuts off. A bound written for a rule
        // that removed a row or a column is counted like any other, but it is not a link in such a
        // chain: there are at most as many of those as the problem has rows and columns.
        if !was_added {
            match direction {
                BoundDirection::Lower => self.tightenings[column].0 += 1,
                BoundDirection::Upper => self.tightenings[column].1 += 1,
            }
        }

        // Only now that the bound is going to be written is the row worth copying.
        let origin = match origin {
            Origin::Substitution => BoundOrigin::Substitution,
            Origin::Row(row) => BoundOrigin::Activity {
                row,
                coefficient: self
                    .problem
                    .matrix
                    .get(row, column)
                    .expect("a bound read off a row is read off a coefficient")
                    .clone(),
                coefficients: self
                    .problem
                    .matrix
                    .row(row)
                    .map(|(other, value)| (other, value.clone()))
                    .collect(),
            },
            Origin::Given(origin) => origin,
        };

        *self.problem.columns[column].bound_mut(direction) = Some(value.clone());
        self.postsolve.push(Reduction::BoundTightened {
            column,
            direction,
            old,
            new: value,
            origin,
        });
        self.statistics.per_rule[rule].bounds_tightened += 1;

        // Two bounds that have crossed mean there is no value left for the variable. Nothing else
        // checks for this, and a variable with no feasible value that is left in the problem
        // surfaces much later as a confusing failure.
        if self.problem.columns[column].is_infeasible() {
            return Err(Stop::Infeasible);
        }

        self.after_bound_change(column, direction, was_added);
        Ok(())
    }

    /// Whether a tightening buys enough to be worth the bigger numbers it costs.
    fn is_worth_adopting(&self, column: usize, difference: &F) -> bool {
        if self.limits.min_relative_tightening.is_zero() {
            return true;
        }

        let variable = &self.problem.columns[column];
        let scale = match (&variable.lower_bound, &variable.upper_bound) {
            (Some(lower), Some(upper)) => upper - lower,
            // With an unbounded domain there is no width to compare against, so compare against
            // the size of the bound itself, never letting the yardstick fall below one.
            _ => {
                let magnitude = match (&variable.lower_bound, &variable.upper_bound) {
                    (Some(bound), _) | (_, Some(bound)) => {
                        if bound.is_negative() { -bound } else { bound.clone() }
                    }
                    (None, None) => F::one(),
                };
                if magnitude > F::one() { magnitude } else { F::one() }
            }
        };

        let improvement = if difference.is_negative() {
            -difference
        } else {
            difference.clone()
        };
        improvement > &self.limits.min_relative_tightening * &scale
    }

    /// Keep the activity bookkeeping and the work lists in step with a bound that moved.
    fn after_bound_change(&mut self, column: usize, direction: BoundDirection, was_added: bool) {
        let affected = self
            .problem
            .matrix
            .column(column)
            .map(|(row, coefficient)| (row, coefficient.non_zero_signum()))
            .collect::<Vec<_>>();

        for (row, sign) in affected {
            if was_added {
                self.activity.bound_added(row, contributes_to(direction, sign));
            }
            // The cached activity was computed from the bound that just moved.
            self.activity.invalidate(row);
            self.enqueue_row(row);
        }

        if self.problem.columns[column].fixed_value().is_some() {
            self.queues.fixed_column.push(column);
        }
    }

    // ---------------------------------------------------------------- rows

    /// Change a row's relation and right hand side together.
    pub fn change_row(
        &mut self,
        row: usize,
        relation: RangedConstraintRelation<F>,
        right_hand_side: F,
        rule: RuleId,
    ) {
        debug_assert!(self.problem.rows[row].is_active);

        let old_relation = std::mem::replace(&mut self.problem.rows[row].relation, relation);
        let old_right_hand_side =
            std::mem::replace(&mut self.problem.rows[row].right_hand_side, right_hand_side);

        self.postsolve.push(Reduction::RowChanged {
            row,
            old_relation,
            old_right_hand_side,
        });
        self.statistics.per_rule[rule].rows_changed += 1;

        // The relation changed, so what the activity has to be compared against changed with it.
        self.activity.invalidate(row);
        self.enqueue_row(row);
    }

    /// Multiply a row through by a positive factor.
    ///
    /// The row admits exactly the points it did before, so no rule has to reconsider anything and
    /// no primal value changes. What changes is the size of the numbers, and what a multiplier for
    /// the row means, which is what the record is for.
    pub fn scale_row(&mut self, row: usize, factor: F, rule: RuleId) {
        debug_assert!(self.problem.rows[row].is_active);
        debug_assert!(factor > F::zero(), "a row may only be scaled by a positive factor");
        debug_assert!(!factor.is_one(), "scaling a row by one is not a reduction");

        self.problem.matrix.scale_row(row, &factor);

        let constraint = &mut self.problem.rows[row];
        constraint.right_hand_side *= &factor;
        if let RangedConstraintRelation::Range(range) = &mut constraint.relation {
            *range *= &factor;
        }

        self.postsolve.push(Reduction::RowScaled { row, factor });
        self.statistics.per_rule[rule].rows_scaled += 1;

        // The activity of this row moved by the same factor, but what is cached is a value rather
        // than a formula, so it is dropped instead of scaled.
        self.activity.invalidate(row);
        self.enqueue_row(row);
    }

    /// Move a row's right hand side by an amount.
    pub fn shift_right_hand_side(&mut self, row: usize, by: F, rule: RuleId) {
        let relation = self.problem.rows[row].relation.clone();
        let new = &self.problem.rows[row].right_hand_side + &by;
        self.change_row(row, relation, new, rule);
    }

    /// Take a row out of the problem, leaving its coefficients behind for postsolve.
    ///
    /// Every column the row held is offered back to the rules: losing a coefficient may have made
    /// it a singleton or emptied it.
    pub fn remove_row(&mut self, row: usize, rule: RuleId) -> SparseTupleVec<F> {
        debug_assert!(self.problem.rows[row].is_active);

        let coefficients = self.problem.matrix.take_row(row);
        self.problem.rows[row].is_active = false;
        self.activity.invalidate(row);
        self.statistics.per_rule[rule].rows_removed += 1;

        for &(column, _) in &coefficients {
            self.enqueue_column(column);
        }

        coefficients
    }

    /// Take a column out of the problem, leaving its coefficients behind for postsolve.
    ///
    /// Every row the column was in is offered back to the rules.
    pub fn remove_column(&mut self, column: usize, rule: RuleId) -> SparseTupleVec<F> {
        debug_assert!(self.problem.columns[column].is_active);

        // Which activity ends were waiting on this variable has to be read before it goes.
        let missing = [BoundDirection::Lower, BoundDirection::Upper]
            .map(|direction| self.problem.columns[column].bound(direction).is_none());

        let coefficients = self.problem.matrix.take_column(column);
        self.problem.columns[column].is_active = false;
        self.statistics.per_rule[rule].columns_removed += 1;

        for &(row, ref coefficient) in &coefficients {
            let sign = coefficient.non_zero_signum();
            for direction in [BoundDirection::Lower, BoundDirection::Upper] {
                let index = match direction {
                    BoundDirection::Lower => 0,
                    BoundDirection::Upper => 1,
                };
                if missing[index] {
                    self.activity
                        .missing_bound_left(row, contributes_to(direction, sign));
                }
            }

            // The row lost a term, so its cached activity no longer describes it.
            self.activity.invalidate(row);
            self.enqueue_row(row);
        }

        coefficients
    }

    /// Change a coefficient of the matrix.
    ///
    /// A coefficient appearing or disappearing changes which variables an activity end is waiting
    /// on, and a coefficient changing sign changes which of a variable's bounds it waits on. All
    /// three are handled by taking the old one out and putting the new one in.
    pub fn set_coefficient(&mut self, row: usize, column: usize, value: F, rule: RuleId) {
        let missing = [BoundDirection::Lower, BoundDirection::Upper]
            .map(|direction| self.problem.columns[column].bound(direction).is_none());

        if let Some(old) = self.problem.matrix.take(row, column) {
            let sign = old.non_zero_signum();
            for (index, direction) in [BoundDirection::Lower, BoundDirection::Upper]
                .into_iter()
                .enumerate()
            {
                if missing[index] {
                    self.activity
                        .missing_bound_left(row, contributes_to(direction, sign));
                }
            }
        }

        if !value.is_zero() {
            let sign = value.non_zero_signum();
            for (index, direction) in [BoundDirection::Lower, BoundDirection::Upper]
                .into_iter()
                .enumerate()
            {
                if missing[index] {
                    self.activity
                        .missing_bound_arrived(row, contributes_to(direction, sign));
                }
            }
            if self.problem.matrix.set(row, column, value).is_none() {
                self.statistics.per_rule[rule].nonzeros_created += 1;
            }
        }

        self.activity.invalidate(row);
        self.enqueue_row(row);
        self.enqueue_column(column);
    }

    /// Note that the bounds of a variable changed by something other than a tightening.
    ///
    /// Only the presence of a bound drives the counters, so a rule that replaces a bound with
    /// another one has nothing to count, but every row the variable is in has to forget the
    /// activity it worked out from the old value.
    pub fn bounds_replaced(&mut self, column: usize) {
        let rows = self
            .problem
            .matrix
            .column_indices(column)
            .collect::<Vec<_>>();
        for row in rows {
            self.activity.invalidate(row);
            self.enqueue_row(row);
        }
        self.enqueue_column(column);
    }

    /// Fold a variable held at a value into the right hand sides and the fixed cost.
    ///
    /// Used by every rule that determines a variable outright.
    pub fn substitute_value(&mut self, column: usize, value: &F, rule: RuleId) {
        let coefficients = self.problem.matrix.column(column).map(|(row, coefficient)| {
            (row, coefficient * value)
        }).collect::<Vec<_>>();

        for (row, shift) in coefficients {
            self.shift_right_hand_side(row, -&shift, rule);
        }
        self.problem.fixed_cost += &self.problem.columns[column].cost * value;
    }

}
