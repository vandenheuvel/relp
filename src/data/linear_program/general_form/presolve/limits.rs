//! # Bounding the work presolve does
//!
//! Two things need bounding. One is time: a rule can be tried far more often than it succeeds.
//! The other, specific to exact arithmetic, is the size of the numbers. There is no rounding to
//! wash away a long chain of derivations, so a bound that is tightened over and over converges
//! towards a limit while its numerator and denominator grow without one. Feasibility based bound
//! tightening is known not to terminate finitely even on purely linear constraints, so this is a
//! real phenomenon rather than a precaution.
//!
//! The limits here bound the *length of the chains*, which is what growth is proportional to.
use enum_map::EnumMap;
use relp_num::{OrderedField, OrderedFieldRef};

use crate::data::linear_program::general_form::presolve::rule::RuleId;

/// How much work presolve may do.
///
/// Every one of these says how far presolving gets. None of them says what the answer is: a limit
/// is asked before a rule decides to act, never after, so what it can do is leave a reduction
/// undone. See the invariant in [`rule`](super::rule).
#[derive(Clone, Debug)]
pub struct Limits<F> {
    /// Rounds of "local rules to a fixed point, then the sweeps".
    pub max_rounds: u32,
    /// How often a single bound of a single variable may be tightened.
    ///
    /// This is what makes termination a fact rather than a hope: it is the only rule that is not
    /// monotone in the size of the problem.
    ///
    /// Asked of bounds a row still in the problem implies, and of those only: it is a question
    /// about whether to go on deriving, and only a rule that is deriving gets to be stopped. See
    /// [`Reducer::weigh_bound`](super::reducer::Reducer::weigh_bound).
    pub max_bound_tightenings: u32,
    /// How much tighter a bound has to get to be worth adopting, as a fraction of the width of the
    /// variable's current domain.
    ///
    /// An exact threshold is a modelling decision, not a numerical tolerance: it says that a
    /// hundredth of a percent off a domain is not worth a bigger denominator.
    ///
    /// Like [`Limits::max_bound_tightenings`], it is asked before a rule acts and only of a rule
    /// whose reduction is the bound itself.
    pub min_relative_tightening: F,
    /// New coefficients a single substitution may create.
    pub max_fill_in: usize,
    /// Candidate comparisons one sweep may make.
    pub max_sweep_comparisons: u64,
    /// How much larger a row's encoding may get when its fractions are cleared, as a fraction.
    ///
    /// Clearing denominators usually shrinks a row and always makes what is computed from it
    /// cheaper, but it is not free in general: the least common multiple of the denominators
    /// multiplies into every numerator, so a row whose denominators share nothing pays for all of
    /// them in every entry. Refusing the ones that grow is what keeps that bounded.
    pub max_scaling_growth: (u32, u32),
    /// Rules that may not run.
    ///
    /// Presolve bugs show up as a wrong answer on a problem that a dozen rules touched. Being able
    /// to turn rules off one at a time is what turns that into a question with an answer.
    pub disabled: EnumMap<RuleId, bool>,
}

impl<F> Limits<F>
where
    F: OrderedField,
    for<'r> &'r F: OrderedFieldRef<F>,
{
    /// Limits that let presolve run as long as it terminates at all.
    ///
    /// Only for tests that want to see what a rule does without a budget getting in the way.
    pub fn unlimited() -> Self {
        Self {
            max_rounds: u32::MAX,
            max_bound_tightenings: u32::MAX,
            min_relative_tightening: F::zero(),
            max_fill_in: usize::MAX,
            max_sweep_comparisons: u64::MAX,
            max_scaling_growth: (u32::MAX, 1),
            disabled: EnumMap::default(),
        }
    }

    /// The default limits, with only one rule allowed to run.
    ///
    /// Including one that the defaults turn off.
    pub fn only(rule: RuleId) -> Self {
        let mut limits = Self::default();
        for (other, disabled) in &mut limits.disabled {
            *disabled = other != rule;
        }
        limits
    }

    /// The default limits, with one rule turned on.
    pub fn with(rule: RuleId) -> Self {
        let mut limits = Self::default();
        limits.disabled[rule] = false;
        limits
    }

    /// The default limits, with one rule turned off.
    pub fn without(rule: RuleId) -> Self {
        let mut limits = Self::default();
        limits.disabled[rule] = true;
        limits
    }

    /// The default limits, with every rule turned off whose effect on the multipliers of the rows
    /// is not undone by postsolving.
    ///
    /// Reducing a problem and recovering a dual solution for it are not independent: a reduction
    /// changes what a valid multiplier is, and one whose effect is not undone leaves the
    /// multipliers of the reduced problem meaning nothing about the original. Rather than let that
    /// happen quietly, a run that needs a dual solution runs fewer rules.
    ///
    /// What is left is everything but one rule, which is not a matter of the recovery not being
    /// written: fixing a dominated column removes feasible points, so what is left has a different
    /// dual problem and there is nothing to recover.
    pub fn preserving_duals() -> Self {
        Self::without(RuleId::DominatedColumn)
    }
}

impl<F> Default for Limits<F>
where
    F: OrderedField,
    for<'r> &'r F: OrderedFieldRef<F>,
{
    fn default() -> Self {
        // 1 / 1024, built by halving so that no parsing or conversion is needed.
        let mut threshold = F::one();
        for _ in 0..10 {
            threshold = &threshold / &(F::one() + F::one());
        }

        Self {
            max_rounds: 8,
            max_bound_tightenings: 8,
            min_relative_tightening: threshold,
            // The number PaPILO uses for the same purpose.
            max_fill_in: 10,
            max_sweep_comparisons: 1 << 20,
            // Twice. A row of fractions that clears to integers of no greater total size has paid
            // for itself; one that doubles has not.
            max_scaling_growth: (2, 1),
            disabled: {
                let mut disabled = EnumMap::default();
                // Writing a row over a common denominator makes the problem smaller to hold and,
                // measured over netlib, four times slower to solve: the slacks and artificials the
                // algorithm synthesises do not scale when a row does, and a basis of cleared rows
                // is more expensive to factor than one of fractions. The rule says more. It is
                // here to be turned on by anyone who wants the smaller problem and can pay for it.
                disabled[RuleId::RowScaling] = true;
                disabled
            },
        }
    }
}

/// Which limit stopped presolve, when one did.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LimitHit {
    Rounds,
    BoundTightenings(RuleId),
    SweepComparisons(RuleId),
}
