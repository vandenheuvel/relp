//! # Presolve rules
//!
//! A rule recognises a shape in the problem and replaces it with something smaller. There are two
//! kinds, differing in how they are scheduled rather than in what they do.
//!
//! A [`LocalRule`] fires at a place: a row, a column, or one end of a row's activity. It is
//! offered the places where something changed, and is drained to a fixed point.
//!
//! A [`SweepRule`] has no such place to be offered. Recognising that two rows are multiples of
//! each other means comparing rows with each other, so the rule sorts or hashes the whole problem
//! and runs once per round.
use std::hash::Hash;
use std::ops::Neg;

use enum_map::Enum;
use relp_num::{OrderedField, OrderedFieldRef};

use crate::data::linear_algebra::traits::{SparseComparator, SparseElement};
use crate::data::linear_program::general_form::presolve::integrality::Roundable;
use crate::data::linear_program::general_form::presolve::problem::Problem;
use crate::data::linear_program::general_form::presolve::queues::{Queue, Queues};
use crate::data::linear_program::general_form::presolve::reducer::Reducer;

pub mod dominated_column;
pub mod domain_propagation;
pub mod doubleton_equation;
pub mod duplicate_column;
pub mod duplicate_row;
pub mod empty;
pub mod fixed_column;
pub mod implied_free_singleton;
pub mod singleton_column;
pub mod singleton_row;

/// Which rule something is attributed to.
///
/// Used to index counters and budgets. Never matched on to decide behaviour.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Enum)]
pub enum RuleId {
    /// A row with no coefficients left.
    EmptyRow,
    /// A column with no coefficients left.
    EmptyColumn,
    /// A variable whose bounds leave it one value.
    FixedColumn,
    /// A row with one coefficient, which is a bound in disguise.
    SingletonRow,
    /// A variable in one row and not in the objective.
    SingletonColumn,
    /// Reasoning from what a row's activity can reach.
    DomainPropagation,
    /// A variable in one equality row that already confines it, and in the objective.
    ImpliedFreeColumnSingleton,
    /// An equality with two coefficients.
    DoubletonEquation,
    /// A row that is a multiple of another row.
    DuplicateRow,
    /// A column that is a multiple of another column.
    DuplicateColumn,
    /// A variable that only ever loosens the rows it is in.
    DominatedColumn,
}

impl RuleId {
    /// A name to print in a report.
    pub fn name(self) -> &'static str {
        match self {
            RuleId::EmptyRow => "empty row",
            RuleId::EmptyColumn => "empty column",
            RuleId::FixedColumn => "fixed column",
            RuleId::SingletonRow => "singleton row",
            RuleId::SingletonColumn => "singleton column",
            RuleId::DomainPropagation => "domain propagation",
            RuleId::ImpliedFreeColumnSingleton => "implied free singleton",
            RuleId::DoubletonEquation => "doubleton equation",
            RuleId::DuplicateRow => "duplicate row",
            RuleId::DuplicateColumn => "duplicate column",
            RuleId::DominatedColumn => "dominated column",
        }
    }
}

/// What one application of a rule achieved.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Effect {
    /// Nothing. The place no longer had the shape the rule looks for.
    None,
    /// The problem got strictly smaller: a row or a column is gone.
    Reduced,
    /// Progress that does not shrink the problem: a bound moved, a relation was weakened.
    Tightened,
}

/// Why presolve stopped before running out of work.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Stop {
    /// The problem has no feasible point.
    Infeasible,
    /// A variable improves the objective without limit.
    ///
    /// Only an answer about the whole problem if the problem is feasible, which presolve cannot in
    /// general establish. Whoever reports this has to say so.
    UnboundedIfFeasible { column: usize },
}

/// The bounds every rule needs of its number type.
pub trait RuleField:
    OrderedField + SparseElement<Self> + SparseComparator + Roundable
{
}

impl<F> RuleField for F where F: OrderedField + SparseElement<F> + SparseComparator + Roundable {}

/// A rule that fires at a place in the problem.
///
/// Implemented by a type that is never instantiated: every item is an associated function, so
/// there is no state and no `self`.
pub trait LocalRule<F: RuleField, const RESPECT_INTEGRALITY: bool>
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    /// What identifies one application.
    type Trigger: Copy + Eq + Hash + std::fmt::Debug;

    const ID: RuleId;

    /// The work list this rule drains.
    ///
    /// Naming it here is what lets the driver's drain loop be written once for every rule rather
    /// than once per rule.
    fn queue(queues: &mut Queues) -> &mut Queue<Self::Trigger>;

    /// Whether the place still has the shape this rule fires on.
    ///
    /// Work lists are added to eagerly, so a place is often dealt with before it is reached.
    fn is_applicable(trigger: Self::Trigger, problem: &Problem<F>) -> bool;

    /// Do it.
    fn apply(
        trigger: Self::Trigger,
        reducer: &mut Reducer<'_, F, RESPECT_INTEGRALITY>,
    ) -> Result<Effect, Stop>;
}

/// A rule that has to look at the whole problem.
pub trait SweepRule<F: RuleField, const RESPECT_INTEGRALITY: bool>
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    const ID: RuleId;

    /// One pass over the problem.
    fn sweep(reducer: &mut Reducer<'_, F, RESPECT_INTEGRALITY>) -> Result<Effect, Stop>;
}
