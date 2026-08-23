//! # What presolve did, and what it cost
//!
//! Whether a reduction earns its place is an empirical question, and in exact arithmetic it is not
//! the question a floating point solver asks. Removing coefficients is not automatically progress
//! if the ones that remain got bigger. These counters are what makes that visible.
use std::fmt;

use enum_map::EnumMap;

use crate::data::linear_program::general_form::presolve::limits::LimitHit;
use crate::data::linear_program::general_form::presolve::rule::RuleId;

/// Per rule counters.
#[derive(Clone, Debug, Default)]
pub struct RuleStatistics {
    /// Places the rule was offered.
    pub triggers: u64,
    /// Places that had already been dealt with by the time the rule got there.
    pub stale_triggers: u64,
    /// Places the rule looked at and declined.
    pub misses: u64,
    /// Places the rule changed something.
    pub applications: u64,
    /// Places the rule was not allowed to try, because a limit was reached.
    pub budget_refusals: u64,

    pub rows_removed: u64,
    pub columns_removed: u64,
    pub bounds_tightened: u64,
    pub rows_changed: u64,
    /// Coefficients the rule created. The cost side of a substitution.
    pub nonzeros_created: u64,
}

impl RuleStatistics {
    /// Whether this rule did anything at all.
    fn was_used(&self) -> bool {
        self.triggers > 0 || self.applications > 0
    }
}

/// The size of a problem, for comparing before with after.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProblemSize {
    /// Number of constraints.
    pub rows: usize,
    /// Number of variables.
    pub columns: usize,
    /// Number of coefficients in the constraint matrix.
    pub nonzeros: usize,
}

/// What a presolve run did.
#[derive(Clone, Debug, Default)]
pub struct Statistics {
    /// What each rule was offered, did, and cost.
    pub per_rule: EnumMap<RuleId, RuleStatistics>,
    /// Rounds of local rules followed by sweeps.
    pub rounds: u32,
    /// Size of the problem presolving was given.
    pub before: ProblemSize,
    /// Size of the problem presolving left.
    pub after: ProblemSize,
    /// Which limit ended the run, when one did.
    pub stopped_by: Option<LimitHit>,
}

impl fmt::Display for Statistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "presolve: {} rounds, {}x{} with {} nonzeros -> {}x{} with {} nonzeros",
            self.rounds,
            self.before.rows,
            self.before.columns,
            self.before.nonzeros,
            self.after.rows,
            self.after.columns,
            self.after.nonzeros,
        )?;
        if let Some(limit) = self.stopped_by {
            writeln!(f, "  stopped by {:?}", limit)?;
        }

        writeln!(
            f,
            "  {:<28} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "rule", "tried", "applied", "rows-", "cols-", "nz+",
        )?;
        for (rule, statistics) in &self.per_rule {
            if !statistics.was_used() {
                continue;
            }
            writeln!(
                f,
                "  {:<28} {:>8} {:>8} {:>8} {:>8} {:>8}",
                rule.name(),
                statistics.triggers,
                statistics.applications,
                statistics.rows_removed,
                statistics.columns_removed,
                statistics.nonzeros_created,
            )?;
        }

        Ok(())
    }
}
