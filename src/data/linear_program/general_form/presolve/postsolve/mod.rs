//! # Undoing what presolving did
//!
//! Presolve records what it does as an ordered list of reductions. Recovering a solution to the
//! original problem from a solution to the reduced one is a walk back along that list.
//!
//! The ordering is what makes this work. When a reduction is recorded, everything it refers to is
//! still part of the problem, so anything removed after it is restored before it is reached on the
//! way back. The dependencies between reductions are therefore resolved by construction, and no
//! separate argument about cycles is needed.
//!
//! Every index stored here refers to the problem as it was when presolving started. Nothing is
//! renumbered along the way.
use crate::data::linear_algebra::SparseTupleVec;
use crate::data::linear_program::elements::{BoundDirection, RangedConstraintRelation};
use crate::data::linear_program::general_form::presolve::statistics::Statistics;

mod dual;
mod primal;

#[cfg(test)]
mod test;

/// Where a bound presolving installed came from.
///
/// A bound the original problem does not have is a constraint the original problem does not have.
/// A solution held at one has a reduced cost that has to be accounted for somewhere else, and this
/// says where.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BoundOrigin<F> {
    /// Read off what a row's activity can reach.
    ///
    /// The multiplier of a solution held at this bound belongs on that row. Making that work needs
    /// the row as it was when the bound was derived, because pushing a multiplier onto a row moves
    /// the reduced cost of every other column in it.
    Activity {
        /// The row the bound was read off.
        row: usize,
        /// The bounded column's coefficient in that row.
        coefficient: F,
        /// That row, as it was at the time.
        coefficients: SparseTupleVec<F>,
    },
    /// Put there by a substitution, along with the record that describes the substitution.
    ///
    /// The reduction recorded immediately after this one says what to do with a solution held
    /// here, so there is nothing for the bound itself to carry.
    Substitution,
}

/// One thing presolve did to the problem.
///
/// Variants have named fields so that data needed to recover dual values and basis statuses can be
/// added later, field by field, without disturbing what is already here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Reduction<F> {
    /// A column with no coefficients was set to its best feasible value.
    EmptyColumn {
        /// Which column.
        column: usize,
        /// The value it was given.
        value: F,
        /// What it cost. Not needed to recover the primal value; recovering a reduced cost is.
        cost: F,
    },

    /// A row with no coefficients was found satisfied and dropped.
    EmptyRow {
        /// Which row.
        row: usize,
    },

    /// A column whose bounds left it one value was substituted out.
    FixedColumn {
        /// Which column.
        column: usize,
        /// The one value its bounds left it.
        value: F,
        /// The coefficients the column had. Not needed to recover the primal value; kept because
        /// recovering a reduced cost needs them.
        coefficients: SparseTupleVec<F>,
        /// What it cost, for the same reason.
        cost: F,
    },

    /// A row with a single coefficient was turned into a bound on that variable and dropped.
    SingletonRow {
        /// Which row.
        row: usize,
        /// The one variable it was about.
        column: usize,
        /// That variable’s coefficient in the row.
        coefficient: F,
        /// The relation the row had.
        relation: RangedConstraintRelation<F>,
        /// The right hand side the row had.
        right_hand_side: F,
    },

    /// A variable bound was tightened.
    BoundTightened {
        /// Which variable.
        column: usize,
        /// Which of its two bounds moved.
        direction: BoundDirection,
        /// What it was, when there was one.
        old: Option<F>,
        /// What it became.
        new: F,
        /// Where the new bound came from.
        origin: BoundOrigin<F>,
    },

    /// A row's relation, right hand side, or both were changed.
    RowChanged {
        /// Which row.
        row: usize,
        /// The relation it had.
        old_relation: RangedConstraintRelation<F>,
        /// The right hand side it had.
        old_right_hand_side: F,
    },

    /// A row that the variable bounds already imply was dropped.
    RedundantRow {
        /// Which row.
        row: usize,
        /// The relation it had.
        relation: RangedConstraintRelation<F>,
        /// The right hand side it had.
        right_hand_side: F,
        /// The coefficients it had.
        coefficients: SparseTupleVec<F>,
    },

    /// A zero cost column singleton was removed.
    ///
    /// The variable absorbs whatever the rest of its row leaves over, which is
    /// `(right_hand_side - <coefficients, x>) / coefficient`, clamped to its own bounds.
    ///
    /// The clamp is not decoration. Where the row was dropped entirely, it was dropped because the
    /// variable could always take up the slack in the unbounded direction, which says nothing
    /// about the exact residual being within the bounds. Where the row survived in a weakened
    /// form, the residual can likewise fall outside the bound the row no longer mentions. In both
    /// cases moving the value to the nearer bound still satisfies the original row, because it
    /// only moves it in the direction the relation leaves free.
    ColumnSingleton {
        /// Which column.
        column: usize,
        /// The one row it was in.
        row: usize,
        /// Its coefficient in that row.
        coefficient: F,
        /// The rest of the row, excluding `column`.
        coefficients: SparseTupleVec<F>,
        /// The right hand side that row had.
        right_hand_side: F,
        /// Its lower bound, when it had one.
        lower_bound: Option<F>,
        /// Its upper bound, when it had one.
        upper_bound: Option<F>,
    },

    /// A column singleton with a cost, whose bounds its row already implied, was substituted out.
    ///
    /// The cost was moved into the other columns of the row at reduction time, so recovering the
    /// value is the same computation as for a slack, without the clamp: the bounds were implied,
    /// so the residual respects them.
    ImpliedFreeColumnSingleton {
        /// Which column.
        column: usize,
        /// The one row it was in.
        row: usize,
        /// Its coefficient in that row.
        coefficient: F,
        /// The rest of that row.
        coefficients: SparseTupleVec<F>,
        /// The right hand side that row had.
        right_hand_side: F,
        /// What the column cost before the substitution moved it onto the others.
        ///
        /// The multiplier of the row follows from it: the column is in no other row, so its dual
        /// equation reads `cost - coefficient * y = 0`, the reduced cost being zero because the
        /// row confined the variable and its own bounds were dropped.
        cost: F,
    },

    /// An equality row with two coefficients was used to eliminate one of them.
    ///
    /// `x_eliminated = (right_hand_side - coefficient_kept * x_kept) / coefficient_eliminated`.
    DoubletonEquation {
        /// The equality that was used.
        row: usize,
        /// The column that went.
        eliminated: usize,
        /// The column it is now written in terms of.
        kept: usize,
        /// Coefficient of the eliminated column.
        coefficient_eliminated: F,
        /// Coefficient of the kept column.
        coefficient_kept: F,
        /// The right hand side of the equality.
        right_hand_side: F,
        /// What the eliminated column cost.
        cost_eliminated: F,
        /// The eliminated column's coefficients in the rows other than this one.
        ///
        /// Together with the cost these say what the eliminated column's reduced cost would be if
        /// this row's multiplier were zero, which is what the multiplier is solved from.
        occurrences_eliminated: SparseTupleVec<F>,
        /// The kept column's lower bound before the eliminated column's was transplanted onto it.
        kept_lower_bound: Option<F>,
        /// The kept column's upper bound before the eliminated column's was transplanted onto it.
        kept_upper_bound: Option<F>,
    },

    /// A row that was a multiple of another row was dropped.
    ///
    /// What the two rows admitted between them was intersected onto the one that stayed, so the
    /// multiplier the reduced problem gives that row stands for both. Splitting it again means
    /// asking which of the two rows contributed the end that is active.
    DuplicateRow {
        /// The row that went.
        removed: usize,
        /// The row that stayed.
        kept: usize,
        /// `removed = factor * kept`.
        factor: F,
        /// The relation the row that went had.
        removed_relation: RangedConstraintRelation<F>,
        /// The right hand side the row that went had.
        removed_right_hand_side: F,
        /// The relation the row that stayed had, before the merge.
        kept_relation: RangedConstraintRelation<F>,
        /// The right hand side the row that stayed had, before the merge.
        kept_right_hand_side: F,
    },

    /// Two parallel columns were merged into one.
    ///
    /// The kept column temporarily stands for `x_kept + factor * x_removed`. Splitting the merged
    /// value back is a choice; the one made here fills the kept variable as far as its own bounds
    /// allow and gives the rest to the removed one.
    DuplicateColumn {
        /// The column that went.
        removed: usize,
        /// The column that now stands for both.
        kept: usize,
        /// The kept column now stands for `x_kept + factor * x_removed`.
        factor: F,
        /// Lower bound of the removed column.
        removed_lower_bound: Option<F>,
        /// Upper bound of the removed column.
        removed_upper_bound: Option<F>,
        /// Lower bound the kept column had.
        kept_lower_bound: Option<F>,
        /// Upper bound the kept column had.
        kept_upper_bound: Option<F>,
    },

    /// A column that another column dominates was fixed at one of its bounds.
    ///
    /// This is justified by an argument about optimality rather than feasibility: it removes
    /// feasible points, keeping at least one optimal one.
    DominatedColumn {
        /// Which column.
        column: usize,
        /// The bound it was held at.
        value: F,
    },
}

/// The record of a presolve run, and the maps needed to read a solution of the reduced problem.
#[derive(Clone, Debug)]
pub struct Postsolve<F> {
    /// What was done, oldest first.
    reductions: Vec<Reduction<F>>,

    /// Number of columns and rows the problem had when presolving started.
    nr_columns: usize,
    nr_rows: usize,

    /// For each column of the reduced problem, the column it came from.
    surviving_columns: Vec<usize>,
    /// For each row of the reduced problem, the row it came from.
    surviving_rows: Vec<usize>,

    /// Whether a reduction was applied that preserves the feasible set.
    pub applied_primal_reduction: bool,
    /// Whether a reduction was applied that only preserves an optimal point.
    ///
    /// When one of these was applied, an infeasible reduced problem no longer proves the original
    /// problem infeasible: the reduction may have removed the feasible points that were left.
    pub applied_dual_reduction: bool,

    /// Whether integer variables were treated as such.
    ///
    /// Rounding a bound inwards is not something a row implied, so there is nothing to push a
    /// multiplier back onto and no dual solution to be had.
    pub respects_integrality: bool,

    /// What the run did, and what it cost.
    pub statistics: Statistics,
}

impl<F> Postsolve<F> {
    /// An empty record for a problem of the given size.
    pub fn new(nr_rows: usize, nr_columns: usize) -> Self {
        Self {
            reductions: Vec::new(),
            nr_columns,
            nr_rows,
            surviving_columns: Vec::new(),
            surviving_rows: Vec::new(),
            applied_primal_reduction: false,
            applied_dual_reduction: false,
            respects_integrality: false,
            statistics: Statistics::default(),
        }
    }

    /// Record a reduction that preserves the feasible set.
    pub fn push(&mut self, reduction: Reduction<F>) {
        self.applied_primal_reduction = true;
        self.reductions.push(reduction);
    }

    /// Record a reduction that only preserves an optimal point.
    pub fn push_dual(&mut self, reduction: Reduction<F>) {
        self.applied_dual_reduction = true;
        self.reductions.push(reduction);
    }

    /// Say which rows and columns of the original problem the reduced problem consists of.
    ///
    /// Both must be ascending.
    pub fn set_survivors(&mut self, rows: Vec<usize>, columns: Vec<usize>) {
        debug_assert!(rows.is_sorted());
        debug_assert!(columns.is_sorted());

        self.surviving_rows = rows;
        self.surviving_columns = columns;
    }

    /// The columns of the original problem that the reduced problem still has, ascending.
    pub fn surviving_columns(&self) -> &[usize] {
        &self.surviving_columns
    }
    /// The rows of the original problem that the reduced problem still has, ascending.
    pub fn surviving_rows(&self) -> &[usize] {
        &self.surviving_rows
    }
    /// Number of columns the problem had before presolving.
    pub fn nr_columns(&self) -> usize {
        self.nr_columns
    }
    /// Number of rows the problem had before presolving.
    pub fn nr_rows(&self) -> usize {
        self.nr_rows
    }
    /// What was done, oldest first.
    pub fn reductions(&self) -> &[Reduction<F>] {
        &self.reductions
    }
    /// Whether nothing was done at all.
    pub fn is_empty(&self) -> bool {
        self.reductions.is_empty()
    }
}
