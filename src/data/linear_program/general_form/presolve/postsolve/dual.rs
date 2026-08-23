//! # Recovering the dual solution
//!
//! A multiplier belongs to a row, so recovering one means saying what each row that presolving
//! removed is worth, and checking that what the reduced problem said about the rows that stayed
//! still holds for the problem that was given.
//!
//! # Why this is harder than the primal side
//!
//! A primal value only has to be restored. A multiplier has to be *right*: the whole vector has to
//! satisfy the dual constraints of the original problem, and a reduction that changes those
//! constraints changes what a valid multiplier is.
//!
//! # What the walk maintains
//!
//! Just before a reduction is undone, the problem that the walk describes is the problem as
//! presolve left it at the moment that reduction was applied. Three things are carried along, all
//! indexed by the problem as it was when presolving started, and all in minimisation orientation so
//! that one set of sign rules covers both directions:
//!
//! * `values`, the primal solution, whole, recovered first;
//! * `duals`, a multiplier per row, correct for the rows that problem still had;
//! * `costs`, the reduced costs `c - A^T y` of that same problem, correct for the columns it still
//!   had.
//!
//! Undoing a reduction restores both: the multipliers of the rows it brings back, and the reduced
//! costs of the columns. Only the multipliers are returned — the reduced costs are what every
//! decision along the way is made on.
//!
//! # Bounds presolve derived
//!
//! A bound presolving worked out is a constraint the original problem does not have, so a solution
//! sitting at one with a nonzero reduced cost is dual infeasible for the original. Putting that
//! right means pushing the reduced cost onto the row the bound came from:
//!
//! ```text
//! y[row]  += theta / a[row][column]
//! z[k]    -= a[row][k] * theta / a[row][column]      for every k in that row
//! ```
//!
//! The first line makes the offending reduced cost exactly zero. The second is what makes it
//! sound, and the reason is worth stating: a variable sitting exactly at a bound read off a row's
//! activity forces that row to be tight *and every other variable in it to be exactly at the bound
//! the activity was computed from*. So each reduced cost moves in the direction that variable's
//! bound admits, and the multiplier moves in the direction the row's tight end admits. Nothing can
//! be pushed out of feasibility, and no chain of these can either, because each one only ever
//! helps the ones that follow.
//!
//! # What is still not recovered
//!
//! Fixing a dominated column is justified by an argument about optimality rather than feasibility.
//! It removes feasible points, so the dual of what is left is not the dual of what was given, and
//! there is nothing to recover rather than something not yet written.
//! [`Postsolve::preserves_duals`] says so in advance, and [`Postsolve::recover_dual`] returns
//! nothing rather than a multiplier that means nothing.
use std::ops::Neg;

use relp_num::{OrderedField, OrderedFieldRef};

use crate::data::linear_algebra::SparseTupleVec;
use crate::data::linear_program::elements::{BoundDirection, RangedConstraintRelation};
use crate::data::linear_program::general_form::presolve::postsolve::{
    BoundOrigin, Postsolve, Reduction,
};

impl<F: Clone> Postsolve<F> {
    /// Extend the multipliers of the reduced problem to ones for the original problem.
    ///
    /// # Arguments
    ///
    /// * `minimizing`: whether the problem that was read asked for a minimum. Multipliers are
    ///   reported as `d(objective) / d(right hand side)` either way, so the direction only decides
    ///   which signs are admissible.
    /// * `reduced_duals`: one multiplier per row of the reduced problem, in the order of the rows
    ///   the original problem still has.
    /// * `reduced_costs`: one reduced cost per column of the reduced problem, in that same sense.
    /// * `values`: the recovered primal solution, one value per column the problem started with.
    ///
    /// # Return value
    ///
    /// `None` when a reduction was applied whose effect on the multipliers is not undone here.
    pub fn recover_dual<G>(
        &self,
        minimizing: bool,
        reduced_duals: &[G],
        reduced_costs: &[G],
        values: &[G],
    ) -> Option<Vec<G>>
    where
        G: OrderedField + From<F>,
        for<'r> &'r G: OrderedFieldRef<G> + Neg<Output = G>,
    {
        debug_assert_eq!(reduced_duals.len(), self.surviving_rows().len());
        debug_assert_eq!(reduced_costs.len(), self.surviving_columns().len());
        debug_assert_eq!(values.len(), self.nr_columns());

        if !self.preserves_duals() {
            return None;
        }

        // Everything below is stated for a minimum. Negating a maximisation problem's objective
        // negates its multipliers and its reduced costs along with it, so working in that
        // orientation and turning the answer back at the end is the whole of the difference.
        let orient = |value: &G| if minimizing { value.clone() } else { -value };

        // A row presolving removed because the rest of the problem implies it asks nothing of the
        // solution on its own account, so zero is a valid multiplier for it. Rows that went for
        // another reason are filled in below.
        let mut walk = Walk {
            minimizing,
            values,
            duals: vec![G::zero(); self.nr_rows()],
            costs: vec![G::zero(); self.nr_columns()],
        };
        for (position, &row) in self.surviving_rows().iter().enumerate() {
            walk.duals[row] = orient(&reduced_duals[position]);
        }
        for (position, &column) in self.surviving_columns().iter().enumerate() {
            walk.costs[column] = orient(&reduced_costs[position]);
        }

        for reduction in self.reductions().iter().rev() {
            walk.undo(reduction)?;
        }

        Some(walk.duals.iter().map(orient).collect())
    }

    /// Whether every reduction applied is one whose effect on the multipliers is undone.
    ///
    /// A reduction not in this set does not make the primal solution any less correct; it makes
    /// the multipliers of the reduced problem stop being multipliers of the original one.
    pub fn preserves_duals(&self) -> bool {
        // Rounding a bound inwards is not something a row implied, so there is no row to push the
        // multiplier of a solution held there back onto.
        !self.respects_integrality
            && !self.applied_dual_reduction
            && self.reductions().iter().all(reduction_preserves_duals)
    }
}

/// The state carried along the walk back. See the module documentation.
struct Walk<'a, G> {
    /// Whether the problem that was read asked for a minimum.
    minimizing: bool,
    /// The primal solution of the original problem, whole.
    values: &'a [G],
    /// A multiplier per row, in minimisation orientation.
    duals: Vec<G>,
    /// A reduced cost per column, in minimisation orientation.
    costs: Vec<G>,
}

impl<G> Walk<'_, G>
where
    G: OrderedField,
    for<'r> &'r G: OrderedFieldRef<G> + Neg<Output = G>,
{
    /// A cost out of the record, in the orientation everything here works in.
    ///
    /// Multipliers and reduced costs are oriented on the way in and back again on the way out, but
    /// a cost read straight out of a reduction has had neither done to it.
    fn cost<F: Clone>(&self, value: &F) -> G
    where
        G: From<F>,
    {
        let value = G::from(value.clone());
        if self.minimizing { value } else { -&value }
    }

    /// `<coefficients, duals>`, over the rows the coefficients name.
    fn along_rows<F: Clone>(&self, coefficients: &SparseTupleVec<F>) -> G
    where
        G: From<F>,
    {
        coefficients
            .iter()
            .map(|(row, coefficient)| &G::from(coefficient.clone()) * &self.duals[*row])
            .sum()
    }

    /// Undo the effect of one reduction on the multipliers and the reduced costs.
    ///
    /// `None` when the reduction is one this code cannot undo, which
    /// [`Postsolve::preserves_duals`] should already have ruled out.
    fn undo<F: Clone>(&mut self, reduction: &Reduction<F>) -> Option<()>
    where
        G: From<F>,
    {
        match reduction {
            // A row nothing else depended on. Zero satisfies every relation and asks nothing of
            // complementary slackness.
            Reduction::EmptyRow { row } | Reduction::RedundantRow { row, .. } => {
                self.duals[*row] = G::zero();
            }

            // The row was multiplied by a positive factor, so a multiplier for it as it stands
            // stands for `factor` times as much of the row as it was. Every reduced cost is
            // untouched: what the row contributes to one is the multiplier times the coefficient,
            // and the two changed by inverse amounts.
            Reduction::RowScaled { row, factor } => {
                self.duals[*row] = &self.duals[*row] * &G::from(factor.clone());
            }

            // The relation and right hand side moved, but the coefficients did not, so no reduced
            // cost changes. Every relation that can be restored here admits the multiplier that is
            // already on the row: a shifted right hand side leaves which side is tight alone, and
            // a range that was weakened to an inequality is weaker on exactly the side no
            // multiplier could be sitting on.
            Reduction::RowChanged { .. } => {}

            // The row is gone and the bounds it turned into carry its multiplier. Those records
            // come after this one, so by now they have already put it there, and nothing later
            // will touch it: this is where the multiplier of a bound row can be checked.
            Reduction::SingletonRow { row, relation, .. } => {
                debug_assert!(
                    admits(relation, &self.duals[*row]),
                    "a bound row was given a multiplier its relation does not allow",
                );
            }

            // A column in no row at all: its reduced cost is its cost.
            Reduction::EmptyColumn { column, cost, .. } => {
                self.costs[*column] = self.cost(cost);
            }

            // A column that is back in the rows it was in.
            Reduction::FixedColumn {
                column,
                coefficients,
                cost,
                ..
            } => {
                self.costs[*column] = &self.cost(cost) - &self.along_rows(coefficients);
            }

            // A slack: no cost, and one row.
            Reduction::ColumnSingleton {
                column,
                row,
                coefficient,
                ..
            } => {
                self.costs[*column] = -&(&G::from(coefficient.clone()) * &self.duals[*row]);
            }

            // The column was in this row and no other, and the row confined it to within its own
            // bounds, so those bounds carry no information and its reduced cost is zero. Its dual
            // equation then says what the multiplier of the row is.
            //
            // Every other reduced cost is left alone: the rule moved this column's cost onto the
            // other columns of the row at the time, and giving the row back its multiplier takes
            // exactly that much off again.
            Reduction::ImpliedFreeColumnSingleton {
                column,
                row,
                coefficient,
                cost,
                ..
            } => {
                self.duals[*row] = &self.cost(cost) / &G::from(coefficient.clone());
                self.costs[*column] = G::zero();
            }

            Reduction::BoundTightened {
                column,
                direction,
                new,
                origin,
                ..
            } => self.undo_bound(*column, *direction, new, origin),

            Reduction::DoubletonEquation { .. } => self.undo_doubleton(reduction),

            Reduction::DuplicateRow { .. } => self.undo_duplicate_row(reduction),

            // The rows saw the two columns only through `x_kept + factor * x_removed`, and so did
            // the objective, which is what the rule checked before merging. So the multipliers do
            // not move at all: what the merged column contributed to a row, the two contribute
            // between them, and the reduced costs stand in the same ratio as the columns.
            Reduction::DuplicateColumn {
                removed,
                kept,
                factor,
                ..
            } => {
                self.costs[*removed] = &G::from(factor.clone()) * &self.costs[*kept];
            }

            // Justified by optimality rather than feasibility: it removes feasible points, so the
            // dual of what is left is not the dual of what was given.
            Reduction::DominatedColumn { .. } => return None,
        }

        Some(())
    }

    /// Give back to the row the reduced cost that a bound it implied was holding.
    fn undo_bound<F: Clone>(
        &mut self,
        column: usize,
        direction: BoundDirection,
        new: &F,
        origin: &BoundOrigin<F>,
    ) where
        G: From<F>,
    {
        let BoundOrigin::Activity {
            row,
            coefficient,
            coefficients,
        } = origin
        else {
            // A bound a substitution put there. The record for that substitution comes right
            // before this one and settles the reduced cost itself.
            return;
        };

        // A bound only holds a reduced cost while the solution is sitting on it, with the sign
        // that says this bound and not the opposite one is what is holding it there.
        if self.values[column] != G::from(new.clone()) {
            return;
        }
        let theta = self.costs[column].clone();
        let zero = G::zero();
        let is_held = match direction {
            BoundDirection::Lower => theta > zero,
            BoundDirection::Upper => theta < zero,
        };
        if !is_held {
            return;
        }

        let step = &theta / &G::from(coefficient.clone());
        self.duals[*row] += &step;
        for (other, value) in coefficients {
            self.costs[*other] -= &G::from(value.clone()) * &step;
        }

        debug_assert!(
            self.costs[column] == G::zero(),
            "the push back exists to make exactly this reduced cost disappear",
        );
    }

    /// Work out the multiplier of an equality that was used to eliminate one of its two columns.
    ///
    /// The two reduced costs are tied together by `z_kept + slope * z_eliminated = z_merged`,
    /// whatever the multiplier is, so the multiplier is what chooses how the reduced cost of the
    /// column that stayed is divided between the two. Which way to divide it follows from which of
    /// the two variables is being held at a bound of its own.
    fn undo_doubleton<F: Clone>(&mut self, reduction: &Reduction<F>)
    where
        G: From<F>,
    {
        let Reduction::DoubletonEquation {
            row,
            eliminated,
            kept,
            coefficient_eliminated,
            coefficient_kept,
            cost_eliminated,
            occurrences_eliminated,
            kept_lower_bound,
            kept_upper_bound,
            ..
        } = reduction
        else {
            unreachable!("only called for this variant");
        };

        let coefficient_eliminated = G::from(coefficient_eliminated.clone());
        let coefficient_kept = G::from(coefficient_kept.clone());
        // `x_eliminated = offset + slope * x_kept`.
        let slope = &(-&coefficient_kept) / &coefficient_eliminated;

        // What the eliminated column's reduced cost would be if this row were worth nothing.
        let free = &self.cost(cost_eliminated) - &self.along_rows(occurrences_eliminated);
        let theta = self.costs[*kept].clone();

        let zero = G::zero();
        let holds_its_own = if theta == zero {
            true
        } else {
            let bound = if theta > zero {
                kept_lower_bound
            } else {
                kept_upper_bound
            };
            bound
                .as_ref()
                .is_some_and(|value| self.values[*kept] == G::from(value.clone()))
        };

        if holds_its_own {
            // The bound doing the holding is one the kept column already had, so it keeps its
            // reduced cost and the eliminated column's goes to zero.
            self.duals[*row] = &free / &coefficient_eliminated;
            self.costs[*eliminated] = G::zero();
        } else {
            // The bound came across from the eliminated column, so the reduced cost goes back with
            // it. The eliminated variable is at the bound that produced it, which is the one that
            // admits the sign this gives.
            self.duals[*row] = &(&theta - &(&slope * &free)) / &coefficient_kept;
            self.costs[*eliminated] = &theta / &slope;
            self.costs[*kept] = G::zero();
        }
    }

    /// Split the multiplier of a merged row between the two rows it was merged from.
    ///
    /// The merged row admits the intersection of what the two admitted, so its multiplier is
    /// sitting on whichever of them contributed the end that is active. Giving it all to that one
    /// and nothing to the other reproduces what the pair contributed, because the two rows are
    /// multiples of each other.
    fn undo_duplicate_row<F: Clone>(&mut self, reduction: &Reduction<F>)
    where
        G: From<F>,
    {
        let Reduction::DuplicateRow {
            removed,
            kept,
            factor,
            removed_relation,
            removed_right_hand_side,
            kept_relation,
            kept_right_hand_side,
        } = reduction
        else {
            unreachable!("only called for this variant");
        };

        let multiplier = self.duals[*kept].clone();
        let zero = G::zero();
        if multiplier == zero {
            self.duals[*removed] = zero;
            return;
        }

        let factor = G::from(factor.clone());
        // What the row that went admitted, rewritten in terms of the activity of the row that
        // stayed. Dividing by a negative factor turns the interval around.
        let (low, high) = admissible::<F, G>(removed_relation, removed_right_hand_side);
        let rescale = |bound: Option<G>| bound.map(|value| &value / &factor);
        let (low, high) = (rescale(low), rescale(high));
        let (removed_low, removed_high) = if factor < zero {
            (high, low)
        } else {
            (low, high)
        };

        let (kept_low, kept_high) = admissible::<F, G>(kept_relation, kept_right_hand_side);

        // A negative multiplier sits on the upper end of the merged interval and a positive one on
        // the lower end. The merged end is the tighter of the two, and the row that supplied it is
        // the row the multiplier belongs to.
        let kept_supplied_it = if multiplier < zero {
            match (&kept_high, &removed_high) {
                (None, _) => false,
                (Some(_), None) => true,
                (Some(kept), Some(removed)) => kept <= removed,
            }
        } else {
            match (&kept_low, &removed_low) {
                (None, _) => false,
                (Some(_), None) => true,
                (Some(kept), Some(removed)) => kept >= removed,
            }
        };

        if kept_supplied_it {
            self.duals[*removed] = zero;
        } else {
            self.duals[*removed] = &multiplier / &factor;
            self.duals[*kept] = zero;
        }

        debug_assert!(
            admits(kept_relation, &self.duals[*kept])
                && admits(removed_relation, &self.duals[*removed]),
            "the multiplier went to a row whose relation does not allow its sign",
        );
    }
}

/// Whether a relation allows a multiplier of this sign, for a minimisation problem.
///
/// Relaxing a `<=` row can only help, so its multiplier cannot be positive, and the other way
/// round for a `>=` row. An equality admits either, and so does a range: there the sign says which
/// of its two ends the multiplier is sitting on.
fn admits<F, G: OrderedField>(relation: &RangedConstraintRelation<F>, multiplier: &G) -> bool {
    match relation {
        RangedConstraintRelation::Equal | RangedConstraintRelation::Range(_) => true,
        RangedConstraintRelation::Less => multiplier <= &G::zero(),
        RangedConstraintRelation::Greater => multiplier >= &G::zero(),
    }
}

/// The interval a relation admits for the activity of its row.
fn admissible<F: Clone, G: OrderedField + From<F>>(
    relation: &RangedConstraintRelation<F>,
    right_hand_side: &F,
) -> (Option<G>, Option<G>)
where
    for<'r> &'r G: OrderedFieldRef<G>,
{
    let right_hand_side = G::from(right_hand_side.clone());
    match relation {
        RangedConstraintRelation::Equal => {
            (Some(right_hand_side.clone()), Some(right_hand_side))
        }
        RangedConstraintRelation::Range(range) => (
            Some(&right_hand_side - &G::from(range.clone())),
            Some(right_hand_side),
        ),
        RangedConstraintRelation::Less => (None, Some(right_hand_side)),
        RangedConstraintRelation::Greater => (Some(right_hand_side), None),
    }
}

/// Whether one reduction leaves the multipliers of the rows that remain valid for the original.
fn reduction_preserves_duals<F>(reduction: &Reduction<F>) -> bool {
    match reduction {
        // A row that the rest of the problem implies asks nothing on its own account, so giving it
        // a multiplier of zero is valid.
        Reduction::EmptyRow { .. } | Reduction::RedundantRow { .. } => true,

        // A column determined outright, or one that was in no row at all. Neither changes what any
        // row asks of the others.
        Reduction::EmptyColumn { .. } | Reduction::FixedColumn { .. } => true,

        // A row that was weakened keeps its multiplier: the side that was dropped was one the
        // bounds already implied, so no multiplier was ever going to sit on it.
        Reduction::RowChanged { .. } => true,

        // The same row written differently. Its multiplier changes by the factor and its sign does
        // not, the factor being positive.
        Reduction::RowScaled { .. } => true,

        // A zero cost column that took up the slack of its row. The row keeps its multiplier and
        // the column's reduced cost follows from it.
        Reduction::ColumnSingleton { .. } => true,

        // A bound presolving derived, whose record says which row implied it.
        Reduction::BoundTightened { .. } | Reduction::SingletonRow { .. } => true,

        // The column was in this row and no other, and the row confined it to within its own
        // bounds, so its bounds could be dropped. A variable with no bounds has a reduced cost of
        // zero, and its dual equation then says what the multiplier of the row is.
        Reduction::ImpliedFreeColumnSingleton { .. } => true,

        // Both work out a multiplier from the reduced cost of a column, or split one between two
        // rows. See the two methods on `Walk` that do it.
        Reduction::DoubletonEquation { .. } | Reduction::DuplicateRow { .. } => true,

        // The two columns are multiples of each other in the rows and in the objective alike, so
        // nothing about the rows changes.
        Reduction::DuplicateColumn { .. } => true,

        // Justified by optimality rather than feasibility: it removes feasible points, so the dual
        // of what is left is not the dual of what was given.
        Reduction::DominatedColumn { .. } => false,
    }
}
