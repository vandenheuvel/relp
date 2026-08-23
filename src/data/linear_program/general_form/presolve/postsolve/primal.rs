//! # Recovering the primal solution
//!
//! Walking the reduction list backwards, filling in the value of every variable the reduced
//! problem does not have.
use std::ops::Neg;

use relp_num::{OrderedField, OrderedFieldRef};

use crate::data::linear_algebra::SparseTupleVec;
use crate::data::linear_program::general_form::presolve::postsolve::{Postsolve, Reduction};

impl<F: Clone> Postsolve<F> {
    /// Extend a solution of the reduced problem to one of the original problem.
    ///
    /// `reduced` holds one value per column of the reduced problem, in that problem's own order.
    ///
    /// The solution may be carried in a wider number type than the problem is stored in, which is
    /// why the two are separate parameters.
    pub fn recover_primal<G>(&self, reduced: &[G]) -> Vec<G>
    where
        G: OrderedField + From<F>,
        for<'r> &'r G: OrderedFieldRef<G> + Neg<Output = G>,
    {
        debug_assert_eq!(reduced.len(), self.surviving_columns().len());

        let mut values: Vec<Option<G>> = vec![None; self.nr_columns()];
        for (position, &column) in self.surviving_columns().iter().enumerate() {
            values[column] = Some(reduced[position].clone());
        }

        for reduction in self.reductions().iter().rev() {
            undo(reduction, &mut values);
        }

        debug_assert!(
            values.iter().all(Option::is_some),
            "every column is either in the reduced problem or restored by a reduction",
        );
        values.into_iter().map(Option::unwrap).collect()
    }
}

/// Clamp a value to a range, either end of which may be absent.
fn clamp<G: OrderedField>(value: G, lower: Option<G>, upper: Option<G>) -> G {
    if let Some(lower) = lower
        && value < lower
    {
        return lower;
    }
    if let Some(upper) = upper
        && value > upper
    {
        return upper;
    }
    value
}

/// `(constant - <coefficients, values>) / coefficient`.
///
/// The shape every substitution recovery takes: a row solved for the one variable removed from it.
fn residual<F, G>(
    constant: &F,
    coefficients: &SparseTupleVec<F>,
    coefficient: &F,
    values: &[Option<G>],
) -> G
where
    F: Clone,
    G: OrderedField + From<F>,
    for<'r> &'r G: OrderedFieldRef<G> + Neg<Output = G>,
{
    let inner_product = coefficients
        .iter()
        .map(|(column, value)| {
            let solved = values[*column]
                .as_ref()
                .expect("a reduction only refers to columns restored before it");
            &G::from(value.clone()) * solved
        })
        .sum::<G>();

    &(&G::from(constant.clone()) - &inner_product) / &G::from(coefficient.clone())
}

/// Restore whatever one reduction removed.
fn undo<F, G>(reduction: &Reduction<F>, values: &mut [Option<G>])
where
    F: Clone,
    G: OrderedField + From<F>,
    for<'r> &'r G: OrderedFieldRef<G> + Neg<Output = G>,
{
    match reduction {
        // Rows carry no primal value of their own, and a tightened bound leaves the variable in
        // place.
        Reduction::EmptyRow { .. }
        | Reduction::RedundantRow { .. }
        | Reduction::RowChanged { .. }
        | Reduction::BoundTightened { .. }
        | Reduction::DuplicateRow { .. }
        | Reduction::RowScaled { .. }
        | Reduction::SingletonRow { .. } => {}

        Reduction::EmptyColumn { column, value, .. }
        | Reduction::FixedColumn { column, value, .. }
        | Reduction::DominatedColumn { column, value } => {
            values[*column] = Some(G::from(value.clone()));
        }

        Reduction::ColumnSingleton {
            column,
            coefficient,
            coefficients,
            right_hand_side,
            lower_bound,
            upper_bound,
            ..
        } => {
            let exact = residual(right_hand_side, coefficients, coefficient, values);
            values[*column] = Some(clamp(
                exact,
                lower_bound.clone().map(G::from),
                upper_bound.clone().map(G::from),
            ));
        }

        Reduction::ImpliedFreeColumnSingleton {
            column,
            coefficient,
            coefficients,
            right_hand_side,
            ..
        } => {
            values[*column] = Some(residual(right_hand_side, coefficients, coefficient, values));
        }

        Reduction::DoubletonEquation {
            eliminated,
            kept,
            coefficient_eliminated,
            coefficient_kept,
            right_hand_side,
            ..
        } => {
            let kept_value = values[*kept]
                .as_ref()
                .expect("the kept column outlives the eliminated one");
            let value = &(&G::from(right_hand_side.clone())
                - &(&G::from(coefficient_kept.clone()) * kept_value))
                / &G::from(coefficient_eliminated.clone());
            values[*eliminated] = Some(value);
        }

        Reduction::DuplicateColumn {
            removed,
            kept,
            factor,
            removed_lower_bound,
            removed_upper_bound,
            kept_lower_bound,
            kept_upper_bound,
        } => {
            // The kept column currently holds `x_kept + factor * x_removed`. Give the kept
            // variable as much of that as its own bounds allow, and the rest to the removed one.
            let merged = values[*kept]
                .as_ref()
                .expect("the kept column outlives the removed one")
                .clone();
            let factor = G::from(factor.clone());

            let kept_value = clamp(
                merged.clone(),
                kept_lower_bound.clone().map(G::from),
                kept_upper_bound.clone().map(G::from),
            );
            let removed_value = clamp(
                &(&merged - &kept_value) / &factor,
                removed_lower_bound.clone().map(G::from),
                removed_upper_bound.clone().map(G::from),
            );

            values[*kept] = Some(&merged - &(&factor * &removed_value));
            values[*removed] = Some(removed_value);
        }
    }
}
