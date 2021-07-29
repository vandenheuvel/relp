//! # Scaling
//!
//! Scaling is a presolving operation that can improve properties of a linear program such that it
//! is quicker to solve using subsequent algorithms.
use std::ops::{Mul, MulAssign};

use num_traits::One;

use crate::data::linear_algebra::traits::{SparseComparator, SparseElement};
use crate::data::linear_algebra::vector::{SparseVector, Vector};
use crate::data::linear_program::elements::RangedConstraintRelation;
use crate::data::linear_program::general_form::GeneralForm;

mod rational;

/// Scaling linear programs.
///
/// Scaling the constraints, bounds and / or cost function of a linear program can make it behave
/// better during solving. This can be because of numerics in case of floating point computations,
/// or because of redundant prime factors in fractional computations.
pub trait Scalable<T> {
    // TODO(ARCHITECTURE): Make the `Scaling` struct a more flexible type.

    /// Compute and apply a scaling.
    ///
    /// Scaling should happen before subsequent algorithms are applied. In order to obtain the final
    /// solution, the scaling will have to be re-applied.
    ///
    /// As a convention, ensure that the scaling is equivalent to
    ///
    /// * a multiplication of the constraint coefficients,
    /// * a division of the bound values and
    /// * a multiplication of the cost coefficients
    ///
    /// with the values in the returned `Scaling`.
    #[must_use = "Use the scaling to transform the solution."]
    fn scale(&mut self) -> Scaling<T>;
    /// Reverse a scaling.
    ///
    /// This should happen before any solution is "transformed back", for example to compensate for
    /// other presolve operations that were done.
    ///
    /// As a convention, ensure that the scaling is equivalent to
    ///
    /// * a division of the constraint coefficients,
    /// * a multiplication of the bound values and
    /// * a division of the cost coefficients
    ///
    /// with the values in the returned `Scaling`.
    fn scale_back(&mut self, scale_info: Scaling<T>);
}

const WARNING_MESSAGE: &str = "WARNING: Not scaling. Does your number type fulfill the necessary \
bound constraints?";

/// A default scaling implementation that does not do anything.
impl<T: One + SparseElement<T> + SparseComparator + Clone> Scalable<T> for GeneralForm<T> {
    /// An identity scaling without effect.
    default fn scale(&mut self) -> Scaling<T> {
        println!("{}", WARNING_MESSAGE);
        // TODO(LOGGING): Log that no scaling is done.
        // TODO(FLOAT): Write the scaling implementation here.
        Scaling {
            cost_factor: T::one(),
            constraint_row_factors: vec![T::one(); self.nr_active_constraints()],
            constraint_column_factors: vec![T::one(); self.nr_active_variables()],
        }
    }

    /// An identity scaling without effect.
    default fn scale_back(&mut self, _scale_info: Scaling<T>) {
        println!("{}", WARNING_MESSAGE);
        // TODO(LOGGING): Log that no scaling is done.
        // TODO(FLOAT): Write the scaling implementation here.
    }
}

/// Scaling for a linear program.
///
/// Describes how the coefficients of the cost function, the rows of the constraints and the
/// variables should be scaled.
///
/// Note that the coefficients of the constraint are influenced by both the values applying to the
/// constraint rows and the values applying to the constraint columns.
///
/// To invert the scaling, apply the opposite scaling operation to each element.
#[derive(Eq, PartialEq, Debug)]
pub struct Scaling<R> {
    /// Multiply the cost coefficients by this factor.
    cost_factor: R,
    /// Multiply the constraint rows (including right-hand side) by these factors.
    constraint_row_factors: Vec<R>,
    /// Multiply the constraint columns by these factors and divide the variable bounds by these
    /// factors.
    constraint_column_factors: Vec<R>,
}

impl<T> Scaling<T> {
    /// Scale a solution vector back.
    ///
    /// When a solution is computed using a scaled linear program, that solution is of course
    /// also a scaled version of the true solution. This method scales the computed solution
    /// to compensate for this scaling.
    pub fn scale_back<S>(&self, vector: &mut SparseVector<S, S>)
    where
        for<'r> S: MulAssign<&'r T>,
        S: SparseElement<S> + SparseComparator,
    {
        debug_assert_eq!(vector.len(), self.constraint_column_factors.len());

        for (j, value) in vector.iter_mut() {
            *value *= &self.constraint_column_factors[*j];
        }
    }
}

/// Scale a linear program.
///
/// This is a helper method to scale and rescale a `GeneralForm` linear program. The two operations
/// `op` and `inverse_op` should be `MulAssign` and `DivAssign` respectively when scaling (forward)
/// and be reversed when scaling back.
fn scale<T, F, G>(
    general_form: &mut GeneralForm<T>, scaling: &Scaling<T>,
    op: F, inverse_op: G,
)
where
    T: SparseElement<T> + SparseComparator,
    for<'r> &'r T: Mul<&'r T, Output=T>,
    F: Fn(&mut T, &T),
    G: Fn(&mut T, &T),
{
    let Scaling {
        cost_factor,
        constraint_row_factors,
        constraint_column_factors,
    } = scaling;

    // Column oriented operations
    for (j, column) in general_form.constraints.data.iter_mut().enumerate() {
        let column_factor = &constraint_column_factors[j];

        let variable = &mut general_form.variables[j];
        op(&mut variable.cost, &(cost_factor * column_factor));
        if let Some(bound) = &mut variable.lower_bound {
            inverse_op(bound, column_factor)
        }
        if let Some(bound) = &mut variable.upper_bound {
            inverse_op(bound, column_factor)
        }

        for (i, value) in column {
            let row_factor = &constraint_row_factors[*i];
            // The row_scales value is large, the column_scales value is small, so combine them
            // first, some factors might cancel
            op(value, &(row_factor * column_factor));
        }
    }

    // Row oriented operations
    for (i, value) in general_form.b.iter_mut().enumerate() {
        let row_factor = &constraint_row_factors[i];
        op(value, row_factor);
        if let RangedConstraintRelation::Range(range) = &mut general_form.constraint_types[i] {
            op(range, row_factor);
        }
    }
}
