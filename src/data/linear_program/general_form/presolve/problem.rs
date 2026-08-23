//! # The problem being presolved
//!
//! An owned, mutable copy of the linear program. Rules read and write it through the `Reducer`;
//! everything they need to know about the current state of the problem is here.
use relp_num::{OrderedField, OrderedFieldRef};

use crate::data::linear_algebra::traits::{SparseComparator, SparseElement};
use crate::data::linear_algebra::vector::Vector;
use crate::data::linear_program::elements::{
    BoundDirection, Objective, RangedConstraintRelation, VariableType,
};
use crate::data::linear_program::general_form::GeneralForm;
use crate::data::linear_program::general_form::presolve::matrix::Matrix;

/// A constraint, without its coefficients.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Row<F> {
    /// Right hand side, read as described by `relation`.
    pub right_hand_side: F,
    pub relation: RangedConstraintRelation<F>,
    /// Whether this row is still part of the problem.
    pub is_active: bool,
}

/// A variable, without its coefficients.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Column<F> {
    pub cost: F,
    pub lower_bound: Option<F>,
    pub upper_bound: Option<F>,
    pub variable_type: VariableType,
    /// Whether this variable is still part of the problem.
    pub is_active: bool,
}

impl<F> Column<F> {
    /// The bound in a direction.
    pub fn bound(&self, direction: BoundDirection) -> Option<&F> {
        match direction {
            BoundDirection::Lower => self.lower_bound.as_ref(),
            BoundDirection::Upper => self.upper_bound.as_ref(),
        }
    }
    /// The bound in a direction, mutably.
    pub fn bound_mut(&mut self, direction: BoundDirection) -> &mut Option<F> {
        match direction {
            BoundDirection::Lower => &mut self.lower_bound,
            BoundDirection::Upper => &mut self.upper_bound,
        }
    }
    /// Whether the variable is unbounded in both directions.
    pub fn is_free(&self) -> bool {
        self.lower_bound.is_none() && self.upper_bound.is_none()
    }
}

impl<F: PartialEq> Column<F> {
    /// The single value the variable can take, if the bounds leave only one.
    pub fn fixed_value(&self) -> Option<&F> {
        match (&self.lower_bound, &self.upper_bound) {
            (Some(lower), Some(upper)) if lower == upper => Some(lower),
            _ => None,
        }
    }
}

impl<F: PartialOrd> Column<F> {
    /// Whether the bounds contradict each other.
    pub fn is_infeasible(&self) -> bool {
        match (&self.lower_bound, &self.upper_bound) {
            (Some(lower), Some(upper)) => lower > upper,
            _ => false,
        }
    }
}

/// The linear program, as presolve is currently seeing it.
#[derive(Clone, Debug)]
pub struct Problem<F> {
    pub matrix: Matrix<F>,
    pub rows: Vec<Row<F>>,
    pub columns: Vec<Column<F>>,
    pub objective: Objective,
    pub fixed_cost: F,
}

impl<F> Problem<F>
where
    F: OrderedField + SparseElement<F> + SparseComparator,
    for<'r> &'r F: OrderedFieldRef<F>,
{
    /// Take a copy of a problem to presolve.
    ///
    /// The coefficients are cloned rather than moved: the caller keeps its problem, and only
    /// learns what to do to it once presolving has finished.
    pub fn from_general_form(general_form: &GeneralForm<F>) -> Self {
        Self {
            matrix: Matrix::from_column_major(general_form.constraints.clone()),
            rows: general_form
                .b
                .iter()
                .zip(general_form.constraint_types.iter())
                .map(|(right_hand_side, relation)| Row {
                    right_hand_side: right_hand_side.clone(),
                    relation: relation.clone(),
                    is_active: true,
                })
                .collect(),
            columns: general_form
                .variables
                .iter()
                .map(|variable| Column {
                    cost: variable.cost.clone(),
                    lower_bound: variable.lower_bound.clone(),
                    upper_bound: variable.upper_bound.clone(),
                    variable_type: variable.variable_type,
                    is_active: true,
                })
                .collect(),
            objective: general_form.objective,
            fixed_cost: general_form.fixed_cost.clone(),
        }
    }

    /// The direction in which a change to a variable improves the objective.
    ///
    /// `Lower` when the variable should be as small as possible, `Upper` when as large as
    /// possible, and `None` when the variable does not appear in the objective.
    pub fn improving_direction(&self, column: usize) -> Option<BoundDirection> {
        let cost = &self.columns[column].cost;
        if cost.is_zero() {
            return None;
        }

        let minimizing = self.objective == Objective::Minimize;
        Some(if cost.is_positive() == minimizing {
            BoundDirection::Lower
        } else {
            BoundDirection::Upper
        })
    }
}
