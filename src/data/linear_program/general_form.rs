//! # Linear programs in "general form"
//!
//! A linear program in general form is defined as a linear optimization problem, where the
//! variables can be either constraint or unconstrained and the constraints are either equalities,
//! or inequalities of the form a x >= b with no further requirements on b. This module contains
//! data structures to represent such problems, and logic to do presolving and conversions from this
//! problem form to e.g. a so called equivalent "canonical" representations where any free variables
//! and inequalities are eliminated.
use std::cmp::Ordering;
use std::collections::HashSet;

use daggy::{Dag, WouldCycle};
use daggy::petgraph::data::Element;
use itertools::repeat_n;

use crate::algorithm::simplex::matrix_provider::matrix_data;
use crate::algorithm::simplex::matrix_provider::matrix_data::MatrixData;
use crate::algorithm::utilities::remove_indices;
use crate::data::linear_algebra::matrix::{ColumnMajorOrdering, RowMajorOrdering};
use crate::data::linear_algebra::matrix::SparseMatrix;
use crate::data::linear_algebra::SparseTuples;
use crate::data::linear_algebra::vector::{DenseVector, Vector};
use crate::data::linear_program::elements::{BoundDirection, ConstraintType, LinearProgramType, Objective, VariableType};
use crate::data::linear_program::general_form::OriginalVariable::Removed;
use crate::data::linear_program::general_form::RemovedVariable::{FunctionOfOthers, Solved};
use crate::data::linear_program::solution::Solution;
use crate::data::number_types::traits::{Field, OrderedField};

/// A linear program in general form.
///
/// This structure is used as a first storage independent representation format for different
/// parse results to be transformed to.
///
/// Can be checked for consistency by the `is_consistent` method in this module. That method can be
/// viewed as documentation for the requirements on the variables in this data structure.
#[derive(Debug, Eq, PartialEq)]
pub struct GeneralForm<F: Field> {
    /// Which direction does the objective function go?
    objective: Objective,

    /// Constant in the cost function.
    fixed_cost: F,

    // Constraint related
    /// All constraint coefficients.
    ///
    /// Has size `constraint_types.len()` in the row direction, size `variables.len()` in the column
    /// direction.
    constraints: SparseMatrix<F, RowMajorOrdering>,
    /// The equation type of all rows, ordered by index.
    ///
    /// These are read "from constraint to constraint value", meaning:
    /// * When a constraint is `ConstraintType::Less`, the equation is <a, x> <= b
    /// * When a constraint is `ConstraintType::Greater`, the equation is <a, x> >= b
    constraint_types: Vec<ConstraintType>,
    /// All right-hands sides of equations.
    b: DenseVector<F>,

    // Variable related
    /// Information about all *active* variables, that is, variables that are not yet presolved.
    variables: Vec<Variable<F>>,
    /// For all variables, presolved or not, a placeholder with a potential solution or method to
    /// derive the solution once the other active variables are solved.
    original_variables: Vec<(String, OriginalVariable<F>)>,
    /// Mapping indices of unsolved variables to their index in the original problem.
    from_active_to_original: Vec<usize>,
}

/// Whether a variable from the original problem has been eliminated.
#[derive(Debug, Eq, PartialEq)]
enum OriginalVariable<F> {
    /// The variable is still active.
    ///
    /// Either presolve has not yet been attempted, or was unsuccessful in eliminating the variable
    /// from the problem.
    ///
    /// Contains the index of the variable in the active part of the problem. This index is used to
    /// index into `GeneralForm.variables`.
    Active(usize),
    /// Variable was removed.
    ///
    /// Probably by a presolve operation. An explicit value might not be known.
    Removed(RemovedVariable<F>),
}

/// A variable from the original problem that was removed.
#[derive(Debug, Eq, PartialEq)]
enum RemovedVariable<F> {
    /// Variable was determined to an explicit value.
    Solved(F),
    /// Variable was determined as a function of other variables.
    ///
    /// Affine function of the form `constant - <coefficients, x>` where some of the `x` might be
    /// unknown (or at least not explicitly known) at this point in the solution process.
    FunctionOfOthers {
        constant: F,
        coefficients: SparseTuples<F>,
    },
}

/// Check whether the dimensions of the `GeneralForm` are consistent.
///
/// This method might be expensive, use it in debugging only. It can be viewed as a piece of
/// documentation on the requirements of a `GeneralForm` struct.
fn is_consistent<F: OrderedField>(general_form: &GeneralForm<F>) -> bool {
    // Reference values
    let nr_active_constraints = general_form.constraints.nr_rows();
    let nr_active_variables = general_form.constraints.nr_columns();

    let b = general_form.b.len() == nr_active_constraints;
    let constraints = general_form.constraint_types.len() == nr_active_constraints;
    let rows = general_form.constraints.nr_rows() == nr_active_constraints;


    let variables = general_form.variables.len() == nr_active_variables;
    let columns = general_form.constraints.nr_columns() == nr_active_variables;
    let original_variables = {
        let nr_original_variables = general_form.original_variables.len();
        let size = nr_original_variables >= nr_active_variables;
        let kept_increasing = general_form.original_variables.iter()
            .filter_map(|(_, variable)| match variable {
                OriginalVariable::Active(index) => Some(*index),
                _ => None,
            })
            .collect::<Vec<_>>() == (0..nr_active_variables).collect::<Vec<_>>();
        let no_cycles = {
            let nodes = repeat_n(Element::Node { weight: (), }, nr_original_variables);
            let edges = general_form.original_variables
                .iter().enumerate()
                .filter_map(|(target, (_, variable))| match variable {
                    OriginalVariable::Removed(FunctionOfOthers{ constant: _, coefficients: els, }) => Some((els, target)),
                    _ => None,
                })
                .flat_map(|(els, target)| {
                    els.iter().map(move |&(j, _)| Element::<(), ()>::Edge { source: j, target, weight: (), })
                });

            if let Err(WouldCycle(_)) = Dag::<(), (), usize>::from_elements(nodes.chain(edges)) {
                false
            } else { true }
        };

        size && kept_increasing && no_cycles
    };
    let from_active_to_original = {
        let size = general_form.from_active_to_original.len() == nr_active_variables;
        let unique = general_form.from_active_to_original.iter().collect::<HashSet<_>>().len() == nr_active_variables;
        let sorted = general_form.from_active_to_original.is_sorted();
        let max = if nr_active_variables > 0 {
            general_form.from_active_to_original[nr_active_variables - 1] < nr_active_variables
        } else { true };

        size && unique && sorted && max
    };

    true
        && b
        && constraints
        && rows
        && variables
        && columns
        && original_variables
        && from_active_to_original
}


impl<OF: OrderedField> GeneralForm<OF> {
    /// Create a new linear program in general form.
    pub fn new(
        objective: Objective,
        constraints: SparseMatrix<OF, RowMajorOrdering>,
        constraint_types: Vec<ConstraintType>,
        b: DenseVector<OF>,
        variables: Vec<Variable<OF>>,
        variable_names: Vec<String>,
        fixed_cost: OF,
    ) -> Self {
        let nr_active_variables = variables.len();

        let general_form = Self {
            objective,
            constraints,
            constraint_types,
            b,
            fixed_cost,
            variables,
            original_variables: variable_names.into_iter().enumerate()
                .map(|(j, name)| (name, OriginalVariable::Active(j))).collect(),
            from_active_to_original: (0..nr_active_variables).collect(),
        };

        debug_assert!(is_consistent(&general_form));

        general_form
    }

    /// Modify this linear problem such that it is representable by a `MatrixData` structure.
    ///
    /// The problem gets transformed into `CanonicalForm`, which also includes a presolve operation.
    /// Note that this call might be expensive.
    ///
    /// TODO(ENHANCEMENT): Make sure that presolving can be skipped.
    ///
    /// See also the documentation of the `GeneralForm::canonicalize` method.
    ///
    /// # Return value
    ///
    /// A `Result` containing either the `MatrixData` form of the presolved and canonicalized
    /// problem, or in case the linear program gets solved during this presolve operation, a
    /// `Result::Err` return value.
    pub fn derive_matrix_data(&mut self) -> Result<MatrixData<OF>, LinearProgramType<OF>> {
        self.canonicalize()?;

        let negative_free_variable_dummy_index = self.variables.iter().enumerate()
            .filter(|&(_, variable)| variable.is_free())
            .map(|(j, _)| j).collect();
        let variables = self.variables.iter()
            .map(|variable| matrix_data::Variable {
                cost: variable.cost,
                upper_bound: variable.upper_bound,
                variable_type: variable.variable_type,
            }).collect();
        let (b, (equality, upper, lower)) = self.split_constraints_by_type();

        Ok(MatrixData::new(
            equality,
            upper,
            lower,
            b,
            variables,
            negative_free_variable_dummy_index,
        ))
    }

    /// Convert this `GeneralForm` problem to a form closer to the canonical form representation.
    ///
    /// This involves:
    ///
    /// * Determining which rows and columns can be removed or differently represented to reduce the
    /// problem size and increase the reading speed from a `MatrixData` structure.
    /// * Determining which variables are implicitly fixed, such that their only feasible value can
    /// be substituted into the problem and the column eliminated.
    /// * Modifying variables such that they are either free or bounded below by zero (with possibly
    /// an upper bound).
    /// * Multiplying some rows such that the constraint value is non-negative.
    ///
    /// To do the above, a column major representation of the constraint data is built. This
    /// requires copying all constraint data once.
    ///
    /// TODO(ENHANCEMENT): Make sure that presolving can be skipped.
    ///
    /// # Return value
    pub(crate) fn canonicalize(&mut self) -> Result<(), LinearProgramType<OF>> {
        self.presolve()?;
        self.transform_variables();
        self.make_b_non_negative();
        self.make_minimization_problem();

        Ok(())
    }

    /// Recursively analyse constraints and variable bounds and eliminating or tightning these.
    ///
    /// In order to make the linear program easier to solve, a set of rules is applied. These rules
    /// are cheaper than the full simplex algorithm and are aimed at making the program easier to
    /// solve by other algorithms.
    ///
    /// A set of queues containing constraint and variable indices are maintained. Each presolve
    /// step attempts to apply a presolve rule to either a constraint or variable, as indicated by
    /// these indices. After a rules is applied and a change occurred, relevant constraint or
    /// variable indices might be added to queues, because a rule might be applicable.
    ///
    /// TODO(ENHANCEMENT): Normalization for numerical stability of floating point types.
    ///
    /// # Return value
    ///
    /// If the linear program gets solved during this presolve operation, a `Result::Err` return
    /// value containing the solution.
    pub (crate) fn presolve(&mut self) -> Result<(), LinearProgramType<OF>> {
        let mut index = PresolveIndex::new(&self);

        while !index.queues_are_empty() {
            self.presolve_step(&mut index)?;
        }

        self.optimize_disjoint_variables(&index.columns_optimized_independently)?;
        self.remove_rows_and_columns(index);
        self.compute_solution_where_possible();

        debug_assert!(is_consistent(&self));
        if let Some(solution) = self.get_solution() {
            Err(LinearProgramType::FiniteOptimum(solution))
        } else {
            Ok(())
        }
    }

    /// Apply a single presolve rule.
    ///
    /// The order of the rules is an estimate of the ratio between how likely a rule yields a useful
    /// result, and how expensive it is to apply the rule.
    ///
    /// TODO: What is the best order to apply the rules in?
    /// TODO: Which element should be removed from the queue first?
    ///
    /// # Arguments
    ///
    /// * `index`: A `PresolveIndex` that lives across repeated calls to this function. It is used
    /// to store which constraints and variables need to be checked for which rules. After the
    /// application of each rule, constraints and variables might be added to queues in this struct.
    ///
    /// # Return value
    ///
    /// If the program is determined to be infeasible, an `Err` type.
    fn presolve_step(&mut self, index: &mut PresolveIndex<OF>) -> Result<(), LinearProgramType<OF>> {
        // Actions that are guaranteed to make the problem smaller
        // Remove a row
        if let Some(&row) = index.empty_row_queue.iter().next() {
            self.presolve_empty_constraint(row, index)?;
            return Ok(());
        }
        // Remove a column
        if let Some(&variable) = index.substitution_queue.iter().next() {
            self.presolve_fixed_variable(variable, index);
            return Ok(());
        }
        // Remove a bound
        if let Some(&constraint) = index.bound_queue.iter().next() {
            self.presolve_simple_bound_constraint(constraint, index)?;
            return Ok(());
        }

        // Actions not guaranteed to make the problem smaller
        // Test whether a variable can be seen as a slack
        if let Some(&variable) = index.slack_queue.iter().next() {
            self.presolve_constraint_if_slack_with_suitable_bounds(variable, index);
            return Ok(());
        }
        // Domain propagation
        if let Some(&constraint) = index.activity_queue.iter().next() {
            self.presolve_constraint_by_domain_propagation(constraint, index)?;
            return Ok(());
        }

        Ok(())
    }


    /// Whether an empty constraint indicates infeasibility.
    ///
    /// This method will be called when a constraints has no coefficients left. In the case, the
    /// constraint should still be satisfied, or the problem is infeasible.
    ///
    /// # Arguments
    ///
    /// * `constraint`: Index of row of constraint to investigate.
    ///
    /// # Return value
    ///
    /// `Result` indicating whether the linear program might still be feasible.
    fn presolve_empty_constraint(
        &self,
        constraint: usize,
        index: &mut PresolveIndex<OF>,
    ) -> Result<(), LinearProgramType<OF>> {
        debug_assert_eq!(index.constraint_counters[constraint], 0);

        match (self.b[constraint].cmp(&OF::additive_identity()), self.constraint_types[constraint]) {
            (Ordering::Equal, _)
            | (Ordering::Greater, ConstraintType::Less)
            | (Ordering::Less, ConstraintType::Greater) => {
                index.remove_constraint(constraint);
                Ok(())
            },
            _ => Err(LinearProgramType::Infeasible),
        }
    }

    /// Substitute a variable with a known value in the constraints in which it appears.
    ///
    /// # Arguments
    ///
    /// * `variable`: Index of column under consideration.
    fn presolve_fixed_variable(
        &mut self,
        variable: usize,
        index: &mut PresolveIndex<OF>,
    ) {
        debug_assert!(self.variables[variable].is_fixed().is_some());

        let value = self.variables[variable].is_fixed().unwrap();
        let column_to_scan = index.iter_active_column(variable).copied().collect::<Vec<_>>();
        index.variable_counters[variable] -= column_to_scan.len();
        for (row, coefficient_value) in column_to_scan {
            self.b.shift_value(row, - coefficient_value * value);
            index.constraint_counters[row] -= 1;
            index.readd_row_to_queues(row);
        }
        self.fixed_cost += self.variables[variable].cost * value;

        self.original_variables[variable].1 = OriginalVariable::Removed(RemovedVariable::Solved(value));
        index.remove_variable(variable);
    }

    /// Remove a constraint that is a bound on a variable.
    ///
    /// This bound is either a bound on a slack variable, or it has no slack variables in it.
    ///
    /// # Arguments
    ///
    /// * `constraint`: Index of a row with only a bound.
    ///
    /// # Return value
    ///
    /// `Result::Err` if the linear program was determined infeasible.
    fn presolve_simple_bound_constraint(
        &mut self,
        constraint: usize,
        index: &mut PresolveIndex<OF>,
    ) -> Result<(), LinearProgramType<OF>> {
        debug_assert_eq!(index.constraint_counters[constraint], 1);
        debug_assert_eq!(self.iter_active_row(constraint, &index.variable_counters).count(), 1);

        let &(column, value) = self.iter_active_row(constraint, &index.variable_counters).next().unwrap();
        debug_assert_ne!(index.variable_counters[column], 0);

        let mut changes = Vec::with_capacity(2);
        match self.constraint_types[constraint] {
            ConstraintType::Greater => changes.push(BoundDirection::Lower),
            ConstraintType::Less => changes.push(BoundDirection::Upper),
            ConstraintType::Equal => {
                changes.push(BoundDirection::Lower);
                changes.push(BoundDirection::Upper);
            },
        }

        let bound_value = self.b[constraint] / value;
        let mut bound_changed = false;
        for direction in changes {
            let change = match direction {
                BoundDirection::Lower => self.variables[column].update_lower_bound(bound_value),
                BoundDirection::Upper => self.variables[column].update_upper_bound(bound_value),
            };
            if change.is_some() {
                bound_changed = true;
            }
            if let Some(Some(amount)) = change {
                index.update_activity_bound(column, direction, amount);
            }
        }

        if !self.variables[column].has_feasible_value() {
            Err(LinearProgramType::Infeasible)
        } else {
            index.constraint_counters[constraint] -= 1;
            index.variable_counters[column] -= 1;
            index.remove_constraint(constraint);
            if bound_changed {
                index.after_bound_change(column, &self.variables);
            }
            index.readd_column_to_queues_based_on_counter(column, &self.variables);
            Ok(())
        }
    }

    /// Try to remove a variable that appears in exactly one constraint.
    ///
    /// This method attempts to remove slack variable that can be viewed as a slack variable. This
    /// is a variable that does not appear in the cost function and only in a single constraint.
    /// The variable does nothing besides supporting the constraint it appears in.
    ///
    /// If the slack is bounded on two sides, we leave things as they are. If not, we can remove the
    /// slack and update the bound on the constraint, perhaps removing it altogether.
    ///
    /// # Arguments
    ///
    /// * `variable_index`: Index of variable that should be removed if it is a slack with suitable
    /// bounds.
    fn presolve_constraint_if_slack_with_suitable_bounds(
        &mut self,
        variable_index: usize,
        index: &mut PresolveIndex<OF>,
    ) {
        debug_assert_eq!(index.variable_counters[variable_index], 1);
        debug_assert_eq!(index.iter_active_column(variable_index).count(), 1);
        debug_assert_eq!(self.variables[variable_index].cost, OF::additive_identity());
        debug_assert!(self.variables[variable_index].is_fixed().is_none());

        let &(row, coefficient) = index.iter_active_column(variable_index).next().unwrap();
        let variable = &self.variables[variable_index];
        let effective_bounds = match coefficient.cmp(&OF::additive_identity()) {
            Ordering::Greater => (variable.lower_bound, variable.upper_bound),
            Ordering::Less => (variable.upper_bound, variable.lower_bound),
            Ordering::Equal => panic!(),
        };
        let change = Self::get_change(effective_bounds, self.constraint_types[row]);

        if let Some(maybe_new_bound) = change {
            self.save_slack_value(row, variable_index, coefficient, index);
            self.maybe_update_bound(row, coefficient, maybe_new_bound, index);
            index.constraint_counters[row] -= 1;
            index.variable_counters[variable_index] -= 1;
            index.remove_variable(variable_index);
        } else {
            index.slack_queue.remove(&variable_index);
        }
    }

    /// Constraint type and bound change for slack bounds and initial constraint type.
    ///
    /// This is a helper method for `presolve_constraint_if_slack_with_suitable_bounds`.
    ///
    /// # Arguments
    ///
    /// * `slack_bounds`: Tuple of lower and upper bounds for the slack variable in the constraint
    /// for when the coefficient in front of this slack would be greater than zero.
    /// * `constraint_type`: Current constraint direction.
    ///
    /// # Return type
    ///
    /// The outer `Option` indicates whether anything needs to change. The inner option is `None`
    /// when the bound can be removed entirely, and `Some` if some bound should remain. If that is
    /// the case, the innermost type indicates the direction of the bound that is left and the
    /// relevant bound (which can be used to compute the new constraint value).
    fn get_change(
        slack_bounds: (Option<OF>, Option<OF>),
        constraint_type: ConstraintType,
    ) -> Option<Option<(BoundDirection, OF)>> {
        match slack_bounds {
            (Some(lower), Some(upper)) => match constraint_type {
                ConstraintType::Less => Some(Some((BoundDirection::Upper, lower))),
                ConstraintType::Equal => None,
                ConstraintType::Greater => Some(Some((BoundDirection::Lower, upper))),
            },
            (Some(lower), None) => match constraint_type {
                ConstraintType::Greater => Some(None),
                _ => Some(Some((BoundDirection::Upper, lower))),
            },
            (None, Some(upper)) => match constraint_type {
                ConstraintType::Less => Some(None),
                _ => Some(Some((BoundDirection::Lower, upper))),
            },
            (None, None) => Some(None),
        }
    }

    /// Write the solution of this slack variable in terms of the constraint activation.
    ///
    /// This is a helper method for `presolve_constraint_if_slack_with_suitable_bounds`.
    ///
    /// The slack has been identified as redundant. It can be expressed as an affine function of the
    /// other variables in the constraint.
    ///
    /// # Arguments
    ///
    /// * `constraint`: Index of the constraint to which the slack belongs.
    /// * `variable`: Index of the slack.
    /// * `coefficient`: Value by which the slack gets multiplied in the constraint.
    fn save_slack_value(
        &mut self,
        constraint: usize,
        variable: usize,
        coefficient: OF,
        index: &PresolveIndex<OF>,
    ) {
        self.original_variables[variable].1 = Removed(FunctionOfOthers {
            constant: self.b[constraint] / coefficient,
            coefficients: self.iter_active_row(constraint, &index.variable_counters)
                .filter(|&&(j, _)| j != variable)
                .map(|&(j, other_coefficient)| {
                    (self.from_active_to_original[j], other_coefficient / coefficient)
                })
                .collect(),
        });
    }

    /// Write the solution of this slack variable in terms of the constraint activation.
    ///
    /// This is a helper method for `presolve_constraint_if_slack_with_suitable_bounds`.
    ///
    /// The slack has been identified as redundant. It can be expressed as an affine function of the
    /// other variables in the constraint.
    ///
    /// # Arguments
    ///
    /// * `constraint`: Index of the constraint to which the slack belongs.
    /// * `coefficient`: Value by which the slack gets multiplied in the constraint.
    /// * `maybe_new_bound`: `Option` describing the new state of the constraint. Either is it
    /// `None`, in which case the constraint should be removed entirely, or `Some` in which case the
    /// `BoundDirection` describes the direction of the new constraint. This value is not of the
    /// `ConstraintType` type because it can never be an equality constraint. The `OF` in the tuple
    /// is the bound value of the slack being removed.
    fn maybe_update_bound(
        &mut self,
        constraint: usize,
        coefficient: OF,
        maybe_new_bound: Option<(BoundDirection, OF)>,
        index: &mut PresolveIndex<OF>,
    ) {
        if let Some((direction, bound)) = maybe_new_bound {
            self.b.shift_value(constraint, - coefficient * bound);
            self.constraint_types[constraint] = match direction {
                BoundDirection::Lower => ConstraintType::Greater,
                BoundDirection::Upper => ConstraintType::Less,
            };
        } else {
            index.remove_constraint_values(constraint, &self.constraints, &self.variables);
            index.remove_constraint(constraint)
        }
    }

    /// Attempt to tighten bounds using activity bounds.
    ///
    /// As described in Achterberg (2007), algorithm 7.1.
    ///
    /// # Arguments
    ///
    /// * `constraint`: Index of the constraint under consideration.
    ///
    /// # Return value
    ///
    /// `Result::Err` if the problem was determined to be infeasible.
    fn presolve_constraint_by_domain_propagation(
        &mut self,
        constraint: usize,
        index: &mut PresolveIndex<OF>,
    ) -> Result<(), LinearProgramType<OF>> {
        let (lower, upper) = self.compute_activity_bounds_if_necessary(constraint, index);

        if self.is_infeasible_due_to_activity_bounds(constraint, lower, upper) {
            return Err(LinearProgramType::Infeasible);
        }

        match (self.constraint_types[constraint], self.activity_implied_constraint(constraint, lower, upper)) {
            (_, None) => (),
            (_, Some(ConstraintType::Equal))
                | (ConstraintType::Greater, Some(ConstraintType::Greater))
                | (ConstraintType::Less, Some(ConstraintType::Less)) => {
                index.remove_constraint_values(constraint, &self.constraints, &self.variables);
                index.remove_constraint(constraint);
                return Ok(());
            },
            (ConstraintType::Equal, Some(ConstraintType::Greater)) => {
                // TODO(OPTIMIZATION): This removing half of this bound actually improve
                //  performance? It might cause a slack.
                self.constraint_types[constraint] = ConstraintType::Less;
            },
            (ConstraintType::Equal, Some(ConstraintType::Less)) => {
                // TODO(OPTIMIZATION): This removing half of this bound actually improve
                //  performance? It might cause a slack.
                self.constraint_types[constraint] = ConstraintType::Greater;
            },
            (ConstraintType::Less, Some(ConstraintType::Greater)) | (ConstraintType::Greater, Some(ConstraintType::Less)) => {},
        }

        // We remove from the activity queue here, but the below call to `tighten_variable_bounds`
        // might readd this constraint to the `index.activity_queue` again.
        index.activity_queue.remove(&constraint);
        self.tighten_variable_bounds(constraint, index);
        Ok(())
    }

    /// If no value is known for one of the activity bounds, compute them.
    ///
    /// This is a helper method for `presolve_constraint_by_domain_propagation`.
    ///
    /// # Arguments
    ///
    /// * `constraint`: Index of the constrain being checked.
    ///
    /// # Return value
    ///
    /// A tuple with a (lower, upper) activity bound. Can be `None` if not all of the relevant
    /// lower or upper bounds are known.
    fn compute_activity_bounds_if_necessary(
        &self,
        constraint: usize,
        presolve_index: &mut PresolveIndex<OF>,
    ) -> (Option<OF>, Option<OF>) {
        if presolve_index.activity_bounds[constraint].0.is_none() {
            presolve_index.activity_bounds[constraint].0 = self.sum_products(
                constraint,
                self.choose_relevant_bounds_to_compute_activity_bounds(
                    constraint,
                    BoundDirection::Lower,
                    presolve_index,
                ),
                presolve_index,
            );
        }
        if presolve_index.activity_bounds[constraint].1.is_none() {
            presolve_index.activity_bounds[constraint].1 = self.sum_products(
                constraint,
                self.choose_relevant_bounds_to_compute_activity_bounds(
                    constraint,
                    BoundDirection::Upper,
                    presolve_index,
                ),
                presolve_index,
            );
        }

        presolve_index.activity_bounds[constraint]
    }

    /// Choose the relevant bound needed to compute the activity bound for each variable.
    ///
    /// What the relevant bound is, is determined by the sign of the coefficient of the variable in
    /// the constraint.
    ///
    /// This is a helper method for `presolve_constraint_by_domain_propagation`.
    ///
    /// # Arguments
    ///
    /// * `constraint`: Index of the constrain being checked.
    /// * `activity_bound_direction`: Whether the activity lower- or upperbound is being computed.
    ///
    /// # Return value
    ///
    /// The relevant bounds (either upper or lower) needed to compute the activity bound.
    fn choose_relevant_bounds_to_compute_activity_bounds<'a>(
        &'a self,
        constraint: usize,
        activity_bound_direction: BoundDirection,
        index: &'a PresolveIndex<OF>,
    ) -> impl Iterator<Item = Option<OF>> + 'a {
        self.iter_active_row(constraint, &index.variable_counters)
            .map(move |&(column, coefficient)| {
                match (coefficient.cmp(&OF::additive_identity()), activity_bound_direction) {
                    (Ordering::Greater, BoundDirection::Lower) | (Ordering::Less, BoundDirection::Upper) =>
                        self.variables[column].lower_bound,
                    (Ordering::Greater, BoundDirection::Upper) | (Ordering::Less, BoundDirection::Lower) =>
                        self.variables[column].upper_bound,
                    (Ordering::Equal, _) => panic!(
                        "No coefficient should be zero at this point. (row, column, coefficient) = ({}, {}, {})",
                        constraint, column, coefficient,
                    ),
                }
            })
    }

    /// Compute the activity lower/upper bound using the relevant upper / lower bounds.
    ///
    /// This is essentially an inner product.
    ///
    /// This is a helper method for `presolve_constraint_by_domain_propagation`.
    ///
    /// # Arguments
    ///
    /// * `constraint`: Index of the constrain being checked.
    /// * `relevant_bounds`: The relevant bounds
    fn sum_products(
        &self,
        constraint: usize,
        relevant_bounds: impl Iterator<Item = Option<OF>>,
        index: &PresolveIndex<OF>,
    ) -> Option<OF> {
        self.iter_active_row(constraint, &index.variable_counters).zip(relevant_bounds)
            .map(|(&(_, coefficient), bound): (_, Option<OF>)| bound.map(|b| b * coefficient))
            .sum()
    }

    /// Whether the problem is infeasible because a constraint exceeds the activity the equation can
    /// obtain.
    ///
    /// This is a helper method for `presolve_constraint_by_domain_propagation`.
    fn is_infeasible_due_to_activity_bounds(
        &self,
        constraint: usize,
        activity_lower_bound: Option<OF>,
        activity_upper_bound: Option<OF>,
    ) -> bool {
        let lower_violated = activity_upper_bound.map_or(false, |bound| bound < self.b[constraint]);
        let upper_violated = activity_lower_bound.map_or(false, |bound| bound > self.b[constraint]);

        match self.constraint_types[constraint] {
            ConstraintType::Greater => lower_violated,
            ConstraintType::Less => upper_violated,
            ConstraintType::Equal => lower_violated || upper_violated,
        }
    }

    /// What constraint the activity bounds imply.
    ///
    /// This is a helper method for `presolve_constraint_by_domain_propagation`.
    fn activity_implied_constraint(
        &self,
        constraint: usize,
        activity_lower_bound: Option<OF>,
        activity_upper_bound: Option<OF>,
    ) -> Option<ConstraintType> {
        let lower_redundant = activity_lower_bound.map_or(false, |bound| bound >= self.b[constraint]);
        let upper_redundant = activity_upper_bound.map_or(false, |bound| bound <= self.b[constraint]);

        match (lower_redundant, upper_redundant) {
            (false, false) => None,
            (false, true) => Some(ConstraintType::Less),
            (true, false) => Some(ConstraintType::Greater),
            (true, true) => Some(ConstraintType::Equal),
        }
    }

    /// Activity bounds imply variable bounds that are processed by this method.
    ///
    /// This is a helper method for `presolve_constraint_by_domain_propagation`. It should be called
    /// after the variable bounds have been updated, and a check for the redundancy of the entire
    /// constraint has been done by `presolve_constraint_by_domain_propagation`.
    fn tighten_variable_bounds(
        &mut self,
        constraint: usize,
        index: &mut PresolveIndex<OF>,
    ) {
        let mut derived_bounds = Vec::new();
        for &(variable, coefficient) in self.iter_active_row(constraint, &index.variable_counters) {
            self.determine_whether_bound_can_be_tightened(
                constraint,
                variable,
                coefficient,
                &index.activity_bounds[constraint],
                &mut derived_bounds,
            );
        }

        self.execute_tightening_changes(derived_bounds, index);
    }

    /// Determine which variable bounds are implied by the activity bound.
    ///
    /// This is a helper method for `presolve_constraint_by_domain_propagation`.
    ///
    /// # Arguments
    ///
    /// * `constraint`: Index of the constraint from which the variable bounds are derived.
    /// * `variable`: Index of the variable whos bound we attempt to tighten.
    /// * `coefficient`: Coefficient of the variable in the constraint.
    /// * `activity_bounds`: Activity bounds for this constraint.
    /// * `changes`: A `Vec` of (variable index, bound direction, bound value) tuples in which
    /// derived bounds are collected by the calling method.
    fn determine_whether_bound_can_be_tightened(
        &self,
        constraint: usize,
        variable: usize,
        coefficient: OF,
        activity_bounds: &(Option<OF>, Option<OF>),
        derived_bounds: &mut Vec<(usize, BoundDirection, OF)>,
    ) {
        let derived_bound = |activity: Option<OF>| {
            activity.map(|bound| (self.b[constraint] - bound) / coefficient)
        };
        let coefficient_sign = match coefficient.cmp(&OF::additive_identity()) {
            Ordering::Greater => BoundDirection::Upper,
            Ordering::Less => BoundDirection::Lower,
            Ordering::Equal => panic!("Coefficients can't be zero"),
        };

        let first = if self.constraint_types[constraint] != ConstraintType::Less {
            derived_bound(activity_bounds.1.map(|bound| {
                bound - coefficient * match coefficient_sign {
                    BoundDirection::Upper => self.variables[variable].upper_bound,
                    BoundDirection::Lower => self.variables[variable].lower_bound,
                }.unwrap()
            }))
        } else { None };
        let second = if self.constraint_types[constraint] != ConstraintType::Greater {
            derived_bound(activity_bounds.0.map(|bound| {
                bound - coefficient * match coefficient_sign {
                    BoundDirection::Upper => self.variables[variable].lower_bound,
                    BoundDirection::Lower => self.variables[variable].upper_bound,
                }.unwrap()
            }))
        } else { None };

        // TODO: Rounding to integer bounds for integer problems
        match coefficient_sign {
            BoundDirection::Upper => {
                if let Some(bound) = first { derived_bounds.push((variable, BoundDirection::Lower, bound)); }
                if let Some(bound) = second { derived_bounds.push((variable, BoundDirection::Upper, bound)); }
            },
            BoundDirection::Lower => {
                if let Some(bound) = first { derived_bounds.push((variable, BoundDirection::Upper, bound)); }
                if let Some(bound) = second { derived_bounds.push((variable, BoundDirection::Lower, bound)); }
            },
        }
    }

    /// Attempt to tighten variable bounds using newfound bounds.
    ///
    /// This is a helper method for `presolve_constraint_by_domain_propagation`.
    ///
    /// # Arguments
    ///
    /// * `derived_bounds`: Collection of (variable index, bound direction, bound value) tuples.
    fn execute_tightening_changes(
        &mut self,
        derived_bounds: Vec<(usize, BoundDirection, OF)>,
        index: &mut PresolveIndex<OF>,
    ) {
        for (variable, direction, bound) in derived_bounds {
            let maybe_change = match direction {
                BoundDirection::Lower => self.variables[variable].update_lower_bound(bound),
                BoundDirection::Upper => self.variables[variable].update_upper_bound(bound),
            };

            if let Some(change) = maybe_change {
                index.after_bound_change(variable, &self.variables);
                if let Some(by_how_much) = change {
                    index.update_activity_bound(variable, direction, by_how_much);
                }
            }
        }
    }

    /// Iterate over a constraint row during presolving.
    ///
    /// During presolving, for each column, a count is being kept of the number of active (belonging
    /// to a constraint that has not yet been removed) column. When that count is equal to zero, the
    /// coefficient in the original matrix of active variable coefficients is neglected.
    ///
    /// # Arguments
    ///
    /// * `row`: Row in the constraint matrix to iterate over.
    /// * `column_counters`: Counters from the presolve index.
    ///
    /// # Return value
    ///
    /// A collection of (column index, coefficient value) tuples.
    fn iter_active_row<'a>(
        &'a self,
        row: usize,
        column_counters: &'a Vec<usize>,
    ) -> impl Iterator<Item = &'a (usize, OF)> {
        self.constraints.iter_row(row)
            .filter(move|&&(j, _)| column_counters[j] != 0)
    }

    /// Sets variables that can be optimized independently of all others to their optimal values.
    ///
    /// # Arguments
    ///
    /// * `to_optimize`: Collection of variable indices that should be optimized.
    fn optimize_disjoint_variables(&mut self, to_optimize: &Vec<usize>) -> Result<(), LinearProgramType<OF>> {
        for &j in to_optimize {
            let variable = &mut self.variables[j];

            let new_value = match (self.objective, variable.cost.cmp(&OF::additive_identity())) {
                (_, Ordering::Equal) => panic!("Should not be called if there is no cost"),
                (Objective::Minimize, Ordering::Less) | (Objective::Maximize, Ordering::Greater) => {
                    match variable.upper_bound {
                        Some(v) => v,
                        None => return Err(LinearProgramType::Unbounded),
                    }
                },
                (Objective::Minimize, Ordering::Greater) | (Objective::Maximize, Ordering::Less) => {
                    match variable.lower_bound {
                        Some(v) => v,
                        None => return Err(LinearProgramType::Unbounded),
                    }
                },
            };

            self.original_variables[j].1 = Removed(Solved(new_value));
            self.fixed_cost += variable.cost * new_value;
        }

        Ok(())
    }

    /// Remove a set of rows and columns from the constraint data.
    ///
    /// Constraints might have been determined redundant, or perhaps they represented a variable
    /// bound. Those can also be represented in the `self.variables` property of `GeneralForm`. This
    /// method is used to clean up those rows and indices that are left after the removal of those
    /// constraints and / or variables.
    ///
    /// Note that this method is somewhat expensive because it removes columns from a row-major
    /// ordered matrix.
    fn remove_rows_and_columns(&mut self, index: PresolveIndex<OF>) {
        let mut constraints = index.constraints_marked_removed;
        constraints.sort();
        let mut variables = index.variables_marked_removed;
        variables.sort();

        self.constraints.remove_rows(&constraints);
        remove_indices(&mut self.constraint_types, &constraints);
        self.b.remove_indices(&constraints);

        self.constraints.remove_columns_although_this_matrix_is_row_ordered(&variables);
        remove_indices(&mut self.variables, &variables);

        // Update the `from_active_to_original` map.
        if variables.len() > 0 {
            let mut skipped = 1;
            let new_length = self.from_active_to_original.len() - variables.len();
            for j in variables[0]..new_length {
                while skipped < variables.len() && variables[skipped] == j + skipped {
                    skipped += 1;
                }
                self.from_active_to_original[j] = self.from_active_to_original[j + skipped];
            }
            self.from_active_to_original.drain(new_length..);
        }

        debug_assert!(is_consistent(&self));
    }

    /// Shift all variables, such that the lower bound is zero.
    ///
    /// This allows the removal of those lower bounds afterwards; this lower bound is the only lower
    /// bound for problems in canonical form. When working with a simplex tableau, this form allows
    /// us to eliminate all rows which describe a lower bound.
    ///
    /// If later on, such as during branch and bound, an extra lower bound needs to be inserted,
    /// this information can be stored regardless in a separate data structure.
    fn transform_variables(&mut self) {
        debug_assert!(self.variables.iter().all(|v| !v.flipped));

        // Compute all changes that need to happen
        for j in 0..self.variables.len() {
            let variable = &mut self.variables[j];

            // Flip such that there is not just an upper bound
            if let (None, Some(upper)) = (variable.lower_bound, variable.upper_bound) {
                variable.flipped = !variable.flipped;

                variable.lower_bound = Some(-upper);
                variable.upper_bound = None;
            }

            // Shift such that any lower bound is zero
            match variable.lower_bound {
                Some(ref mut lower) if *lower != OF::additive_identity() => {
                    variable.shift = -*lower;
                    *lower += variable.shift; // *lower = 0
                    if let Some(ref mut upper) = variable.upper_bound {
                        *upper += variable.shift;
                    }
                    self.fixed_cost += variable.cost * variable.shift;
                },
                _ => (),
            }
        }

        // Do these changes in the coefficients
        for (i, tuples) in self.constraints.data.iter_mut().enumerate() {
            for &mut (j, ref mut coefficient) in tuples {
                self.b.shift_value(i,*coefficient * self.variables[j].shift);
                if self.variables[j].flipped {
                    *coefficient *= -OF::multiplicative_identity();
                }
            }
        }

        debug_assert!(is_consistent(&self));
    }

    /// Multiply the constraints by a constant such that the constraint value is >= 0.
    ///
    /// This is a step towards representing a `GeneralForm` problem in `CanonicalForm`.
    fn make_b_non_negative(&mut self) {
        let rows_to_negate = self.b.iter_values()
            .enumerate()
            .filter(|&(_, &v)| v < OF::additive_identity())
            .map(|(i, _)| i)
            .collect();

        self.constraints.change_row_signs(&rows_to_negate);
        for row in rows_to_negate.into_iter() {
            self.constraint_types[row] = match self.constraint_types[row] {
                ConstraintType::Greater => ConstraintType::Less,
                ConstraintType::Equal => ConstraintType::Equal,
                ConstraintType::Less => ConstraintType::Greater,
            };
            self.b[row] *= -OF::multiplicative_identity();
        }

        debug_assert!(is_consistent(&self));
    }

    /// Make this a minimization problem by multiplying the cost function by -1.
    fn make_minimization_problem(&mut self) {
        if self.objective == Objective::Maximize {
            self.objective = Objective::Minimize;

            for variable in &mut self.variables {
                variable.cost = -variable.cost;
            }
        }
    }

    /// Split the constraints out per type.
    ///
    /// The constraints in a `GeneralForm` linear program are mixed; the of the constraint is saved
    /// in `self.constraint_types`. A `CanonicalForm` linear program has a separate data structure
    /// for each constraint type. This to facilitate the easy creation of a `MatrixData` data
    /// struct, which "simulates" the presence of slack variables based on those different
    /// constraint types.
    fn split_constraints_by_type(
        &self,
    ) -> (DenseVector<OF>, (Vec<SparseTuples<OF>>, Vec<SparseTuples<OF>>, Vec<SparseTuples<OF>>)) {
        let (mut b_equality, mut b_upper, mut b_lower) = (Vec::new(), Vec::new(), Vec::new());
        let (mut equality, mut upper, mut lower) = (Vec::new(), Vec::new(), Vec::new());
        for (i, row) in self.constraints.data.iter().enumerate() {
            match self.constraint_types[i] {
                ConstraintType::Equal => {
                    b_equality.push(self.b[i]);
                    equality.push(row.clone());
                },
                ConstraintType::Less => {
                    b_upper.push(self.b[i]);
                    upper.push(row.clone());
                }
                ConstraintType::Greater => {
                    b_lower.push(self.b[i]);
                    lower.push(row.clone());
                },
            }
        }

        (
            DenseVector::new([b_equality, b_upper, b_lower].concat(), self.b.len()),
            (equality, upper, lower),
        )
    }

    /// Get the known solution value for a variable, if there is one.
    ///
    /// # Arguments
    ///
    /// * `variable`: Index of the variable to get the solution for with respect to the *original,
    /// non-presolved* problem.
    ///
    /// # Return value
    ///
    /// `None` if the variable is still in the problem, `Some` if not. In the latter case, if the
    /// solution is a specific value known at this point in time, it contains another `Some` with
    /// value and `None` otherwise.
    ///
    /// ## Note
    ///
    /// This is only an actual solution for the variable if the problem turns out to be feasible.
    fn is_variable_presolved(&self, variable: usize) -> bool {
        debug_assert!(variable < self.variables.len());

        match self.original_variables[variable].1 {
            OriginalVariable::Active(_) => false,
            OriginalVariable::Removed(RemovedVariable::Solved(_)) => true,
            OriginalVariable::Removed(RemovedVariable::FunctionOfOthers { .. }) => true,
        }
    }

    /// Compute explicit solution from slack variables where possible.
    fn compute_solution_where_possible(&mut self) {
        // To avoid recomputations, we store all computed intermediate values in this collection.
        let mut new_solutions = vec![None; self.original_variables.len()];
        let mut changed = Vec::new();
        for (j, (_, variable)) in self.original_variables.iter().enumerate() {
            if let OriginalVariable::Removed(FunctionOfOthers { .. }) = variable {
                if self.compute_solution_value(j, &mut new_solutions).is_some() {
                    changed.push(j);
                }
            }
        }
        for variable in changed {
            self.original_variables[variable].1 = OriginalVariable::Removed(Solved(new_solutions[variable].unwrap()));
        }
    }

    /// Compute the solution value for a single variable.
    ///
    /// This is a helper method of the `compute_solution_where_possible` function.
    ///
    /// # Arguments
    ///
    /// * `variable`: Index of the variable for which we try to determine an explicit solution.
    /// * `new_solutions`: Collection containing solutions previously computed, that the computed
    /// solution value will also be written in to (if it is determined).
    ///
    /// # Return value
    ///
    /// The solution value, if it could be determined.
    fn compute_solution_value(&self, variable: usize, new_solutions: &mut Vec<Option<OF>>) -> Option<OF> {
        let new_value = match &self.original_variables[variable].1 {
            OriginalVariable::Active(_) => None,
            OriginalVariable::Removed(Solved(value)) => Some(*value),
            OriginalVariable::Removed(FunctionOfOthers { constant, coefficients }) => {
                new_solutions[variable].or_else(|| {
                    coefficients.iter().map(|&(j, coefficient)| {
                        self.compute_solution_value(j, new_solutions).map(|v| coefficient * v)
                    })
                        .sum::<Option<OF>>()
                        .map(|inner_product| *constant - inner_product)
                })
            }
        };

        if let Some(value) = new_value {
            new_solutions[variable] = Some(value);
        }

        new_value
    }

    /// If this problem is fully solved (probably by presolving), get the solution.
    ///
    /// # Return value
    ///
    /// If one of the variables is still undertermined, `None`.
    pub fn get_solution(&self) -> Option<Solution<OF>> {
        debug_assert_eq!(self.nr_variables(), 0);

        let maybe_variable_values = self.original_variables.iter().map(|(name, variable)| {
            if let OriginalVariable::Removed(Solved(value)) = variable {
                Some((name.clone(), *value))
            } else {
                None
            }
        }).collect::<Option<Vec<_>>>();
        maybe_variable_values.map(|variable_values| {
            Solution::new(self.fixed_cost, variable_values)
        })
    }

    /// The number of constraints in this linear program.
    ///
    /// # Return value
    ///
    /// The number of constraints, which excludes any variable bounds.
    pub fn nr_constraints(&self) -> usize {
        self.constraints.nr_rows()
    }

    /// The number of variables in this linear program.
    ///
    /// # Return value
    ///
    /// The number of columns / variables, which includes the slack columns / variables.
    pub fn nr_variables(&self) -> usize {
        self.constraints.nr_columns()
    }
}

/// Container data structure to keep track of presolve status.
struct PresolveIndex<F: Field> {
    // Queues
    /// TODO(OPTIMIZATION): Could a different datastructure back these queues more efficiently?
    ///  How about stacks? Also, some queues might have limited lengths.
    /// Constraints to check for empty row (bound should be suitable).
    /// All elements should have a row count 0.
    empty_row_queue: HashSet<usize>,
    /// Constraints to check to see whether they are a bound (without a slack).
    /// All elements should have row count 1.
    bound_queue: HashSet<usize>,
    /// Constraints to check for activity bound tightening.
    /// All elements need to be checked at least once.
    activity_queue: HashSet<usize>,
    /// Variables that are fixed need substitution.
    /// All elements have a single feasible value.
    substitution_queue: HashSet<usize>,
    /// Variables to check for slack variables.
    /// All elements have column count 1.
    slack_queue: HashSet<usize>,

    // Collectors
    /// Collecting the indices, relative to the active part of the problem, that should be removed.
    ///
    /// This collection is here only to avoid an extra scan over the `column_counters`.
    constraints_marked_removed: Vec<usize>,
    /// Collecting the indices, relative to the active part of the problem, that should be removed.
    ///
    /// This collection is here only to avoid an extra scan over the `column_counters`.
    variables_marked_removed: Vec<usize>,
    /// These columns still need to be optimized independently.
    ///
    /// They could be removed from the problem because they at some point during presolving no
    /// longer interacted with any constraint. These columns do have a nonzero coefficient in the
    /// cost function.
    ///
    /// This is a subset of `columns_marked_for_removal`.
    columns_optimized_independently: Vec<usize>,

    // Counters
    /// Amount of meaningful elements still in the column or row.
    /// The elements should at least be considered when the counter drops below 2. This also
    /// depends on whether the variable appears in the cost function.
    variable_counters: Vec<usize>,
    /// The elements should at least be reconsidered when the counter drops below 2.
    constraint_counters: Vec<usize>,
    /// We maintain the computed activity bounds.
    /// TODO(PRECISION): Include counter for recomputation (numerical errors accumulate)
    activity_bounds: Vec<(Option<F>, Option<F>)>,
    /// TODO(OPTIMIZATION): Add index counting whether an activity bound can be computed (all active
    ///  elements are iterated to determine whether the bound can be computed, can be avoided).

    // Elements from the general form
    /// Column major representation of the constraint matrix (a copy).
    columns: SparseMatrix<F, ColumnMajorOrdering>,
}

impl<OF: OrderedField> PresolveIndex<OF> {
    /// Create a new instance.
    ///
    /// This operation is expensive, it creates a column major copy of the constraint matrix. The
    /// creation of some of the indices might also be expensive.
    /// TODO(OPTIMIZATION): Can this copy be created only when needed?
    /// TODO(OPTIMIZATION): Are these counters and queues created in a single pass? (compiler)
    ///
    /// Note that all indices of columns and variables in the attributes of this struct are
    /// relative to the active variables in the problem in its state before the presolving started.
    /// These indices get updated (e.g. due to the removal of constraints from the general form's
    /// data structures) only after this struct is dropped.
    ///
    /// # Arguments
    ///
    /// * `general_form`: Problem description which is being presolved.
    fn new(general_form: &GeneralForm<OF>) -> Self {
        let columns = SparseMatrix::from_row_ordered_tuples_although_this_is_expensive(
            &general_form.constraints.data, general_form.constraints.nr_columns()
        );
        let row_counters = (0..general_form.constraints.nr_rows())
            .map(|i| general_form.constraints.data[i].len())
            .collect::<Vec<_>>();
        let column_counters = (0..general_form.constraints.nr_columns())
            .map(|j| columns.data[j].len())
            .collect::<Vec<_>>();

        Self {
            // Queues
            empty_row_queue: row_counters.iter().enumerate()
                .filter(|&(_, &count)| count == 0)
                .map(|(i, _)| i).collect(),
            bound_queue: row_counters.iter().enumerate()
                .filter(|&(_, &count)| count == 1)
                .map(|(i, _)| i).collect(),
            activity_queue: (0..general_form.constraints.nr_rows()).collect(),

            substitution_queue: general_form.variables.iter().enumerate()
                .filter_map(|(j, variable)| variable.is_fixed().map(|_| j))
                .collect(),
            slack_queue: column_counters.iter().enumerate()
                .filter(|&(_, &count)| count == 1)
                .filter(|&(j, _)| general_form.variables[j].cost == OF::additive_identity())
                .map(|(j, _)| j).collect(),

            // Collecting removed constraints and variables
            constraints_marked_removed: row_counters.iter().enumerate()
                .filter(|&(_, &count)| count == 0)
                .map(|(i, _)| i).collect(),
            variables_marked_removed: column_counters.iter().enumerate()
                .filter(|&(_, &count)| count == 0)
                .map(|(j, _)| j).collect(),
            columns_optimized_independently: column_counters.iter().enumerate()
                .filter(|&(_, &count)| count == 0)
                .filter(|&(j, _)| general_form.variables[j].cost != OF::additive_identity())
                .map(|(j, _)| j).collect(),

            // Counters
            variable_counters: column_counters,
            constraint_counters: row_counters,
            activity_bounds: vec![(None, None); general_form.nr_constraints()],

            columns,
        }
    }

    /// Performs actions that should be performed after a new variable bound is found.
    ///
    /// These include:
    ///
    /// * If a variable is now fixed, it should be substituted in the problem.
    /// * Otherwise, some rows might need to have the existing activity bound recomputed
    ///
    /// # Arguments
    ///
    /// * `variable`: Variable who's bound was changed.
    /// * `variables`: View into the general form's problem's variables that were active before the
    /// presolving started.
    fn after_bound_change(
        &mut self,
        variable: usize,
        variables: &Vec<Variable<OF>>,
    ) {
        debug_assert_ne!(self.variable_counters[variable], 0);

        if variables[variable].is_fixed().is_some() {
            self.substitution_queue.insert(variable);
        } else {
            let rows_to_recheck = self.iter_active_column(variable)
                .map(|&(i, _)| i)
                .collect::<Vec<_>>();
            self.activity_queue.extend(rows_to_recheck.iter());
        }
    }

    /// Update an activity bound if it exists.
    ///
    /// Activity bounds don't have to be recomputed entirely after a single variable bound change.
    /// You need to know by how much the bound was changed, which gets lost. So it is recomputed
    /// now, even though the result might not be directly used, and other recomputations might be
    /// triggered first (in the future, perhaps due to the counter for the number of recomputations
    /// reaching it's limit (precision)).
    ///
    /// # Arguments
    ///
    /// * `variable`: Variable who's bound was changed.
    /// * `direction`: Whether an upper or lower bound was changed.
    /// * `by_how_much`: Size of the change.
    fn update_activity_bound(&mut self, variable: usize, direction: BoundDirection, by_how_much: OF) {
        debug_assert!(match direction {
            BoundDirection::Lower => by_how_much > OF::additive_identity(),
            BoundDirection::Upper => by_how_much < OF::additive_identity(),
        });

        for &(row, coefficient) in self.columns.iter_column(variable) {
            if !self.is_constraint_still_active(row) {
                continue;
            }

            if let Some(ref mut bound) = match (direction, coefficient.cmp(&OF::additive_identity())) {
                // Update the lower bound
                (BoundDirection::Lower, Ordering::Greater) | (BoundDirection::Upper, Ordering::Less) => self.activity_bounds[row].0,
                // Update the lower bound
                (BoundDirection::Lower, Ordering::Less) | (BoundDirection::Upper, Ordering::Greater) => self.activity_bounds[row].1,
                (_, Ordering::Equal) => panic!("Zero coefficient"),
            } {
                // TODO(NUMERICS): See Achterberg (2007), Algorithm 7.1
                *bound += by_how_much * coefficient;
            }
        }
    }

    /// Whether all queues are empty.
    ///
    /// This indicates whether the repeated application of reduction rules can be stopped.
    fn queues_are_empty(&self) -> bool {
        // Note the reverse order w.r.t. the order in which these queues are tested in the main loop
        self.activity_queue.is_empty()
            && self.slack_queue.is_empty()
            && self.bound_queue.is_empty()
            && self.substitution_queue.is_empty()
            && self.empty_row_queue.is_empty()
    }

    /// Iterate over the constraints of a column who have not (yet) been eliminated.
    fn iter_active_column(&self, column: usize) -> impl Iterator<Item = &(usize, OF)> {
        self.columns.iter_column(column)
            .filter(move |&&(i, _)| self.is_constraint_still_active(i))
    }

    /// Iterate over the columns of a constraint that have not yet been eliminated.
    ///
    /// # Arguments
    ///
    /// * `constraint`: Constraint to iter over.
    /// * `constraints`: The row major representation of the constraints of the general form for
    /// fast iteration.
    fn iter_active_row<'a>(
        &'a self,
        constraint: usize,
        constraints: &'a SparseMatrix<OF, RowMajorOrdering>,
    ) -> impl Iterator<Item = &'a (usize, OF)> {
        constraints.iter_row(constraint)
            .filter(move |&&(j, _)| self.is_variable_still_active(j))
    }

    /// The constraint counter indicates whether the constraint still has any variables left.
    ///
    /// Note that this can be zero, even though the constraint has not yet been eliminated, because
    /// it still needs to be checked by `presolve_empty_constraint`. It can, in that case, however
    /// be ignored during the application of presolving rules.
    fn is_constraint_still_active(&self, constraint: usize) -> bool {
        debug_assert_eq!(
            self.constraints_marked_removed.contains(&constraint),
            self.constraint_counters[constraint] == 0,
        );

        self.constraint_counters[constraint] != 0
    }

    /// The variable counter indicates whether the variable still appears in any constraint.
    fn is_variable_still_active(&self, variable: usize) -> bool {
        debug_assert_eq!(
            self.variables_marked_removed.contains(&variable),
            self.variable_counters[variable] == 0,
        );

        self.variable_counters[variable] != 0
    }

    /// Mark a constraint as removed.
    ///
    /// There should be more than one element in this row, otherwise, the column that can be removed
    /// should be known from the rule application and removed directly, such that an iteration can
    /// be avoided.
    ///
    /// # Arguments
    ///
    /// * `constraint`: Constraint to iter over.
    /// * `constraints`: The row major representation of the constraints of the general form for
    /// fast iteration.
    /// * `variables`: Variables of the active part of the problem.
    ///
    /// TODO(ARCHITECTURE): This `constraints` and `variables` need to be given to this object,
    ///  which seems a bit ugly. Can this be avoided? See also `iter_active_row` from `GeneralForm`.
    fn remove_constraint_values(
        &mut self,
        constraint: usize,
        constraints: &SparseMatrix<OF, RowMajorOrdering>,
        variables: &Vec<Variable<OF>>,
    ) {
        debug_assert!(self.constraint_counters[constraint] >= 2);

        let variables_to_scan = self.iter_active_row(constraint, constraints)
            .map(|&(j, _)| j).collect::<Vec<_>>();
        self.constraint_counters[constraint] -= variables.len();
        for variable in variables_to_scan {
            self.variable_counters[variable] -= 1;
            self.readd_column_to_queues_based_on_counter(variable, variables);
        }
    }

    /// When the variable counter drops low, this has implications for rules that should be tested.
    fn readd_column_to_queues_based_on_counter(&mut self, column: usize, variables: &Vec<Variable<OF>>) {
        match self.variable_counters[column] {
            0 => {
                self.remove_variable(column);
                if variables[column].cost != OF::additive_identity() {
                    self.columns_optimized_independently.push(column);
                }
            },
            1 => if variables[column].cost == OF::additive_identity() {
                self.slack_queue.insert(column);
            },
            _ => (),
        }
    }

    /// When a constraint counter drops low, this has implications for rules that should be tested.
    fn readd_row_to_queues(&mut self, row: usize) {
        match self.constraint_counters[row] {
            0 => self.empty_row_queue.insert(row),
            1 => self.bound_queue.insert(row),
            _ => self.activity_queue.insert(row),
        };
    }

    /// Mark a constraint as removed.
    fn remove_constraint(&mut self, constraint: usize) {
        debug_assert_eq!(self.constraint_counters[constraint], 0);

        self.constraints_marked_removed.push(constraint);

        self.empty_row_queue.remove(&constraint);
        self.bound_queue.remove(&constraint);
        self.activity_queue.remove(&constraint);
    }

    /// Mark a variable as removed.
    fn remove_variable(&mut self, variable: usize) {
        debug_assert_eq!(self.variable_counters[variable], 0);

        self.variables_marked_removed.push(variable);

        self.substitution_queue.remove(&variable);
        self.slack_queue.remove(&variable);
    }
}

/// A variable as part of a linear problem without restrictions (as opposed to for a
/// `CanonicalFrom` variable).
///
/// A variable is named, of continuous or integer type and may be shifted and flipped w.r.t. how it
/// was originally present in the problem.
///
/// The upper bound is relative to the shift; that is, the lower bound is `lower_bound - shift`, the
/// upper bound is `upper_bound - shift`. For example, the range stays the same, regardless of the
/// shift.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Variable<F: Field> {
    /// Whether the variable is integer or not.
    pub variable_type: VariableType,
    /// Coefficient in the objective function.
    pub cost: F,
    /// Describing the accepted values for this variable
    ///
    /// Lower bound should be set to 0 when a variable is nonnegative. If it is `None`, the variable
    /// is considered to be in (-oo, upper_bound).
    pub lower_bound: Option<F>,
    /// Describing the accepted values for this variable
    ///
    /// If it is `None`, the variable is considered to be in (lower_bound, oo).
    pub upper_bound: Option<F>,
    /// How much this variable is shifted to have a zero lower bound.
    ///
    /// To find the "true" solution value, one needs to subtract this shift from the solution value
    /// produced by an optimization routine using the lower bound of 0.
    pub shift: F,
    /// Whether this variable was originally negative.
    ///
    /// To find the "true" solution value, one needs to multiply the solutionvalue found by -1, and
    /// then shift the value by the `shifted_by` field value.
    pub flipped: bool,
}

impl<OF: OrderedField> Variable<OF> {
    /// Whether the variable allows only a single value.
    ///
    /// # Return value
    ///
    /// `Some` with the value if so, `None` otherwise.
    fn is_fixed(&self) -> Option<OF> {
        match (self.lower_bound, self.upper_bound) {
            (Some(lower), Some(upper)) if lower == upper => Some(lower),
            _ => None,
        }
    }

    /// Whether a variable is unconstrained (has no bounds).
    fn is_free(&self) -> bool {
        self.lower_bound.is_none() && self.upper_bound.is_none()
    }

    /// Whether the variable admits a feasible value (and the upper bound is not below the lower
    /// bound)
    fn has_feasible_value(&self) -> bool {
        match (self.lower_bound, self.upper_bound) {
            (Some(lower), Some(upper)) => lower <= upper,
            _ => true,
        }
    }

    /// Change the lower bound if the given bound is higher.
    ///
    /// # Arguments
    ///
    /// * `new`: New value to compare the existing bound against (if there is any)
    ///
    /// # Return value
    ///
    /// `None` if the bound was not updated, `Some` if it was. If there was no bound before, a
    /// `None`, otherwise, a `Some` with the difference between the old and new bound. This
    /// difference is always strictly positive, as the lower bound can only be increased.
    fn update_lower_bound(&mut self, new: OF) -> Option<Option<OF>> {
        Self::update_bound(&mut self.lower_bound, new, |new, existing| new > existing)
    }

    /// Change the upper bound if the given bound is lower.
    ///
    /// # Arguments
    ///
    /// * `new`: New value to compare the existing bound against (if there is any)
    ///
    /// # Return value
    ///
    /// `None` if the bound was not updated, `Some` if it was. If there was no bound before, a
    /// `None`, otherwise, a `Some` with the difference between the old and new bound. This
    /// difference is always strictly negative, as the upper bound can only be decreased.
    fn update_upper_bound(&mut self, new: OF) -> Option<Option<OF>> {
        Self::update_bound(&mut self.upper_bound, new, |new, existing| new < existing)
    }

    /// Update either the lower or upper bound, if it makes the bound tighter.
    ///
    /// This is a helper method.
    ///
    /// # Arguments
    ///
    /// * `existing_bound`: Bound that could be updated.
    /// * `new`: New bound value.
    /// * `is_better`: A predicate indicating whether the new bound is better than the old bound, or
    /// not.
    ///
    /// # Return value
    ///
    /// `None` if the bound was not updated, `Some` if it was. If there was no bound before, a
    /// `None`, otherwise, a `Some` with the difference between the old and new bound.
    fn update_bound<P: Fn(OF, OF) -> bool>(
        existing_bound: &mut Option<OF>,
        new: OF,
        is_better: P,
    ) -> Option<Option<OF>> {
        match existing_bound {
            Some(existing) => {
                if is_better(new, *existing) {
                    *existing = new;
                    Some(Some(new - *existing))
                } else {
                    None
                }
            }
            None => {
                *existing_bound = Some(new);
                Some(None)
            }
        }
    }
}

#[cfg(test)]
mod test {
    use num::rational::Ratio;
    use num::traits::FromPrimitive;

    use crate::data::linear_algebra::matrix::{MatrixOrder, RowMajorOrdering, SparseMatrix};
    use crate::data::linear_algebra::matrix::ColumnMajorOrdering;
    use crate::data::linear_algebra::vector::DenseVector;
    use crate::data::linear_algebra::vector::test::TestVector;
    use crate::data::linear_program::elements::{ConstraintType, LinearProgramType, Objective, VariableType};
    use crate::data::linear_program::general_form::{GeneralForm, Variable};
    use crate::data::linear_program::solution::Solution;
    use crate::R32;

    type T = Ratio<i32>;

    /// MIN Z = 211x1 + 223x2 + 227x3 - 229x4 + 233x5 + 0x6
    /// subject to
    /// 2x1 = 101
    /// 3x1 + 5x2 <= 103
    /// 7x1 + 11x2 + 13x3 >= 107
    /// x2 >= -97/10
    /// 17x1 + 19x2 + 23x3 + 29x5 + 31x6 = 109
    /// x4 <= 131
    /// x5 >= -30736/1885
    /// x5 <= 123
    /// x6 >= 5
    /// and x1,x2,x3,x4,x5,x6 unrestricted in sign
    #[test]
    fn test_presolve() {
        let data = vec![
            // Column 3 should be removed because empty
            vec![2f64, 0f64, 0f64, 0f64, 0f64, 0f64], // Should be removed because simple bound
            vec![3f64, 5f64, 0f64, 0f64, 0f64, 0f64], // Should be removed because simple bound after removal of the row above
            vec![7f64, 11f64, 13f64, 0f64, 0f64, 0f64], // Should be removed because of fixed variable after the removal of above two
            vec![17f64, 19f64, 23f64, 0f64, 29f64, 31f64], // Row that should stay
        ];
        let rows = RowMajorOrdering::from_test_data(&data, 6);
        let b = DenseVector::from_test_data(vec![
            101f64,
            103f64,
            107f64,
            109f64,
        ]);
        let constraints = vec![
            ConstraintType::Equal,
            ConstraintType::Less,
            ConstraintType::Greater,
            ConstraintType::Equal,
        ];
        let column_info = vec![
            Variable {
                variable_type: VariableType::Continuous,
                cost: R32!(211),
                lower_bound: None,
                upper_bound: None,
                shift: R32!(0),
                flipped: false
            }, Variable {
                variable_type: VariableType::Continuous,
                cost: R32!(223),
                lower_bound: Some((R32!(103) - R32!(101) / R32!(2) * R32!(3)) / R32!(5)),
                upper_bound: None,
                shift: R32!(0),
                flipped: false
            }, Variable {
                variable_type: VariableType::Continuous,
                cost: R32!(227),
                lower_bound: None,
                upper_bound: None,
                shift: R32!(0),
                flipped: false
            }, Variable {
                variable_type: VariableType::Continuous,
                cost: R32!(-229),
                lower_bound: None,
                upper_bound: Some(R32!(131)),
                shift: R32!(0),
                flipped: false
            }, Variable {
                variable_type: VariableType::Continuous,
                cost: R32!(233),
                lower_bound: Some(R32!(-30736, 65 * 29)),
                upper_bound: Some(R32!(123)),
                shift: R32!(0),
                flipped: false
            }, Variable {
                variable_type: VariableType::Continuous,
                cost: R32!(0),
                lower_bound: Some(R32!(5)),
                upper_bound: None,
                shift: R32!(0),
                flipped: false
            },
        ];
        let variable_names = vec![
            "XONE".to_string(),
            "XTWO".to_string(),
            "XTHREE".to_string(),
            "XFOUR".to_string(),
            "XFIVE".to_string(),
            "XSIX".to_string(),
        ];
        let mut initial = GeneralForm::new(
            Objective::Minimize,
            rows,
            constraints,
            b,
            column_info,
            variable_names,
            R32!(1),
        );
        debug_assert_eq!(
            initial.presolve(),
            Err(LinearProgramType::FiniteOptimum(Solution::new(
                R32!(1)
                    + R32!(211 * 101, 2)
                    + R32!(223 * -97, 10)
                    + R32!(227 * -699, 65)
                    + R32!(-229 * 131)
                    + R32!(233 * -30736, 1885),
                vec![
                    ("XONE".to_string(), R32!(101, 2)),
                    ("XTWO".to_string(), (R32!(103) - R32!(101) / R32!(2) * R32!(3)) / R32!(5)),
                    ("XTHREE".to_string(), (R32!(-3601, 5) + R32!(29 * 30736, 1885)) / R32!(23)),
                    ("XFOUR".to_string(), R32!(131)),
                    ("XFIVE".to_string(), R32!(-30736, 65 * 29)),
                    ("XSIX".to_string(), R32!(5)),
                ],
            ))),
        );
    }

    /// If a simple equality bound is found, remember the solution value and remove the row and
    /// column
    #[test]
    fn test_substitute_fixed() {
        let data = vec![
            vec![1f64, 0f64, 0f64],
            vec![1f64, 2f64, 3f64],
        ];
        let constraint_types = vec![
            ConstraintType::Equal,
            ConstraintType::Less,
        ];
        let columns: SparseMatrix<Ratio<i32>, _> = ColumnMajorOrdering::from_test_data(&data, 3);
        let constraints = RowMajorOrdering::from_test_data(&data, 3);
        let b = DenseVector::from_test_data(vec![3f64, 8f64]);
        let variables = vec![Variable {
            variable_type: VariableType::Continuous,
            cost: R32!(1),
            lower_bound: None,
            upper_bound: None,
            shift: R32!(0),
            flipped: false
        }, Variable {
            variable_type: VariableType::Continuous,
            cost: R32!(2),
            lower_bound: None,
            upper_bound: None,
            shift: R32!(0),
            flipped: false
        }, Variable {
            variable_type: VariableType::Continuous,
            cost: R32!(3),
            lower_bound: None,
            upper_bound: None,
            shift: R32!(0),
            flipped: false
        }];
        let variable_names = vec!["XONE".to_string(), "XTWO".to_string(), "XTHREE".to_string()];
        let mut initial = GeneralForm::new(
            Objective::Minimize,
            constraints,
            constraint_types,
            b,
            variables,
            variable_names,
            R32!(0),
        );
        initial.presolve();

        let data = vec![vec![2f64, 3f64]];
        let rows = RowMajorOrdering::from_test_data(&data, 2);
        let constraints = vec![ConstraintType::Less];
        let b = DenseVector::from_test_data(vec![5f64]);
        let fixed_cost = R32!(3);
        let variables = vec![
            Variable {
                variable_type: VariableType::Continuous,
                cost: R32!(2),
                lower_bound: None,
                upper_bound: None,
                shift: R32!(0),
                flipped: false
            },
            Variable {
                variable_type: VariableType::Continuous,
                cost: R32!(3),
                lower_bound: None,
                upper_bound: None,
                shift: R32!(0),
                flipped: false
            },
        ];
        let variable_names = vec!["XONE".to_string(), "XTWO".to_string(), "XTHREE".to_string()];
        let expected = GeneralForm::new(
            Objective::Minimize,
            rows,
            constraints,
            b,
            variables,
            variable_names,
            fixed_cost,
        );

        assert_eq!(initial, expected);
    }

    /// Shifting a variable
    #[test]
    fn test_shift_variables() {
        let bound_value = 2.5f64;

        let data = vec![
            vec![1f64, 0f64],
            vec![2f64, 1f64],
        ];
        let rows = RowMajorOrdering::from_test_data(&data, 2);
        let columns: SparseMatrix<Ratio<i32>, _> = ColumnMajorOrdering::from_test_data(&data, 2);
        let b = DenseVector::from_test_data(vec![
            2f64,
            8f64,
        ]);
        let constraints = vec![
            ConstraintType::Greater,
            ConstraintType::Less,
        ];
        let variables = vec![Variable {
            variable_type: VariableType::Continuous,
            cost: R32!(1),
            lower_bound: None,
            upper_bound: None,
            shift: R32!(0),
            flipped: false
        }, Variable {
            variable_type: VariableType::Continuous,
            cost: R32!(1),
            lower_bound: Some(T::from_f64(bound_value).unwrap()),
            upper_bound: None,
            shift: R32!(0),
            flipped: false
        }];
        let variable_names = vec!["XONE".to_string(), "XTWO".to_string()];
        let mut general_form = GeneralForm::new(
            Objective::Minimize,
            rows,
            constraints,
            b,
            variables,
            variable_names,
            R32!(0),
        );
        general_form.transform_variables();

        let data = vec![
            vec![1f64, 0f64],
            vec![2f64, 1f64],
        ];
        let rows = RowMajorOrdering::from_test_data(&data, 2);
        let b = DenseVector::from_test_data(vec![
            2f64 - bound_value * 0f64,
            8f64 - bound_value * 1f64,
        ]);
        let constraints = vec![
            ConstraintType::Greater,
            ConstraintType::Less,
        ];
        let variables = vec![
            Variable {
                variable_type: VariableType::Continuous,
                cost: R32!(1),
                lower_bound: None,
                upper_bound: None,
                shift: R32!(0),
                flipped: false
            }, Variable {
                variable_type: VariableType::Continuous,
                cost: R32!(1),
                lower_bound: Some(R32!(0)),
                upper_bound: None,
                shift: -T::from_f64(bound_value).unwrap(),
                flipped: false
            },
        ];
        let variable_names = vec!["XONE".to_string(), "XTWO".to_string(), "XTHREE".to_string()];
        let expected = GeneralForm::new(
            Objective::Minimize,
            rows,
            constraints,
            b,
            variables,
            variable_names,
            -T::from_f64(bound_value).unwrap(),
        );

        assert_eq!(general_form, expected);
    }

    #[test]
    fn test_make_b_non_negative() {
        let rows = RowMajorOrdering::from_test_data(&vec![vec![2f64]], 1);
        let b = DenseVector::from_test_data(vec![-1f64]);
        let variables = vec![
            Variable {
                variable_type: VariableType::Continuous,
                cost: R32!(1),
                lower_bound: None,
                upper_bound: None,
                shift: R32!(0),
                flipped: false
            },
        ];
        let variable_names = vec!["XONE".to_string()];
        let constraints = vec![ConstraintType::Equal];
        let mut result = GeneralForm::new(
            Objective::Minimize,
            rows,
            constraints,
            b,
            variables,
            variable_names,
            R32!(0),
        );
        result.make_b_non_negative();

        let data = RowMajorOrdering::from_test_data(&vec![vec![-2f64]], 1);
        let b = DenseVector::from_test_data(vec![1f64]);
        let constraints = vec![ConstraintType::Equal];
        let variables = vec![Variable {
            variable_type: VariableType::Continuous,
            cost: R32!(1),
            lower_bound: None,
            upper_bound: None,
            shift: R32!(0),
            flipped: false
        }];
        let variable_names = vec!["XONE".to_string()];
        let expected = GeneralForm::new(
            Objective::Minimize,
            data,
            constraints,
            b,
            variables,
            variable_names,
            R32!(0),
        );

        assert_eq!(result, expected);
    }
}
