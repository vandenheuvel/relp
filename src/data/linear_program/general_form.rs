//! # Linear programs in "general form"
//!
//! A linear program in general form is defined as a linear optimization problem, where the
//! variables need to be either nonnegative, or are "free" and unconstrained. The constraints are
//! either equalities, or inequalities of the form a x >= b with no further requirements on b. This
//! module contains data structures to represent such problems, and logic to do conversions from
//! this problem form to e.g. a so called equivalent "canonical" representations where any free
//! variables and inequalities are eliminated.
use std::cmp::Ordering;
use std::collections::HashSet;

use daggy::{Dag, WouldCycle};
use daggy::petgraph::data::Element;

use crate::algorithm::simplex::matrix_provider::matrix_data;
use crate::algorithm::simplex::matrix_provider::matrix_data::MatrixData;
use crate::algorithm::utilities::remove_indices;
use crate::data::linear_algebra::matrix::{ColumnMajorOrdering, RowMajorOrdering};
use crate::data::linear_algebra::matrix::SparseMatrix;
use crate::data::linear_algebra::SparseTuples;
use crate::data::linear_algebra::vector::{DenseVector, SparseVector, Vector};
use crate::data::linear_program::elements::{BoundDirection, ConstraintType, LinearProgramType, Objective, VariableType};
use crate::data::linear_program::general_form::OriginalVariable::Removed;
use crate::data::linear_program::general_form::RemovedVariable::{FunctionOfOthers, Solved};
use crate::data::number_types::traits::{Field, OrderedField};

/// A linear program in general form.
///
/// This structure is used as a first storage independent representation format for different
/// parse results to be transformed to.
#[derive(Debug, Eq, PartialEq)]
pub struct GeneralForm<F: Field> {
    /// Which direction does the objective function go?
    objective: Objective,

    /// All coefficients.
    constraints: SparseMatrix<F, RowMajorOrdering>,
    /// The equation type of all rows, ordered by index.
    constraint_types: Vec<ConstraintType>,
    /// All right-hands sides of equations.
    b: DenseVector<F>,

    /// The names of all variables and their type, ordered by index.
    variables: Vec<Variable<F>>,

    /// Constant in the cost function.
    fixed_cost: F,
    /// Variables previously eliminated from the problem.
    original_variables: Vec<(String, OriginalVariable<F>)>,
    /// Mapping indices of unsolved variables to their index in the original problem.
    from_active_to_original: Vec<usize>,
}
#[derive(Debug, Eq, PartialEq)]
enum OriginalVariable<F> {
    /// Index in the active problem.
    Kept(usize),
    /// Variable was removed
    Removed(RemovedVariable<F>),
}
#[derive(Debug, Eq, PartialEq)]
enum RemovedVariable<F> {
    /// Simply a known value.
    Solved(F),
    /// Affine function of the form `b - <a, x>` where some of the `x` might be unknown.
    FunctionOfOthers {
        constant: F,
        coefficients: SparseTuples<F>,
    },
}

/// Check whether the dimensions of the `GeneralForm` are consistent. Meant for debugging.
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
                OriginalVariable::Kept(index) => Some(*index),
                _ => None,
            })
            .collect::<Vec<_>>() == (0..nr_active_variables).collect::<Vec<_>>();
        let no_cycles = {
            let nodes = (0..nr_original_variables).map(|j| {
                Element::Node { weight: (), }
            });
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

    return true
        && b
        && constraints
        && rows
        && variables
        && columns
        && original_variables
        && from_active_to_original
    ;
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
                .map(|(j, name)| (name, OriginalVariable::Kept(j))).collect(),
            from_active_to_original: (0..nr_active_variables).collect(),
        };

        debug_assert!(is_consistent(&general_form));

        general_form
    }

    /// Modify this linear problem such that it is representable by a `MatrixData` structure.
    ///
    /// See also the documentation of the `GeneralForm::canonicalize` method.
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
    pub(crate) fn canonicalize(&mut self) -> Result<(), LinearProgramType<OF>> {
        self.presolve()?;
        self.transform_variables();
        self.make_b_non_negative();
        self.make_minimization_problem();

        Ok(())
    }

    /// Recursively analyse rows and columns to see if they can be removed.
    ///
    /// This can be seen as a presolve operation.
    ///
    /// All rows should be considered only once, whereas column are considered more than once only
    /// when new bounds are added to the variable.
    ///
    /// TODO: Normalization for numerical stability
    ///
    /// # Arguments
    ///
    /// * `columns` - A column major representation of the constraint data.
    ///
    /// # Return value
    ///
    /// If the problem is not determined infeasible, a tuple containing:
    /// * A set of rows that should be removed.
    /// * A set of columns that should be removed.
    /// * A vec of columns that can be independently maximized, also contained in the above set.
    fn presolve(&mut self) -> Result<(), LinearProgramType<OF>> {
        let mut index = PresolveIndex::new(&self);

        while !index.queues_are_empty() {
            self.presolve_step(&mut index)?;
        }

        self.optimize_disjoint_variables(&index.columns_optimized_independently)?;
        self.remove_rows_and_columns(index);

        debug_assert!(is_consistent(&self));
        Ok(())
    }

    fn presolve_step(&mut self, index: &mut PresolveIndex<OF>) -> Result<(), LinearProgramType<OF>> {
        // Actions that are guaranteed to make the problem smaller
        // Remove a row
        if let Some(&row) = index.empty_row_queue.iter().next() {
            self.remove_empty_row(row, index)?;
            return Ok(());
        }
        // Remove a column
        if let Some(&variable) = index.substitution_queue.iter().next() {
            self.substitute_fixed_variable(variable, index);
            return Ok(());
        }
        // Remove a bound
        if let Some(&constraint) = index.bound_queue.iter().next() {
            self.treat_bound_without_slack(constraint, index)?;
            return Ok(());
        }

        // Actions not guaranteed to make the problem smaller
        // Test whether a variable can be seen as a slack
        if let Some(&variable) = index.slack_queue.iter().next() {
            self.remove_if_slack_with_suitable_bounds(variable, index);
            return Ok(());
        }
        // Domain propagation
        if let Some(&constraint) = index.activity_queue.iter().next() {
            self.domain_propagate_by_activity_bounds(constraint, index)?;
            index.activity_queue.remove(&constraint);
            return Ok(());
        }

        Ok(())
    }

    /// See whether a row can be removed.
    ///
    /// This might happen when a row has no constraints, or when the row has just a single
    /// constraint, which indicates that it is a bound (that can be represented differently in a
    /// `GeneralForm`.
    ///
    /// # Arguments
    ///
    /// * `constraint` - Index of row to investigate.
    /// * `row_counters` - Amount of "meaningful" (no known solution value, not yet marked for
    /// deletion) elements that are still left in a row.
    ///
    /// # Return value
    ///
    /// Result that indicates whether the linear program might still have a feasible solution (if
    /// not, the `Result::Err` type will indicate that it is not). Inside there is an `Option` which
    /// might contain the index of a variable that would be effected by the removal of the row under
    /// consideration.


    /// Whether an empty constraint can be safely discarded.
    ///
    /// When this method gets called, the constraint should have already been marked for deletion.
    ///
    /// # Arguments
    ///
    /// * `constraint` - Index of row to investigate.
    ///
    /// # Return value
    ///
    /// `Result` indicating whether the linear program might still be feasible.
    fn remove_empty_row(
        &self,
        constraint: usize,
        index: &mut PresolveIndex<OF>,
    ) -> Result<(), LinearProgramType<OF>> {
        debug_assert_eq!(index.row_counters[constraint], 0);

        if self.b.get_value(constraint) != OF::additive_identity() {
            Err(LinearProgramType::Infeasible)
        } else {
            Ok(index.remove_constraint(constraint))
        }
    }

    /// If a variable is determined, substitute it in constraints in which it appears.
    ///
    /// # Arguments
    ///
    /// * `column` - Index of column under consideration.
    /// * `columns` - Column major ordered sparsematrix of the constraints.
    ///
    /// # Return value
    ///
    /// `Option` indicating whether the variable was fixed. Inside it, a `Vec` of indices of rows
    /// that were affected.
    fn substitute_fixed_variable(
        &mut self,
        column: usize,
        index: &mut PresolveIndex<OF>,
    ) {
        debug_assert!(self.variables[column].is_fixed().is_some());

        let value = self.variables[column].is_fixed().unwrap();
        let column_to_scan = index.iter_active_column(column)
            .map(|&e| e).collect::<Vec<_>>();
        index.column_counters[column] -= column_to_scan.len();
        for (row, coefficient_value) in column_to_scan {
            let old_b = self.b.get_value(row);
            self.b.set_value(row, old_b - coefficient_value * value);
            index.row_counters[row] -= 1;
            index.readd_row_to_queues(row);
        }
        self.fixed_cost += self.variables[column].cost * value;

        self.original_variables[column].1 = OriginalVariable::Removed(RemovedVariable::Solved(value));
        index.remove_variable(column);
    }

    /// Modify bounds of a non-slack variable.
    ///
    /// # Arguments
    ///
    /// * `constraint` - Index of a row with only a bound.
    ///
    /// # Return value
    ///
    /// `Result` indicating whether the linear program might still be feasible. Inside it, the
    /// column that should be checked next.
    fn treat_bound_without_slack(
        &mut self,
        constraint: usize,
        index: &mut PresolveIndex<OF>,
    ) -> Result<(), LinearProgramType<OF>> {
        debug_assert_eq!(index.row_counters[constraint], 1);
        debug_assert_eq!(self.iter_active_row(constraint, &index.column_counters).count(), 1);

        let &(column, value) = self.iter_active_row(constraint, &index.column_counters)
            .nth(0).unwrap();
        debug_assert_ne!(index.column_counters[column], 0);

        let bound_value = self.b.get_value(constraint) / value;
        let mut changes = Vec::with_capacity(2);
        match self.constraint_types[constraint] {
            ConstraintType::Greater => changes.push(BoundDirection::Lower),
            ConstraintType::Less => changes.push(BoundDirection::Upper),
            ConstraintType::Equal => {
                changes.push(BoundDirection::Lower);
                changes.push(BoundDirection::Upper);
            },
        }
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
            index.row_counters[constraint] -= 1;
            index.column_counters[column] -= 1;
            index.remove_constraint(constraint);
            if bound_changed {
                index.after_bound_change(column, &self.variables);
            }
            index.readd_column_to_queues_based_on_counter(column, &self.variables);
            Ok(())
        }
    }

    /// If the variable is a slack variable, it might be possible to remove it.
    ///
    /// This method attempts to remove slack variables that "don't matter". E.g. a variable appears
    /// only in a single row, while not having a cost coefficient. In that case, it is essentially
    /// a slack variable, supporting the definition of constraint on the equation in that row.
    ///
    /// In that latter case, if the slack is bounded on two sides, we leave things as they are. If
    /// not, we can remove the slack and update the bound on the constraint.
    ///
    /// # Arguments
    ///
    /// * `column` - Index of column that should be removed if it is a slack.
    /// * `row_counters` - Amount of meaningful elements left in each row.
    /// * `columns` - Column major representation of the constraint data.
    ///
    /// # Return value
    ///
    /// `Some` if the slack actually is removed, containing the index of the row affected. `None` if
    /// not removed.
    fn remove_if_slack_with_suitable_bounds(
        &mut self,
        column: usize,
        index: &mut PresolveIndex<OF>,
    ) {
        debug_assert_eq!(index.column_counters[column], 1);
        debug_assert_eq!(self.variables[column].cost, OF::additive_identity());
        debug_assert!(self.variables[column].is_fixed().is_none());

        let (row, coefficient, effective_bounds) = self.get_info(column, index);
        let change = Self::get_change(effective_bounds, self.constraint_types[row]);

        if let Some(maybe_new_bound) = change {
            self.save_slack_value(row, column, coefficient, index);
            self.maybe_update_bound(row, coefficient, maybe_new_bound, index);
            index.row_counters[row] -= 1;
            index.column_counters[column] -= 1;
            index.remove_variable(column);
        } else {
            index.slack_queue.remove(&column);
        }
    }

    fn get_info(&self, column: usize, index: &PresolveIndex<OF>) -> (usize, OF, (Option<OF>, Option<OF>)) {
        let (row, coefficient) = if let &[&pair] = index.iter_active_column(column)
            .collect::<Vec<_>>().as_slice() {
            pair
        } else { panic!("Should be exactly one value.") };
        let variable = &self.variables[column];
        let effective_bounds = match coefficient.cmp(&OF::additive_identity()) {
            Ordering::Greater => (variable.lower_bound, variable.upper_bound),
            Ordering::Less => (variable.upper_bound, variable.lower_bound),
            Ordering::Equal => panic!(),
        };

        (row, coefficient, effective_bounds)
    }

    fn get_change(bounds: (Option<OF>, Option<OF>), constraint_type: ConstraintType) -> Option<Option<(BoundDirection, OF)>> {
        match bounds {
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

    fn save_slack_value(&mut self, row: usize, column: usize, coefficient: OF, index: &PresolveIndex<OF>) {
        self.original_variables[column].1 = Removed(FunctionOfOthers {
            constant: self.b.get_value(row) / coefficient,
            coefficients: self.iter_active_row(row, &index.column_counters)
                .filter(|&&(j, _)| j != column)
                .map(|&(j, other_coefficient)| {
                    (self.from_active_to_original[j], other_coefficient / coefficient)
                })
                .collect(),
        });
    }

    fn maybe_update_bound(&mut self, row: usize, coefficient: OF, maybe_new_bound: Option<(BoundDirection, OF)>, index: &mut PresolveIndex<OF>) {
        if let Some((direction, bound)) = maybe_new_bound {
            let old_b = self.b.get_value(row);
            self.b.set_value(row, old_b - coefficient * bound);
            self.constraint_types[row] = match direction {
                BoundDirection::Lower => ConstraintType::Greater,
                BoundDirection::Upper => ConstraintType::Less,
            };
        } else {
            index.remove_constraint_values(row, &self.constraints, &self.variables);
            index.remove_constraint(row)
        }
    }

    /// Attempt to tighten bounds using activity bounds.
    ///
    /// See Achterberg (2007), algorithm 7.1.
    ///
    /// TODO: parts 3 / 4 of 7.1
    ///
    /// Should be called at most once on each constraint, due to the way elements are added to
    /// `row_queue` in `substitute_extract_eliminate`.
    ///
    fn domain_propagate_by_activity_bounds(
        &mut self,
        constraint: usize,
        index: &mut PresolveIndex<OF>,
    ) -> Result<(), LinearProgramType<OF>> {
        // TODO: Recompute these (probably not here, but when things get substituted or bounds adjusted)
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
                // TODO: Reconsider this
                // Could be removed, probably doesn't increase performance (introduces a slack)
                self.constraint_types[constraint] = ConstraintType::Less;
            },
            (ConstraintType::Equal, Some(ConstraintType::Less)) => {
                self.constraint_types[constraint] = ConstraintType::Greater;
            },
            (ConstraintType::Less, Some(ConstraintType::Greater)) | (ConstraintType::Greater, Some(ConstraintType::Less)) => {},
        }

        Ok(self.tighten_variable_bounds(constraint, index))
    }

    fn compute_activity_bounds_if_necessary(
        &self,
        constraint: usize,
        presolve_index: &mut PresolveIndex<OF>,
    ) -> (Option<OF>, Option<OF>) {
        if presolve_index.activity_bounds[constraint].0.is_none() {
            presolve_index.activity_bounds[constraint].0 = self.sum_products(
                constraint,
                self.activity_relevant_bounds(constraint, ConstraintType::Greater, presolve_index),
                presolve_index,
            );
        }
        if presolve_index.activity_bounds[constraint].1.is_none() {
            presolve_index.activity_bounds[constraint].1 = self.sum_products(
                constraint,
                self.activity_relevant_bounds(constraint, ConstraintType::Less, presolve_index),
                presolve_index,
            );
        }

        presolve_index.activity_bounds[constraint]
    }

    fn activity_relevant_bounds(
        &self,
        constraint: usize,
        constraint_type: ConstraintType,
        index: &PresolveIndex<OF>,
    ) -> Vec<Option<OF>> {
        self.iter_active_row(constraint, &index.column_counters)
            .map(|&(column, coefficient)| {
                if coefficient > OF::additive_identity() {
                    match constraint_type {
                        ConstraintType::Greater => self.variables[column].lower_bound,
                        ConstraintType::Less => self.variables[column].upper_bound,
                        ConstraintType::Equal => panic!(),
                    }

                } else if coefficient < OF::additive_identity() {
                    match constraint_type {
                        ConstraintType::Greater => self.variables[column].upper_bound,
                        ConstraintType::Less => self.variables[column].lower_bound,
                        ConstraintType::Equal => panic!(),
                    }
                } else {
                    panic!(
                        "No coefficient should be zero at this point. (row, column, coefficient) = ({}, {}, {})",
                        constraint, column, coefficient,
                    )
                }
            })
            .collect()
    }

    fn sum_products(
        &self,
        constraint: usize,
        relevant_bounds: Vec<Option<OF>>,
        index: &PresolveIndex<OF>,
    ) -> Option<OF> {
        self.iter_active_row(constraint, &index.column_counters).zip(relevant_bounds)
            .map(|(&(_, coefficient), bound): (_, Option<OF>)| bound.map(|b| b * coefficient))
            .sum()
    }

    // 4.
    fn is_infeasible_due_to_activity_bounds(
        &self,
        constraint: usize,
        activity_lower_bound: Option<OF>,
        activity_upper_bound: Option<OF>,
    ) -> bool {
        let lower_violated = activity_upper_bound.map_or(false, |bound| bound < self.b.get_value(constraint));
        let upper_violated = activity_lower_bound.map_or(false, |bound| bound > self.b.get_value(constraint));

        match self.constraint_types[constraint] {
            ConstraintType::Greater => lower_violated,
            ConstraintType::Less => upper_violated,
            ConstraintType::Equal => lower_violated || upper_violated,
        }
    }

    fn activity_implied_constraint(
        &self,
        constraint: usize,
        activity_lower_bound: Option<OF>,
        activity_upper_bound: Option<OF>,
    ) -> Option<ConstraintType> {
        let lower_redundant = activity_lower_bound.map_or(false, |bound| bound >= self.b.get_value(constraint));
        let upper_redundant = activity_upper_bound.map_or(false, |bound| bound <= self.b.get_value(constraint));

        match (lower_redundant, upper_redundant) {
            (false, false) => None,
            (false, true) => Some(ConstraintType::Less),
            (true, false) => Some(ConstraintType::Greater),
            (true, true) => Some(ConstraintType::Equal),
        }
    }

    /// TODO: Can this method show infeasibility?
    /// TODO: When should variables be rechecked?
    fn tighten_variable_bounds(
        &mut self,
        constraint: usize,
        index: &mut PresolveIndex<OF>,
    ) {
        let (activity_lower_bound, activity_upper_bound) = index.activity_bounds[constraint];
        let mut changes = Vec::new();

        for &(j, coefficient) in self.iter_active_row(constraint, &index.column_counters) {
            let derived_bound = |activity: Option<OF>| {
                activity.map(|bound| (self.b.get_value(constraint) - bound) / coefficient)
            };
            let first = if self.constraint_types[constraint] != ConstraintType::Less {
                derived_bound(activity_upper_bound.map(|bound| {
                        bound - coefficient * match coefficient.cmp(&OF::additive_identity()) {
                            Ordering::Greater => self.variables[j].upper_bound.unwrap(),
                            Ordering::Less => self.variables[j].lower_bound.unwrap(),
                            Ordering::Equal => panic!(),
                        }
                }))
            } else { None };
            let second = if self.constraint_types[constraint] != ConstraintType::Greater {
                derived_bound(activity_lower_bound.map(|bound| {
                    bound - coefficient * match coefficient.cmp(&OF::additive_identity()) {
                        Ordering::Greater => self.variables[j].lower_bound.unwrap(),
                        Ordering::Less => self.variables[j].upper_bound.unwrap(),
                        Ordering::Equal => panic!(),
                    }
                }))
            } else { None };

            // TODO: Rounding to integer bounds for integer problems
            if coefficient > OF::additive_identity() {
                if let Some(bound) = first { changes.push((j, BoundDirection::Lower, bound)); }
                if let Some(bound) = second { changes.push((j, BoundDirection::Upper, bound)); }
            } else if coefficient < OF::additive_identity() {
                if let Some(bound) = first { changes.push((j, BoundDirection::Upper, bound)); }
                if let Some(bound) = second { changes.push((j, BoundDirection::Lower, bound)); }
            } else {
                panic!(
                    "No coefficient should be zero at this point. (row, column, coefficient) = ({}, {}, {})",
                    constraint, j, coefficient,
                )
            }
        }

        for (variable, direction, bound) in changes {
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

    fn iter_active_row<'a>(
        &'a self,
        row: usize,
        column_counters: &'a Vec<usize>,
    ) -> impl Iterator<Item = &'a (usize, OF)> {
        self.constraints.iter_row(row)
            .filter(move|&&(j, _)| column_counters[j] != 0)
    }

    /// Sets variables that can be optimized independently of all others to their maximum values.
    ///
    /// # Arguments
    ///
    /// * `to_optimize` - Collection of variable indices that should be optimized.
    fn optimize_disjoint_variables(
        &mut self,
        to_optimize: &Vec<usize>,
    ) -> Result<(), LinearProgramType<OF>> {
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
    ///
    /// # Arguments
    ///
    /// * `rows` - Collection of rows to delete.
    /// * `columns` - Collection of columns to delete.
    fn remove_rows_and_columns(&mut self, index: PresolveIndex<OF>) {
        let mut rows = index.constraints_marked_for_removal.into_iter().collect::<Vec<_>>();
        rows.sort();

        let mut columns = index.columns_marked_removed.into_iter().collect::<Vec<_>>();
        columns.sort();

        self.constraints.remove_rows(&rows);
        remove_indices(&mut self.constraint_types, &rows);
        self.b.remove_indices(&rows);

        self.constraints.remove_columns_although_this_matrix_is_row_ordered(&columns);
        remove_indices(&mut self.variables, &columns);

        if columns.len() > 0 {
            let mut skipped = 1;
            let new_length = self.from_active_to_original.len() - columns.len();
            for j in columns[0]..new_length {
                while skipped < columns.len() && columns[skipped] == j + skipped {
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
                let old_b = self.b.get_value(i);
                self.b.set_value(i, old_b + *coefficient * self.variables[j].shift);
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
            let old = self.b.get_value(row);
            self.b.set_value(row, -old);
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
                    b_equality.push(self.b.get_value(i));
                    equality.push(row.clone());
                },
                ConstraintType::Less => {
                    b_upper.push(self.b.get_value(i));
                    upper.push(row.clone());
                }
                ConstraintType::Greater => {
                    b_lower.push(self.b.get_value(i));
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
    /// * `variable` - Index of the variable to get the solution for _with respect to the
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
            OriginalVariable::Kept(_) => false,
            OriginalVariable::Removed(RemovedVariable::Solved(_)) => true,
            OriginalVariable::Removed(RemovedVariable::FunctionOfOthers { .. }) => true,
        }
    }

    /// Output all names of nonzero variables and their values.
    ///
    /// # Arguments
    ///
    /// * `bfs` - A basic feasible solution (a value for each of the open variables in the problem).
    ///
    /// # Return value
    ///
    /// A vector of (variable name, basic feasible solution value) tuples.
    ///
    /// # Note
    ///
    /// Would probably only be used for printing the final solution and debugging.
    ///
    /// TODO: Use the `solution_values` parameter while outputting the solution to this LP
    pub fn complete_solution(&self, bfs: SparseVector<OF>) -> Vec<(&str, OF)> {
        debug_assert_eq!(bfs.len(), self.variables.len());
        // TODO: write assert that checks that this is actually a solution.

        let mut solutions = vec![None; self.original_variables.len()];
        for j in 0..solutions.len() {
            solutions[j] = Some(self.compute_solution_value(j, &bfs, &mut solutions));
        }
        self.original_variables.iter().zip(solutions.into_iter()).map(|((name, _), value)| {
            (name.as_str(), value.unwrap())
        }).collect()
    }

    fn compute_solution_value(
        &self,
        variable: usize,
        bfs: &SparseVector<OF>,
        solutions: &mut Vec<Option<OF>>,
    ) -> OF {
        if let Some(value) = solutions[variable] {
            return value;
        }

        let new_value = match &self.original_variables[variable].1 {
            OriginalVariable::Kept(index) => bfs.get_value(*index),
            OriginalVariable::Removed(Solved(value)) => *value,
            OriginalVariable::Removed(FunctionOfOthers { constant, coefficients }) => {
                let mut value = *constant;
                for &(j, coefficient) in coefficients {
                    value -= coefficient * self.compute_solution_value(j, bfs, solutions);
                }
                value
            }
        };

        solutions[variable] = Some(new_value);
        new_value
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

struct PresolveIndex<F: Field> {
    constraints_marked_for_removal: HashSet<usize>,
    columns_marked_removed: HashSet<usize>,
    /// This is a subset of `columns_marked_for_removal`.
    columns_optimized_independently: Vec<usize>,

    /// Constraints to check for empty row (bound should be suitable)
    empty_row_queue: HashSet<usize>,
    /// Constraints to check to see whether they are a bound.
    bound_queue: HashSet<usize>,
    /// Constraints to check for activity bound tightening.
    activity_queue: HashSet<usize>,
    /// Variables that are fixed need substitution.
    substitution_queue: HashSet<usize>,
    /// Variables to check for slack variables.
    slack_queue: HashSet<usize>,

    /// Amount of meaningful elements still in the column or row.
    /// The elements should be considered when the counter drops below 2. Variables should also be
    /// considered when a new bound on them is found.
    column_counters: Vec<usize>,
    row_counters: Vec<usize>,
    /// We maintain the computed activity bounds.
    /// TODO: Include counter for recomputation (numerical errors accumulate)
    activity_bounds: Vec<(Option<F>, Option<F>)>,
    /// TODO: Add index counting whether an activity bound can be computed

    /// Column major representation of the constraint matrix.
    columns: SparseMatrix<F, ColumnMajorOrdering>,
}

impl<OF: OrderedField> PresolveIndex<OF> {
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
            constraints_marked_for_removal: row_counters.iter().enumerate()
                .filter(|&(_, &count)| count == 0)
                .map(|(i, _)| i).collect(),
            columns_marked_removed: column_counters.iter().enumerate()
                .filter(|&(_, &count)| count == 0)
                .map(|(j, _)| j).collect(),
            columns_optimized_independently: column_counters.iter().enumerate()
                .filter(|&(_, &count)| count == 0)
                .filter(|&(j, _)| general_form.variables[j].cost != OF::additive_identity())
                .map(|(j, _)| j).collect(),

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

            column_counters,
            row_counters,
            activity_bounds: vec![(None, None); general_form.nr_constraints()],

            columns,
        }
    }

    ///
    /// # Arguments
    ///
    /// * `variable` - Variable who's bound was changed.
    /// * `direction` - Whether an upper or lower bound was changed.
    /// * `by_how_much` - Size of the change.
    fn after_bound_change(
        &mut self,
        variable: usize,
        variables: &Vec<Variable<OF>>,
    ) {
        debug_assert_ne!(self.column_counters[variable], 0);

        if variables[variable].is_fixed().is_some() {
            self.substitution_queue.insert(variable);
        } else {
            let rows_to_recheck = self.iter_active_column(variable)
                .map(|&(i, _)| i)
                .collect::<Vec<_>>();
            self.activity_queue.extend(rows_to_recheck.iter());
        }
    }

    fn update_activity_bound(&mut self, variable: usize, direction: BoundDirection, by_how_much: OF) {
        debug_assert!(match direction {
            BoundDirection::Lower => by_how_much > OF::additive_identity(),
            BoundDirection::Upper => by_how_much < OF::additive_identity(),
        });

        for &(row, coefficient) in self.columns.iter_column(variable) {
            if self.constraints_marked_for_removal.contains(&row) {
                continue;
            }

            match (direction, coefficient.cmp(&OF::additive_identity())) {
                (BoundDirection::Lower, Ordering::Greater)
                    | (BoundDirection::Upper, Ordering::Less) => {
                    if let Some(ref mut bound) = self.activity_bounds[row].0 {
                        // TODO: Numerics
                        *bound += by_how_much * coefficient;
                    }
                },
                (BoundDirection::Lower, Ordering::Less)
                    | (BoundDirection::Upper, Ordering::Greater) => {
                    if let Some(ref mut bound) = self.activity_bounds[row].1 {
                        // TODO: Numerics
                        *bound += by_how_much * coefficient;
                    }
                },
                (_, Ordering::Equal) => panic!("Zero element")
            }
        }
    }

    fn queues_are_empty(&self) -> bool {
        // Note the reverse order w.r.t. the order in which these queues are tested in the main loop
        self.activity_queue.is_empty()
            && self.slack_queue.is_empty()
            && self.bound_queue.is_empty()
            && self.substitution_queue.is_empty()
            && self.empty_row_queue.is_empty()
    }

    fn iter_active_column(&self, column: usize) -> impl Iterator<Item = &(usize, OF)> {
        self.columns.iter_column(column)
            .filter(move |&&(i, _)| self.is_constraint_still_active(i))
    }

    fn iter_active_row<'a>(
        &'a self,
        constraint: usize,
        constraints: &'a SparseMatrix<OF, RowMajorOrdering>,
    ) -> impl Iterator<Item = &'a (usize, OF)> {
        constraints.iter_row(constraint)
            .filter(move |&&(j, _)| self.is_variable_still_active(j))
    }

    fn is_constraint_still_active(&self, constraint: usize) -> bool {
        debug_assert_eq!(
            self.constraints_marked_for_removal.contains(&constraint),
            self.row_counters[constraint] == 0,
        );

        self.row_counters[constraint] != 0
    }

    fn is_variable_still_active(&self, variable: usize) -> bool {
        debug_assert_eq!(
            self.columns_marked_removed.contains(&variable),
            self.column_counters[variable] == 0,
        );

        self.column_counters[variable] != 0
    }

    fn remove_constraint_values(
        &mut self,
        constraint: usize,
        constraints: &SparseMatrix<OF, RowMajorOrdering>,
        variables: &Vec<Variable<OF>>,
    ) {
        debug_assert!(self.row_counters[constraint] >= 2);

        let variables_to_scan = self.iter_active_row(constraint, constraints)
            .map(|&(j, _)| j).collect::<Vec<_>>();
        self.row_counters[constraint] -= variables.len();
        for variable in variables_to_scan {
            self.column_counters[variable] -= 1;
            self.readd_column_to_queues_based_on_counter(variable, variables);
        }
    }

    fn readd_column_to_queues_based_on_counter(&mut self, column: usize, variables: &Vec<Variable<OF>>) {
        match self.column_counters[column] {
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

    fn readd_row_to_queues(&mut self, row: usize) {
        match self.row_counters[row] {
            0 => self.empty_row_queue.insert(row),
            1 => self.bound_queue.insert(row),
            _ => self.activity_queue.insert(row),
        };
    }

    fn remove_constraint(
        &mut self,
        constraint: usize,
    ) {
        debug_assert_eq!(self.row_counters[constraint], 0);

        self.constraints_marked_for_removal.insert(constraint);

        self.empty_row_queue.remove(&constraint);
        self.bound_queue.remove(&constraint);
        self.activity_queue.remove(&constraint);
    }

    fn remove_variable(&mut self, variable: usize) {
        debug_assert_eq!(self.column_counters[variable], 0);

        self.columns_marked_removed.insert(variable);

        self.substitution_queue.remove(&variable);
        self.slack_queue.remove(&variable);
    }
}

/// A variable is named, of continuous or integer type and may be shifted.
///
/// TODO: Check the below calculation and logic.
/// The upper bound is relative to the offset; that is, the lower bound is `offset`, the upper
/// bound is `upper_bound - offset`. For example, the range stays the same, regardless of the shift.
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
    fn is_fixed(&self) -> Option<OF> {
        match (self.lower_bound, self.upper_bound) {
            (Some(lower), Some(upper)) if lower == upper => Some(lower),
            _ => None,
        }
    }
    fn is_free(&self) -> bool {
        self.lower_bound.is_none() && self.upper_bound.is_none()
    }
    fn has_feasible_value(&self) -> bool {
        match (self.lower_bound, self.upper_bound) {
            (Some(lower), Some(upper)) => lower <= upper,
            _ => true,
        }
    }
    fn update_lower_bound(&mut self, new_bound: OF) -> Option<Option<OF>> {
        match self.lower_bound {
            Some(existing_bound) => {
                if new_bound > existing_bound {
                    self.lower_bound = Some(new_bound);
                    Some(Some(new_bound - existing_bound))
                } else {
                    None
                }
            },
            None => {
                self.lower_bound = Some(new_bound);
                Some(None)
            }
        }
    }
    fn update_upper_bound(&mut self, new_bound: OF) -> Option<Option<OF>> {
        match self.upper_bound {
            Some(existing_bound) => {
                if new_bound < existing_bound {
                    self.upper_bound = Some(new_bound);
                    Some(Some(new_bound - existing_bound))
                } else {
                    None
                }
            },
            None => {
                self.upper_bound = Some(new_bound);
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
    use crate::data::linear_program::elements::{ConstraintType, Objective, VariableType};
    use crate::data::linear_program::general_form::{GeneralForm, Variable};
    use crate::data::linear_program::general_form::OriginalVariable::Removed;
    use crate::data::linear_program::general_form::RemovedVariable::{Solved, FunctionOfOthers};
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
        initial.presolve().unwrap();
        println!("{:?}", initial.fixed_cost);

        let expected = GeneralForm {
            objective: Objective::Minimize,
            constraints: RowMajorOrdering::from_test_data(&vec![], 0),
            constraint_types: vec![],
            b: DenseVector::from_test_data(vec![]),
            variables: vec![],
            fixed_cost: R32!(1)
                + R32!(211 * 101, 2)
                + R32!(223 * -97, 10)
                + R32!(227 * -699, 65)
                + R32!(-229 * 131)
                + R32!(233 * -30736, 1885)
            ,
            original_variables: vec![
                ("XONE".to_string(), Removed(Solved(R32!(101, 2)))),
                ("XTWO".to_string(), Removed(Solved((R32!(103) - R32!(101) / R32!(2) * R32!(3)) / R32!(5)))),
                ("XTHREE".to_string(), Removed(Solved((R32!(-3601, 5) + R32!(29 * 30736, 1885)) / 23))),
                ("XFOUR".to_string(), Removed(Solved(R32!(131)))),
                ("XFIVE".to_string(), Removed(Solved(R32!(-30736, 65 * 29)))),
                ("XSIX".to_string(), Removed(FunctionOfOthers {
                    constant: R32!(-2826, 5 * 31),
                    coefficients: vec![(2, R32!(23, 31)), (4, R32!(29, 31))],
                })),
            ],
            from_active_to_original: vec![],
        };

        assert_eq!(initial, expected);
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
