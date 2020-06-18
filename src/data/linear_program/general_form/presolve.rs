//! # Presolving linear programs
//!
//! A `GeneralForm` can be presolved by building an index, repeatedly applying reduction rules and
//! applying the changes proposed. This module contains data structures and logic for presolving.
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::iter::Iterator;
use std::ops::BitXor;

use crate::data::linear_algebra::matrix::{RowMajor, Sparse};
use crate::data::linear_algebra::SparseTuple;
use crate::data::linear_algebra::traits::SparseElementZero;
use crate::data::linear_program::elements::{BoundDirection, ConstraintType, LinearProgramType, Objective};
use crate::data::linear_program::general_form::{GeneralForm, RemovedVariable};
use crate::data::number_types::traits::{Field, OrderedField, OrderedFieldRef};

/// Container data structure to keep track of presolve status.
///
/// Queues are used to determine which rules still have to be applied to which constraint or
/// variable indices. Proposed changes that don't change the solution to the final problem are
/// collected in the Updates field. A few indices, counters, are kept to speed up the process: they
/// indicate when which rules can be applied.
pub(super) struct Index<'a, F: Field, FZ: SparseElementZero<F>> {
    /// Which rule should still be applied to which constraint or variable.
    pub queues: Queues,
    /// Set of changes that doesn't change the solution to the problem.
    pub updates: Updates<'a, F, FZ>,
    /// Indices to speed up solving.
    pub counters: Counters<'a, F, FZ>,

    /// We maintain the computed activity bounds.
    /// TODO(PRECISION): Include counter for recomputation (numerical errors accumulate)
    activity_bounds: Vec<(Option<F>, Option<F>)>,

    /// Column major representation of the constraint matrix, borrowed from problem being presolved.
    general_form: &'a GeneralForm<F, FZ>,
}

/// Which rules still need to be applied to which constraint or variable.
///
/// Note that a value could be removed from a queue before the rule is actually applied. A variable
/// might be a slack, for example, but is then removed because a bound has been adjusted.
pub(super) struct Queues {
    /// TODO(OPTIMIZATION): Could a different datastructure back these queues more efficiently?
    ///  How about stacks? Also, some queues might have limited lengths.
    /// Constraints to check for empty row (bound should be suitable).
    /// All elements should have a row count 0.
    empty_row: HashSet<usize>,
    /// Constraints to check to see whether they are a bound (without a slack).
    /// All elements should have row count 1.
    bound: HashSet<usize>,
    /// Constraints to check for activity bound tightening.
    ///
    /// The relevant activity counter (`.0` for `Lower`, `.1` for `Upper`) should be either 0 or 1.
    /// TODO: Consider tracking which indices are still missing to avoid a scan when count is 1.
    activity: HashSet<(usize, BoundDirection)>,
    /// Variables that are fixed need substitution.
    /// All elements have a single feasible value.
    substitution: HashSet<usize>,
    /// Variables to check for slack variables.
    /// All elements have column count 1.
    slack: HashSet<usize>,
}
impl Queues {
    /// Create a new instance.
    ///
    /// Requires iteration over all counters. This method also demonstrates, which constraints or
    /// variables get added to which queue under which condition.
    ///
    /// TODO(OPTIMIZATION): Are these iterations efficiently done after compiler optimizations?
    ///
    /// # Arguments
    ///
    /// * `general_form`: Problem being presolved.
    /// * `counters`: Counters initialized on the problem in the other argument.
    ///
    /// # Return value
    ///
    /// A new instance.
    fn new<OF: OrderedField, OFZ: SparseElementZero<OF>>(
        general_form: &GeneralForm<OF, OFZ>, counters: &Counters<OF, OFZ>,
    ) -> Self
    where
        for<'r> &'r OF: OrderedFieldRef<OF>,
    {
        Self {
            empty_row: counters.constraint.iter().enumerate()
                .filter(|&(_, &count)| count == 0)
                .map(|(i, _)| i).collect(),
            bound: counters.constraint.iter().enumerate()
                .filter(|&(_, &count)| count == 1)
                .map(|(i, _)| i).collect(),
            activity: counters.activity.iter().enumerate()
                .flat_map(|(i, &(lower_count, upper_count))| {
                    let mut sides = Vec::with_capacity(2);
                    if counters.constraint[i] > 1 {
                        if lower_count <= 1 {
                            sides.push((i, BoundDirection::Lower));
                        }
                        if upper_count <= 1 {
                            sides.push((i, BoundDirection::Upper));
                        }
                    }
                    sides
                }).collect(),

            substitution: general_form.variables.iter().enumerate()
                // Columns with no values get substituted right away, as they removal doesn't lead
                // to any further actions. The only values that should be put in the queue are those
                // that lead to other actions, that is, have a nonzero counter.
                .filter(|&(j, _)| counters.variable[j] > 0)
                .filter_map(|(j, variable)| variable.is_fixed().map(|_| j))
                .collect(),
            slack: counters.variable.iter().enumerate()
                .filter(|&(_, &count)| count == 1)
                .filter(|&(j, _)| general_form.variables[j].cost.is_zero())
                .map(|(j, _)| j).collect(),
        }
    }
}

/// Collection of changes that can be applied to the `GeneralForm` without changing its solution.
///
/// This struct doesn't just collect changes, it also presents the latest "version" of the problem
/// to the presolving logic, and as such it is read from continuously.
///
/// Some value are replacements, others are changes, so be careful. The one attribute that isn't a
/// change that can be made, is the borrow of the original problem to which these change can be
/// applied. This borrow is there only to enable a few methods to be shifted down from the `Index`
/// struct to this struct, such that borrows on `self` within `Index` block less.
///
/// All indices in this struct are relative to the problem without anything removed (couldn't really
/// be different, but for clarity).
pub(super) struct Updates<'a, F: Field, FZ: SparseElementZero<F>> {
    /// New constraint values (replace the current value), by constraint index.
    b: HashMap<usize, F>,
    /// New constraint types.
    constraints: HashMap<usize, ConstraintType>,
    /// Change to the fixed cost.
    fixed_cost: F,
    /// New or tightened bounds for variables, replace old bounds.
    bounds: HashMap<(usize, BoundDirection), F>,
    /// Variables that were eliminated, and their solution.
    ///
    /// This solution might depend on the solution of other variables, see the documentation of the
    /// `RemovedVariable` struct. All variables for which a solution was found should be part of the
    /// `variables_marked_removed` field.
    removed_variables: Vec<(usize, RemovedVariable<F>)>,
    /// Collecting the indices, relative to the active part of the problem, that should be removed.
    ///
    /// This collection is here only to avoid an extra scan over the `column_counters`.
    constraints_marked_removed: Vec<usize>,
    /// These columns still need to be optimized independently.
    ///
    /// They could be removed from the problem because they at some point during presolving no
    /// longer interacted with any constraint. These columns do have a nonzero coefficient in the
    /// cost function.
    ///
    /// This is a subset of `columns_marked_for_removal`.
    columns_optimized_independently: Vec<usize>,

    /// Original problem being solved. Included as an attribute to allow nicer code, see above.
    general_form: &'a GeneralForm<F, FZ>,
}
impl<'a, OF: OrderedField, OFZ: SparseElementZero<OF>> Updates<'a, OF, OFZ>
    where
            for<'r> &'r OF: OrderedFieldRef<OF>,
{
    /// Create a new instance.
    ///
    /// Requires iteration over all counters.
    ///
    /// TODO(OPTIMIZATION): Are these iterations efficiently done after compiler optimizations?
    ///
    /// # Arguments
    ///
    /// * `general_form`: Problem being presolved.
    /// * `counters`: Counters initialized on the problem in the other argument.
    ///
    /// # Return value
    ///
    /// A new instance.
    fn new(
        general_form: &'a GeneralForm<OF, OFZ>,
        counters: &Counters<OF, OFZ>,
    ) -> Result<Self, LinearProgramType<OF>> {
        let mut fixed_cost = OF::zero();
        let removed_variables = counters.variable.iter().enumerate()
            .filter(|&(_, &count)| count == 0)
            .map(|(j, _)| {
                let variable = &general_form.variables[j];
                // There should be a feasible value, otherwise, reading of the problem should
                // have been ended due to a trivial infeasibility.
                debug_assert!(variable.has_feasible_value());

                let value = if variable.cost.is_zero() {
                    variable.get_feasible_value().unwrap()
                } else {
                    let value = Self::get_optimized_value(
                        general_form.objective,
                        &variable.cost,
                        (variable.lower_bound.as_ref(), variable.upper_bound.as_ref()),
                    )?;
                    fixed_cost += &variable.cost * &value;
                    value
                };

                Ok((j, RemovedVariable::Solved(value)))
            }).collect::<Result<_, _>>()?;

        Ok(Self {
            b: HashMap::new(),
            constraints: HashMap::new(),
            fixed_cost,
            bounds: HashMap::new(),
            // Variables that don't appear in any constraint are removed right away.
            removed_variables,
            constraints_marked_removed: Vec::default(),
            columns_optimized_independently: Vec::default(),

            general_form,
        })
    }

    /// Get the latest constraint value.
    fn b(&self, constraint: usize) -> &OF {
        // There is no reason to want to know the latest value for a constraint already removed.
        debug_assert!(!self.constraints_marked_removed.contains(&constraint));

        self.b.get(&constraint).unwrap_or(&self.general_form.b[constraint])
    }
    /// Change the constraint value by a certain amount.
    fn change_b(&mut self, constraint: usize, change: OF) {
        // There is no reason to change the value for a constraint already removed.
        debug_assert!(!self.constraints_marked_removed.contains(&constraint));

        if let Some(existing_value) = self.b.get_mut(&constraint) {
            *existing_value += change;
        } else {
            self.b.insert(constraint, &self.general_form.b[constraint] + change);
        }
    }

    /// Get the latest constraint type on a constraint.
    ///
    /// When the constraint has been removed (that is, the index is in the
    /// `constraints_marked_removed` field), this function will still return some value (that is
    /// then meaningless).
    fn constraint_types(&self, constraint: usize) -> ConstraintType {
        // There is no reason to want to know the latest value for a constraint already removed.
        debug_assert!(!self.constraints_marked_removed.contains(&constraint));

        self.constraints.get(&constraint)
            .copied()
            .unwrap_or(self.general_form.constraint_types[constraint])
    }

    /// Whether the latest version of a variable has exactly one feasible value.
    fn is_variable_fixed(&self, variable: usize) -> Option<&OF> {
        // There is no reason to want to know the latest value for a variable already removed.
        debug_assert!(!self.removed_variables.iter().any(|&(j, _)| j == variable));

        match self.variable_bounds(variable) {
            (Some(lower), Some(upper)) if lower == upper => Some(lower),
            _ => None,
        }
    }
    /// Whether a the latest version of variable has any feasible value.
    fn variable_has_feasible_value(&self, variable: usize) -> bool {
        // There is no reason to want to know the latest value for a variable already removed.
        debug_assert!(!self.removed_variables.iter().any(|&(j, _)| j == variable));

        if let (Some(lower), Some(upper)) = self.variable_bounds(variable) {
            lower <= upper
        } else {
            true
        }
    }
    /// Any feasible value for a variable, if there is one.
    fn get_feasible_value(&self, variable: usize) -> Option<OF> {
        if self.variable_has_feasible_value(variable) {
            let (lower, upper) = self.variable_bounds(variable);
            // It is often positive slacks that get removed here, so we pick the upper bound if
            // there is one, because it makes it more likely that a meaningful variable is zero,
            // which in turn might lead to sparser solutions.
            upper.cloned().or(lower.cloned()).or(Some(OF::zero()))
        } else { None }
    }
    /// Latest version of both bounds of a variable at once.
    ///
    /// Convenience method.
    fn variable_bounds(&self, variable: usize) -> (Option<&OF>, Option<&OF>) {
        // There is no reason to want to know the latest value for a variable already removed.
        debug_assert!(!self.removed_variables.iter().any(|&(j, _)| j == variable));

        let lower = self.variable_bound(variable, BoundDirection::Lower);
        let upper = self.variable_bound(variable, BoundDirection::Upper);
        (lower, upper)
    }
    /// Latest version of a variable bound.
    fn variable_bound(&self, variable: usize, direction: BoundDirection) -> Option<&OF> {
        // There is no reason to want to know the latest value for a variable already removed.
        debug_assert!(!self.removed_variables.iter().any(|&(j, _)| j == variable));

        self.bounds.get(&(variable, direction))
            .or_else(|| {
                let variable = &self.general_form.variables[variable];
                match direction {
                    BoundDirection::Lower => variable.lower_bound.as_ref(),
                    BoundDirection::Upper => variable.upper_bound.as_ref(),
                }
            })
    }
    /// Update a variable bound.
    ///
    /// # Arguments
    ///
    /// * `variable`: Index of the variable to be changed.
    /// * `direction`: Whether the lower or upper bound should be changed.
    /// * `new`: New value for the bound.
    ///
    /// # Return value
    ///
    /// Nexted `Option`. First indicates whether the bound was changed at all (`None` for no change,
    /// `Some` for a change). The inner `Option` indicates whether the bound was changed by a
    /// specific amount (`Some` for the amount, `None` if there was no bound before).
    #[allow(clippy::option_option)]
    fn update_bound(&mut self, variable: usize, direction: BoundDirection, new: &OF) -> Option<Option<OF>> {
        // There is no reason to update the bound for a variable already removed.
        debug_assert!(!self.removed_variables.iter().any(|&(j, _)| j == variable));


        if let Some(existing) = self.variable_bound(variable, direction) {
            if match direction {
                BoundDirection::Lower => new > existing,
                BoundDirection::Upper => new < existing,
            } {
                let difference = new - &*existing;
                self.bounds.insert((variable, direction), new.clone());
                Some(Some(difference))
            } else {
                None
            }
        } else {
            self.bounds.insert((variable, direction), new.clone());
            Some(None)
        }
    }

    /// Apply domain propagation through activation bounds for a constraint where all of the
    /// relevant variable bounds are known.
    ///
    /// # Arguments
    ///
    /// * `constraint`: Index of the constraint under consideration.
    /// * `variable`: Index of variable who's relevant variable bound is not known, with all other
    /// variable bounds in this constraint being known.
    /// * `coefficient`: Nonzero coefficient of the variable in the constraint.
    /// * `direction`: Whether the activity lower- or upperbound should be used.
    /// * `activity_bound`: Value of the activity lower- or upperbound.
    ///
    /// # Return value
    ///
    /// Tuple of:
    ///
    /// * Direction of the bound that was changed (lower or upper).
    /// * Change that was done (any at all, and if so, by how much or was there no bound before? See
    /// the `update_bound` method for more).
    fn tighten_variable_bound(
        &mut self,
        constraint: usize,
        variable: usize,
        coefficient: &OF,
        direction: BoundDirection,
        activity_bound: &OF,
    ) -> (BoundDirection, Option<Option<OF>>) {
        let bound_direction = !direction.bitxor(Index::<_, OFZ>::direction_from_sign(coefficient));
        let bound_value = self.variable_bound(variable, bound_direction).unwrap();
        let bound = activity_bound - coefficient * bound_value;

        let variable_bound = (self.b(constraint) - bound) / coefficient;
        let change = self.update_bound(variable, !bound_direction, &variable_bound);
        (!bound_direction, change)
    }

    /// Sets variables that can be optimized independently of all others to their optimal values.
    fn optimize_column_independently(
        &mut self,
        column: usize,
    ) -> Result<RemovedVariable<OF>, LinearProgramType<OF>> {
        let variable = &self.general_form.variables[column];
        debug_assert_ne!(variable.cost, OF::zero());

        let new_value = Self::get_optimized_value(
            self.general_form.objective,
            &variable.cost,
            (variable.lower_bound.as_ref(), variable.upper_bound.as_ref()),
        )?;
        self.fixed_cost += &self.general_form.variables[column].cost * &new_value;

        Ok(RemovedVariable::Solved(new_value))
    }

    /// Gets the value that maximizes a column's contribution in the objective direction.
    fn get_optimized_value(
        objective: Objective,
        cost: &OF,
        (lower_bound, upper_bound): (Option<&OF>, Option<&OF>),
    ) -> Result<OF, LinearProgramType<OF>> {
        let cost_sign = cost.cmp(&OF::zero());
        match (objective, cost_sign) {
            (Objective::Minimize, Ordering::Less) | (Objective::Maximize, Ordering::Greater) => {
                match upper_bound {
                    Some(v) => Ok(v),
                    None => Err(LinearProgramType::Unbounded),
                }
            },
            (Objective::Minimize, Ordering::Greater) | (Objective::Maximize, Ordering::Less) => {
                match lower_bound {
                    Some(v) => Ok(v),
                    None => Err(LinearProgramType::Unbounded),
                }
            },
            (_, Ordering::Equal) => panic!("Should not be called if there is no cost"),
        }.cloned()
    }

    /// Get all changes that the presolving resulted in.
    ///
    /// This method only exists because this struct also holds a borrow to the original problem,
    /// the caller of this method.
    ///
    /// # Return value
    ///
    /// Almost all values of this struct; just not the borrow to the original problem.
    #[allow(clippy::type_complexity)]
    pub fn get_updates(mut self) -> (
        HashMap<usize, OF>,
        HashMap<usize, ConstraintType>,
        OF,
        HashMap<(usize, BoundDirection), OF>,
        Vec<(usize, RemovedVariable<OF>)>,
        Vec<usize>,
    ) {
        for constraint in &self.constraints_marked_removed {
            self.b.remove(constraint);
            self.constraints.remove(constraint);
        }
        for variable in self.removed_variables.iter().map(|&(j, _)| j) {
            self.bounds.remove(&(variable, BoundDirection::Lower));
            self.bounds.remove(&(variable, BoundDirection::Upper));
        }

        (
            self.b,
            self.constraints,
            self.fixed_cost,
            self.bounds,
            self.removed_variables,
            self.constraints_marked_removed,
        )
    }
}

pub struct Counters<'a, F: Field, FZ: SparseElementZero<F>> {
    /// Amount of meaningful elements still in the column or row.
    /// The elements should at least be considered when the counter drops below 2. This also
    /// depends on whether the variable appears in the cost function.
    variable: Vec<usize>,
    /// The elements should at least be reconsidered when the counter drops below 2.
    constraint: Vec<usize>,
    /// bound that are missing before the activity bound can be computed.
    ///
    /// If only one bound is missing, a variable bound can be computed. If none are missing, the
    /// entire variable bound can be computed.
    activity: Vec<(usize, usize)>,

    /// Row major representation of the constraint matrix (a copy of `generalform.constraints`).
    rows: Sparse<&'a F, FZ, F, RowMajor>,
    general_form: &'a GeneralForm<F, FZ>,
}
impl<'a, OF, OFZ> Counters<'a, OF, OFZ>
    where
        OF: OrderedField,
        OFZ: SparseElementZero<OF>,
        for<'r> &'r OF: OrderedFieldRef<OF>,
{
    /// Create a new instance.
    ///
    /// Create a row major representation of the problem, which is expensive. Doing this once allows
    /// quick iteration over rows, which is helpful for interacting with the constraints.
    ///
    /// # Arguments
    ///
    /// * `general_form`: Problem being presolved.
    ///
    /// # Return value
    ///
    /// A new instance.
    fn new(general_form: &'a GeneralForm<OF, OFZ>) -> Self {
        let rows: Sparse<_, OFZ, _, _> = Sparse::from_column_major_ordered_matrix_although_this_is_expensive(
            &general_form.constraints,
        );

        Self {
            constraint: (0..general_form.nr_active_constraints())
                .map(|i| rows.data[i].len())
                .collect(),
            variable: (0..general_form.nr_active_variables())
                .map(|j| general_form.constraints.data[j].len())
                .collect(),
            activity: rows.iter_rows().map(|row| {
                row.iter().map(|&(j, coefficient)| {
                    let (lower, upper) =  (&general_form.variables[j].lower_bound, &general_form.variables[j].upper_bound);
                    match Index::<OF, OFZ>::direction_from_sign(coefficient) {
                        BoundDirection::Upper => (lower, upper),
                        BoundDirection::Lower => (upper, lower),
                    }
                })
                    .fold((0, 0), |(lower_total, upper_total), (lower, upper)| {
                        let is_missing = |option| match option {
                            &Some(_) => 0,
                            &None => 1,
                        };
                        (lower_total + is_missing(lower), upper_total + is_missing(upper))
                    })
            }).collect(),

            rows,
            general_form,
        }
    }

    /// Iterate over the constraints of a column who have not (yet) been eliminated.
    ///
    /// # Arguments
    ///
    /// * `column`: Column to iter over.
    /// * `constraints`: The row major representation of the constraints of the general form for
    /// fast iteration.
    ///
    /// # Return value
    ///
    /// An iterator of (row index, reference to value) tuples.
    fn iter_active_column(
        &self,
        variable: usize,
    ) -> impl Iterator<Item = SparseTuple<&OF>> {
        self.general_form.constraints.iter_column(variable)
            .map(|&(i, ref v)| (i, v))
            .filter(move |&(i, _)| self.is_constraint_still_active(i))
    }

    /// Iterate over the columns of a constraint that have not yet been eliminated.
    ///
    /// During presolving, for each column, a count is being kept of the number of active (belonging
    /// to a constraint that has not yet been removed) column. When that count is equal to zero, the
    /// coefficient in the original matrix of active variable coefficients is neglected.
    ///
    /// # Arguments
    ///
    /// * `constraint`: Constraint to iter over.
    ///
    /// # Return value
    ///
    /// A collection of (column index, coefficient value) tuples.
    fn iter_active_row(&self, constraint: usize) -> impl Iterator<Item = SparseTuple<&OF>> {
        self.rows.iter_row(constraint)
            .map(|&(i, v)| (i, v))
            .filter(move |&(j, _)| self.is_variable_still_active(j))
    }

    /// The constraint counter indicates whether the constraint still has any variables left.
    ///
    /// Note that this can be zero, even though the constraint has not yet been eliminated, because
    /// it still needs to be checked by `presolve_empty_constraint`. It can, in that case, however
    /// be ignored during the application of presolving rules.
    fn is_constraint_still_active(&self, constraint: usize) -> bool {
        self.constraint[constraint] != 0
    }

    /// The variable counter indicates whether the variable still appears in any constraint.
    fn is_variable_still_active(&self, variable: usize) -> bool {
        self.variable[variable] != 0
    }
}

impl<'a, OF, OFZ> Index<'a, OF, OFZ>
    where
        OF: OrderedField,
        for<'r> &'r OF: OrderedFieldRef<OF>,
        OFZ: SparseElementZero<OF>,
{
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
    pub(super) fn new(general_form: &'a GeneralForm<OF, OFZ>) -> Result<Self, LinearProgramType<OF>> {
        let counters = Counters::new(general_form);
        let updates = Updates::new(general_form, &counters)?;

        Ok(Self {
            queues: Queues::new(general_form, &counters),
            updates,
            counters,

            activity_bounds: vec![(None, None); general_form.nr_constraints()],

            general_form: &general_form,
        })
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
    pub(super) fn presolve_step(&mut self) -> Result<(), LinearProgramType<OF>> {
        // Actions that are guaranteed to make the problem smaller
        // Remove a row
        if let Some(&row) = self.queues.empty_row.iter().next() {
            return self.presolve_empty_constraint(row);
        }
        // Remove a column
        if let Some(&variable) = self.queues.substitution.iter().next() {
            self.presolve_fixed_variable(variable);
            return Ok(());
        }
        // Remove a bound
        if let Some(&constraint) = self.queues.bound.iter().next() {
            return self.presolve_simple_bound_constraint(constraint);
        }

        // Actions not guaranteed to make the problem smaller
        // Test whether a variable can be seen as a slack
        if let Some(&variable) = self.queues.slack.iter().next() {
            return self.presolve_constraint_if_slack_with_suitable_bounds(variable);
        }
        // Domain propagation
        if let Some(&(constraint, direction)) = self.queues.activity.iter().next() {
            return self.presolve_constraint_by_domain_propagation(constraint, direction);
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
        &mut self,
        constraint: usize,
    ) -> Result<(), LinearProgramType<OF>> {
        debug_assert_eq!(self.counters.constraint[constraint], 0);

        match (self.updates.b(constraint).cmp(&OF::zero()), self.updates.constraint_types(constraint)) {
            (Ordering::Equal, _)
            | (Ordering::Greater, ConstraintType::Less)
            | (Ordering::Less, ConstraintType::Greater) => {
                self.remove_constraint(constraint);
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
    ) {
        debug_assert!(!self.updates.removed_variables.iter().any(|&(j, _)| j == variable));
        debug_assert!(self.updates.is_variable_fixed(variable).is_some());

        let value = self.updates.is_variable_fixed(variable).unwrap().clone();
        let mut rows_to_recheck = Vec::new();
        for (row, coefficient_value) in self.counters.iter_active_column(variable) {
            self.updates.change_b(row, -coefficient_value * &value);
            match self.counters.constraint[row] {
                // Values matched on are one higher than the actually should be, because the counter
                // can only be updated after this loop due to borrowck.
                1 => { self.queues.empty_row.insert(row); },
                2 => { self.queues.bound.insert(row); },
                _ => {},
            };

            rows_to_recheck.push(row);
        }
        self.counters.variable[variable] -= rows_to_recheck.len(); // Value only used for debug asserts
        debug_assert_eq!(self.counters.variable[variable], 0);
        for row in rows_to_recheck {
            self.counters.constraint[row] -= 1;
        }
        self.updates.fixed_cost += &self.general_form.variables[variable].cost * &value;

        self.remove_variable(variable, RemovedVariable::Solved(value));
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
    ) -> Result<(), LinearProgramType<OF>> {
        debug_assert_eq!(self.counters.constraint[constraint], 1);
        debug_assert_eq!(self.counters.iter_active_row(constraint).count(), 1);

        let (column, value) = self.counters.iter_active_row(constraint).next().unwrap();
        debug_assert_ne!(self.counters.variable[column], 0);

        let mut changes = Vec::with_capacity(2);
        match self.updates.constraint_types(constraint) {
            ConstraintType::Greater => changes.push(BoundDirection::Lower),
            ConstraintType::Less => changes.push(BoundDirection::Upper),
            ConstraintType::Equal => {
                changes.push(BoundDirection::Lower);
                changes.push(BoundDirection::Upper);
            },
        }

        let bound_value = self.updates.b(constraint) / value;
        for direction in changes {
            if let Some(change) = self.updates.update_bound(column, direction, &bound_value) {
                self.after_bound_change(column, direction, change);
            }
        }

        if self.updates.variable_has_feasible_value(column) {
            self.counters.constraint[constraint] -= 1;
            self.counters.variable[column] -= 1;
            self.remove_constraint(constraint);
            self.readd_column_to_queues_based_on_counter(column)
        } else {
            Err(LinearProgramType::Infeasible)
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
    ) -> Result<(), LinearProgramType<OF>> {
        debug_assert_eq!(self.counters.variable[variable_index], 1);
        debug_assert_eq!(self.counters.iter_active_column(variable_index).count(), 1);
        debug_assert_eq!(self.general_form.variables[variable_index].cost, OF::zero());
        debug_assert!(self.updates.is_variable_fixed(variable_index).is_none());

        // Only coefficient in the problem
        let (constraint, coefficient) = self.counters.iter_active_column(variable_index)
            .next().unwrap();
        let coefficient = coefficient.clone(); // TODO: Try to avoid this clone

        let bounds = self.updates.variable_bounds(variable_index);
        let effective_bounds = match Self::direction_from_sign(&coefficient) {
            BoundDirection::Upper => (bounds.0, bounds.1),
            BoundDirection::Lower => (bounds.1, bounds.0),
        };
        let change = Self::get_change(effective_bounds, self.updates.constraint_types(constraint));
        let (lower_is_none, upper_is_none) = (effective_bounds.0.is_none(), effective_bounds.1.is_none());

        if let Some(new_situation) = change {
            // TODO: Try to avoid this clone
            let new_situation = new_situation.map(|(direction, value)| (direction, value.clone()));

            // Variable is a function of others, save and eliminate
            let constant = self.updates.b(constraint) / &coefficient;
            let coefficients = self.counters.iter_active_row(constraint)
                .filter(|&(j, _)| j != variable_index)
                .map(|(j, other_coefficient)| {
                    (self.general_form.from_active_to_original[j], other_coefficient / &coefficient)
                })
                .collect();
            let removed = RemovedVariable::FunctionOfOthers { constant, coefficients };

            // Administration to keep the index up to date
            self.counters.constraint[constraint] -= 1;
            self.counters.variable[variable_index] -= 1;
            if lower_is_none {
                self.counters.activity[constraint].0 -= 1;
                if self.counters.activity[constraint].0 <= 1 {
                    self.queues.activity.insert((constraint, BoundDirection::Lower));
                }
            }
            if upper_is_none {
                self.counters.activity[constraint].1 -= 1;
                if self.counters.activity[constraint].1 <= 1 {
                    self.queues.activity.insert((constraint, BoundDirection::Upper));
                }
            }

            self.remove_variable(variable_index, removed);
            self.update_or_remove_bound(constraint, &coefficient, new_situation)
        } else {
            self.queues.slack.remove(&variable_index);
            Ok(())
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
    fn get_change<'b>(
        slack_bounds: (Option<&'b OF>, Option<&'b OF>),
        constraint_type: ConstraintType,
    ) -> Option<Option<(BoundDirection, &'b OF)>> {
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
    /// * `variable`: Index of the variable of the slack.
    /// * `coefficient`: Value by which the slack gets multiplied in the constraint.
    /// * `maybe_new_bound`: `Option` describing the new state of the constraint. Either is it
    /// `None`, in which case the constraint should be removed entirely, or `Some` in which case the
    /// `BoundDirection` describes the direction of the new constraint. This value is not of the
    /// `ConstraintType` type because it can never be an equality constraint. The `OF` in the tuple
    /// is the bound value of the slack being removed.
    fn update_or_remove_bound(
        &mut self,
        constraint: usize,
        coefficient: &OF,
        maybe_new_bound: Option<(BoundDirection, OF)>,
    ) -> Result<(), LinearProgramType<OF>> {
        if let Some((direction, bound)) = maybe_new_bound {
            self.updates.change_b(constraint, -coefficient * bound);

            self.updates.constraints.insert(constraint, match direction {
                BoundDirection::Lower => ConstraintType::Greater,
                BoundDirection::Upper => ConstraintType::Less,
            });
        } else {
            self.remove_constraint_values(constraint)?;
            self.remove_constraint(constraint)
        }

        Ok(())
    }

    /// Attempt to tighten bounds using activity bounds.
    ///
    /// As described in Achterberg (2007), algorithm 7.1.
    ///
    /// # Arguments
    ///
    /// * `constraint`: Index of the constraint under consideration.
    /// * `direction`: Whether the activity lower- or upperbound should be used.
    ///
    /// # Return value
    ///
    /// `Result::Err` if the problem was determined to be infeasible.
    fn presolve_constraint_by_domain_propagation(
        &mut self,
        constraint: usize,
        direction: BoundDirection,
    ) -> Result<(), LinearProgramType<OF>> {
        // We remove from the activity queue here, but the below calls might readd this constraint
        // to the `self.queues.activity` again.
        self.queues.activity.remove(&(constraint, direction));
        let counter = match direction {
            BoundDirection::Lower => self.counters.activity[constraint].0,
            BoundDirection::Upper => self.counters.activity[constraint].1,
        };
        match counter {
            0 => self.for_entire_constraint(constraint, direction),
            1 => {
                self.create_variable_bound(constraint, direction);
                Ok(())
            },
            _ => panic!(),
        }
    }

    /// Apply domain propagation through activation bounds for a constraint where all relevant
    /// variable bounds are known.
    ///
    /// Attempts to both prove infeasibility and removal of the entire constraint. Afterwards, it
    /// attempts to add any missing "opposite" bounds.
    ///
    /// # Arguments
    ///
    /// * `constraint`: Index of the constraint under consideration.
    /// * `direction`: Whether the activity lower- or upperbound should be used.
    fn for_entire_constraint(
        &mut self,
        constraint: usize,
        direction: BoundDirection, // Activity bound that can be calculated
    ) -> Result<(), LinearProgramType<OF>> {
        debug_assert_eq!(
            match direction {
                BoundDirection::Lower => self.counters.activity[constraint].0,
                BoundDirection::Upper => self.counters.activity[constraint].1,
            },
            0,
        );

        // TODO: Try to avoid this clone
        let bound = self.compute_activity_bound_if_needed(constraint, direction).clone();

        if self.is_infeasible_due_to_activity_bounds(constraint, &bound, direction) {
            return Err(LinearProgramType::Infeasible);
        }

        if self.is_constraint_removable(constraint, &bound, direction) {
            self.remove_constraint_values(constraint)?;
            self.remove_constraint(constraint);
            return Ok(());
        }

        match (direction, self.updates.constraint_types(constraint)) {
            (BoundDirection::Lower, ConstraintType::Less | ConstraintType::Equal)
        | (BoundDirection::Upper, ConstraintType::Greater | ConstraintType::Equal) => {
            let targets = self.counters.iter_active_row(constraint)
                .map(|(i, v)| (i, v.clone())) // TODO: Avoid this clone
                .collect::<Vec<_>>();
            for (variable, coefficient) in targets {
                let (direction, change) = self.updates.tighten_variable_bound(
                    constraint, variable, &coefficient, direction, &bound,
                );
                if let Some(change) = change {
                    self.after_bound_change(variable, direction, change);
                }
            }
        },
        _ => (),
    }

        Ok(())
    }

    /// Apply domain propagation through activation bounds for a constraint where all but one of the
    /// relevant variable bounds are known.
    ///
    /// # Arguments
    ///
    /// * `constraint`: Index of the constraint under consideration.
    /// * `direction`: Whether the activity lower- or upperbound should be used.
    fn create_variable_bound(
        &mut self,
        constraint: usize,
        direction: BoundDirection,
    ) {
        // We should be exactly one variable bound away from being able to compute an activity
        // bound, otherwise, a different rule should be applied.
        debug_assert_eq!(
            match direction {
                BoundDirection::Lower => self.counters.activity[constraint].0,
                BoundDirection::Upper => self.counters.activity[constraint].1,
            },
            1,
        );
        // All necessary variable bounds are present, except for one.
        debug_assert_eq!(self.counters.iter_active_row(constraint)
                             .filter(|&(j, c)| {
                                 let bound_direction = !direction.bitxor(Self::direction_from_sign(c));
                                 self.updates.variable_bound(j, bound_direction).is_none()
                             })
                             .count(), 1,
        );

        // There is no point in calculating any variable bound if the directions are wrong.
        if matches!(
            (direction, self.updates.constraint_types(constraint)),
            (BoundDirection::Lower, ConstraintType::Greater) | (BoundDirection::Upper, ConstraintType::Less)
        ) { return; }

        // Compute the activity bound and notice to which variable it can be applied (because it
        // is the only one that doesn't have the relevant bound yet).
        let mut total_activity = OF::zero();
        let mut target = None;
        for (column, coefficient) in self.counters.iter_active_row(constraint) {
            let bound_direction = !direction.bitxor(Self::direction_from_sign(coefficient));
            match self.updates.variable_bound(column, bound_direction) {
                Some(bound) => total_activity += coefficient * bound,
                None => target = Some((column, coefficient)),
            }
        }
        let (target_column, coefficient) = target.unwrap();

        // Compute the variable bound and apply the change.
        let direction = direction.bitxor(Self::direction_from_sign(coefficient));
        let value = (self.updates.b(constraint) - total_activity) / coefficient;
        if let Some(change) = self.updates.update_bound(target_column, direction, &value) {
            self.after_bound_change(target_column, direction, change);
        };
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
    fn compute_activity_bound_if_needed(
        &mut self,
        constraint: usize,
        direction: BoundDirection,
    ) -> &OF {
        // Relevant activity counter.
        let bound = match direction {
            BoundDirection::Lower => &mut self.activity_bounds[constraint].0,
            BoundDirection::Upper => &mut self.activity_bounds[constraint].1,
        };

        // If there is no bound, compute it.
        if bound.is_none() {
            let updates = &self.updates;
            *bound = Some(self.counters.iter_active_row(constraint)
                .map(|(variable, coefficient)| {
                    let bound_direction = !direction.bitxor(Self::direction_from_sign(coefficient));
                    let bound = updates.variable_bound(variable, bound_direction).unwrap();
                    coefficient * bound
                })
                .sum());
        }

        // We unwrap because we just computed it.
        bound.as_ref().unwrap()
    }

    /// Whether the problem is infeasible because a constraint exceeds the activity the equation can
    /// obtain.
    ///
    /// This is a helper method for `presolve_constraint_by_domain_propagation`.
    ///
    /// # Arguments
    ///
    /// * `constraint`: Index of constraint under consideration.
    /// * `bound_value`: Value computed for the activity bound.
    /// * `direction`: Which activity bound it is.
    fn is_infeasible_due_to_activity_bounds(
        &self,
        constraint: usize,
        bound_value: &OF,
        direction: BoundDirection
    ) -> bool {
        match (direction, self.updates.constraint_types(constraint)) {
            (BoundDirection::Lower, ConstraintType::Less | ConstraintType::Equal) => {
                bound_value > &self.updates.b(constraint)
            },
            (BoundDirection::Upper, ConstraintType::Greater | ConstraintType::Equal) => {
                bound_value < &self.updates.b(constraint)
            },
            _ => false,
        }
    }

    /// Whether a constraint can be removed.
    ///
    /// # Arguments
    ///
    /// * `constraint`: Index of the constraint under consideration.
    /// * `value`: Value of the lower- or upperbound.
    /// * `direction`: Whether the activity lower- or upperbound is known.
    fn is_constraint_removable(
        &self,
        constraint: usize,
        bound_value: &OF,
        direction: BoundDirection,
    ) -> bool {
        match (direction, self.updates.constraint_types(constraint)) {
            (BoundDirection::Lower, ConstraintType::Greater) => bound_value >= self.updates.b(constraint),
            (BoundDirection::Upper, ConstraintType::Less) => bound_value <= self.updates.b(constraint),
            _ => false,
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
    /// * `direction`: Whether the lower- or upper variable bound was updated.
    /// * `change`: Whether another bound was previously known, and the difference between the
    /// former and current bound.
    /// * `change`: Change with respect to the previous bound value.
    /// * `variables`: View into the general form's problem's variables that were active before the
    /// presolving started.
    fn after_bound_change(
        &mut self,
        variable: usize,
        direction: BoundDirection,
        change: Option<OF>,
    ) {
        debug_assert_ne!(self.counters.variable[variable], 0);
        debug_assert!(match direction {
            BoundDirection::Lower => change.as_ref().map_or(true, |v| v > &OF::zero()),
            BoundDirection::Upper => change.as_ref().map_or(true, |v| v < &OF::zero()),
        });

        if self.updates.is_variable_fixed(variable).is_some() {
            self.queues.substitution.insert(variable);
        }

        if let Some(difference) = change {
            self.update_activity_bounds(variable, direction, difference);
        } else {
            self.update_activity_counters(variable, direction);
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
    /// * `direction`: Whether a variable upper or lower bound was changed.
    /// * `by_how_much`: Size of the change.
    fn update_activity_bounds(
        &mut self,
        variable: usize,
        direction: BoundDirection,
        by_how_much: OF,
    ) {
        debug_assert!(match direction {
            BoundDirection::Lower => &by_how_much > &OF::zero(),
            BoundDirection::Upper => &by_how_much < &OF::zero(),
        });

        let rows_to_check = self.counters.iter_active_column(variable).collect::<Vec<_>>();
        for (row, coefficient) in rows_to_check {
            if !self.counters.is_constraint_still_active(row) {
                continue;
            }

            let bound_to_edit = !direction.bitxor(Self::direction_from_sign(coefficient));
            if let Some(ref mut bound) = match bound_to_edit {
                BoundDirection::Lower => &mut self.activity_bounds[row].0,
                BoundDirection::Upper => &mut self.activity_bounds[row].1,
            } {
                // TODO(NUMERICS): See Achterberg (2007), Algorithm 7.1
                *bound += &by_how_much * coefficient;
                self.queues.activity.insert((row, bound_to_edit));
            }
        }
    }

    /// Update the activity counters after a new bound was found.
    ///
    /// # Arguments
    ///
    /// * `variable`: Variable who's bound was changed.
    /// * `direction`: Whether a variable upper or lower bound was changed.
    fn update_activity_counters(
        &mut self,
        variable: usize,
        direction: BoundDirection,
    ) {
        let constraints_to_check = self.counters.iter_active_column(variable)
            .map(|(i, v)| (i, v.clone())).collect::<Vec<_>>(); // TODO: Avoid This clone
        for (constraint, coefficient) in constraints_to_check {
            let activity_direction = !direction.bitxor(Self::direction_from_sign(&coefficient));
            let counter = match activity_direction {
                BoundDirection::Lower => &mut self.counters.activity[constraint].0,
                BoundDirection::Upper => &mut self.counters.activity[constraint].1,
            };
            *counter -= 1;
            if *counter <= 1 {
                self.queues.activity.insert((constraint, activity_direction));
            }
        }
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
    fn remove_constraint_values(
        &mut self,
        constraint: usize,
    ) -> Result<(), LinearProgramType<OF>> {
        let variables_to_scan = self.counters.iter_active_row(constraint).map(|(j, _)| j).collect::<Vec<_>>();
        self.counters.constraint[constraint] -= variables_to_scan.len();
        for variable in variables_to_scan {
            self.counters.variable[variable] -= 1;
            self.readd_column_to_queues_based_on_counter(variable)?;
        }

        debug_assert_eq!(self.counters.constraint[constraint], 0);
        Ok(())
    }

    /// When the variable counter drops low, this has implications for rules that should be tested.
    fn readd_column_to_queues_based_on_counter(&mut self, column: usize) -> Result<(), LinearProgramType<OF>> {
        match self.counters.variable[column] {
            0 => {
                // If a variable is unfeasible before presolving, should have been detected during
                // loading as a trivial infeasibility. If unfeasible later, should be detected at
                // bound change.
                debug_assert!(self.updates.variable_has_feasible_value(column));

                let value = if self.general_form.variables[column].cost.is_zero() {
                    RemovedVariable::Solved(self.updates.get_feasible_value(column).unwrap())
                } else {
                    self.updates.optimize_column_independently(column)?
                };
                self.remove_variable(column, value);
            },
            1 => if self.general_form.variables[column].cost.is_zero() {
                self.queues.slack.insert(column);
            },
            _ => (),
        }

        Ok(())
    }

    /// Mark a constraint as removed.
    fn remove_constraint(&mut self, constraint: usize) {
        debug_assert_eq!(self.counters.constraint[constraint], 0);

        self.updates.constraints_marked_removed.push(constraint);
        self.queues.remove_constraint_from_all(constraint);
    }

    /// Mark a variable as removed.
    fn remove_variable(&mut self, variable: usize, solution: RemovedVariable<OF>) {
        debug_assert_eq!(self.counters.variable[variable], 0);

        self.updates.removed_variables.push((variable, solution));
        self.queues.remove_variable_from_all(variable);
    }

    fn direction_from_sign(value: &OF) -> BoundDirection {
        match value.cmp(&OF::zero()) {
            Ordering::Greater => BoundDirection::Upper,
            Ordering::Less => BoundDirection::Lower,
            Ordering::Equal => unreachable!("No coefficient should be zero at this point. Value: {}", value),
        }
    }
}

impl Queues {
    /// Whether all queues are empty.
    ///
    /// This indicates whether the repeated application of reduction rules can be stopped.
    pub(super) fn are_empty(&self) -> bool {
        // Note the reverse order w.r.t. the order in which these queues are tested in the main loop
        self.activity.is_empty()
            && self.slack.is_empty()
            && self.bound.is_empty()
            && self.substitution.is_empty()
            && self.empty_row.is_empty()
    }

    /// When a constraint is removed from the problem, it shouldn't be in any queue.
    fn remove_constraint_from_all(&mut self, constraint: usize) {
        self.empty_row.remove(&constraint);
        self.bound.remove(&constraint);
        self.activity.remove(&(constraint, BoundDirection::Lower));
        self.activity.remove(&(constraint, BoundDirection::Upper));
    }

    /// When a variable is removed from the problem, it shouldn't be in any queue.
    fn remove_variable_from_all(&mut self, variable: usize) {
        self.substitution.remove(&variable);
        self.slack.remove(&variable);
    }
}

#[cfg(test)]
mod test {
    use num::rational::Ratio;
    use num::traits::FromPrimitive;

    use crate::data::linear_algebra::matrix::ColumnMajor;
    use crate::data::linear_algebra::matrix::Order;
    use crate::data::linear_algebra::vector::{Dense, Vector};
    use crate::data::linear_algebra::vector::test::TestVector;
    use crate::data::linear_program::elements::{BoundDirection, ConstraintType, LinearProgramType, Objective, VariableType};
    use crate::data::linear_program::general_form::{GeneralForm, RemovedVariable, Variable};
    use crate::data::linear_program::general_form::presolve::Index;
    use crate::data::linear_program::solution::Solution;
    use crate::R32;

    type T = Ratio<i32>;

    #[test]
    fn presolve_empty_constraint() {
        let create = |constraint_type, value| {
            GeneralForm::<_, T>::new(
                Objective::Minimize,
                ColumnMajor::from_test_data(&vec![vec![0], vec![1]], 1),
                vec![constraint_type, ConstraintType::Equal],
                Dense::from_test_data(vec![value, 123]),
                vec![Variable {
                    variable_type: VariableType::Continuous,
                    cost: R32!(1),
                    lower_bound: None,
                    upper_bound: None,
                    shift: R32!(0),
                    flipped: false,
                }],
                vec!["X".to_string()],
                R32!(0),
            )
        };

        let initial = create(ConstraintType::Equal, 0);
        let mut index = Index::new(&initial).unwrap();
        assert!(index.presolve_empty_constraint(0).is_ok());
        assert_eq!(index.counters.constraint, vec![0, 1]);
        assert_eq!(index.counters.variable, vec![1]);
        assert_eq!(index.updates.constraints_marked_removed, vec![0]);

        let initial = create(ConstraintType::Greater, 0);
        let mut index = Index::new(&initial).unwrap();
        assert!(index.presolve_empty_constraint(0).is_ok());
        assert_eq!(index.counters.constraint, vec![0, 1]);
        assert_eq!(index.counters.variable, vec![1]);
        assert_eq!(index.updates.constraints_marked_removed, vec![0]);

        let initial = create(ConstraintType::Less, 1);
        let mut index = Index::new(&initial).unwrap();
        assert!(index.presolve_empty_constraint(0).is_ok());
        assert_eq!(index.counters.constraint, vec![0, 1]);
        assert_eq!(index.counters.variable, vec![1]);
        assert_eq!(index.updates.constraints_marked_removed, vec![0]);

        let initial = create(ConstraintType::Greater, 1);
        let mut index = Index::new(&initial).unwrap();
        assert!(index.presolve_empty_constraint(0).is_err());
    }

    #[test]
    fn presolve_fixed_variable() {
        let initial = GeneralForm::<_, T>::new(
            Objective::Minimize,
            ColumnMajor::from_test_data(&vec![vec![1], vec![2]], 1),
            vec![ConstraintType::Equal; 2],
            Dense::from_test_data(vec![1; 2]),
            vec![Variable {
                variable_type: VariableType::Continuous,
                cost: R32!(1),
                lower_bound: Some(R32!(1)),
                upper_bound: Some(R32!(1)),
                shift: R32!(0),
                flipped: false
            }],
            vec!["X".to_string()],
            R32!(7),
        );
        let mut index = Index::new(&initial).unwrap();

        index.presolve_fixed_variable(0);
        assert_eq!(index.counters.constraint, vec![0, 0]);
        assert_eq!(index.counters.variable, vec![0]);
        assert_eq!(index.updates.constraints_marked_removed, vec![]);
        assert_eq!(index.updates.removed_variables, vec![(0, RemovedVariable::Solved(R32!(1)))]);
        assert_eq!(index.updates.b.get(&0), Some(&R32!(0)));
        assert_eq!(index.updates.b.get(&1), Some(&R32!(-1)));
        assert_eq!(index.updates.fixed_cost, R32!(1) * R32!(1));
        assert!(!index.queues.are_empty());
    }

    #[test]
    fn presolve_simple_bound_constraint() {
        let initial = GeneralForm::<T, T>::new(
            Objective::Minimize,
            ColumnMajor::from_test_data(&vec![vec![1]; 2], 1),
            vec![ConstraintType::Equal; 2],
            Dense::from_test_data(vec![2; 2]),
            vec![Variable {
                variable_type: VariableType::Continuous,
                cost: R32!(1),
                lower_bound: None,
                upper_bound: None,
                shift: R32!(0),
                flipped: false
            }],
            vec!["X".to_string()],
            R32!(0),
        );
        let mut index = Index::new(&initial).unwrap();

        assert_eq!(index.presolve_simple_bound_constraint(0), Ok(()));
        assert_eq!(index.counters.constraint, vec![0, 1]);
        assert_eq!(index.counters.variable, vec![1]);
        assert_eq!(index.updates.constraints_marked_removed, vec![0]);
        assert_eq!(index.updates.removed_variables, vec![]);
        assert_eq!(index.updates.bounds.get(&(0, BoundDirection::Lower)), Some(&R32!(2)));
        assert_eq!(index.updates.bounds.get(&(0, BoundDirection::Upper)), Some(&R32!(2)));
        assert!(!index.queues.are_empty());
        assert_eq!(initial.b, Dense::from_test_data(vec![2; 2]));
    }

    #[test]
    fn presolve_constraint_if_slack_with_suitable_bounds() {
        let create = |constraint_type, lower, upper| {
            let nr_variables = 3;
            GeneralForm::<T, T>::new(
                Objective::Minimize,
                ColumnMajor::from_test_data(&vec![vec![2; nr_variables]], nr_variables),
                vec![constraint_type],
                Dense::from_test_data(vec![3]),
                vec![Variable {
                    variable_type: VariableType::Continuous,
                    cost: R32!(0),
                    lower_bound: lower,
                    upper_bound: upper,
                    shift: R32!(0),
                    flipped: false
                }; nr_variables],
                vec!["X".to_string(); nr_variables],
                R32!(0),
            )
        };

        // Don't change a thing
        let initial = create(ConstraintType::Equal, Some(R32!(1)), Some(R32!(2)));
        let mut index = Index::new(&initial).unwrap();
        assert!(index.presolve_constraint_if_slack_with_suitable_bounds(0).is_ok());
        assert_eq!(index.counters.constraint, vec![3]);
        assert_eq!(index.counters.variable, vec![1, 1, 1]);
        assert_eq!(index.updates.constraints_marked_removed, vec![]);
        assert_eq!(index.updates.removed_variables, vec![]);
        assert_eq!(initial.constraint_types, vec![ConstraintType::Equal]);
        assert_eq!(initial.b, Dense::from_test_data(vec![3]));

        // Change a thing
        let initial = create(ConstraintType::Equal, Some(R32!(1)), None);
        let mut index = Index::new(&initial).unwrap();
        assert!(index.presolve_constraint_if_slack_with_suitable_bounds(0).is_ok());
        assert_eq!(index.counters.constraint, vec![2]);
        assert_eq!(index.counters.variable, vec![0, 1, 1]);
        assert_eq!(index.updates.constraints_marked_removed, vec![]);
        assert_eq!(index.updates.constraint_types(0), ConstraintType::Less);
        assert_eq!(index.updates.removed_variables, vec![(0, RemovedVariable::FunctionOfOthers {
            constant: R32!(3, 2),
            coefficients: vec![(1, R32!(1)), (2, R32!(1))],
        })]);
        assert_eq!(index.updates.b(0), &R32!(3 - 2));

        let initial = create(ConstraintType::Equal, None, Some(R32!(1)));
        let mut index = Index::new(&initial).unwrap();
        assert!(index.presolve_constraint_if_slack_with_suitable_bounds(0).is_ok());
        assert_eq!(index.counters.constraint, vec![2]);
        assert_eq!(index.counters.variable, vec![0, 1, 1]);
        assert_eq!(index.updates.constraints_marked_removed, vec![]);
        assert_eq!(index.updates.constraint_types(0), ConstraintType::Greater);
        assert_eq!(index.updates.removed_variables, vec![(0, RemovedVariable::FunctionOfOthers {
            constant: R32!(3, 2),
            coefficients: vec![(1, R32!(1)), (2, R32!(1))],
        })]);
        assert_eq!(index.updates.b(0), &R32!(3 - 2));

        let initial = create(ConstraintType::Greater, Some(R32!(1)), None);
        let mut index = Index::new(&initial).unwrap();
        assert!(index.presolve_constraint_if_slack_with_suitable_bounds(0).is_ok());
        assert_eq!(index.counters.constraint, vec![0]);
        assert_eq!(index.counters.variable, vec![0; 3]);
        assert_eq!(index.updates.constraints_marked_removed, vec![0]);
        assert_eq!(initial.constraint_types.len(), 1); // Removed after
        assert_eq!(index.updates.removed_variables, vec![
            (0, RemovedVariable::FunctionOfOthers {
                constant: R32!(3, 2),
                coefficients: vec![(1, R32!(1)), (2, R32!(1))],
            }),
            (1, RemovedVariable::Solved(R32!(1))),
            (2, RemovedVariable::Solved(R32!(1))),
        ]);
        assert_eq!(initial.b.len(), 1); // Removed after
    }

    /// Shifting a variable.
    #[test]
    fn shift_variables() {
        let bound_value = 2.5_f64;

        let data = vec![
            vec![1, 0],
            vec![2, 1],
        ];
        let constraints = ColumnMajor::from_test_data::<T, T, T, _>(&data, 2);
        let b = Dense::from_test_data(vec![
            2,
            8,
        ]);
        let constraint_types = vec![
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
            cost: R32!(3),
            lower_bound: Some(R32!(bound_value)),
            upper_bound: None,
            shift: R32!(0),
            flipped: false
        }];
        let variable_names = vec!["XONE".to_string(), "XTWO".to_string()];
        let mut general_form = GeneralForm::new(
            Objective::Minimize,
            constraints,
            constraint_types,
            b,
            variables,
            variable_names,
            R32!(1),
        );
        general_form.transform_variables();

        let constraints = ColumnMajor::from_test_data(&data, 2);
        let b = Dense::from_test_data(vec![
            2f64 - bound_value * 0f64,
            8f64 - bound_value * 1f64,
        ]);
        let constraint_types = vec![
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
                cost: R32!(3),
                lower_bound: Some(R32!(0)),
                upper_bound: None,
                shift: -R32!(bound_value),
                flipped: false
            },
        ];
        let variable_names = vec!["XONE".to_string(), "XTWO".to_string()];
        let expected = GeneralForm::new(
            Objective::Minimize,
            constraints,
            constraint_types,
            b,
            variables,
            variable_names,
            R32!(1) + R32!(3) * R32!(bound_value),
        );

        assert_eq!(general_form, expected);
    }

    #[test]
    fn make_b_non_negative() {
        let rows = ColumnMajor::from_test_data::<T, T, T, _>(&vec![vec![2]], 1);
        let b = Dense::from_test_data(vec![-1]);
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

        let data = ColumnMajor::from_test_data(&vec![vec![-2]], 1);
        let b = Dense::from_test_data(vec![1]);
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
    fn presolve() {
        let data = vec![
            // Column 3 should be removed because empty
            vec![2, 0, 0, 0, 0, 0], // Should be removed because simple bound
            vec![3, 5, 0, 0, 0, 0], // Should be removed because simple bound after removal of the row above
            vec![7, 11, 13, 0, 0, 0], // Should be removed because of fixed variable after the removal of above two
            vec![17, 19, 23, 0, 29, 31], // Removal by variable bounds
        ];
        let constraints = ColumnMajor::from_test_data::<T, T, T, _>(&data, 6);
        let b = Dense::from_test_data(vec![
            101,
            103,
            107,
            109,
        ]);
        let constraint_types = vec![
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
            constraints,
            constraint_types,
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
}
