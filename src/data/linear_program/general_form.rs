//! # Linear programs in "general form"
//!
//! A linear program in general form is defined as a linear optimization problem, where the
//! variables need to be either nonnegative, or are "free" and unconstrained. The constraints are
//! either equalities, or inequalities of the form a x >= b with no further requirements on b. This
//! module contains data structures to represent such problems, and logic to do conversions from
//! this problem form to e.g. a so called equivalent "canonical" representations where any free
//! variables and inequalities are eliminated.
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::algorithm::simplex::matrix_provider::matrix_data;
use crate::algorithm::simplex::matrix_provider::matrix_data::MatrixData;
use crate::algorithm::utilities::remove_indices;
use crate::data::linear_algebra::matrix::{ColumnMajorOrdering, RowMajorOrdering};
use crate::data::linear_algebra::matrix::SparseMatrix;
use crate::data::linear_algebra::SparseTuples;
use crate::data::linear_algebra::vector::{DenseVector, SparseVector, Vector};
use crate::data::linear_program::elements::{ConstraintType, LinearProgramType, Objective, VariableType};
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
    /// Already known solution values.
    solution_values: HashMap<String, F>,
}

/// Check whether the dimensions of the `GeneralForm` are consistent meant for debugging.
///
/// # Note
///
/// Use only debug asserts in this macro.
fn is_consistent<F: OrderedField>(general_form: &GeneralForm<F>) -> bool {
    // Take the number of rows as a reference value
    let m = general_form.constraints.nr_rows();
    let b = general_form.b.len() == m;
    let constraints = general_form.constraint_types.len() == m;
    let rows = general_form.constraints.nr_rows() == m;

    // Take the number of columns as a reference value
    let n = general_form.constraints.nr_columns();
    let variables = general_form.variables.len() == n;
    let columns = general_form.constraints.nr_columns() == n;

    return b && constraints && rows && variables && columns;
}


impl<OF: OrderedField> GeneralForm<OF> {
    /// Create a new linear program in general form.
    pub fn new(
        objective: Objective,
        constraints: SparseMatrix<OF, RowMajorOrdering>,
        constraint_types: Vec<ConstraintType>,
        b: DenseVector<OF>,
        variables: Vec<Variable<OF>>,
        fixed_cost: OF,
    ) -> Self {
        let general_form = GeneralForm {
            objective,
            constraints,
            constraint_types,
            b,
            fixed_cost,
            variables,
            solution_values: HashMap::new(),
        };

        debug_assert!(is_consistent(&general_form));

        general_form
    }

    /// Modify this linear problem such that it is representable by a `MatrixData` structure.
    ///
    /// See also the documentation of the `GeneralForm::canonicalize` method.
    pub fn derive_matrix_data(&mut self) -> Result<MatrixData<OF>, LinearProgramType<OF>> {
        self.canonicalize()?;

        let variables = self.variables.iter()
            .map(|variable| matrix_data::Variable {
                cost: match self.objective {
                    Objective::Minimize => variable.cost,
                    Objective::Maximize => -variable.cost,
                },
                upper_bound: variable.upper_bound,
                variable_type: variable.variable_type,
            }).collect();
        let negative_free_variable_dummy_index = self.variables.iter().enumerate()
            .filter(|&(_, variable)| variable.is_free())
            .map(|(j, _)| j).collect();
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
    pub(crate) fn canonicalize(&mut self) -> Result<(), LinearProgramType<OF>>{
        let column_major = SparseMatrix::from_row_ordered_tuples_although_this_is_expensive(
            &self.constraints.data, self.constraints.nr_rows()
        );

        let (rows_to_remove, columns_to_remove, optimize_independently) =
            self.substitute_extract_eliminate(&column_major)?;
        self.optimize_disjoint_variables(optimize_independently)?;
        self.transform_variables(&rows_to_remove, &columns_to_remove, &column_major);
        {
            self.remove_rows_and_columns(rows_to_remove, columns_to_remove);
            drop(column_major); // This structure no longer matches the data in `self`
        }
        self.make_b_non_negative();

        Ok(())
    }

    /// Recursively analyse rows and columns to see if they can be removed.
    ///
    /// This can be seen as a presolve operation.
    ///
    /// All rows should be considered only once, whereas column are considered more than once only
    /// when new bounds are added to the variable.
    ///
    /// # Arguments
    ///
    /// * `columns` - A column major representation of the constraint data.
    ///
    /// # Return value
    ///
    /// If the problem is not determined infeasible, a tuple containing:
    ///
    /// * A set of rows that should be removed.
    /// * A set of columns that should be removed.
    /// * A vec of columns that can be independently maximized, also contained in the above set.
    fn substitute_extract_eliminate(
        &mut self,
        columns: &SparseMatrix<OF, ColumnMajorOrdering>,
    ) -> Result<(HashSet<usize>, HashSet<usize>, Vec<usize>), LinearProgramType<OF>> {
        let mut columns_marked_for_removal = HashSet::new();
        // This is a subset of `columns_marked_for_removal`.
        let mut columns_optimized_independently = Vec::new();
        let mut rows_marked_for_removal = HashSet::new();

        // Amount of meaningful elements still in the column or row.
        // The elements should be considered when the counter drops below 2. Variables should also
        // be considered when a new bound on them is found.
        let mut column_counters = (0..self.constraints.nr_columns()).map(|j| {
            columns.data[j].len() + if self.variables[j].cost != OF::additive_identity() { 1 } else { 0 }
        }).collect::<Vec<_>>();
        let mut row_counters = (0..self.constraints.nr_rows()).map(|i| {
            self.constraints.data[i].len()
        }).collect::<Vec<_>>();

        // Columns and rows that needs to be considered.
        let mut column_queue = (0..self.constraints.nr_columns()).collect::<HashSet<_>>();
        let mut row_queue = (0..self.constraints.nr_rows())
            .filter(|&i| row_counters[i] < 2).collect::<HashSet<_>>();

        while !column_queue.is_empty() || !row_queue.is_empty() {
            // Each row in the queue will be deleted and never looked at again. That's why we do
            // these first.
            while let Some(&index) = row_queue.iter().next() {
                row_queue.remove(&index);
                if let Some(modified_index) = self.treat_row(
                    index,
                    &row_counters,
                    &column_counters,
                )? {
                    column_counters[modified_index] -= 1;
                    if column_counters[modified_index] < 2 && !columns_marked_for_removal.contains(&modified_index){
                        column_queue.insert(modified_index);
                    }
                }
                rows_marked_for_removal.insert(index);
            }

            // Columns, on the other hand, are not necessarily deleted when they are checked for
            // whether they are fixed by their bounds.
            if let Some(&index) = column_queue.iter().next() {
                column_queue.remove(&index);
                if let Some(item_removed_from_indices) = self.treat_column(
                    index,
                    &column_counters,
                    &row_counters,
                    &mut columns_optimized_independently,
                    columns,
                ) {
                    columns_marked_for_removal.insert(index);
                    for modified_index in item_removed_from_indices {
                        row_counters[modified_index] -= 1;
                        if row_counters[modified_index] < 2 && !rows_marked_for_removal.contains(&modified_index){
                            row_queue.insert(modified_index);
                        }
                    }
                }
            }
        }

        Ok((rows_marked_for_removal, columns_marked_for_removal, columns_optimized_independently))
    }

    /// See if a presolve operation can be applied to this column.
    ///
    /// This operation can either marks the column for removal, or does nothing.
    ///
    /// # Arguments
    ///
    /// * `columns` - Index of the column under consideration.
    /// * `column_counters` - Amount of meaningful elements left in each column. Directly determines
    /// which method gets called from this one.
    /// * `row_counters` - Amount of meaningful elements left in each row.
    /// * `columns_optimized_independently` - Collects variables marked for independent
    /// optimization.
    /// * `columns` - Column major representation of the constraint data.
    ///
    /// # Return value
    ///
    /// A `Some` if the column was removed, with a `Vec` of all indices of rows a affected by this
    /// removal.
    fn treat_column(
        &mut self,
        column: usize,
        column_counters: &Vec<usize>,
        row_counters: &Vec<usize>,
        columns_optimized_independently: &mut Vec<usize>,
        columns: &SparseMatrix<OF, ColumnMajorOrdering>,
    ) -> Option<Vec<usize>> {
        debug_assert!(self.solution_value(column).is_none());

        match column_counters[column] {
            0 => {
                // No interaction with any row or cost
                Some(Vec::with_capacity(0))
            },
            1 => if self.variables[column].cost == OF::additive_identity() {
                // Interaction with one row, this is essentially a slack column
                    self.remove_if_slack_with_suitable_bounds(column, row_counters, columns)
                        .map(|row| vec![row])
                } else {
                // No interaction with any row, this variable is independent from the problem
                columns_optimized_independently.push(column);

                Some(Vec::with_capacity(0))
            },
            // This variable can only be removed if it is fixed by it's bounds.
            _ => self.substitute_if_fixed_variable(column, columns),
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
        row_counters: &Vec<usize>,
        columns: &SparseMatrix<OF, ColumnMajorOrdering>,
    ) -> Option<usize> {
        let variable = &mut self.variables[column];

        let (row, value)  = *columns.iter_column(column)
            .filter(|&&(i, _)| row_counters[i] >= 2)
            .next().unwrap();
        let old_b = self.b.get_value(row);
        match self.constraint_types[row] {
            ConstraintType::Equal => {
                if variable.lower_bound.is_none() || variable.upper_bound.is_none() {
                    if let Some(lower_bound) = variable.lower_bound {
                        self.constraint_types[row] = ConstraintType::Less;
                        self.b.set_value(row, old_b - value * lower_bound);
                    }
                    if let Some(upper_bound) = variable.upper_bound {
                        self.constraint_types[row] = ConstraintType::Greater;
                        self.b.set_value(row, old_b - value * upper_bound);
                    }
                    Some(row)
                } else {
                    None
                }
            },
            ConstraintType::Less => {
                if let Some(lower_bound) = variable.lower_bound {
                    self.b.set_value(row, old_b - value * lower_bound);
                }
                Some(row)
            },
            ConstraintType::Greater => {
                if let Some(upper_bound) = variable.upper_bound {
                    self.b.set_value(row, old_b - value * upper_bound);
                }
                Some(row)
            },
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
    fn substitute_if_fixed_variable(
        &mut self,
        column: usize,
        columns: &SparseMatrix<OF, ColumnMajorOrdering>,
    ) -> Option<Vec<usize>> {
        if let Some(value) = self.variables[column].is_fixed() {
            let mut edited_rows = Vec::with_capacity(columns.data[column].len());
            for &(i, coefficient_value) in columns.iter_column(column) {
                let old_b = self.b.get_value(i);
                self.b.set_value(i, old_b - coefficient_value * value);
                edited_rows.push(i);
            }
            self.fixed_cost += self.variables[column].cost * value;
            Some(edited_rows)
        } else {
            None
        }
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
    /// * `column_counters` -  Amount of "meaningful" (no known solution value, corresponding row is
    /// not yet marked for deletion) elements that are still left in a column.
    ///
    /// # Return value
    ///
    /// Result that indicates whether the linear program might still have a feasible solution (if
    /// not, the `Result::Err` type will indicate that it is not). Inside there is an `Option` which
    /// might contain the index of a variable that would be effected by the removal of the row under
    /// consideration.
    fn treat_row(
        &mut self,
        constraint: usize,
        row_counters: &Vec<usize>,
        column_counters: &Vec<usize>,
    ) -> Result<Option<usize>, LinearProgramType<OF>> {
        match row_counters[constraint] {
            0 => {
                self.treat_empty_row(constraint)?;
                Ok(None)
            },
            // Remove if bound without slack
            1 => {
                let column_to_check =  self.treat_bound_without_slack(constraint, column_counters)?;
                Ok(Some(column_to_check))
            },
            _ => panic!(),
        }
    }

    /// Whether an empty constraint can be safely discarded.
    ///
    /// # Arguments
    ///
    /// * `constraint` - Index of row to investigate.
    ///
    /// # Return value
    ///
    /// `Result` indicating whether the linear program might still be feasible.
    fn treat_empty_row(&self, constraint: usize) -> Result<(), LinearProgramType<OF>> {
        if self.b.get_value(constraint) == OF::additive_identity() {
            Err(LinearProgramType::Infeasible)
        } else {
            Ok(())
        }
    }

    /// Modify bounds of a non-slack variable.
    ///
    /// # Arguments
    ///
    /// * `constraint` - Index of a row with only a bound.
    /// * `column_counters` - Amount of tuples in each column corresponding to a row that has not
    /// yet been marked for deletion.
    ///
    /// # Return value
    ///
    /// `Result` indicating whether the linear program might still be feasible. Inside it, the
    /// column that should be checked next.
    fn treat_bound_without_slack(
        &mut self,
        constraint: usize,
        column_counters: &Vec<usize>,
    ) -> Result<usize, LinearProgramType<OF>> {
        let &(column, value) = self.constraints.iter_row(constraint)
            .filter(|&&(j, _)| column_counters[j] >= 2)
            .filter(|&&(j, _)| self.solution_value(j).is_none())
            .nth(0).unwrap();
        let bound_value = self.b.get_value(constraint) / value;
        match self.constraint_types[constraint] {
            ConstraintType::Equal => {
                self.variables[column].update_upper_bound(bound_value);
                self.variables[column].update_lower_bound(bound_value);
            },
            ConstraintType::Greater => {
                self.variables[column].update_lower_bound(bound_value);
            },
            ConstraintType::Less => {
                self.variables[column].update_upper_bound(bound_value);
            },
        }

        if !self.variables[column].has_feasible_value() {
            Err(LinearProgramType::Infeasible)
        } else {
            Ok(column)
        }
    }

    /// Sets variables that can be optimized independently of all others to their maximum values.
    ///
    /// # Arguments
    ///
    /// * `to_optimize` - Collection of variable indices that should be optimized.
    fn optimize_disjoint_variables(&mut self, to_optimize: Vec<usize>) -> Result<(), LinearProgramType<OF>> {
        for j in to_optimize {
            let variable = &mut self.variables[j];

            let new_value = match (self.objective, variable.cost.cmp(&OF::additive_identity())) {
                (_, Ordering::Equal) => variable.lower_bound
                    .or(variable.upper_bound)
                    // We choose zero if the variable has no bounds
                    .unwrap_or(OF::additive_identity()),
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
            self.solution_values.insert(
                variable.name.clone(),
                new_value,
            );
            self.fixed_cost += variable.cost * new_value;
        }

        Ok(())
    }

    /// Get the known solution value for a variable, if there is one.
    ///
    /// This is a helper method.
    ///
    /// # Arguments
    ///
    /// * `variable` - Variable to get the solution value for.
    ///
    /// # Return value
    ///
    /// The solution found, it there is one.
    ///
    /// # Note
    ///
    /// This is only an actual solution for the variable if the problem turns out to be feasible.
    fn solution_value(&self, variable: usize) -> Option<OF> {
        debug_assert!(variable < self.variables.len());

        self.solution_values.get(&self.variables[variable].name).map(|&v| v)
    }

    /// Shift all variables, such that the lower bound is zero.
    ///
    /// This allows the removal of those lower bounds afterwards; this lower bound is the only lower
    /// bound for problems in canonical form. When working with a simplex tableau, this form allows
    /// us to eliminate all rows which describe a lower bound.
    ///
    /// If later on, such as during branch and bound, an extra lower bound needs to be inserted,
    /// this information can be stored regardless in a separate data structure.
    fn transform_variables(
        &mut self,
        rows_to_be_removed: &HashSet<usize>,
        columns_to_be_removed: &HashSet<usize>,
        columns: &SparseMatrix<OF, ColumnMajorOrdering>,
    ) {
        let mut columns_to_flip = Vec::new();

        for j in 0..self.variables.len() {
            if self.solution_value(j).is_some() && columns_to_be_removed.contains(&j) {
                continue;
            }

            let variable = &mut self.variables[j];

            // Flip such that there is not just an upper bound
            if let (None, Some(upper)) = (variable.lower_bound, variable.upper_bound) {
                variable.flipped = !variable.flipped;

                variable.lower_bound = Some(-upper);
                variable.upper_bound = None;

                columns_to_flip.push(j);
            }

            // Shift such that any lower bound is zero
            if let Some(ref mut lower) = variable.lower_bound {
                variable.shift = -*lower;
                *lower += variable.shift;
                if let Some(ref mut upper) = variable.upper_bound {
                    *upper += variable.shift;
                }
                for &(i, coefficient) in columns.iter_column(j) {
                    let old_b = self.b.get_value(i);
                    self.b.set_value(i, old_b + coefficient * variable.shift);
                }
                self.fixed_cost += variable.cost * variable.shift;
            }
        }

        let mut i = 0;
        for (row_index, row) in self.constraints.data.iter_mut().enumerate() {
            if rows_to_be_removed.contains(&row_index) {
                continue;
            }

            for (j, ref mut value) in row {
                while i < columns_to_flip.len() && columns_to_flip[i] < *j {
                    i += 1;
                }
                if i < columns_to_flip.len() && columns_to_flip[i] == *j {
                    *value *= -OF::multiplicative_identity()
                }
                if i == columns_to_flip.len() {
                    break;
                }
            }
            i = 0;
        }
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
    fn remove_rows_and_columns(&mut self, rows: HashSet<usize>, columns: HashSet<usize>) {
        let mut rows = rows.into_iter().collect::<Vec<_>>();
        rows.sort();

        let mut columns = columns.into_iter().collect::<Vec<_>>();
        columns.sort();

        self.constraints.remove_rows(&rows);
        remove_indices(&mut self.constraint_types, &rows);
        self.b.remove_indices(&rows);

        self.constraints.remove_columns_although_this_matrix_is_row_ordered(&columns);
        remove_indices(&mut self.variables, &columns);
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

    /// Combines the variable names to the values of a basic feasible solution.
    ///
    /// * `bfs` - A basic feasible solution.
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
    pub fn human_readable_bfs(&self, bfs: SparseVector<OF>) -> Vec<(String, OF)> {
        debug_assert_eq!(bfs.len(), self.variables.len());

        self.variables.iter().zip(bfs.iter_values())
            .map(|(variable, &(_, value))| (variable.name.clone(), value))
            .collect()
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

/// A variable is named, of continuous or integer type and may be shifted.
///
/// TODO: Check the below calculation and logic.
/// The upper bound is relative to the offset; that is, the lower bound is `offset`, the upper
/// bound is `upper_bound - offset`. For example, the range stays the same, regardless of the shift.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Variable<F: Field> {
    /// Human semi-readable name from the original problem.
    pub name: String,
    /// Whether the variable is integer or not.
    pub variable_type: VariableType,
    /// Coefficient in the objective function.
    pub cost: F,
    /// Describing the accepted values for this variable
    ///
    /// If it is `None`, the variable is considered to be in (lower_bound, oo).
    pub upper_bound: Option<F>,
    /// Describing the accepted values for this variable
    ///
    /// Lower bound should be set to 0 when a variable is nonnegative. If it is `None`, the variable
    /// is considered to be in (-oo, upper_bound).
    pub lower_bound: Option<F>,
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
    fn update_upper_bound(&mut self, new_bound: OF) {
        self.upper_bound = Some(match self.upper_bound {
            Some(existing_bound) => existing_bound.min(new_bound),
            None => new_bound,
        });
    }
    fn update_lower_bound(&mut self, new_bound: OF) {
        self.lower_bound = Some(match self.lower_bound {
            Some(existing_bound) => existing_bound.max(new_bound),
            None => new_bound,
        });
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use num::rational::Ratio;
    use num::traits::FromPrimitive;

    use crate::data::linear_algebra::matrix::{MatrixOrder, RowMajorOrdering};
    use crate::data::linear_algebra::matrix::ColumnMajorOrdering;
    use crate::data::linear_algebra::vector::DenseVector;
    use crate::data::linear_algebra::vector::test::TestVector;
    use crate::data::linear_program::elements::{ConstraintType, Objective, VariableType};
    use crate::data::linear_program::general_form::{GeneralForm, Variable};
    use crate::R32;

    type T = Ratio<i32>;

    /// If a simple equality bound is found, remember the solution value and remove the row and
    /// column
    #[test]
    fn test_substitute_fixed() {
        let data = vec![
            vec![1f64, 0f64, 0f64],
            vec![1f64, 2f64, 3f64],
        ];
        let columns = ColumnMajorOrdering::from_test_data(&data);
        let rows = RowMajorOrdering::from_test_data(&data);
        let b = DenseVector::from_test_data(vec![3f64, 8f64]);
        let column_info = vec![Variable {
            name: "XONE".to_string(),
            variable_type: VariableType::Continuous,
            cost: R32!(1),
            lower_bound: None,
            upper_bound: None,
            shift: R32!(0),
            flipped: false
        }, Variable {
            name: "XTWO".to_string(),
            variable_type: VariableType::Continuous,
            cost: R32!(2),
            lower_bound: None,
            upper_bound: None,
            shift: R32!(0),
            flipped: false
        }, Variable {
            name: "XTHREE".to_string(),
            variable_type: VariableType::Continuous,
            cost: R32!(3),
            lower_bound: None,
            upper_bound: None,
            shift: R32!(0),
            flipped: false
        }];
        let constraints = vec![
            ConstraintType::Equal,
            ConstraintType::Less,
        ];
        let mut initial = GeneralForm::new(
            Objective::Minimize,
            rows,
            constraints,
            b,
            column_info,
            R32!(0),
        );
        let (remove_rows, remove_columns, _) = initial.substitute_extract_eliminate(&columns).unwrap();
        initial.remove_rows_and_columns(remove_rows, remove_columns);

        let data = vec![vec![2f64, 3f64]];
        let rows = RowMajorOrdering::from_test_data(&data);
        let b = DenseVector::from_test_data(vec![5f64]);
        let fixed_cost = R32!(3);
        let column_info = vec![
            Variable {
                name: "XTWO".to_string(),
                variable_type: VariableType::Continuous,
                cost: R32!(2),
                lower_bound: None,
                upper_bound: None,
                shift: R32!(0),
                flipped: false
            },
            Variable {
                name: "XTHREE".to_string(),
                variable_type: VariableType::Continuous,
                cost: R32!(3),
                lower_bound: None,
                upper_bound: None,
                shift: R32!(0),
                flipped: false
            },
        ];
        let constraints = vec![ConstraintType::Less];
        let expected = GeneralForm::new(
            Objective::Minimize,
            rows,
            constraints,
            b,
            column_info,
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
        let rows = RowMajorOrdering::from_test_data(&data);
        let columns = ColumnMajorOrdering::from_test_data(&data);
        let b = DenseVector::from_test_data(vec![
            2f64,
            8f64,
        ]);
        let variables = vec![Variable {
            name: "XONE".to_string(),
            variable_type: VariableType::Continuous,
            cost: R32!(1),
            lower_bound: None,
            upper_bound: None,
            shift: R32!(0),
            flipped: false
        }, Variable {
            name: "XTWO".to_string(),
            variable_type: VariableType::Continuous,
            cost: R32!(1),
            lower_bound: Some(T::from_f64(bound_value).unwrap()),
            upper_bound: None,
            shift: R32!(0),
            flipped: false
        }];
        let constraints = vec![
            ConstraintType::Greater,
            ConstraintType::Less,
        ];
        let mut general_form = GeneralForm::new(
            Objective::Minimize,
            rows,
            constraints,
            b,
            variables,
            R32!(0),
        );
        general_form.transform_variables(
            &HashSet::with_capacity(0),
            &HashSet::with_capacity(0),
            &columns,
        );

        let data = vec![
            vec![1f64, 0f64],
            vec![2f64, 1f64],
        ];
        let rows = RowMajorOrdering::from_test_data(&data);
        let b = DenseVector::from_test_data(vec![
            2f64 - bound_value * 0f64,
            8f64 - bound_value * 1f64,
        ]);
        let variables = vec![
            Variable {
                name: "XONE".to_string(),
                variable_type: VariableType::Continuous,
                cost: R32!(1),
                lower_bound: None,
                upper_bound: None,
                shift: R32!(0),
                flipped: false
            }, Variable {
                name: "XTWO".to_string(),
                variable_type: VariableType::Continuous,
                cost: R32!(1),
                lower_bound: Some(R32!(0)),
                upper_bound: None,
                shift: -T::from_f64(bound_value).unwrap(),
                flipped: false
            },
        ];
        let constraints = vec![
            ConstraintType::Greater,
            ConstraintType::Less,
        ];
        let expected = GeneralForm::new(
            Objective::Minimize,
            rows,
            constraints,
            b,
            variables,
            -T::from_f64(bound_value).unwrap(),
        );

        assert_eq!(general_form, expected);
    }

    #[test]
    fn test_make_b_non_negative() {
        let rows = RowMajorOrdering::from_test_data(&vec![vec![2f64]]);
        let b = DenseVector::from_test_data(vec![-1f64]);
        let variables = vec![
            Variable {
                name: "X".to_string(),
                variable_type: VariableType::Continuous,
                cost: R32!(1),
                lower_bound: None,
                upper_bound: None,
                shift: R32!(0),
                flipped: false
            },
        ];
        let constraints = vec![ConstraintType::Equal];
        let mut result = GeneralForm::new(
            Objective::Minimize,
            rows,
            constraints,
            b,
            variables,
            R32!(0),
        );
        result.make_b_non_negative();

        let data = RowMajorOrdering::from_test_data(&vec![vec![-2f64]]);
        let b = DenseVector::from_test_data(vec![1f64]);
        let variables = vec![Variable {
            name: "X".to_string(),
            variable_type: VariableType::Continuous,
            cost: R32!(1),
            lower_bound: None,
            upper_bound: None,
            shift: R32!(0),
            flipped: false
        }];
        let constraints = vec![ConstraintType::Equal];
        let expected = GeneralForm::new(
            Objective::Minimize,
            data,
            constraints,
            b,
            variables,
            R32!(0),
        );

        assert_eq!(result, expected);
    }
}
