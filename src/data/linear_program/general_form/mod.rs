//! # Linear programs in "general form"
//!
//! Data structure for manipulation of linear programs.
use std::cmp::Ordering;
use std::mem;
use std::ops::Neg;

use cumsum::cumsum_array;
use relp_num::{OrderedField, OrderedFieldRef};

pub use presolve::scale::{Scalable, Scaling};

use crate::algorithm::two_phase::matrix_provider::matrix_data::MatrixData;
use crate::data::linear_algebra::matrix::SparseMatrix;
use crate::data::linear_algebra::matrix::{ColumnMajor, MatrixOrder};
use crate::data::linear_algebra::traits::{
    SparseComparator, SparseElement as LinearAlgebraElement,
};
use crate::data::linear_algebra::vector::{DenseVector, SparseVector, Vector};
use crate::data::linear_program::elements::{
    Objective, RangedConstraintRelation, RangedConstraintRelationKind, VariableType,
};
use crate::data::linear_program::general_form::presolve::{
    PresolveReducer, PresolveStop, Problem,
};
use crate::data::linear_program::solution::Solution;

pub use presolve::{BoundOrigin, Limits, Postsolve, Presolved, Reduction, RuleId, Statistics};

mod presolve;

/// A linear program in general form.
///
/// This structure is used as a first storage independent representation format for different
/// parse results to be transformed to.
///
/// Can be checked for consistency by the `is_consistent` method in this module. That method can be
/// viewed as documentation for the requirements on the variables in this data structure.
#[derive(Debug, Clone)]
pub struct GeneralForm<F> {
    /// Which direction does the objective function go?
    objective: Objective,

    /// Constant in the cost function.
    fixed_cost: F,

    // Constraint related
    /// All constraint coefficients.
    ///
    /// Has size `constraint_types.len()` in the row direction, size `variables.len()` in the column
    /// direction.
    constraints: SparseMatrix<F, F, ColumnMajor>,
    /// The equation type of all rows, ordered by index.
    ///
    /// These are read "from constraint to constraint value", meaning:
    /// * When a constraint is `ConstraintType::Equal`, the equation is `<a, x> == b`
    /// * When a constraint is `ConstraintType::Range(r)`, the equation is `b - r <= <a, x> <= b`
    /// * When a constraint is `ConstraintType::Less`, the equation is `<a, x> <= b`
    /// * When a constraint is `ConstraintType::Greater`, the equation is `<a, x> >= b`
    constraint_types: Vec<RangedConstraintRelation<F>>,
    /// All right-hands sides of equations.
    b: DenseVector<F>,

    // Variable related
    /// Information about all *active* variables, that is, variables that are not yet presolved.
    variables: Vec<Variable<F>>,
    /// Names of the variables of the problem as it was read, in that order.
    ///
    /// Never shrinks: a variable that presolving removes still has to be named in the solution.
    variable_names: Vec<String>,
    /// Names of the constraints of the problem as it was read, in that order.
    ///
    /// Never shrinks either: a dual value has to be reported against a row that can be named.
    constraint_names: Vec<String>,

    /// What presolving did, and which of the original columns are still here.
    ///
    /// Before presolving this says that nothing was done and every column survived.
    postsolve: Postsolve<F>,

    /// Where the value of each of the current columns is found in the standardized problem.
    ///
    /// The identity until `standardize` has run.
    standardization: Vec<StandardizedColumn>,

    /// Whether `standardize` turned a maximization problem into a minimization one.
    ///
    /// The costs and the constant were negated to do that, so the objective value of the
    /// standardized problem is the negation of the one asked for.
    objective_was_flipped: bool,

    /// Where each row of the presolved problem ended up once `standardize` sorted them by type.
    ///
    /// The identity until then.
    row_destination: Vec<usize>,
    /// Which rows `standardize` multiplied by `-1` to make their right hand side non negative.
    negated_rows: Vec<bool>,
}

/// Whether a row with no coefficients left is satisfied.
///
/// Its activity is zero, so this asks where zero sits relative to what the relation admits.
fn empty_row_holds<F>(relation: &RangedConstraintRelation<F>, right_hand_side: &F) -> bool
where
    F: OrderedField,
    for<'r> &'r F: OrderedFieldRef<F>,
{
    let zero = F::zero();
    match relation {
        RangedConstraintRelation::Equal => right_hand_side == &zero,
        RangedConstraintRelation::Less => right_hand_side >= &zero,
        RangedConstraintRelation::Greater => right_hand_side <= &zero,
        RangedConstraintRelation::Range(range) => {
            right_hand_side >= &zero && right_hand_side - range <= zero
        }
    }
}

/// Two problems are the same when they describe the same thing.
///
/// The record of how this problem came to be is deliberately not part of that: two problems that
/// pose the same question are equal however each of them was arrived at.
impl<F: PartialEq> PartialEq for GeneralForm<F> {
    fn eq(&self, other: &Self) -> bool {
        self.objective == other.objective
            && self.fixed_cost == other.fixed_cost
            && self.constraints == other.constraints
            && self.constraint_types == other.constraint_types
            && self.b == other.b
            && self.variables == other.variables
            && self.variable_names == other.variable_names
            && self.constraint_names == other.constraint_names
    }
}

impl<F: Eq> Eq for GeneralForm<F> {}

/// Where the value of a column of the presolved problem is found in the standardized problem.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum StandardizedColumn {
    /// One column, at this index.
    Single(usize),
    /// Two columns, `positive - negative`.
    ///
    /// The simplex implementation works with variables that are bounded below, so a variable with
    /// no bounds at all is split into a difference of two that are.
    Split { positive: usize, negative: usize },
}

/// Check whether the dimensions of the `GeneralForm` are consistent.
///
/// This method might be expensive, use it in debugging only. It can be viewed as a piece of
/// documentation on the requirements of a `GeneralForm` struct.
fn is_consistent<F: LinearAlgebraElement<F> + SparseComparator>(
    general_form: &GeneralForm<F>,
) -> bool {
    let nr_rows = general_form.constraints.nr_rows();
    let nr_columns = general_form.constraints.nr_columns();

    let rows = general_form.b.len() == nr_rows && general_form.constraint_types.len() == nr_rows;
    let columns = general_form.variables.len() == nr_columns;

    // Every column of the presolved problem says where its value is to be found, and every index
    // it names is a column of the problem as it now stands.
    let standardization = general_form.standardization.len()
        == general_form.postsolve.surviving_columns().len()
        && general_form.standardization.iter().all(|entry| match entry {
            StandardizedColumn::Single(index) => *index < nr_columns,
            StandardizedColumn::Split { positive, negative } => {
                *positive < nr_columns && *negative < nr_columns
            }
        });

    // The permutation `standardize` recorded has to be one.
    let permutation = {
        let mut seen = vec![false; general_form.row_destination.len()];
        general_form.row_destination.iter().all(|&destination| {
            destination < seen.len() && !std::mem::replace(&mut seen[destination], true)
        })
    };
    let negations = general_form.negated_rows.len() == general_form.row_destination.len();

    // Every original variable is named, and every surviving column refers to one of them.
    let names = general_form.variable_names.len() == general_form.postsolve.nr_columns();
    let survivors = general_form.postsolve.surviving_columns().is_sorted()
        && general_form
            .postsolve
            .surviving_columns()
            .iter()
            .all(|&column| column < general_form.postsolve.nr_columns());

    rows && columns && standardization && names && survivors && permutation && negations
}

impl<F: LinearAlgebraElement<F> + SparseComparator> GeneralForm<F> {
    /// Create a new linear program in general form.
    pub fn new(
        objective: Objective,
        constraints: SparseMatrix<F, F, ColumnMajor>,
        constraint_types: Vec<RangedConstraintRelation<F>>,
        b: DenseVector<F>,
        variables: Vec<Variable<F>>,
        variable_names: Vec<String>,
        fixed_cost: F,
    ) -> Self {
        let constraint_names = (0..constraints.nr_rows())
            .map(|row| format!("r{}", row))
            .collect();
        Self::new_named(
            objective,
            constraints,
            constraint_types,
            b,
            variables,
            variable_names,
            constraint_names,
            fixed_cost,
        )
    }

    /// Create a new linear program in general form, naming the constraints as well.
    ///
    /// A dual value belongs to a row, so a row has to be nameable for one to be reported.
    #[allow(clippy::too_many_arguments)]
    pub fn new_named(
        objective: Objective,
        constraints: SparseMatrix<F, F, ColumnMajor>,
        constraint_types: Vec<RangedConstraintRelation<F>>,
        b: DenseVector<F>,
        variables: Vec<Variable<F>>,
        variable_names: Vec<String>,
        constraint_names: Vec<String>,
        fixed_cost: F,
    ) -> Self {
        let nr_rows = constraints.nr_rows();
        let nr_columns = variables.len();

        let mut postsolve = Postsolve::new(nr_rows, nr_columns);
        postsolve.set_survivors((0..nr_rows).collect(), (0..nr_columns).collect());

        let general_form = Self {
            objective,
            constraints,
            constraint_types,
            b,
            fixed_cost,
            variables,
            variable_names,
            constraint_names,
            postsolve,
            standardization: (0..nr_columns).map(StandardizedColumn::Single).collect(),
            objective_was_flipped: false,
            row_destination: (0..nr_rows).collect(),
            negated_rows: vec![false; nr_rows],
        };

        debug_assert!(is_consistent(&general_form));

        general_form
    }

    /// What presolving did to this problem.
    pub fn postsolve(&self) -> &Postsolve<F> {
        &self.postsolve
    }

    /// The names of the constraints, in order. For checking a solution against this problem.
    pub(crate) fn constraint_names_for_verification(&self) -> &[String] {
        &self.constraint_names
    }

    /// The names of the variables, in order. For checking a solution against this problem.
    pub(crate) fn variable_names_for_verification(&self) -> &[String] {
        &self.variable_names
    }

    /// The bounds of each variable, in order. For checking a solution against this problem.
    pub(crate) fn variable_bounds_for_verification(
        &self,
    ) -> impl Iterator<Item = (Option<&F>, Option<&F>)> {
        self.variables
            .iter()
            .map(|variable| (variable.lower_bound.as_ref(), variable.upper_bound.as_ref()))
    }

    /// Each row, as a relation, a right hand side and its coefficients. For checking a solution.
    pub(crate) fn rows_for_verification(
        &self,
    ) -> impl Iterator<
        Item = (
            &RangedConstraintRelation<F>,
            &F,
            impl Iterator<Item = (usize, &F)>,
        ),
    > {
        (0..self.constraints.nr_rows()).map(move |row| {
            (
                &self.constraint_types[row],
                &self.b[row],
                (0..self.constraints.nr_columns()).filter_map(move |column| {
                    self.constraints
                        .iter_column(column)
                        .find(|(i, _)| *i == row)
                        .map(|(_, value)| (column, value))
                }),
            )
        })
    }

    /// The optimization direction, the constant and the costs. For checking a solution.
    pub(crate) fn objective_for_verification(
        &self,
    ) -> (Objective, &F, impl Iterator<Item = &F>) {
        (
            self.objective,
            &self.fixed_cost,
            self.variables.iter().map(|variable| &variable.cost),
        )
    }

    /// The relations of the constraints, for diagnostics.
    #[cfg(test)]
    pub(crate) fn constraint_types_for_test(&self) -> &[RangedConstraintRelation<F>] {
        &self.constraint_types
    }
    /// The right hand sides, for diagnostics.
    #[cfg(test)]
    pub(crate) fn b_for_test(&self) -> &DenseVector<F> {
        &self.b
    }
    /// One variable, for diagnostics.
    #[cfg(test)]
    pub(crate) fn variable_for_test(&self, column: usize) -> &Variable<F> {
        &self.variables[column]
    }
    /// One coefficient, for diagnostics.
    #[cfg(test)]
    pub(crate) fn coefficient_for_test(&self, row: usize, column: usize) -> Option<&F> {
        self.constraints
            .iter_column(column)
            .find(|(i, _)| *i == row)
            .map(|(_, v)| v)
    }
}

impl<OF: 'static> GeneralForm<OF>
where
    OF: OrderedField + LinearAlgebraElement<OF>,
    for<'r> &'r OF: OrderedFieldRef<OF>,
{
    /// Modify this linear problem such that it is representable by a `MatrixData` structure.
    ///
    /// The problem gets transformed into standard form, which also includes a presolve operation.
    /// Note that this call might be expensive.
    ///
    /// TODO(ENHANCEMENT): Make sure that presolving can be skipped.
    ///
    /// See also the documentation of the `GeneralForm::standardize` method.
    ///
    /// # Return value
    ///
    /// A `Result` containing either the `MatrixData` form of the presolved and standardized
    /// problem.
    ///
    /// # Errors
    ///
    /// In case the linear program gets solved during this presolve operation, a solution.
    pub fn derive_matrix_data(&self, constraint_type_counts: [usize; 4]) -> MatrixData<'_, OF> {
        debug_assert!({
            let boundaries = cumsum_array(&constraint_type_counts);
            self.constraint_types
                .iter()
                .enumerate()
                .all(|(i, constraint_type)| {
                    use RangedConstraintRelation::*;
                    if i < boundaries[0] {
                        matches!(constraint_type, Equal)
                    } else if i < boundaries[1] {
                        matches!(constraint_type, Range(_))
                    } else if i < boundaries[2] {
                        matches!(constraint_type, Less)
                    } else if i < boundaries[3] {
                        matches!(constraint_type, Greater)
                    } else {
                        panic!();
                    }
                })
        });

        let [nr_equality, nr_range, nr_upper, nr_lower] = constraint_type_counts;
        debug_assert_eq!(
            nr_equality + nr_range + nr_upper + nr_lower,
            self.nr_constraints()
        );

        let ranges = self.constraint_types[nr_equality..(nr_equality + nr_range)]
            .iter()
            .map(|constraint_type| match constraint_type {
                RangedConstraintRelation::Range(r) => r,
                _ => unreachable!(
                    "Constraint types were just sorted, and there should only be ranges here."
                ),
            })
            .collect();

        MatrixData::new(
            &self.constraints,
            &self.b,
            ranges,
            nr_equality,
            nr_range,
            nr_upper,
            nr_lower,
            &self.variables,
        )
    }

    /// Convert this `GeneralForm` problem to a form closer to the standard form representation.
    ///
    /// This involves:
    ///
    /// * Determining which rows and columns can be removed or differently represented to reduce the
    ///   problem size and increase the reading speed from a `MatrixData` structure.
    /// * Determining which variables are implicitly fixed, such that their only feasible value can
    ///   be substituted into the problem and the column eliminated.
    /// * Modifying variables such that they are either free or bounded below by zero (with possibly
    ///   an upper bound).
    /// * Multiplying some rows such that the constraint value is non-negative.
    /// * Reorder the constraints by their type (equal, range, less, greater).
    ///
    /// To do the above, a column major representation of the constraint data is built. This
    /// requires copying all constraint data once.
    ///
    /// TODO(ENHANCEMENT): Make sure that presolving can be skipped.
    ///
    /// # Errors
    ///
    /// In case the linear program gets solved during this presolve operation, a solution.
    pub fn standardize(&mut self) -> [usize; 4] {
        self.transform_variables();
        self.make_b_non_negative();
        self.make_minimization_problem();

        self.reorder_constraints_by_type()
    }

    /// Replace this problem with a smaller one that has the same answer.
    ///
    /// A set of rules is applied until none of them has anything left to do. Each removes a row or
    /// a column, tightens a bound, or weakens a relation, and records what it did so that a
    /// solution of what remains can be turned back into a solution of what was given.
    ///
    /// The constant says whether the integrality of variables has to be respected. With `false`,
    /// every variable is treated as continuous and the result is a reduction of the relaxation;
    /// with `true`, the reductions are ones that also hold for the integer problem.
    ///
    /// # Return value
    ///
    /// What could be concluded. When the answer is [`Presolved::Reduced`], this problem has been
    /// replaced by the reduced one and its [`GeneralForm::postsolve`] says how to get back.
    pub fn presolve<const RESPECT_INTEGRALITY: bool>(&mut self) -> Presolved<OF> {
        self.presolve_with::<RESPECT_INTEGRALITY>(&Limits::default(), true)
    }

    /// Presolve, saying how much work may be done and which arguments may be used.
    ///
    /// `allow_dual_reductions` permits rules that keep an optimal solution without keeping every
    /// feasible one. They make the problem smaller, but the result is no longer a relaxation of
    /// the original, so an infeasible reduced problem no longer proves the original infeasible,
    /// and a dual solution of the reduced problem is not one of the original.
    pub fn presolve_with<const RESPECT_INTEGRALITY: bool>(
        &mut self,
        limits: &Limits<OF>,
        allow_dual_reductions: bool,
    ) -> Presolved<OF> {
        debug_assert!(
            self.postsolve.is_empty(),
            "presolving twice, or after standardizing, is not supported",
        );

        let problem = Problem::from_general_form(self);
        let mut reducer =
            PresolveReducer::<OF, RESPECT_INTEGRALITY>::new(problem, limits, allow_dual_reductions);

        let outcome = presolve::run(&mut reducer);
        let applied_dual = reducer.postsolve.applied_dual_reduction;
        let unbounded = reducer.unbounded_if_feasible.is_some();

        match outcome {
            Err(PresolveStop::Infeasible) => {
                // A rule that only preserves an optimal point may have removed the last feasible
                // one, so an infeasible reduced problem no longer settles the question.
                return if applied_dual {
                    Presolved::InfeasibleOrUnbounded
                } else {
                    Presolved::Infeasible
                };
            }
            Err(PresolveStop::UnboundedIfFeasible { .. }) => {
                // Presolve found a variable that improves the objective without limit. Whether the
                // problem is unbounded also depends on it having a feasible point at all, which
                // presolve cannot establish in general.
                return Presolved::Unbounded {
                    assuming_feasible: true,
                };
            }
            Ok(()) => {}
        }

        self.adopt(reducer);

        if !self.variables.is_empty() {
            return if unbounded {
                // A problem is left, so whether it has a feasible point is still open, and that is
                // what stands between this and an answer.
                Presolved::Unbounded {
                    assuming_feasible: true,
                }
            } else {
                Presolved::Reduced
            };
        }

        // Nothing is left to decide, so whatever rows remain either hold or do not. A rule being
        // switched off can leave one behind, and reporting a solution to a problem that has none
        // would be the worst way to find that out.
        for (relation, right_hand_side) in self.constraint_types.iter().zip(self.b.iter()) {
            if !empty_row_holds(relation, right_hand_side) {
                return Presolved::Infeasible;
            }
        }

        if unbounded {
            return Presolved::Unbounded {
                // Everything was determined and holds, so the problem does have a feasible point.
                assuming_feasible: false,
            };
        }

        Presolved::Solved(self.solution_of_empty_problem())
    }

    /// Replace this problem with the reduced one the reducer produced.
    fn adopt<const RESPECT_INTEGRALITY: bool>(
        &mut self,
        reducer: PresolveReducer<'_, OF, RESPECT_INTEGRALITY>,
    ) {
        let PresolveReducer {
            problem,
            mut postsolve,
            statistics,
            ..
        } = reducer;

        let surviving_rows = (0..problem.rows.len())
            .filter(|&row| problem.rows[row].is_active)
            .collect::<Vec<_>>();
        let surviving_columns = (0..problem.columns.len())
            .filter(|&column| problem.columns[column].is_active)
            .collect::<Vec<_>>();

        // The reduced problem's own indices, for translating the matrix.
        let mut position_of_column = vec![usize::MAX; problem.columns.len()];
        for (position, &column) in surviving_columns.iter().enumerate() {
            position_of_column[column] = position;
        }
        let mut position_of_row = vec![usize::MAX; problem.rows.len()];
        for (position, &row) in surviving_rows.iter().enumerate() {
            position_of_row[row] = position;
        }

        self.constraints = ColumnMajor::new(
            surviving_columns
                .iter()
                .map(|&column| {
                    problem
                        .matrix
                        .column(column)
                        .map(|(row, value)| (position_of_row[row], value.clone()))
                        .collect()
                })
                .collect(),
            surviving_rows.len(),
            surviving_columns.len(),
        );
        self.constraint_types = surviving_rows
            .iter()
            .map(|&row| problem.rows[row].relation.clone())
            .collect();
        self.b = DenseVector::new(
            surviving_rows
                .iter()
                .map(|&row| problem.rows[row].right_hand_side.clone())
                .collect(),
            surviving_rows.len(),
        );
        self.variables = surviving_columns
            .iter()
            .map(|&column| Variable {
                variable_type: problem.columns[column].variable_type,
                cost: problem.columns[column].cost.clone(),
                lower_bound: problem.columns[column].lower_bound.clone(),
                upper_bound: problem.columns[column].upper_bound.clone(),
                shift: OF::zero(),
                flipped: false,
            })
            .collect();
        self.fixed_cost = problem.fixed_cost;

        self.standardization = (0..surviving_columns.len())
            .map(StandardizedColumn::Single)
            .collect();
        // Standardizing happens after presolving, so its record starts again from the problem that
        // is left rather than the one that was given.
        self.row_destination = (0..surviving_rows.len()).collect();
        self.negated_rows = vec![false; surviving_rows.len()];
        postsolve.set_survivors(surviving_rows, surviving_columns);
        postsolve.statistics = statistics;
        self.postsolve = postsolve;

        debug_assert!(is_consistent(self));
    }

    /// The solution of a problem that presolving left with no variables at all.
    ///
    fn solution_of_empty_problem(&self) -> Solution<OF> {
        debug_assert!(self.variables.is_empty());

        let values = self.postsolve.recover_primal(&[]);
        Solution::new(
            self.fixed_cost.clone(),
            self.variable_names.iter().cloned().zip(values).collect(),
        )
    }

    /// Split free variables and shift all variables, such that the lower bound is zero.
    ///
    /// This allows the removal of those lower bounds afterwards; this lower bound is the only lower
    /// bound for problems in standard form. When working with a simplex tableau, this form allows
    /// us to eliminate all rows which describe a lower bound.
    ///
    /// If later on, such as during branch and bound, an extra lower bound needs to be inserted,
    /// this information can be stored regardless in a separate data structure.
    fn transform_variables(&mut self) {
        // After this call, all active variables have at least one bound.
        self.split_free_variables();
        debug_assert!(is_consistent(self));

        for j in 0..self.variables.len() {
            let variable = &mut self.variables[j];

            // Flip such that there is at least a lower bound.
            if let (None, Some(upper)) = (&variable.lower_bound, &variable.upper_bound) {
                variable.flipped = !variable.flipped;
                variable.shift *= -OF::one();
                variable.cost *= -OF::one();

                variable.lower_bound = Some(-upper);
                variable.upper_bound = None;

                for (_, coefficient) in &mut self.constraints.data[j] {
                    *coefficient *= -OF::one();
                }
            };

            // Shift such that any lower bound is zero.
            if let Some(lower) = &mut variable.lower_bound {
                variable.shift -= &*lower;
                if let Some(upper) = &mut variable.upper_bound {
                    *upper -= &*lower;
                }

                self.fixed_cost += &*lower * &variable.cost;
                // Update the bounds
                for (i, coefficient) in &self.constraints.data[j] {
                    self.b[*i] -= coefficient * &*lower;
                }

                *lower = OF::zero();
            }
        }

        debug_assert!(is_consistent(self));
    }

    /// Split all free variables.
    ///
    /// The simplex implementation works with variables bounded below, which a free variable is
    /// not. Writing it as the difference of two non negative variables gives it that form; which
    /// two columns stand for it is recorded so that the difference can be taken again afterwards.
    fn split_free_variables(&mut self) {
        let (free, columns) = (0..self.nr_variables())
            .filter(|&j| self.variables[j].is_free())
            .map(|j| {
                (
                    j,
                    self.constraints
                        .iter_column(j)
                        .map(|(i, v)| (*i, -v))
                        .collect(),
                )
            })
            .unzip::<_, _, Vec<_>, _>();

        let as_matrix = ColumnMajor::new(columns, self.nr_constraints(), free.len());
        self.constraints = mem::replace(
            &mut self.constraints,
            ColumnMajor::new(Vec::with_capacity(0), 0, 0),
        )
        .concatenate_horizontally(as_matrix);

        for positive in free {
            let negative = self.variables.len();
            self.variables.push(Variable {
                variable_type: self.variables[positive].variable_type,
                cost: -&self.variables[positive].cost,
                lower_bound: Some(OF::zero()),
                upper_bound: None,
                shift: OF::zero(),
                flipped: false,
            });
            self.variables[positive].lower_bound = Some(OF::zero());

            // The entry to update is the one that currently points at this single column.
            let entry = self
                .standardization
                .iter_mut()
                .find(|entry| **entry == StandardizedColumn::Single(positive))
                .expect("every column of the presolved problem has an entry");
            *entry = StandardizedColumn::Split { positive, negative };
        }
    }

    /// Multiply the constraints by a constant such that the constraint value is >= 0.
    ///
    /// This is a step towards representing a `GeneralForm` problem in standard form.
    fn make_b_non_negative(&mut self) {
        let rows_to_negate = self
            .b
            .iter()
            .enumerate()
            .filter(|&(_, v)| v < &OF::zero())
            .map(|(i, _)| i)
            .collect();

        for &row in &rows_to_negate {
            self.negated_rows[row] = true;
        }
        self.constraints.change_row_signs(&rows_to_negate);
        for row in rows_to_negate {
            match &self.constraint_types[row] {
                RangedConstraintRelation::Less => {
                    self.constraint_types[row] = RangedConstraintRelation::Greater;
                    self.b[row] *= -OF::one();
                }
                RangedConstraintRelation::Equal => {
                    self.b[row] *= -OF::one();
                }
                RangedConstraintRelation::Greater => {
                    self.constraint_types[row] = RangedConstraintRelation::Less;
                    self.b[row] *= -OF::one();
                }
                RangedConstraintRelation::Range(range) => {
                    let minus_old_lower_bound = range - &self.b[row];
                    self.b[row] = minus_old_lower_bound;
                }
            };
        }

        debug_assert!(is_consistent(self));
    }

    /// Make this a minimization problem by multiplying the cost function by `-1`.
    ///
    /// The constant is negated along with the coefficients, so that the whole objective, and not
    /// just its variable part, is the negation of the one asked for. Leaving the constant alone
    /// would make the reported objective value wrong by twice that constant, which is the kind of
    /// error that only shows up on the problems nobody writes tests for.
    fn make_minimization_problem(&mut self) {
        if self.objective == Objective::Maximize {
            self.objective = Objective::Minimize;
            self.objective_was_flipped = true;

            for variable in &mut self.variables {
                variable.cost *= -OF::one();
            }
            self.fixed_cost *= -OF::one();
        }
    }

    /// Split the constraints out per type.
    ///
    /// The constraints in a `GeneralForm` linear program are mixed; the of the constraint is saved
    /// in `self.constraint_types`. A standard form linear program has a separate data structure
    /// for each constraint type. This to facilitate the easy creation of a `MatrixData` data
    /// struct, which "simulates" the presence of slack variables based on those different
    /// constraint types.
    ///
    /// The constraints are sorted as:
    ///
    /// 1. Equality,
    /// 2. Range,
    /// 3. Upper bounded,
    /// 4. Lower bounded.
    ///
    /// # Return value
    ///
    /// The number of equality, range, upper and lower bounds.
    fn reorder_constraints_by_type(&mut self) -> [usize; 4] {
        let (mut e_counter, mut r_counter, mut l_counter, mut g_counter) = (0, 0, 0, 0);
        let map = self
            .constraint_types
            .iter()
            .map(|constraint_type| match constraint_type {
                RangedConstraintRelation::Equal => {
                    e_counter += 1;
                    e_counter - 1
                }
                RangedConstraintRelation::Range(_) => {
                    r_counter += 1;
                    r_counter - 1
                }
                RangedConstraintRelation::Less => {
                    l_counter += 1;
                    l_counter - 1
                }
                RangedConstraintRelation::Greater => {
                    g_counter += 1;
                    g_counter - 1
                }
            })
            .collect::<Vec<usize>>();

        let old_constraint_types = self
            .constraint_types
            .iter()
            .map(From::from)
            .collect::<Vec<RangedConstraintRelationKind>>();
        self.constraint_types.sort_by(|a, b| {
            // Ordering is: ==  <  =r= < <=  <  >=
            // Sorting needs to be stable for the range constraints.
            match a {
                RangedConstraintRelation::Equal => match b {
                    RangedConstraintRelation::Equal => Ordering::Equal,
                    _ => Ordering::Less,
                },
                RangedConstraintRelation::Range(_) => match b {
                    RangedConstraintRelation::Range(_) => Ordering::Equal,
                    RangedConstraintRelation::Equal => Ordering::Greater,
                    _ => Ordering::Less,
                },
                RangedConstraintRelation::Less => match b {
                    RangedConstraintRelation::Less => Ordering::Equal,
                    RangedConstraintRelation::Greater => Ordering::Less,
                    _ => Ordering::Greater,
                },
                RangedConstraintRelation::Greater => match b {
                    RangedConstraintRelation::Greater => Ordering::Equal,
                    _ => Ordering::Greater,
                },
            }
        });

        let get_destination = |source| match old_constraint_types[source] {
            RangedConstraintRelationKind::Equal => map[source],
            RangedConstraintRelationKind::Range => e_counter + map[source],
            RangedConstraintRelationKind::Less => e_counter + r_counter + map[source],
            RangedConstraintRelationKind::Greater => {
                e_counter + r_counter + l_counter + map[source]
            }
        };

        let mut new_b = vec![None; self.b.len()];
        for i in 0..self.b.len() {
            // TODO(ARCHITECTURE): Avoid this mem::replace
            new_b[get_destination(i)] = Some(mem::replace(&mut self.b[i], OF::zero()));
        }
        self.b = DenseVector::new(
            new_b.into_iter().collect::<Option<Vec<_>>>().unwrap(),
            self.b.len(),
        );

        for column in &mut self.constraints.data {
            for (i, _) in column.iter_mut() {
                *i = get_destination(*i);
            }
            column.sort_unstable_by_key(|&(i, _)| i);
        }

        // The closure above is the permutation; keeping it is what lets a dual value be carried
        // back to the row it belongs to.
        self.row_destination = (0..self.constraint_types.len())
            .map(get_destination)
            .collect();

        debug_assert!(is_consistent(self));

        [e_counter, r_counter, l_counter, g_counter]
    }

    /// Turn a solution of the standardized problem into one of the problem as it was read.
    ///
    /// Two steps, in this order. First the transformations `standardize` made are undone, which
    /// gives a value for each column of the presolved problem. Then the record of what presolving
    /// did is walked backwards, which fills in the columns presolving removed.
    ///
    /// # Arguments
    ///
    /// * `reduced_solution`: one value per column of the standardized problem.
    pub fn compute_full_solution_with_reduced_solution<G>(
        &self,
        reduced_solution: SparseVector<G, G>,
    ) -> Solution<G>
    where
        G: LinearAlgebraElement<G> + SparseComparator + OrderedField + From<OF>,
        for<'r> &'r G: OrderedFieldRef<G> + Neg<Output = G>,
    {
        debug_assert_eq!(reduced_solution.len(), self.variables.len());

        let objective_value = self.objective_value(&reduced_solution);
        let presolved = self.undo_standardization(&reduced_solution);
        let values = self.postsolve.recover_primal::<G>(&presolved);

        Solution::new(
            objective_value,
            self.variable_names.iter().cloned().zip(values).collect(),
        )
    }

    /// The objective value of a solution of the standardized problem, in the original direction.
    fn objective_value<G>(&self, reduced_solution: &SparseVector<G, G>) -> G
    where
        G: LinearAlgebraElement<G> + SparseComparator + OrderedField + From<OF>,
        for<'r> &'r G: OrderedFieldRef<G> + Neg<Output = G>,
    {
        // The shifts applied to the variables were folded into the constant, so this is computed
        // before they are undone.
        let value = reduced_solution
            .iter()
            .map(|(j, v)| v * &G::from(self.variables[j].cost.clone()))
            .sum::<G>()
            + G::from(self.fixed_cost.clone());

        if self.objective_was_flipped { -&value } else { value }
    }

    /// Turn a solution of the standardized problem into one of the problem as it was read,
    /// carrying the multipliers of the constraints with it.
    ///
    /// `constraint_duals` holds one multiplier per constraint of the standardized problem, in that
    /// problem's own row order; [`MatrixData::constraint_duals`] produces exactly that.
    ///
    /// # Return value
    ///
    /// `None` when presolving applied a reduction whose effect on the multipliers this code does
    /// not know how to undo. The primal solution is unaffected by that and can still be had from
    /// [`Self::compute_full_solution_with_reduced_solution`].
    pub fn compute_full_solution_with_duals<G>(
        &self,
        reduced_solution: SparseVector<G, G>,
        constraint_duals: Vec<G>,
    ) -> Option<Solution<G>>
    where
        G: LinearAlgebraElement<G> + SparseComparator + OrderedField + From<OF>,
        for<'r> &'r G: OrderedFieldRef<G> + Neg<Output = G>,
    {
        let objective_value = self.objective_value(&reduced_solution);
        let presolved = self.undo_standardization(&reduced_solution);
        let values = self.postsolve.recover_primal::<G>(&presolved);
        let duals = self.duals_for_original_rows(constraint_duals, &values)?;

        Some(Solution::with_duals(
            objective_value,
            self.variable_names.iter().cloned().zip(values).collect(),
            self.constraint_names.iter().cloned().zip(duals).collect(),
        ))
    }

    /// Carry the multipliers of the standardized problem back to the rows that were read.
    ///
    /// Three things have to be undone, in this order: the sorting of the rows by type, the
    /// negation of the rows whose right hand side was negative, and the negation of the whole
    /// objective if the problem asked for a maximum. Then the rows presolving removed have to be
    /// given multipliers of their own, which also needs the reduced costs of the problem
    /// presolving left and the primal solution of the problem that was read.
    fn duals_for_original_rows<G>(
        &self,
        constraint_duals: Vec<G>,
        values: &[G],
    ) -> Option<Vec<G>>
    where
        G: LinearAlgebraElement<G> + SparseComparator + OrderedField + From<OF>,
        for<'r> &'r G: OrderedFieldRef<G> + Neg<Output = G>,
    {
        debug_assert_eq!(constraint_duals.len(), self.row_destination.len());

        let reduced_costs = self.reduced_costs_of_presolved_columns(&constraint_duals);

        let mut presolved = Vec::with_capacity(self.row_destination.len());
        for (row, &destination) in self.row_destination.iter().enumerate() {
            let mut value = constraint_duals[destination].clone();
            // Multiplying a row by `-1` multiplies its multiplier by `-1` too, leaving what the
            // row contributes to the objective unchanged.
            if self.negated_rows[row] {
                value = -&value;
            }
            // Minimizing the negation of an objective negates every multiplier along with it.
            if self.objective_was_flipped {
                value = -&value;
            }
            presolved.push(value);
        }

        self.postsolve.recover_dual::<G>(
            !self.objective_was_flipped,
            &presolved,
            &reduced_costs,
            values,
        )
    }

    /// The reduced costs of the problem presolving left, one per column it left.
    ///
    /// Computed on the standardized problem and carried back, which is where the row negations
    /// cancel out: a row that was negated has its multiplier negated too, so what the two
    /// contribute to `c - A^T y` is unchanged. What does not cancel is the flipping of a column
    /// and the flipping of the whole objective, both of which negate a reduced cost.
    fn reduced_costs_of_presolved_columns<G>(&self, constraint_duals: &[G]) -> Vec<G>
    where
        G: LinearAlgebraElement<G> + SparseComparator + OrderedField + From<OF>,
        for<'r> &'r G: OrderedFieldRef<G> + Neg<Output = G>,
    {
        let mut standardized = self
            .variables
            .iter()
            .map(|variable| G::from(variable.cost.clone()))
            .collect::<Vec<_>>();
        for (column, reduced_cost) in standardized.iter_mut().enumerate() {
            for (row, coefficient) in self.constraints.iter_column(column) {
                *reduced_cost -= &G::from(coefficient.clone()) * &constraint_duals[*row];
            }
        }

        self.standardization
            .iter()
            .map(|entry| {
                // A free variable was written as the difference of two columns, whose costs and
                // coefficients are negations of each other; either half says the same thing.
                let column = match entry {
                    StandardizedColumn::Single(column) => *column,
                    StandardizedColumn::Split { positive, .. } => *positive,
                };
                let value = &standardized[column];
                if self.variables[column].flipped != self.objective_was_flipped {
                    -value
                } else {
                    value.clone()
                }
            })
            .collect()
    }

    /// Undo the shifting, flipping and splitting that `standardize` did.
    ///
    /// # Return value
    ///
    /// One value per column of the presolved problem.
    fn undo_standardization<G>(&self, reduced_solution: &SparseVector<G, G>) -> Vec<G>
    where
        G: LinearAlgebraElement<G> + SparseComparator + OrderedField + From<OF>,
        for<'r> &'r G: OrderedFieldRef<G> + Neg<Output = G>,
    {
        let of = |column: usize| -> G {
            let value = reduced_solution
                .get(column)
                .cloned()
                .unwrap_or_else(G::zero);
            let variable = &self.variables[column];

            // The shift moved the variable's lower bound to zero, and the flip turned a variable
            // bounded only above into one bounded only below.
            let value = &value - &G::from(variable.shift.clone());
            if variable.flipped { -&value } else { value }
        };

        self.standardization
            .iter()
            .map(|entry| match entry {
                StandardizedColumn::Single(column) => of(*column),
                StandardizedColumn::Split { positive, negative } => &of(*positive) - &of(*negative),
            })
            .collect()
    }

}

impl<OF> GeneralForm<OF>
where
    OF: LinearAlgebraElement<OF> + SparseComparator,
{
    /// Number of constraints that have not been eliminated after a presolving operation.
    ///
    /// During a presolving operation, the number of variables that is both active and not yet
    /// marked for elimination in the `PresolveIndex` can only be derived using that data structure.
    fn nr_active_constraints(&self) -> usize {
        self.constraints.nr_rows()
    }

    /// The number of constraints in this linear program.
    ///
    /// # Return value
    ///
    /// The number of constraints, which excludes any variable bounds.
    pub fn nr_constraints(&self) -> usize {
        self.nr_active_constraints()
    }

    /// Number of variables that have not been eliminated after a presolving operation.
    ///
    /// During a presolving operation, the number of variables that is both active and not yet
    /// marked for elimination in the `PresolveIndex` can only be derived using that data structure.
    fn nr_active_variables(&self) -> usize {
        self.constraints.nr_columns()
    }

    /// The number of variables in this linear program.
    ///
    /// # Return value
    ///
    /// The number of columns / variables, which includes the slack columns / variables.
    pub fn nr_variables(&self) -> usize {
        self.nr_active_variables()
    }
}
/// A variable as part of a linear problem without restrictions (as opposed to for a `MatrixData` variable).
///
/// A variable is named, of continuous or integer type and may be shifted and flipped w.r.t. how it
/// was originally present in the problem.
///
/// The upper bound is relative to the shift; that is, the lower bound is `lower_bound - shift`, the
/// upper bound is `upper_bound - shift`. For example, the range stays the same, regardless of the
/// shift.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Variable<F> {
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
    /// How much this variable was shifted to have a zero lower bound.
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

impl<OF> Variable<OF>
where
    OF: OrderedField,
    for<'r> &'r OF: OrderedFieldRef<OF>,
{
    /// Whether the variable is unconstrained in both directions.
    ///
    /// Such a variable has to be split before the simplex can work with it.
    fn is_free(&self) -> bool {
        self.lower_bound.is_none() && self.upper_bound.is_none()
    }
}

#[cfg(test)]
mod test {
    use relp_num::R32;
    use relp_num::Rational32;

    use crate::data::linear_algebra::matrix::{ColumnMajor, MatrixOrder};
    use crate::data::linear_algebra::vector::DenseVector;
    use crate::data::linear_algebra::vector::test::TestVector;
    use crate::data::linear_program::elements::{
        Objective, RangedConstraintRelation, VariableType,
    };
    use crate::data::linear_program::general_form::{GeneralForm, StandardizedColumn, Variable};

    /// Shifting a variable.
    #[test]
    fn shift_variables() {
        type T = Rational32;

        let bound_value = R32!(5, 2);

        let data = vec![vec![1, 0], vec![2, 1]];
        let constraints = ColumnMajor::from_test_data::<T, T, _>(&data, 2);
        let b = DenseVector::from_test_data(vec![2, 8]);
        let constraint_types = vec![
            RangedConstraintRelation::Greater,
            RangedConstraintRelation::Less,
        ];
        let variables = vec![
            Variable {
                variable_type: VariableType::Continuous,
                cost: R32!(1),
                lower_bound: None,
                upper_bound: None,
                shift: R32!(0),
                flipped: false,
            },
            Variable {
                variable_type: VariableType::Continuous,
                cost: R32!(3),
                lower_bound: Some(bound_value),
                upper_bound: None,
                shift: R32!(0),
                flipped: false,
            },
        ];
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

        let expected = GeneralForm {
            objective: Objective::Minimize,
            fixed_cost: R32!(1) + R32!(3) * bound_value,
            constraints: ColumnMajor::from_test_data(&[vec![1, 0, -1], vec![2, 1, -2]], 3),
            constraint_types: vec![
                RangedConstraintRelation::Greater,
                RangedConstraintRelation::Less,
            ],
            b: DenseVector::from_test_data(vec![
                (2, 1),
                (11, 2), // 8 - 5/2
            ]),
            variables: vec![
                Variable {
                    variable_type: VariableType::Continuous,
                    cost: R32!(1),
                    lower_bound: Some(R32!(0)),
                    upper_bound: None,
                    shift: R32!(0),
                    flipped: false,
                },
                Variable {
                    variable_type: VariableType::Continuous,
                    cost: R32!(3),
                    lower_bound: Some(R32!(0)),
                    upper_bound: None,
                    shift: -bound_value,
                    flipped: false,
                },
                Variable {
                    variable_type: VariableType::Continuous,
                    cost: R32!(-1),
                    lower_bound: Some(R32!(0)),
                    upper_bound: None,
                    shift: R32!(0),
                    flipped: false,
                },
            ],
            variable_names: vec!["XONE".to_string(), "XTWO".to_string()],
            constraint_names: general_form.constraint_names.clone(),
            postsolve: general_form.postsolve.clone(),
            standardization: vec![],
            objective_was_flipped: false,
            row_destination: general_form.row_destination.clone(),
            negated_rows: general_form.negated_rows.clone(),
        };

        assert_eq!(general_form, expected);
        assert_eq!(
            general_form.standardization,
            vec![
                // The free variable became the difference of the columns 0 and 2.
                StandardizedColumn::Split {
                    positive: 0,
                    negative: 2,
                },
                StandardizedColumn::Single(1),
            ],
        );
    }

    #[test]
    fn make_b_non_negative() {
        type T = Rational32;

        let rows = ColumnMajor::from_test_data::<T, T, _>(&[vec![2]], 1);
        let b = DenseVector::from_test_data(vec![-1]);
        let variables = vec![Variable {
            variable_type: VariableType::Continuous,
            cost: R32!(1),
            lower_bound: None,
            upper_bound: None,
            shift: R32!(0),
            flipped: false,
        }];
        let variable_names = vec!["XONE".to_string()];
        let constraints = vec![RangedConstraintRelation::Equal];
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

        let data = ColumnMajor::from_test_data(&[vec![-2]], 1);
        let b = DenseVector::from_test_data(vec![1]);
        let constraints = vec![RangedConstraintRelation::Equal];
        let variables = vec![Variable {
            variable_type: VariableType::Continuous,
            cost: R32!(1),
            lower_bound: None,
            upper_bound: None,
            shift: R32!(0),
            flipped: false,
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
