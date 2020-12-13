//! # Simple matrix data
//!
//! A combination of a sparse matrix of constraints and a list of upper bounds for variables.
use std::fmt::{Display, Formatter};
use std::fmt;

use itertools::repeat_n;

use crate::algorithm::two_phase::matrix_provider::{Column as ColumnTrait, OrderedColumn};
use crate::algorithm::two_phase::matrix_provider::filter::generic_wrapper::IntoFilteredColumn;
use crate::algorithm::two_phase::matrix_provider::MatrixProvider;
use crate::algorithm::two_phase::phase_one::PartialInitialBasis;
use crate::algorithm::two_phase::tableau::kind::artificial::IdentityColumn;
use crate::algorithm::utilities::remove_sparse_indices;
use crate::data::linear_algebra::matrix::{ColumnMajor, Sparse as SparseMatrix};
use crate::data::linear_algebra::SparseTuple;
use crate::data::linear_algebra::traits::{Element, SparseElement};
use crate::data::linear_algebra::vector::{Dense, Sparse as SparseVector, Vector};
use crate::data::linear_program::elements::{BoundDirection, VariableType};
use crate::data::number_types::traits::{Field, FieldRef};

/// Describes a linear program using a combination of a sparse matrix of constraints, and a vector
/// with simple variable bounds.
///
/// Created once from a `GeneralForm`. Should allow for reasonably quick data access.
///
/// TODO(ENHANCEMENT): Is there a faster constraint storage backend possible?
/// TODO(ENHANCEMENT): Should this data structure hold less references?
///
/// The indexing for the variables and constraints is as follows:
///
/// /                           || Vars of which we want a solution | Inequality slack vars |   Bound slack vars   |
/// ============================||==================================|=======================|======================| -----
/// Equality constraints        ||            constants             |           0           |           0          | |   |
/// ----------------------------||----------------------------------|-----------------------|----------------------| |   |
/// Inequality (<=) constraints ||            constants             |     I     |     0     |           0          | | b |
/// ----------------------------||----------------------------------|-----------------------|----------------------| |   |
/// Inequality (>=) constraints ||            constants             |     0     |    - I    |           0          | |   |
/// ----------------------------||----------------------------------|-----------------------|----------------------| |---|
///                             ||                                  |                       |  +/- 1               |
/// Bound constraints           ||    constants (one 1 per row)     |           0           |         +/- 1        |
///                             ||                                  |                       |                +/- 1 |
/// ----------------------------------------------------------------------------------------------------------------
#[allow(non_snake_case)]
#[derive(Debug, PartialEq)]
pub struct MatrixData<'a, F> {
    /// Coefficient matrix.
    ///
    /// This should not contain variable bounds.
    constraints: &'a SparseMatrix<F, F, ColumnMajor>,
    /// Constraint values for the constraints, excludes the simple bounds.
    b: &'a Dense<F>,
    /// How many of which constraint do we have?
    nr_equality_constraints: usize,
    nr_upper_bounded_constraints: usize,
    nr_lower_bounded_constraints: usize,

    variables: Vec<Variable<F>>,

    /// If there is an upper bound, the "virtual constraint row" index is given together with the
    /// bound value.
    ///
    /// (non-slack variable -> bound index)
    non_slack_variable_index_to_bound_index: Vec<Option<usize>>,
    /// (bound index -> non-slack variable)
    bound_index_to_non_slack_variable_index: Vec<usize>,
}

/// Variable for a standard form linear program.
///
/// Has a lower bound of zero.
#[allow(missing_docs)]
#[derive(Eq, PartialEq, Copy, Clone, Debug)]
pub struct Variable<F> {
    pub cost: F,
    pub upper_bound: Option<F>,
    pub variable_type: VariableType,
}

/// Indexing within the different column types (see struct description).
enum ColumnType {
    Normal(usize),
    UpperInequalitySlack(usize),
    LowerInequalitySlack(usize),
    BoundSlack(usize),
}

impl<'a, F: 'static> MatrixData<'a, F>
where
    F: SparseElement<F> + Field,
    for <'r> &'r F: FieldRef<F>,
{
    /// Create a new `MatrixData` struct.
    ///
    /// # Arguments
    ///
    /// * `constraints`: Reading from the column major constraint data (probably from a
    /// `GeneralForm`).
    /// * `b`: Constraint values.
    /// * `negative_free_variable_dummy_index`: Index i contains the index of the i'th free
    /// variable.
    #[must_use]
    pub fn new(
        constraints: &'a SparseMatrix<F, F, ColumnMajor>,
        b: &'a Dense<F>,
        nr_equality_constraints: usize,
        nr_upper_bounded_constraints: usize,
        nr_lower_bounded_constraints: usize,
        variables: Vec<Variable<F>>,
    ) -> Self {
        debug_assert_eq!(
            nr_equality_constraints + nr_upper_bounded_constraints + nr_lower_bounded_constraints,
            constraints.nr_rows(),
        );

        let mut bound_index_to_non_slack_variable_index = Vec::new();
        let mut non_slack_variable_index_to_bound_index = Vec::new();
        for (j, variable) in variables.iter().enumerate() {
            if variable.upper_bound.is_some() {
                non_slack_variable_index_to_bound_index.push(Some(bound_index_to_non_slack_variable_index.len()));
                bound_index_to_non_slack_variable_index.push(j);
            } else {
                non_slack_variable_index_to_bound_index.push(None);
            }
        }
        debug_assert!(bound_index_to_non_slack_variable_index.len() <= variables.len());
        debug_assert_eq!(non_slack_variable_index_to_bound_index.len(), variables.len());

        MatrixData {
            constraints,
            b,
            nr_equality_constraints,
            nr_upper_bounded_constraints,
            nr_lower_bounded_constraints,
            variables,
            non_slack_variable_index_to_bound_index,
            bound_index_to_non_slack_variable_index,
        }
    }

    /// Classify a column by type using the column index.
    ///
    /// See the struct documentation for a visualization.
    fn column_type(&self, j: usize) -> ColumnType {
        debug_assert!(j < self.nr_columns());

        let separating_indices = self.get_variable_separating_indices();
        debug_assert_eq!(separating_indices[3], self.nr_columns());

        // TODO(OPTIMIZATION): Consider binary search here
        if j < separating_indices[0] {
            ColumnType::Normal(j)
        } else if j < separating_indices[1] {
            ColumnType::UpperInequalitySlack(j - separating_indices[0])
        } else if j < separating_indices[2] {
            ColumnType::LowerInequalitySlack(j - separating_indices[1])
        } else if j < separating_indices[3] {
            ColumnType::BoundSlack(j - separating_indices[2])
        } else {
            unreachable!("Should have `self.nr_columns() == separating_indices[3]`")
        }
    }

    /// Indices at which a different type of variable starts.
    ///
    /// See the struct documentation for a visualization.
    fn get_variable_separating_indices(&self) -> [usize; 4] {
        let normal = self.nr_normal_variables();
        let normal_upper = normal + self.nr_upper_bounded_constraints;
        let normal_upper_lower = normal_upper + self.nr_lower_bounded_constraints;
        let normal_upper_lower_bound = normal_upper_lower + self.nr_bounds();
        debug_assert_eq!(normal_upper_lower_bound, self.nr_columns());

        [
            normal,
            normal_upper,
            normal_upper_lower,
            normal_upper_lower_bound,
        ]
    }

    /// Indices at which a different type of constraints starts.
    ///
    /// See the struct documentation for a visualization.
    fn get_constraint_separating_indices(&self) -> [usize; 4] {
        let eq = self.nr_equality_constraints;
        let eq_upper = eq + self.nr_upper_bounded_constraints;
        let eq_upper_lower = eq_upper + self.nr_lower_bounded_constraints;
        let eq_upper_lower_bound = eq_upper_lower + self.nr_bounds();
        debug_assert_eq!(eq_upper_lower_bound, self.nr_rows());

        [
            eq,
            eq_upper,
            eq_upper_lower,
            eq_upper_lower_bound,
        ]
    }

    fn nr_normal_variables(&self) -> usize {
        self.constraints.nr_columns()
    }
}

impl<'data, F: 'static> MatrixProvider for MatrixData<'data, F>
where
    F: Field,
    for<'r> &'r F: FieldRef<F>,
{
    type Column = Column<F>;
    type Cost<'a> = Option<&'a <Self::Column as ColumnTrait>::F>;

    fn column(&self, j: usize) -> Self::Column {
        debug_assert!(j < self.nr_columns());

        // TODO(ARCHITECTURE): Can the +/- F::one() constants be avoided? They might be large and
        //  require an allocation.
        match self.column_type(j) {
            ColumnType::Normal(j) => {
                Column::Sparse {
                    // TODO(ENHANCEMENT): Avoid this cloning by using references (needs the GAT
                    //  feature to be more mature).
                    constraint_values: self.constraints.iter_column(j).cloned().collect(),
                    slack: self.bound_row_index(j, BoundDirection::Upper)
                        .map(|i| (i, F::one())),
                }
            },
            ColumnType::UpperInequalitySlack(j) => {
                let row_index = self.nr_equality_constraints + j;
                Column::Slack((row_index, F::one()), [])
            },
            ColumnType::LowerInequalitySlack(j) => {
                let row_index = self.nr_equality_constraints + self.nr_upper_bounded_constraints + j;
                Column::Slack((row_index, -F::one()), [])
            },
            ColumnType::BoundSlack(j) => {
                let row_index = self.nr_constraints() + j;
                Column::Slack((row_index, F::one()), [])
            },
        }
    }

    fn cost_value(&self, j: usize) -> Self::Cost<'_> {
        debug_assert!(j < self.nr_columns());

        match self.column_type(j) {
            ColumnType::Normal(j) => Some(&self.variables[j].cost),
            _ => None,
        }
    }

    fn constraint_values(&self) -> Dense<F> {
        let mut constraint_coefficients = self.b.clone();
        constraint_coefficients.extend_with_values(
            self.bound_index_to_non_slack_variable_index.iter()
                .map(|&j| self.variables[j].upper_bound.clone().unwrap())
                .collect()
        );
        constraint_coefficients
    }

    /// Index of row that represents a bound.
    fn bound_row_index(
        &self,
        j: usize,
        bound_type: BoundDirection,
    ) -> Option<usize> {
        debug_assert!(j < self.nr_columns());

        match bound_type {
            BoundDirection::Lower => None,
            BoundDirection::Upper => match self.column_type(j) {
                ColumnType::Normal(j) => {
                    self.non_slack_variable_index_to_bound_index[j]
                        .map(|index| self.nr_constraints() + index)
                },
                _ => None,
            }
        }
    }

    fn nr_constraints(&self) -> usize {
        self.nr_equality_constraints
            + self.nr_upper_bounded_constraints
            + self.nr_lower_bounded_constraints
    }

    fn nr_bounds(&self) -> usize {
        self.bound_index_to_non_slack_variable_index.len()
    }

    fn nr_columns(&self) -> usize {
        self.nr_normal_variables()
            + self.nr_upper_bounded_constraints
            + self.nr_lower_bounded_constraints
            + self.nr_bounds()
    }

    fn reconstruct_solution<G: Element>(&self, mut column_values: SparseVector<G, G>) -> SparseVector<G, G> {
        debug_assert_eq!(column_values.len(), self.nr_columns());

        column_values.remove_indices(&(self.nr_normal_variables()..self.nr_columns()).collect::<Vec<_>>());
        column_values
    }
}

impl<'a, F: 'static> PartialInitialBasis for MatrixData<'a, F>
where
    F: SparseElement<F> + Field,
    for <'r> &'r F: FieldRef<F>,
{
    fn pivot_element_indices(&self) -> Vec<(usize, usize)> {
        let ineq_start = self.get_variable_separating_indices()[0];
        let upper_bounded_constraints = (0..self.nr_upper_bounded_constraints)
            .map(|index| {
                (self.nr_equality_constraints + index, ineq_start + index)
            });

        let bound_constraint_start = self.get_constraint_separating_indices()[2];
        let bound_slack_start = self.get_variable_separating_indices()[2];
        let upper_bounded_variables = self.non_slack_variable_index_to_bound_index
            .iter().enumerate()
            .filter_map(|(_, &index)| index)
            .map(|index| (bound_constraint_start + index, bound_slack_start + index));

        upper_bounded_constraints.chain(upper_bounded_variables).collect()
    }

    fn nr_initial_elements(&self) -> usize {
        self.nr_upper_bounded_constraints + self.nr_bounds()
    }
}

/// Describes a column from a `MatrixData` struct.
///
/// Either a vector with (probably several) elements, or a slack column. They are always sparsely
/// represented.
///
/// TODO(ARCHITECTURE): Can this type be simplified?
#[derive(Eq, PartialEq, Debug)]
pub enum Column<F> {
    /// The case where there is at least one constraint value and possibly a slack.
    Sparse {
        /// Values belonging to constraints (and not variable bounds).
        // TODO(ARCHITECTURE): This cloning can be avoided if we can use GATs to store a type
        //  referencing the data instead.
        constraint_values: Vec<SparseTuple<F>>,
        /// Positive slack for a variable bound.
        ///
        /// Note that this slack value is always positive, because the matrix data should be in
        /// canonical form. As such, the value in this tuple is always the same: `1`.
        ///
        /// The values are grouped in a tuple, such that the `ColumnIntoIter` can yield a reference
        /// to it directly.
        ///
        /// TODO(ARCHITECTURE): Can this value be eliminated?
        slack: Option<(usize, F)>, // Is always a positive slack: `1`.
    },
    /// The case where there is only a variable bound being represented.
    ///
    /// Note that the constant value is always +/- 1, because these bound columns are normalized.
    ///
    /// TODO(ENHANCEMENT): Consider relaxing the requirement that bound columns are normalized.
    // TODO(ARCHITECTURE): F is always 1 or -1, can we optimize that? Changing it's type to
    //  BoundDirection can be bit tricky because you can't convert it to a reference while
    //  iterating, and we're currently iterating over references.
    // TODO(PERFORMANCE): Can we eliminate the zero sized array? It's currently there to avoid
    //  wrapping the `ColumnIntoIter::parent_iter_cloned` field in an `Option`, but that might be
    //  equivalent after optimizations.
    Slack((usize, F), [(usize, F); 0]),
}

impl<F: 'static> IdentityColumn for Column<F>
where
    F: Field,
{
    #[must_use]
    fn identity(i: usize, len: usize) -> Self {
        assert!(i < len);

        Self::Slack((i, F::one()), [])
    }
}

#[allow(clippy::type_repetition_in_bounds)]
impl<F> ColumnTrait for Column<F>
where
    // TODO(ARCHITECTURE): Once GATs are more developed, it could be possible to replace this bound
    //  with a where clause on the method. Then, the 'static bound doesn't propagate through the
    //  entire codebase. Once this is done, remove the `clippy::type_repetition_in_bounds`
    //  annotation.
    F: 'static,
    F: Field,
{
    type F = F;
    type Iter<'a> = ColumnIter<'a, F, impl Iterator<Item = &'a SparseTuple<F>> + Clone>;

    fn iter(&self) -> Self::Iter<'_> {
        ColumnIter {
            constraints: match self {
                Column::Sparse { constraint_values, .. } => constraint_values.iter(),
                Column::Slack(_, mock_array) => mock_array.iter(),
            },
            slack: match self {
                Column::Sparse { slack, .. } => slack.as_ref(),
                Column::Slack(pair, _) => Some(pair),
            },
        }
    }

    fn index_to_string(&self, i: usize) -> String {
        match &self {
            Column::Sparse { constraint_values, slack } => {
                let value = constraint_values.iter()
                    .find(|&&(index, _)| index == i);
                match value {
                    None => match slack {
                        Some((index, v)) if *index == i => v.to_string(),
                        _ => "0".to_string(),
                    }
                    Some((_, v)) => v.to_string(),
                }
            }
            Column::Slack((index, value), ..) => {
                if *index == i {
                    value.to_string()
                } else {
                    "0".to_string()
                }
            }
        }
    }
}

/// Mark this column implementation as ordered.
impl<F: 'static> OrderedColumn for Column<F>
where
    F: Field,
{
}

/// Describes how to iterate over a matrix data column.
///
/// Stores only references to the column struct.
#[derive(Clone)]
pub struct ColumnIter<'a, F, I> {
    /// Iterates over the nonzero constraint values (i.e. not a variable bound).
    constraints: I,

    /// Slack value always comes after the elements in `parent_iter_cloned`.
    slack: Option<&'a (usize, F)>,
}

impl<'a, F: 'static, I> Iterator for ColumnIter<'a, F, I>
where
    I: Iterator<Item = &'a SparseTuple<F>> + Clone,
    F: Field,
{
    type Item = &'a SparseTuple<F>;

    fn next(&mut self) -> Option<Self::Item> {
        self.constraints.next().or_else(|| self.slack.take())
    }
}

impl<F: 'static> IntoFilteredColumn for Column<F>
where
    F: Field,
{
    type Filtered = Column<F>;

    fn into_filtered(self, to_filter: &[usize]) -> Self::Filtered {
        debug_assert!(to_filter.is_sorted());

        match self {
            Column::Sparse { mut constraint_values, slack, .. } => {
                remove_sparse_indices(&mut constraint_values, to_filter);

                let new_row_index = slack
                    .map(|(row_index, value)| ({
                        to_filter
                            .binary_search(&row_index)
                            .map_err(|data_index| row_index - data_index)
                    }, value));
                let slack = if let Some((Err(row_index), value)) = new_row_index {
                    Some((row_index, value))
                } else { None };

                Column::Sparse { constraint_values, slack, }
            },
            Column::Slack((row_index, direction), ..) => {
                // Reasonable applications of row filtering would not create empty columns
                debug_assert!(!to_filter.contains(&row_index));

                match to_filter.binary_search(&row_index) {
                    Ok(_) => unreachable!("Deleting this row would have created an empty column"),
                    Err(skipped_before) => {
                        Column::Slack((row_index - skipped_before, direction), [])
                    },
                }
            }
        }
    }
}

impl<'a, F: 'static> Display for MatrixData<'a, F>
where
    F: Field + 'a,
    for<'r> &'r F: FieldRef<F>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let width = 8;
        let separator_width = (1 + self.nr_columns()) * width + self.get_variable_separating_indices().len();

        write!(f, "{:>width$}", "|", width = width)?;
        for column in 0..self.nr_columns() {
            if self.get_variable_separating_indices().contains(&column) {
                write!(f, "|")?;
            }
            write!(f, "{:^width$}", column, width = width)?;
        }
        writeln!(f)?;
        writeln!(f, "{}", repeat_n("=",separator_width).collect::<String>())?;
        write!(f, "{:>width$}", "cost |", width = width)?;
        for column in 0..self.nr_columns() {
            if self.get_variable_separating_indices().contains(&column) {
                write!(f, "|")?;
            }
            let cost = self.cost_value(column).map_or("0".to_string(), ToString::to_string);
            write!(f, "{:^width$}", cost, width = width)?;
        }
        writeln!(f)?;
        writeln!(f, "{}", repeat_n("=",separator_width).collect::<String>())?;

        for row in 0..self.nr_rows() {
            if self.get_constraint_separating_indices().contains(&row) {
                writeln!(f, "{}", repeat_n("-",separator_width).collect::<String>())?;
            }
            write!(f, "{:>width$}", format!("{} |", row), width = width)?;
            for column in 0..self.nr_columns() {
                if self.get_variable_separating_indices().contains(&column) {
                    write!(f, "|")?;
                }
                let value = self.column(column).iter()
                    .find(|&&(index, _)| index == row)
                    .map_or_else(|| "0".to_string(), |(_, value)| value.to_string());
                write!(f, "{:^width$}", value, width = width)?;
            }
            writeln!(f)?;
        }
        writeln!(f)
    }
}

#[cfg(test)]
mod test {
    use num::traits::FromPrimitive;

    use crate::algorithm::two_phase::matrix_provider::matrix_data::Column;
    use crate::algorithm::two_phase::matrix_provider::MatrixProvider;
    use crate::data::linear_algebra::vector::Dense;
    use crate::data::linear_algebra::vector::test::TestVector;
    use crate::data::linear_program::elements::BoundDirection;
    use crate::data::number_types::rational::Rational64;
    use crate::R64;
    use crate::tests::problem_1;

    #[test]
    fn from_general_form() {
        let (constraints, b) = problem_1::create_matrix_data_data();
        let matrix_data = problem_1::matrix_data_form(&constraints, &b);

        assert_eq!(matrix_data.nr_normal_variables(), 3);

        // Variable column with bound constant
        assert_eq!(
            matrix_data.column(0),
            Column::Sparse {
                constraint_values: vec![(1, R64!(1))],
                slack: Some((2, R64!(1))),
            },
        );
        // Variable column without bound constant
        assert_eq!(
            matrix_data.column(2),
            Column::Sparse {
                constraint_values: vec![(0,  R64!(1)), (1, R64!(1))],
                slack: Some((4, R64!(1))),
            },
        );
        // Lower bounded inequality slack
        assert_eq!(
            matrix_data.column(3),
            Column::Slack((1, R64!(-1)), []),
        );
        // Upper bounded inequality slack
        assert_eq!(
            matrix_data.column(4),
            Column::Slack((2, R64!(1)), []),
        );
        // Variable upper bound slack
        assert_eq!(
            matrix_data.column(5),
            Column::Slack((3, R64!(1)), []),
        );
        assert_eq!(
            matrix_data.column(6),
            Column::Slack((4, R64!(1)), []),
        );

        // Variable cost
        assert_eq!(matrix_data.cost_value(0),  Some(&R64!(1)));
        // Upper bounded inequality slack variable cost
        assert_eq!(matrix_data.cost_value(3), None);
        // Lower bounded inequality slack variable cost
        assert_eq!(matrix_data.cost_value(4), None);
        // Bound slack variable cost
        assert_eq!(matrix_data.cost_value(5), None);

        assert_eq!(
            matrix_data.constraint_values(),
            Dense::from_test_data(vec![0, 2, 2, 2, 2]),
        );

        assert_eq!(matrix_data.bound_row_index(0, BoundDirection::Upper), Some(2));
        assert_eq!(matrix_data.bound_row_index(2, BoundDirection::Upper), Some(4));
        assert_eq!(matrix_data.bound_row_index(3, BoundDirection::Upper), None);
        assert_eq!(matrix_data.bound_row_index(4, BoundDirection::Upper), None);
        assert_eq!(matrix_data.bound_row_index(5, BoundDirection::Upper), None);
        assert_eq!(matrix_data.bound_row_index(0, BoundDirection::Lower), None);
        assert_eq!(matrix_data.bound_row_index(3, BoundDirection::Lower), None);
        assert_eq!(matrix_data.bound_row_index(4, BoundDirection::Lower), None);
        assert_eq!(matrix_data.bound_row_index(5, BoundDirection::Lower), None);

        assert_eq!(matrix_data.nr_constraints(), 2);
        assert_eq!(matrix_data.nr_bounds(), 3);
        assert_eq!(matrix_data.nr_rows(), 5);
        assert_eq!(matrix_data.nr_columns(), 7);
    }
}
