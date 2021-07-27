//! # Simple matrix data
//!
//! A combination of a sparse matrix of constraints and a list of upper bounds for variables.
use std::fmt::{Display, Formatter};
use std::fmt;

use cumsum::cumsum_array_owned;
use enum_map::{Enum, enum_map, EnumMap};
use relp_num::{Field, FieldRef};
use relp_num::NonZero;

use crate::algorithm::two_phase::matrix_provider::column::{Column as ColumnTrait, OrderedColumn};
use crate::algorithm::two_phase::matrix_provider::column::identity::IdentityColumn;
use crate::algorithm::two_phase::matrix_provider::filter::generic_wrapper::IntoFilteredColumn;
use crate::algorithm::two_phase::matrix_provider::MatrixProvider;
use crate::algorithm::two_phase::phase_one::PartialInitialBasis;
use crate::algorithm::utilities::remove_sparse_indices;
use crate::data::linear_algebra::matrix::{ColumnMajor, Sparse as SparseMatrix};
use crate::data::linear_algebra::SparseTuple;
use crate::data::linear_algebra::traits::{SparseComparator, SparseElement};
use crate::data::linear_algebra::vector::{DenseVector, SparseVector, Vector};
use crate::data::linear_program::elements::BoundDirection;
use crate::data::linear_program::general_form::Variable;

/// The `MatrixData` struct represents variables in 6 different categories.
///
/// They are sorted and grouped together in their representation. This constant is used to define
/// the size of an array the contains all the indices at the boundaries of the groups, to be able to
/// search quickly what type a variable is by index.
const NR_GROUPS: usize = 6;

/// Describes a linear program using a combination of a sparse matrix of constraints, and a vector
/// with simple variable bounds.
///
/// Created once from a `GeneralForm`. Should allow for reasonably quick data access, given that
/// we're reading directly from the underlying `GeneralForm`.
///
/// TODO(ENHANCEMENT): Is there a faster constraint storage backend possible?
/// TODO(ENHANCEMENT): Should this data structure hold less references?
///
/// The indexing for the variables and constraints is as follows:
///
/// /               || Vars of which we want a solution | Constraint slack vars | Bound slack vars | Slack slack vars |
/// ================||==================================|=======================|==================|==================| -----
/// Equality   (==) ||           coefficients           |           0           |         0        |         0        | |   |
/// ----------------||----------------------------------|-----------------------|------------------|------------------| |   |
/// Range     (=r=) ||           coefficients           |   I   |   0   |   0   |         0        |         0        | |   |
/// ----------------||----------------------------------|-----------------------|------------------|------------------| | b |
/// Inequality (<=) ||           coefficients           |   0   |   I   |   0   |         0        |         0        | |   |
/// ----------------||----------------------------------|-----------------------|------------------|------------------| |   |
/// Inequality (>=) ||           coefficients           |   0   |   0   |  -I   |         0        |         0        | |   |
/// ----------------||----------------------------------|-----------------------|------------------|------------------| |---|
/// Variable bound  ||   coefficients (one 1 per row)   |           0           |         I        |         0        | |*b*|
/// ----------------||----------------------------------|-----------------------|------------------|------------------| |---|
/// Slack bound     ||                 0                |   I   |   0   |   0   |         0        |         I        | | r |
/// ------------------------------------------------------------------------------------------------------------------| |---|
#[derive(Debug, PartialEq)]
pub struct MatrixData<'a, F> {
    /// Coefficient matrix.
    ///
    /// This should not contain variable bounds.
    constraints: &'a SparseMatrix<F, F, ColumnMajor>,
    /// Constraint values for the constraints, excludes the simple bounds.
    b: &'a DenseVector<F>,

    /// Ranges for the range constraints.
    ///
    /// Used as the right hand side for the range slacks; the range is their upper bound.
    ranges: Vec<&'a F>,

    /// How many of which constraint do we have?
    /// ==
    nr_equality_constraints: usize,
    /// =r=
    nr_range_constraints: usize,
    /// <=
    nr_upper_bounded_constraints: usize,
    /// >=
    nr_lower_bounded_constraints: usize,
    /// Indices that separate the different groups.
    row_group_end: EnumMap<RowType, usize>,
    column_group_end: EnumMap<ColumnType, usize>,

    /// Used to read the variable upper bounds, see `*b*` in the matrix in the struct documentation.
    ///
    /// The lower bound of these variables should always be zero, the upper bound may or may not be
    /// there.
    variables: &'a [Variable<F>],

    /// If there is an upper bound, the "virtual constraint row" index is given together with the
    /// bound value.
    ///
    /// (non-slack variable -> bound index)
    non_slack_variable_index_to_bound_index: Vec<Option<usize>>,
    /// (bound index -> non-slack variable)
    bound_index_to_non_slack_variable_index: Vec<usize>,
}

#[derive(Enum, Debug)]
enum RowType {
    EqualityConstraint,
    RangeConstraint,
    UpperInequalityConstraint,
    LowerInequalityConstraint,
    VariableBound,
    SlackBound,
}

/// Indexing within the different column types (see struct description).
#[derive(Enum, Debug)]
enum ColumnType {
    /// Variables for which we want a solution.
    ///
    /// These are explicitly represented in the `GeneralForm` that `MatrixData` reads from. These
    /// are not slacks, so if the problem is properly presolved, at least two `Normal` columns appear
    /// on each row.
    Normal,
    /// Slack for a range constraint.
    ///
    /// These are the only slacks that are themselves bounded. In that way, they are not really
    /// slacks. The alternative is to duplicate an entire row, which would cost sparsity.
    ///
    /// TODO(ENHANCEMENT): Use these slacks as part of the initial basis.
    RangeSlack,
    /// Slacks `s` for an equation like `<a, x> + s = b`.
    ///
    /// Can be used as part of the initial basis.
    UpperInequalitySlack,
    /// Slacks `s` for an equation like `<a, x> - s = b`.
    LowerInequalitySlack,
    /// Slacks for a variable bound like `x + s = b`.
    ///
    /// Only upper bounds are represented; lower bounds should have been eliminated beforehand.
    /// Can be used as part of the initial basis.
    VariableBoundSlack,
    /// Slack variables for the range slacks (see the `RangeSlack` variant).
    ///
    /// Can be used as part of the initial basis.
    SlackBoundSlack,
}

impl<'a, F: 'static> MatrixData<'a, F>
where
    F: SparseElement<F> + Field + NonZero,
    for <'r> &'r F: FieldRef<F>,
{
    /// Create a new `MatrixData` instance.
    ///
    /// This is mostly just a simple constructor, although it builds up two indices to help locate
    /// the row index of variable upper bound constraints.
    ///
    /// # Arguments
    ///
    /// * `constraints`: Column major constraint matrix. These are all the equality, range, upper
    /// and lower constraints, and they should be sorted in that order.
    /// * `b`: The problem's right hand side. Should have a value for each row in `constraints`.
    /// * `ranges`: Range values, one for each range constraint. Should be at least non negative and
    /// preferably positive, as one extra row is introduced for each of these elements.
    /// * `nr_equality_constraints`: Number of equality (`==`) constraints.
    /// * `nr_range_constraints`: Number of range (`=r=`) constraints.
    /// * `nr_upper_bounded_constraints`: Number of upper bounded (`<=`) constraints.
    /// * `nr_lower_bounded_constraints`: Number of lower bounded(`>=`) constraints.
    /// * `variables`: All the non slack variables in the problem (ideally, all slack values are
    /// presolved out). Lower bounds should all be equal to zero, the upper bound is optional and
    /// will be represented by this matrix.
    #[must_use]
    pub fn new(
        constraints: &'a SparseMatrix<F, F, ColumnMajor>,
        b: &'a DenseVector<F>,
        ranges: Vec<&'a F>,
        nr_equality_constraints: usize,
        nr_range_constraints: usize,
        nr_upper_bounded_constraints: usize,
        nr_lower_bounded_constraints: usize,
        variables: &'a [Variable<F>],
    ) -> Self {
        debug_assert_eq!(ranges.len(), nr_range_constraints);

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

        let nr_bounds = bound_index_to_non_slack_variable_index.len();

        let cumulative = cumsum_array_owned([
            nr_equality_constraints,
            nr_range_constraints,
            nr_upper_bounded_constraints,
            nr_lower_bounded_constraints,
            nr_bounds,
            nr_range_constraints,
        ]);
        let row_group_end = enum_map!{
            RowType::EqualityConstraint        => cumulative[0],
            RowType::RangeConstraint           => cumulative[1],
            RowType::UpperInequalityConstraint => cumulative[2],
            RowType::LowerInequalityConstraint => cumulative[3],
            RowType::VariableBound             => cumulative[4],
            RowType::SlackBound                => cumulative[5],
        };
        debug_assert_eq!(row_group_end[RowType::LowerInequalityConstraint], constraints.nr_rows());

        let cumulative = cumsum_array_owned([
            variables.len(),
            nr_range_constraints,
            nr_upper_bounded_constraints,
            nr_lower_bounded_constraints,
            nr_bounds,
            nr_range_constraints,
        ]);
        let column_group_end = enum_map!{
            ColumnType::Normal               => cumulative[0],
            ColumnType::RangeSlack           => cumulative[1],
            ColumnType::UpperInequalitySlack => cumulative[2],
            ColumnType::LowerInequalitySlack => cumulative[3],
            ColumnType::VariableBoundSlack   => cumulative[4],
            ColumnType::SlackBoundSlack      => cumulative[5],
        };

        MatrixData {
            constraints,
            b,
            ranges,
            nr_equality_constraints,
            nr_range_constraints,
            nr_upper_bounded_constraints,
            nr_lower_bounded_constraints,
            row_group_end,
            column_group_end,
            variables,
            non_slack_variable_index_to_bound_index,
            bound_index_to_non_slack_variable_index,
        }
    }

    /// Classify a column by type using the column index.
    ///
    /// See the struct documentation for a visualization.
    fn column_type(&self, j: usize) -> (ColumnType, usize) {
        debug_assert!(j < self.nr_columns());

        // TODO(OPTIMIZATION): Binary search might be faster, but it would need to select the
        //  left-most element of equal elements in the cumulative group indices.

        if j < self.column_group_end[ColumnType::Normal] {
            (ColumnType::Normal, j - 0)
        } else if j < self.column_group_end[ColumnType::RangeSlack] {
            (ColumnType::RangeSlack, j - self.column_group_end[ColumnType::Normal])
        } else if j < self.column_group_end[ColumnType::UpperInequalitySlack] {
            (ColumnType::UpperInequalitySlack, j - self.column_group_end[ColumnType::RangeSlack])
        } else if j < self.column_group_end[ColumnType::LowerInequalitySlack] {
            (ColumnType::LowerInequalitySlack, j - self.column_group_end[ColumnType::UpperInequalitySlack])
        } else if j < self.column_group_end[ColumnType::VariableBoundSlack] {
            (ColumnType::VariableBoundSlack, j - self.column_group_end[ColumnType::LowerInequalitySlack])
        } else {
            // j < self.column_type_indices[ColumnType::VariableBoundSlack] == self.nr_columns()
            (ColumnType::SlackBoundSlack, j - self.column_group_end[ColumnType::VariableBoundSlack])
        }
    }

    /// The number of variables for which we want a solution.
    fn nr_normal_variables(&self) -> usize {
        self.constraints.nr_columns()
    }
}

impl<'data, F: 'static> MatrixProvider for MatrixData<'data, F>
where
    F: Field + SparseElement<F> + NonZero,
    for<'r> &'r F: FieldRef<F>,
{
    type Column = Column<F>;
    type Cost<'a> = Option<&'a <Self::Column as ColumnTrait>::F>;
    type Rhs = F;

    #[inline]
    fn column(&self, j: usize) -> Self::Column {
        debug_assert!(j < self.nr_columns());

        // TODO(ARCHITECTURE): Can the +/- F::one() constants be avoided? They might be large and
        //  require an allocation.
        let (column_type, j) = self.column_type(j);
        match column_type {
            ColumnType::Normal => {
                Column::Sparse {
                    // TODO(ENHANCEMENT): Avoid this cloning by using references (needs the GAT
                    //  feature to be more mature).
                    constraint_values: self.constraints.iter_column(j).cloned().collect(),
                    slack: self.bound_row_index(j, BoundDirection::Upper)
                        .map(|i| 0 + i)
                        .map(|i| [(i, F::one())]),
                    mock_array: [],
                }
            }
            ColumnType::RangeSlack => Column::TwoSlack([
                    (self.row_group_end[RowType::EqualityConstraint] + j, F::one()),
                    (self.row_group_end[RowType::VariableBound] + j, F::one()),
                ],
                [],
            ),
            ColumnType::UpperInequalitySlack => {
                let row_index = self.row_group_end[RowType::RangeConstraint] + j;
                Column::Slack([(row_index, F::one())], [])
            }
            ColumnType::LowerInequalitySlack => {
                let row_index = self.row_group_end[RowType::UpperInequalityConstraint] + j;
                Column::Slack([(row_index, -F::one())], [])
            }
            ColumnType::VariableBoundSlack => {
                let row_index = self.row_group_end[RowType::LowerInequalityConstraint] + j;
                Column::Slack([(row_index, F::one())], [])
            }
            ColumnType::SlackBoundSlack => {
                let row_index = self.row_group_end[RowType::VariableBound] + j;
                Column::Slack([(row_index, F::one())], [])
            }
        }
    }

    fn cost_value(&self, j: usize) -> Self::Cost<'_> {
        debug_assert!(j < self.nr_columns());

        let (column_type, j) = self.column_type(j);
        match column_type {
            ColumnType::Normal => Some(&self.variables[j].cost),
            _ => None,
        }
    }

    fn right_hand_side(&self) -> DenseVector<Self::Rhs> {
        // TODO(ENHANCEMENT): Avoid this cloning

        let mut values = self.b.clone();
        values.extend_with_values(
            self.bound_index_to_non_slack_variable_index.iter()
                .map(|&j| self.variables[j].upper_bound.clone().unwrap())
                .collect()
        );
        let ranges = self.ranges.iter().map(|&v| v.clone()).collect();
        values.extend_with_values(ranges);
        values
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
            BoundDirection::Upper => {
                let (column_type, j) = self.column_type(j);
                match column_type {
                    ColumnType::Normal => {
                        self.non_slack_variable_index_to_bound_index[j]
                            .map(|index| self.row_group_end[RowType::LowerInequalityConstraint] + index)
                    },
                    ColumnType::RangeSlack => {
                        Some(self.row_group_end[RowType::VariableBound] + j)
                    },
                    _ => None,
                }
            }
        }
    }

    fn nr_constraints(&self) -> usize {
        self.row_group_end[RowType::LowerInequalityConstraint]
        // == self.nr_equality_constraints
        //     + self.nr_range_constraints
        //     + self.nr_upper_bounded_constraints
        //     + self.nr_lower_bounded_constraints
    }

    fn nr_variable_bounds(&self) -> usize {
        self.bound_index_to_non_slack_variable_index.len() + self.nr_range_constraints
    }

    fn nr_columns(&self) -> usize {
        self.column_group_end[ColumnType::SlackBoundSlack]
        // == self.nr_normal_variables()
        //     + self.nr_range_constraints
        //     + self.nr_upper_bounded_constraints
        //     + self.nr_lower_bounded_constraints
        //     + self.nr_variable_bounds()
    }

    fn reconstruct_solution<G>(&self, mut column_values: SparseVector<G, G>) -> SparseVector<G, G>
    where
        G: SparseElement<G> + SparseComparator,
    {
        debug_assert_eq!(column_values.len(), self.nr_columns());

        let to_remove = (self.nr_normal_variables()..self.nr_columns()).collect::<Vec<_>>();
        column_values.remove_indices(&to_remove);
        column_values
    }
}

impl<'a, F: 'static> PartialInitialBasis for MatrixData<'a, F>
where
    F: SparseElement<F> + Field + NonZero,
    for <'r> &'r F: FieldRef<F>,
{
    fn pivot_element_indices(&self) -> Vec<(usize, usize)> {
        // TODO(ENHANCEMENT): Use also the pivot values from the range slack variables. This
        //  requires that the range constraint rows are subtracted from the range slack variable
        //  bound rows, such that identity vectors appear in the "second group". Note that this is
        //  not always possible: the range needs to be larger than the range upper bound for the
        //  right hand side to remain nonnegative.
        let upper = (0..self.nr_upper_bounded_constraints)
            .map(|j| {
                let row = self.row_group_end[RowType::RangeConstraint] + j;
                let column = self.column_group_end[ColumnType::RangeSlack] + j;
                (row, column)
            });
        let variable = (0..self.bound_index_to_non_slack_variable_index.len())
            .map(|j| {
                let row = self.row_group_end[RowType::LowerInequalityConstraint] + j;
                let column = self.column_group_end[ColumnType::LowerInequalitySlack] + j;
                (row, column)
            });
        let slack = (0..self.nr_range_constraints)
            .map(|j| {
                let row = self.row_group_end[RowType::VariableBound] + j;
                let column = self.column_group_end[ColumnType::VariableBoundSlack] + j;
                (row, column)
            });

        upper.chain(variable).chain(slack).collect()
    }

    fn nr_initial_elements(&self) -> usize {
        self.nr_upper_bounded_constraints + self.nr_variable_bounds()
    }
}

/// Describes a column from a `MatrixData` struct.
///
/// Either a vector with (probably several) elements, or a slack column. They are always sparsely
/// represented.
///
/// TODO(ARCHITECTURE): Can this type be simplified?
#[derive(Eq, PartialEq, Clone, Debug)]
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
        slack: Option<[(usize, F); 1]>, // Is always a positive slack: `1`.
        /// Used for creating a chaining iterator if there is no slack.
        mock_array: [(usize, F); 0],
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
    Slack([(usize, F); 1], [(usize, F); 0]),
    /// For range variables that have exactly two `1`'s in their column.
    TwoSlack([(usize, F); 2], [(usize, F); 0]),
}

impl<F: 'static> IdentityColumn for Column<F>
where
    F: Field,
{
    #[must_use]
    fn identity(i: usize, len: usize) -> Self {
        assert!(i < len);

        Self::Slack([(i, F::one())], [])
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
    type Iter<'a> = impl Iterator<Item = &'a SparseTuple<F>> + Clone;

    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        match self {
            Column::Sparse {
                constraint_values,
                slack,
                mock_array
            } => {
                match slack {
                    Some(slack) => constraint_values.iter().chain(slack.iter()),
                    None => constraint_values.iter().chain(mock_array.iter()),
                }
            },
            Column::Slack(single_value, mock_array) =>
                single_value.iter().chain(mock_array.iter()),
            Column::TwoSlack(two_values, mock_array) =>
                two_values.iter().chain(mock_array.iter()),
        }
    }

    fn index_to_string(&self, i: usize) -> String {
        match &self {
            Column::Sparse { constraint_values, slack, .. } => {
                let value = constraint_values.iter()
                    .find(|&&(index, _)| index == i);
                match value {
                    None => match slack {
                        Some([(index, v)]) if *index == i => v.to_string(),
                        _ => "0".to_string(),
                    }
                    Some((_, v)) => v.to_string(),
                }
            },
            Column::Slack([(index, value)], ..) => {
                if *index == i {
                    value.to_string()
                } else {
                    "0".to_string()
                }
            },
            Column::TwoSlack([(index1, value1), (index2, value2)], ..) => {
                if *index1 == i {
                    value1.to_string()
                } else if *index2 == i {
                    value2.to_string()
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

impl<F: 'static> IntoFilteredColumn for Column<F>
where
    F: Field,
{
    type Filtered = Column<F>;

    fn into_filtered(mut self, to_filter: &[usize]) -> Self::Filtered {
        debug_assert!(to_filter.is_sorted());

        // Note that the only rows that can be linearly dependent are the equality rows. Because
        // slack variables are always at a higher index, we are certain that all removed rows are
        // below that index, so we can shift it with a fixed amount.
        let shift = to_filter.len();
        match &mut self {
            Column::Sparse { constraint_values, slack, .. } => {
                remove_sparse_indices(constraint_values, to_filter);
                if let Some([(i, _)]) = slack {
                    *i -= shift;
                }
            },
            Column::Slack([(row_index, _)], ..) => *row_index -= shift,
            Column::TwoSlack([(row1, _), (row2, _)], ..) => {
                *row1 -= shift;
                *row2 -= shift;
            }
        }

        self
    }
}

impl<'a, F: 'static> Display for MatrixData<'a, F>
where
    F: Field + SparseElement<F> + NonZero + 'a,
    for<'r> &'r F: FieldRef<F>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let width = 8;
        let separator_width = (1 + self.nr_columns()) * width + NR_GROUPS;

        write!(f, "{:>width$}", "|", width = width)?;
        for column in 0..self.nr_columns() {
            if column == 0 || self.column_group_end.values().any(|&j| j == column) {
                write!(f, "|")?;
            }
            write!(f, "{:^width$}", column, width = width)?;
        }
        writeln!(f)?;
        f.write_str(&"=".repeat(separator_width))?;
        write!(f, "{:>width$}", "cost |", width = width)?;
        for column in 0..self.nr_columns() {
            if column == 0 || self.column_group_end.values().any(|&j| j == column) {
                write!(f, "|")?;
            }
            let cost = self.cost_value(column).map_or("0".to_string(), ToString::to_string);
            write!(f, "{:^width$}", cost, width = width)?;
        }
        writeln!(f)?;
        f.write_str(&"=".repeat(separator_width))?;

        for row in 0..self.nr_rows() {
            if row == 0 || self.row_group_end.values().any(|&i| i == row) {
                // Start of new section
                f.write_str(&"-".repeat(separator_width))?;
            }
            write!(f, "{:>width$}", format!("{} |", row), width = width)?;
            for column in 0..self.nr_columns() {
                if column == 0 || self.column_group_end.values().any(|&j| j == column) {
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
    use relp_num::R64;

    use crate::algorithm::two_phase::matrix_provider::matrix_data::Column;
    use crate::algorithm::two_phase::matrix_provider::MatrixProvider;
    use crate::data::linear_algebra::vector::DenseVector;
    use crate::data::linear_algebra::vector::test::TestVector;
    use crate::data::linear_program::elements::BoundDirection;
    use crate::tests::problem_1;

    #[test]
    fn from_general_form() {
        let (constraints, b, variables) = problem_1::create_matrix_data_data();
        let matrix_data = problem_1::matrix_data_form(&constraints, &b, &variables);

        assert_eq!(matrix_data.nr_normal_variables(), 3);

        // Variable column with bound constant
        assert_eq!(
            matrix_data.column(0),
            Column::Sparse {
                constraint_values: vec![(1, R64!(1))],
                slack: Some([(2, R64!(1))]),
                mock_array: [],
            },
        );
        // Variable column without bound constant
        assert_eq!(
            matrix_data.column(2),
            Column::Sparse {
                constraint_values: vec![(0,  R64!(1)), (1, R64!(1))],
                slack: None,
                mock_array: [],
            },
        );
        // Upper bounded inequality slack
        assert_eq!(
            matrix_data.column(3),
            Column::Slack([(1, R64!(-1))], []),
        );
        // Variable upper bound slack
        assert_eq!(
            matrix_data.column(4),
            Column::Slack([(2, R64!(1))], []),
        );
        // Lower bounded inequality slack
        assert_eq!(
            matrix_data.column(5),
            Column::Slack([(3, R64!(1))], []),
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
            matrix_data.right_hand_side(),
            DenseVector::from_test_data(vec![6, 10, 4, 2]),
        );

        assert_eq!(matrix_data.bound_row_index(0, BoundDirection::Upper), Some(2));
        assert_eq!(matrix_data.bound_row_index(2, BoundDirection::Upper), None);
        assert_eq!(matrix_data.bound_row_index(3, BoundDirection::Upper), None);
        assert_eq!(matrix_data.bound_row_index(4, BoundDirection::Upper), None);
        assert_eq!(matrix_data.bound_row_index(5, BoundDirection::Upper), None);
        assert_eq!(matrix_data.bound_row_index(0, BoundDirection::Lower), None);
        assert_eq!(matrix_data.bound_row_index(3, BoundDirection::Lower), None);
        assert_eq!(matrix_data.bound_row_index(4, BoundDirection::Lower), None);
        assert_eq!(matrix_data.bound_row_index(5, BoundDirection::Lower), None);

        assert_eq!(matrix_data.nr_constraints(), 2);
        assert_eq!(matrix_data.nr_variable_bounds(), 2);
        assert_eq!(matrix_data.nr_rows(), 4);
        assert_eq!(matrix_data.nr_columns(), 6);
    }
}
