//! # Simple matrix data
//!
//! A combination of a sparse matrix of constraints and a list of upper bounds for variables.
use std::fmt::{Display, Formatter};
use std::fmt;

use itertools::repeat_n;

use crate::algorithm::simplex::matrix_provider::{Column, MatrixProvider};
use crate::data::linear_algebra::matrix::{ColumnMajor, Sparse};
use crate::data::linear_algebra::traits::SparseElementZero;
use crate::data::linear_algebra::vector::{Dense, Sparse as SparseVector, Vector};
use crate::data::linear_program::elements::{BoundDirection, VariableType};
use crate::data::number_types::traits::{Field, FieldRef};

/// Describes a linear program using a combination of a sparse matrix of constraints, and a vector
/// with simple variable bounds.
///
/// Created once from a `GeneralForm`. Should allow for reasonably quick data access.
///
/// TODO: Is there a faster constraint storage backend possible?
/// TODO: Should this data structure hold less references?
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
/// ---------------------------------------------------------------------------------------------------------------
#[allow(non_snake_case)]
#[derive(Debug, PartialEq)]
pub struct MatrixData<'a, F: Field, FZ: SparseElementZero<F>> {
    /// Coefficient matrix.
    ///
    /// This should not contain variable bounds.
    constraints: &'a Sparse<F, FZ, F, ColumnMajor>,
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

    /// Constants to refer to when retuning SparseVector<&F> types.
    ONE: F,
    MINUS_ONE: F,
    ZERO: F,
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

impl<'a, F, FZ> MatrixData<'a, F, FZ>
where
    F: Field,
    for<'r> &'r F: FieldRef<F>,
    FZ: SparseElementZero<F>,
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
    pub fn new(
        constraints: &'a Sparse<F, FZ, F, ColumnMajor>,
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

            ONE: F::one(),
            MINUS_ONE: -F::one(),
            ZERO: F::zero(),
        }
    }

    /// Classify a column by type using the column index.
    ///
    /// See the struct documentation for a visualization.
    fn column_type(&self, j: usize) -> ColumnType {
        debug_assert!(j < self.nr_columns());

        let separating_indices = self.get_variable_separating_indices();
        debug_assert_eq!(separating_indices[3], self.nr_columns());

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

impl<'a, F, FZ> MatrixProvider<F, FZ> for MatrixData<'a, F, FZ>
where
    F: Field,
    for<'r> &'r F: FieldRef<F>,
    FZ: SparseElementZero<F>,
{
    fn column(&self, j: usize) -> Column<&F, FZ, F> {
        debug_assert!(j < self.nr_columns());

        match self.column_type(j) {
            ColumnType::Normal(j) => {
                // TODO: Avoid cloning, allocating (need existential type on matrix provider?)
                let mut tuples = self.constraints.iter_column(j)
                    .map(|(i, v)| (*i, v)).collect::<Vec<_>>();
                if let Some(bound_row_index) = self.bound_row_index(j, BoundDirection::Upper) {
                    tuples.push((bound_row_index, &self.ONE)); // It's always an upper bound, so `1f64`
                }
                Column::Sparse(SparseVector::new(tuples, self.nr_rows()))
            },
            ColumnType::UpperInequalitySlack(j) => {
                let row_index = self.nr_equality_constraints + j;
                Column::Slack(row_index, BoundDirection::Upper)
            },
            ColumnType::LowerInequalitySlack(j) => {
                let row_index = self.nr_equality_constraints + self.nr_upper_bounded_constraints + j;
                Column::Slack(row_index, BoundDirection::Lower)
            },
            ColumnType::BoundSlack(j) => {
                let row_index = self.nr_constraints() + j;
                Column::Slack(row_index, BoundDirection::Upper)
            },
        }
    }

    fn cost_value(&self, j: usize) -> &F {
        debug_assert!(j < self.nr_columns());

        match self.column_type(j) {
            ColumnType::Normal(j) => &self.variables[j].cost,
            _ => &self.ZERO,
        }
    }

    fn constraint_values(&self) -> Dense<F> {
        let mut constraint_coefficients = self.b.clone();
        constraint_coefficients.extend_with_values(self.bound_index_to_non_slack_variable_index.iter()
            .map(|&j| self.variables[j].upper_bound.clone().unwrap()).collect());
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

    fn bounds(&self, j: usize) -> (&F, &Option<F>) {
        debug_assert!(j < self.nr_columns());

        (&self.ZERO, match self.column_type(j) {
            ColumnType::Normal(j) => &self.variables[j].upper_bound,
            _ => &None,
        })
    }

    fn positive_slack_indices(&self) -> Vec<(usize, usize)> {
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

    fn nr_positive_slacks(&self) -> usize {
        self.nr_upper_bounded_constraints + self.nr_bounds()
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

    fn reconstruct_solution<FZ2: SparseElementZero<F>>(
        &self, mut column_values: SparseVector<F, FZ2, F>,
    ) -> SparseVector<F, FZ2, F> {
        debug_assert_eq!(column_values.len(), self.nr_columns());

        column_values.remove_indices(&(self.nr_normal_variables()..self.nr_columns()).collect());
        column_values
    }
}

impl<'a, F, FZ> Display for MatrixData<'a, F, FZ>
where
    F: Field + 'a,
    for<'r> &'r F: FieldRef<F>,
    FZ: SparseElementZero<F>,
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
            write!(f, "{:^width$}", format!("{}", self.cost_value(column)), width = width)?;
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
                let value = match self.column(column) {
                    Column::Sparse(vector) => vector[row].clone(),
                    Column::Slack(index, direction) => if row == index {
                        direction.into()
                    } else {
                        F::zero()
                    },
                };
                write!(f, "{:^width$}", format!("{}", value), width = width)?;
            }
            writeln!(f)?;
        }
        writeln!(f)
    }
}

#[cfg(test)]
mod test {
    use num::rational::Ratio;
    use num::traits::FromPrimitive;

    use crate::algorithm::simplex::matrix_provider::Column;
    use crate::algorithm::simplex::matrix_provider::MatrixProvider;
    use crate::data::linear_algebra::vector::{Dense, Sparse as SparseVector, Vector};
    use crate::data::linear_algebra::vector::test::TestVector;
    use crate::data::linear_program::elements::BoundDirection;
    use crate::R32;
    use crate::tests::problem_1;

    #[test]
    fn from_general_form() {
        let (constraints, b) = problem_1::create_matrix_data_data();
        let matrix_data = problem_1::matrix_data_form(&constraints, &b);

        assert_eq!(matrix_data.nr_normal_variables(), 3);

        // // Variables with bound
        assert_eq!(matrix_data.bounds(0), (&R32!(0), &Some(R32!(2))));
        assert_eq!(matrix_data.bounds(1), (&R32!(0), &Some(R32!(2))));
        // Variable without bound
        assert_eq!(matrix_data.bounds(2), (&R32!(0), &Some(R32!(2))));
        // Slack variable, as such without bound
        assert_eq!(matrix_data.bounds(3), (&R32!(0), &None));

        // Variable column with bound constant
        assert_eq!(
            matrix_data.column(0),
            Column::Sparse(SparseVector::new(vec![(1, &R32!(1)), (2,  &R32!(1))], 5)),
        );
        // Variable column without bound constant
        assert_eq!(
            matrix_data.column(2),
            Column::Sparse(SparseVector::new(vec![(0,  &R32!(1)), (1, &R32!(1)), (4, &R32!(1))], 5)),
        );
        // Upper bounded inequality slack
        assert_eq!(
            matrix_data.column(3),
            Column::Slack(1, BoundDirection::Lower),
        );
        // Lower bounded inequality slack
        assert_eq!(
            matrix_data.column(4),
            Column::Slack(2, BoundDirection::Upper),
        );
        // Variable upper bound slack
        assert_eq!(
            matrix_data.column(5),
            Column::Slack(3, BoundDirection::Upper),
        );
        assert_eq!(
            matrix_data.column(6),
            Column::Slack(4, BoundDirection::Upper),
        );

        // Variable cost
        assert_eq!(matrix_data.cost_value(0),  &R32!(1));
        // Upper bounded inequality slack variable cost
        assert_eq!(matrix_data.cost_value(3), &R32!(0));
        // Lower bounded inequality slack variable cost
        assert_eq!(matrix_data.cost_value(4), &R32!(0));
        // Bound slack variable cost
        assert_eq!(matrix_data.cost_value(5), &R32!(0));

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

        assert_eq!(matrix_data.bounds(0), (&R32!(0), &Some(R32!(2))));
        assert_eq!(matrix_data.bounds(2), (&R32!(0), &Some(R32!(2))));
        assert_eq!(matrix_data.bounds(3), (&R32!(0), &None));
        assert_eq!(matrix_data.bounds(4), (&R32!(0), &None));
        assert_eq!(matrix_data.bounds(5), (&R32!(0), &None));

        assert_eq!(matrix_data.nr_constraints(), 2);
        assert_eq!(matrix_data.nr_bounds(), 3);
        assert_eq!(matrix_data.nr_rows(), 5);
        assert_eq!(matrix_data.nr_columns(), 7);
    }
}
