//! # Simple matrix data
//!
//! A combination of a sparse matrix of constraints and a list of upper bounds for variables.
use std::collections::HashSet;
use std::fmt::{Display, Formatter, Result as FormatResult};

use crate::algorithm::simplex::matrix_provider::{MatrixProvider, VariableFeasibilityLogic, BoundType};
use crate::data::linear_algebra::matrix::{ColumnMajorOrdering, SparseMatrix};
use crate::data::linear_algebra::SparseTuples;
use crate::data::linear_algebra::vector::{DenseVector, SparseVector, Vector};
use crate::data::linear_program::elements::VariableType;
use crate::data::number_types::traits::{Field, OrderedField, RealField};

/// Describes a linear program using a combination of a sparse matrix of constraints, and a vector
/// with simple variable bounds.
///
/// Created once from `CanonicalForm`. Should allow for quick data access.
///
/// The indexing for the variables and constraints is as follows:
///
/// /                           || Vars of which we want a solution | Free variable dummies | Inequality slack vars |   Bound slack vars   |
/// ----------------------------||----------------------------------|-----------------------|-----------------------|----------------------|
/// ----------------------------||----------------------------------|-----------------------|-----------------------|----------------------| -----
/// Eqquality constraints       ||            constants             |       constants       |           0           |           0          | |   |
/// ----------------------------||----------------------------------|-----------------------|-----------------------|----------------------| |   |
/// Inequality (<=) constraints ||            constants             |       constants       |     I     |     0     |           0          | | b |
/// ----------------------------||----------------------------------|-----------------------|-----------------------|----------------------| |   |
/// Inequality (>=) constraints ||            constants             |       constants       |     0     |    - I    |           0          | |   |
/// ----------------------------||----------------------------------|-----------------------|-----------------------|----------------------| |---|
///                             ||                                  |                       |                       |  +/- 1               |
/// Bound constraints           ||    constants (one 1 per row)     |           0           |           0           |         +/- 1        |
///                             ||                                  |                       |                       |                +/- 1 |
/// ----------------------------------------------------------------------------------------------------------------------------------------
#[derive(Debug, PartialEq)]
pub struct MatrixData<F: Field> {
    /// Coefficient matrix.
    ///
    /// This should not contain variable bounds.
    constraints: SparseMatrix<F, ColumnMajorOrdering>,
    /// Constraint values for the constraints, excludes the simple bounds.
    b: DenseVector<F>,
    /// How many of which constraint do we have?
    nr_equality_constraints: usize,
    nr_upper_bounded_constraints: usize,
    nr_lower_bounded_constraints: usize,

    variables: Vec<Variable<F>>,
    negative_free_variable_dummy_index: Vec<usize>,

    ///  If there is an upper bound, the "virtual constraint row" index is given together with the
    ///  bound value.
    ///
    /// (non-slack variable -> bound index)
    non_slack_variable_index_to_bound_index: Vec<Option<usize>>,
    /// (bound index -> non-slack variable)
    bound_index_to_non_slack_variable_index: Vec<usize>,
}

/// Variable for a canonical linear program.
///
/// Has a lower bound of zero.
#[allow(missing_docs)]
#[derive(Eq, PartialEq, Copy, Clone, Debug)]
pub struct Variable<F> {
    pub cost: F,
    pub upper_bound: Option<F>,
    pub variable_type: VariableType,
}
impl<RF: RealField> Variable<RF> {
    fn is_feasible(&self, value: RF) -> bool {
        value >= RF::zero() && {
            if let Some(bound) = self.upper_bound {
                value <= bound
            } else { true }
        } && {
            self.variable_type == VariableType::Continuous || value == value.round()
        }
    }

    fn closest_feasible(&self, value: RF) -> (Option<RF>, Option<RF>) {
        if value < RF::zero() {
            (None, Some(RF::zero()))
        } else {
            if let Some(bound) = self.upper_bound {
                if value > bound {
                    (Some(match self.variable_type {
                        VariableType::Continuous => bound,
                        VariableType::Integer => bound.floor(),
                    }), None)
                } else {
                    match self.variable_type {
                        VariableType::Continuous => (Some(value), Some(value)),
                        VariableType::Integer => (Some(value.floor()), if value.ceil() <= bound { Some(value.ceil()) } else { None }),
                    }
                }
            } else {
                match self.variable_type {
                    VariableType::Continuous => (Some(value), Some(value)),
                    VariableType::Integer => (Some(value.floor()), Some(value.floor())),
                }
            }
        }
    }
}

enum ColumnType {
    NormalVariable(usize),
    FreeVariableDummy(usize),
    UpperInequalitySlackVariable(usize),
    LowerInequalitySlackVariable(usize),
    BoundSlackVariable(usize),
}

impl<F: Field> MatrixData<F> {
    /// Create a new `MatrixData` struct.
    ///
    /// # Arguments
    ///
    /// * `data` - Sparse representation of constraints, which excludes simple bounds.
    /// * `cost` - Sparse representation of the cost function.
    /// * `b` - Constraint values.
    /// * `upper_bound` - For each variable,
    pub fn new(
        equality_constraints: Vec<SparseTuples<F>>,
        upper_bounded_constraints: Vec<SparseTuples<F>>,
        lower_bounded_constraints: Vec<SparseTuples<F>>,
        b: DenseVector<F>,
        variables: Vec<Variable<F>>,
        negative_free_variable_dummy_index: Vec<usize>,
    ) -> Self {
        let nr_equality_constraints = equality_constraints.len();
        let nr_upper_bounded_constraints = upper_bounded_constraints.len();
        let nr_lower_bounded_constraints = lower_bounded_constraints.len();
        let nr_constraints = nr_equality_constraints
            + nr_upper_bounded_constraints
            + nr_lower_bounded_constraints;
        let nr_non_slack_variables = variables.len();

        debug_assert_eq!(b.len(), nr_constraints);

        let constraints: SparseMatrix<_, ColumnMajorOrdering> = SparseMatrix::from_row_ordered_tuples_although_this_is_expensive(
            &[equality_constraints, upper_bounded_constraints, lower_bounded_constraints].concat(),
            nr_non_slack_variables,
        );
        debug_assert_eq!(constraints.nr_rows(), nr_constraints);
        debug_assert_eq!(constraints.nr_columns(), nr_non_slack_variables);

        debug_assert!(negative_free_variable_dummy_index.iter().max() < Some(&constraints.nr_columns()));
        debug_assert_eq!(
            negative_free_variable_dummy_index.iter().collect::<HashSet<_>>().len(),
            negative_free_variable_dummy_index.len(),
        );
        debug_assert!(negative_free_variable_dummy_index.is_sorted());

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
            negative_free_variable_dummy_index,
            non_slack_variable_index_to_bound_index,
            bound_index_to_non_slack_variable_index,
        }
    }

    fn column_type(&self, j: usize) -> ColumnType {
        let normal = self.nr_normal_variables();
        let normal_free = normal + self.nr_free_variable_dummies();
        let normal_free_upper = normal_free + self.nr_upper_bounded_constraints;
        let normal_free_upper_lower = normal_free_upper + self.nr_lower_bounded_constraints;
        let normal_free_upper_lower_bound = normal_free_upper_lower + self.nr_bounds();

        if j < normal {
            ColumnType::NormalVariable(j)
        } else if j < normal_free {
            ColumnType::FreeVariableDummy(j - normal)
        } else if j < normal_free_upper {
            ColumnType::UpperInequalitySlackVariable(j - normal_free)
        } else if j < normal_free_upper_lower {
            ColumnType::LowerInequalitySlackVariable(j - normal_free_upper)
        } else if j < normal_free_upper_lower_bound {
            ColumnType::BoundSlackVariable(j - normal_free_upper_lower)
        } else {
            panic!()
        }
    }

    fn nr_normal_variables(&self) -> usize {
        self.constraints.nr_columns()
    }

    fn nr_free_variable_dummies(&self) -> usize {
        self.negative_free_variable_dummy_index.len()
    }
}

impl<F: Field> MatrixProvider<F> for MatrixData<F> {
    fn column(&self, j: usize) -> SparseVector<F> {
        debug_assert!(j < self.nr_columns());

        match self.column_type(j) {
            ColumnType::NormalVariable(j) => {
                let mut tuples = self.constraints.iter_column(j).map(|&(i, v)| (i, v)).collect::<Vec<_>>();
                if let Some(bound_row_index) = self.bound_row_index(j, BoundType::Upper) {
                    tuples.push((bound_row_index, F::multiplicative_identity())); // It's always an upper bound, so `1f64`
                }
                SparseVector::new(tuples, self.nr_rows())
            },
            ColumnType::FreeVariableDummy(j) => {
                let mut copy_modifying = self.column(j);
                copy_modifying.element_wise_multiply(-F::multiplicative_identity());
                copy_modifying
            },
            ColumnType::UpperInequalitySlackVariable(j) => {
                let row_index = self.nr_equality_constraints + j;
                SparseVector::standard_basis_vector(row_index, self.nr_rows())
            },
            ColumnType::LowerInequalitySlackVariable(j) => {
                let row_index = self.nr_equality_constraints + self.nr_upper_bounded_constraints + j;
                let mut column = SparseVector::standard_basis_vector(row_index, self.nr_rows());
                column.element_wise_multiply(-F::multiplicative_identity());
                column
            },
            ColumnType::BoundSlackVariable(j) => {
                let row_index = self.nr_constraints() + j;
                SparseVector::standard_basis_vector(row_index, self.nr_rows()) // Slack coefficients are always `1f64`
            },
        }
    }

    fn cost_value(&self, j: usize) -> F {
        debug_assert!(j < self.nr_columns());

        match self.column_type(j) {
            ColumnType::NormalVariable(j) => self.variables[j].cost,
            _ => F::additive_identity(),
        }
    }

    fn constraint_values(&self) -> DenseVector<F> {
        let mut constraint_coefficients = self.b.clone();
        constraint_coefficients.extend_with_values(self.bound_index_to_non_slack_variable_index.iter()
            .map(|&j| self.variables[j].upper_bound.unwrap()).collect());
        constraint_coefficients
    }

    fn bound_row_index(
        &self,
        j: usize,
        bound_type: BoundType,
    ) -> Option<usize> {
        debug_assert!(j < self.nr_columns());

        match bound_type {
            BoundType::Lower => None,
            BoundType::Upper => match self.column_type(j) {
                ColumnType::NormalVariable(j) => {
                    self.non_slack_variable_index_to_bound_index[j]
                        .map(|index| self.nr_constraints() + index)
                },
                _ => None,
            }
        }
    }

    fn bounds(&self, j: usize) -> (F, Option<F>) {
        debug_assert!(j < self.nr_columns());

        (F::additive_identity(), match self.column_type(j) {
            ColumnType::NormalVariable(j) => self.variables[j].upper_bound,
            _ => None,
        })
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
            + self.nr_free_variable_dummies()
            + self.nr_upper_bounded_constraints
            + self.nr_lower_bounded_constraints
            + self.nr_bounds()
    }
}

impl<RF: RealField> VariableFeasibilityLogic<RF> for MatrixData<RF> {
    fn is_feasible(&self, j: usize, value: RF) -> bool {
        debug_assert!(j < self.nr_columns());

        match self.column_type(j) {
            ColumnType::NormalVariable(j) => self.variables[j].is_feasible(value),
            _ => value >= RF::zero(),
        }
    }

    fn closest_feasible(&self, j: usize, value: RF) -> (Option<RF>, Option<RF>) {
        debug_assert!(j < self.nr_columns());

        match self.column_type(j) {
            ColumnType::NormalVariable(j) => self.variables[j].closest_feasible(value),
            _ => (
                if value >= RF::zero() {
                    Some(value)
                } else {
                    None
                },
                Some(RF::zero().max(value)),
            ),
        }
    }
}

impl<F: OrderedField> Display for MatrixData<F> {
    fn fmt(&self, f: &mut Formatter) -> FormatResult {
        writeln!(f, "Matrix Data")?;
        writeln!(f, "Rows: {}\tColumns: {}", self.nr_rows(), self.nr_columns())?;

        let column_width = 10;
        let counter_width = 5;
        // Column counter
        write!(f, "{0:width$}", "", width = counter_width)?;
        for column_index in 0..self.nr_columns() {
            write!(f, "{0:>width$}", column_index, width = column_width)?;
        }
        writeln!(f, "")?;

        // Row counter and row data
        for row_index in 0..self.nr_rows() {
            write!(f, "{0: <width$}", row_index, width = counter_width)?;
            for column_index in 0..self.nr_columns() {
                write!(f, "{0:>width$.5}", self.constraints.get_value(row_index, column_index), width = column_width)?;
            }
            writeln!(f, "")?;
        }
        write!(f, "")
    }
}

#[cfg(test)]
mod test {
    use num::rational::Ratio;
    use num::traits::FromPrimitive;

    use crate::algorithm::simplex::matrix_provider::{MatrixProvider, VariableFeasibilityLogic, BoundType};
    use crate::data::linear_algebra::vector::{DenseVector, SparseVector, Vector};
    use crate::data::linear_algebra::vector::test::TestVector;
    use crate::R32;
    use crate::tests::problem_1;

    #[test]
    fn from_general_form() {
        let matrix_data = problem_1::matrix_data_form();

        assert_eq!(matrix_data.nr_normal_variables(), 3);

        // // Variables with bound
        assert_eq!(matrix_data.bounds(0), (R32!(0), Some(R32!(4))));
        assert_eq!(matrix_data.bounds(1), (R32!(0), Some(R32!(2))));
        // Variable without bound
        assert_eq!(matrix_data.bounds(2), (R32!(0), None));
        // Slack variable, as such without bound
        assert_eq!(matrix_data.bounds(3), (R32!(0), None));

        // Variable column with bound constant
        assert_eq!(
            matrix_data.column(0),
            SparseVector::new(vec![(1, R32!(1)), (2,  R32!(1)), (3,  R32!(1))], 5),
        );
        // Variable column without bound constant
        assert_eq!(
            matrix_data.column(2),
            SparseVector::new(vec![(0,  R32!(1)), (2,  R32!(1))], 5),
        );
        // Upper bounded inequality slack
        assert_eq!(
            matrix_data.column(3),
            SparseVector::new(vec![(1,  R32!(1))], 5),
        );
        // Lower bounded inequality slack
        assert_eq!(
            matrix_data.column(4),
            SparseVector::new(vec![(2, - R32!(1))], 5),
        );
        // Variable upper bound slack
        assert_eq!(
            matrix_data.column(5),
            SparseVector::new(vec![(3,  R32!(1))], 5),
        );
        assert_eq!(
            matrix_data.column(6),
            SparseVector::new(vec![(4,  R32!(1))], 5),
        );

        // Variable cost
        assert_eq!(matrix_data.cost_value(0),  R32!(1));
        // Upper bounded inequality slack variable cost
        assert_eq!(matrix_data.cost_value(3), R32!(0));
        // Lower bounded inequality slack variable cost
        assert_eq!(matrix_data.cost_value(4), R32!(0));
        // Bound slack variable cost
        assert_eq!(matrix_data.cost_value(5), R32!(0));

        assert_eq!(
            matrix_data.constraint_values(),
            DenseVector::from_test_data(vec![6f64, 6f64, 10f64, 4f64, 2f64]),
        );

        assert_eq!(matrix_data.bound_row_index(0, BoundType::Upper), Some(3));
        assert_eq!(matrix_data.bound_row_index(2, BoundType::Upper), None);
        assert_eq!(matrix_data.bound_row_index(3, BoundType::Upper), None);
        assert_eq!(matrix_data.bound_row_index(4, BoundType::Upper), None);
        assert_eq!(matrix_data.bound_row_index(5, BoundType::Upper), None);
        assert_eq!(matrix_data.bound_row_index(0, BoundType::Lower), None);
        assert_eq!(matrix_data.bound_row_index(3, BoundType::Lower), None);
        assert_eq!(matrix_data.bound_row_index(4, BoundType::Lower), None);
        assert_eq!(matrix_data.bound_row_index(5, BoundType::Lower), None);

        assert_eq!(matrix_data.is_feasible(0, R32!(0)), true);
        assert_eq!(matrix_data.is_feasible(1, R32!(1)), true);
        assert_eq!(matrix_data.is_feasible(1, R32!(1.5)), false);
        assert_eq!(matrix_data.is_feasible(3, R32!(0.5)), true);
        assert_eq!(matrix_data.is_feasible(4, R32!(-1)), false);
        assert_eq!(matrix_data.is_feasible(5, R32!(2)), true);

        assert_eq!(matrix_data.bounds(0).0, R32!(0));
        assert_eq!(matrix_data.bounds(0).1.unwrap(), R32!(4));
        assert_eq!(matrix_data.bounds(2), (R32!(0), None));
        assert_eq!(matrix_data.bounds(3), (R32!(0), None));
        assert_eq!(matrix_data.bounds(4), (R32!(0), None));
        assert_eq!(matrix_data.bounds(5), (R32!(0), None));

        assert_eq!(matrix_data.nr_constraints(), 3);
        assert_eq!(matrix_data.nr_bounds(), 2);
        assert_eq!(matrix_data.nr_rows(), 5);
        assert_eq!(matrix_data.nr_columns(), 7);
    }
}
