//! # Simple matrix data
//!
//! A combination of a sparse matrix of constraints and a list of upper bounds for variables.
use std::collections::HashSet;
use std::fmt::{Display, Formatter, Result as FormatResult};

use crate::algorithm::simplex::matrix_provider::{BoundType, MatrixProvider, VariableFeasibilityLogic};
use crate::data::linear_algebra::matrix::{ColumnMajor, Sparse};
use crate::data::linear_algebra::SparseTuples;
use crate::data::linear_algebra::vector::{DenseVector, SparseVector, Vector};
use crate::data::linear_program::elements::VariableType;
use crate::data::number_types::traits::{Field, OrderedField, RealField};
use std::fmt;
use itertools::repeat_n;

/// Describes a linear program using a combination of a sparse matrix of constraints, and a vector
/// with simple variable bounds.
///
/// Created once from `CanonicalForm`. Should allow for quick data access.
///
/// The indexing for the variables and constraints is as follows:
///
/// /                           || Vars of which we want a solution | Free variable dummies | Inequality slack vars |   Bound slack vars   |
/// ============================||==================================|=======================|=======================|======================| -----
/// Equality constraints        ||            constants             |       constants       |           0           |           0          | |   |
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
pub struct MatrixData<'a, F: Field> {
    /// Coefficient matrix.
    ///
    /// This should not contain variable bounds.
    constraints: &'a Sparse<F, ColumnMajor>,
    /// Constraint values for the constraints, excludes the simple bounds.
    b: &'a DenseVector<F>,
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

impl<'a, F: Field> MatrixData<'a, F> {
    /// Create a new `MatrixData` struct.
    ///
    /// # Arguments
    ///
    /// TODO
    /// * `b`: Constraint values.
    /// * `negative_free_variable_dummy_index`: Index i contains the index of the i'th free
    /// variable.
    pub fn new(
        constraints: &'a Sparse<F, ColumnMajor>,
        b: &'a DenseVector<F>,
        nr_equality_constraints: usize,
        nr_upper_bounded_constraints: usize,
        nr_lower_bounded_constraints: usize,
        variables: Vec<Variable<F>>,
        negative_free_variable_dummy_index: Vec<usize>,
    ) -> Self {
        debug_assert_eq!(
            nr_equality_constraints + nr_upper_bounded_constraints + nr_lower_bounded_constraints,
            constraints.nr_rows(),
        );
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
        let separating_indices = self.get_variable_separating_indices();

        if j < separating_indices[0] {
            ColumnType::NormalVariable(j)
        } else if j < separating_indices[1] {
            ColumnType::FreeVariableDummy(j - separating_indices[0])
        } else if j < separating_indices[2] {
            ColumnType::UpperInequalitySlackVariable(j - separating_indices[1])
        } else if j < separating_indices[3] {
            ColumnType::LowerInequalitySlackVariable(j - separating_indices[2])
        } else if j < separating_indices[4] {
            ColumnType::BoundSlackVariable(j - separating_indices[3])
        } else {
            panic!()
        }
    }

    fn get_variable_separating_indices(&self) -> [usize; 5] {
        let normal = self.nr_normal_variables();
        let normal_free = normal + self.nr_free_variable_dummies();
        let normal_free_upper = normal_free + self.nr_upper_bounded_constraints;
        let normal_free_upper_lower = normal_free_upper + self.nr_lower_bounded_constraints;
        let normal_free_upper_lower_bound = normal_free_upper_lower + self.nr_bounds();
        debug_assert_eq!(normal_free_upper_lower_bound, self.nr_columns());

        [
            normal,
            normal_free,
            normal_free_upper,
            normal_free_upper_lower,
            normal_free_upper_lower_bound,
        ]
    }

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

    fn nr_free_variable_dummies(&self) -> usize {
        self.negative_free_variable_dummy_index.len()
    }
}

impl<'a, F: Field> MatrixProvider<F> for MatrixData<'a, F> {
    fn column(&self, j: usize) -> SparseVector<F> {
        debug_assert!(j < self.nr_columns());

        match self.column_type(j) {
            ColumnType::NormalVariable(j) => {
                let mut tuples = self.constraints.iter_column(j).copied().collect::<Vec<_>>();
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

    /// Index of row that represents a bound.
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

    fn reconstruct_solution(&self, column_values: SparseVector<F>) -> SparseVector<F> {
        debug_assert_eq!(column_values.len(), self.nr_columns());

        let mut out = SparseVector::new(
            Vec::with_capacity(self.nr_normal_variables()),
            self.nr_normal_variables(),
        );
        for (i, v) in column_values.values() {
            if i < self.nr_normal_variables() {
                out.set_value(i, v);
            } else if i < self.nr_normal_variables() + self.nr_free_variable_dummies() {
                out.shift_value(i, -v);
            } else {
                break
            }
        }

        out
    }
}

impl<'a, RF: RealField> VariableFeasibilityLogic<RF> for MatrixData<'a, RF> {
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

impl<'a, F: Field> Display for MatrixData<'a, F> {
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
                write!(f, "{:^width$}", format!("{}", self.column(column)[row]), width = width)?;
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

    use crate::algorithm::simplex::matrix_provider::{BoundType, MatrixProvider, VariableFeasibilityLogic};
    use crate::data::linear_algebra::vector::{DenseVector, SparseVector, Vector};
    use crate::data::linear_algebra::vector::test::TestVector;
    use crate::R32;
    use crate::tests::problem_1;

    #[test]
    fn from_general_form() {
        let (constraints, b) = problem_1::create_matrix_data_data();
        let matrix_data = problem_1::matrix_data_form(&constraints, &b);

        assert_eq!(matrix_data.nr_normal_variables(), 3);

        // // Variables with bound
        assert_eq!(matrix_data.bounds(0), (R32!(0), Some(R32!(2))));
        assert_eq!(matrix_data.bounds(1), (R32!(0), Some(R32!(2))));
        // Variable without bound
        assert_eq!(matrix_data.bounds(2), (R32!(0), Some(R32!(2))));
        // Slack variable, as such without bound
        assert_eq!(matrix_data.bounds(3), (R32!(0), None));

        // Variable column with bound constant
        assert_eq!(
            matrix_data.column(0),
            SparseVector::new(vec![(1, R32!(1)), (2,  R32!(1))], 5),
        );
        // Variable column without bound constant
        assert_eq!(
            matrix_data.column(2),
            SparseVector::new(vec![(0,  R32!(1)), (1, R32!(1)), (4, R32!(1))], 5),
        );
        // Upper bounded inequality slack
        assert_eq!(
            matrix_data.column(3),
            SparseVector::new(vec![(1,  R32!(-1))], 5),
        );
        // Lower bounded inequality slack
        assert_eq!(
            matrix_data.column(4),
            SparseVector::new(vec![(2, R32!(1))], 5),
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
            DenseVector::from_test_data(vec![0f64, 2f64, 2f64, 2f64, 2f64]),
        );

        assert_eq!(matrix_data.bound_row_index(0, BoundType::Upper), Some(2));
        assert_eq!(matrix_data.bound_row_index(2, BoundType::Upper), Some(4));
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

        assert_eq!(matrix_data.bounds(0), (R32!(0), Some(R32!(2))));
        assert_eq!(matrix_data.bounds(2), (R32!(0), Some(R32!(2))));
        assert_eq!(matrix_data.bounds(3), (R32!(0), None));
        assert_eq!(matrix_data.bounds(4), (R32!(0), None));
        assert_eq!(matrix_data.bounds(5), (R32!(0), None));

        assert_eq!(matrix_data.nr_constraints(), 2);
        assert_eq!(matrix_data.nr_bounds(), 3);
        assert_eq!(matrix_data.nr_rows(), 5);
        assert_eq!(matrix_data.nr_columns(), 7);
    }
}
