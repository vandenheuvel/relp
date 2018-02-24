use std::fmt::Debug;

use data::linear_algebra::matrix::{Matrix, SparseMatrix};
use data::linear_algebra::vector::{DenseVector, SparseVector, Vector};
use data::linear_program::canonical_form::CanonicalForm;
use data::linear_program::elements::{ConstraintType, Variable, VariableType};


/// Checks whether the dimensions of the GeneralForm are consistent.
/// Note: Use only debug asserts in this macro.
macro_rules! debug_assert_consistent {
    ($general_form:expr) => {
        debug_assert_eq!($general_form.b.len(), $general_form.data.nr_rows());
        debug_assert_eq!($general_form.row_info.len(), $general_form.data.nr_rows());

        debug_assert_eq!($general_form.cost.len(), $general_form.data.nr_columns());
        debug_assert_eq!($general_form.column_info.len(), $general_form.data.nr_columns());
    };
}

/// A linear program in general form.
#[derive(Debug, PartialEq)]
pub struct GeneralForm {
    /// All coefficients
    data: SparseMatrix,
    /// All right-hands sides of equations
    b: DenseVector,
    /// Constant in the cost function
    fixed_cost: f64,
    /// The linear combination of costs
    cost: SparseVector,

    /// The names of all variables and their type, ordered by index
    column_info: Vec<Variable>,
    /// The equation type of all rows, ordered by index
    row_info: Vec<ConstraintType>,
    /// Already known solution values
    solution_values: Vec<(String, f64)>,
}

impl GeneralForm {
    /// Create a new linear program in general form.
    pub fn new(data: SparseMatrix, b: DenseVector, cost: SparseVector, fixed_cost: f64,
               column_info: Vec<Variable>, solution_values: Vec<(String, f64)>,
               row_info: Vec<ConstraintType>) -> GeneralForm {

        let general_form = GeneralForm { data, b, fixed_cost, cost, column_info, row_info, solution_values, };
        debug_assert_consistent!(general_form);
        general_form
    }
    /// Convert this linear program into canonical form.
    pub fn to_canonical(&mut self) -> CanonicalForm {
        self.substitute_fixed();
        self.shift_variables();
        self.make_b_non_negative();
        self.introduce_slack_variables();

        CanonicalForm::new(self.data.clone(), self.b.clone(), self.cost.clone(), self.fixed_cost, self.column_info.clone(), self.solution_values.clone())
    }
    /// Substitute known variables in all constraints in which it is nonzero.
    fn substitute_fixed(&mut self) {
        let mut row = 0;
        while row < self.nr_rows() {
            if self.row_info[row] == ConstraintType::Equal && self.data.row(row).len() == 1 {
                let (column, value) = self.data.row(row).last().unwrap().to_owned();
                let fixed_value = self.b.get_value(row) / value;
                for (index, value) in self.data.column(column) {
                    let old_value = self.b.get_value(*index);
                    self.b.set_value(*index, old_value - value * fixed_value);
                }
                let name = self.column_info[column].name.clone();
                self.solution_values.push((name, fixed_value));

                self.data.remove_row(row);
                self.b.remove_value(row);
                self.row_info.remove(row);

                self.data.remove_column(column);
                self.fixed_cost += self.cost.get_value(column) * fixed_value;
                self.cost.remove_value(column);
                self.column_info.remove(column);
            }
            row += 1;

            debug_assert_consistent!(self);
        }
    }
    /// Shift all variables, such that the lower bound is zero.
    fn shift_variables(&mut self) {
        let mut i = 0;
        while i < self.nr_rows() {
            match self.is_lower_bound(i) {
                Some((column, bound_value)) => {
                    // TODO: Handle the existence of multiple lower bounds
                    self.shift_variable(column, bound_value);
                    self.data.remove_row(i);
                    self.b.remove_value(i);
                    self.row_info.remove(i);
                },
                None => i += 1,
            }

            debug_assert_consistent!(self);
        }
    }
    /// Shift a variable with a specific value.
    fn shift_variable(&mut self, j: usize, shift_value: f64) {
        for (row, value) in self.data.column(j) {
            let previous_b = self.b.get_value(*row);
            self.b.set_value(*row, previous_b - value * shift_value);
        }
        self.column_info[j].set_shift(shift_value);
        self.fixed_cost += self.cost.get_value(j) * shift_value;
    }
    /// Test whether a constraint is a lower bound.
    fn is_lower_bound(&self, i: usize) -> Option<(usize, f64)> {
        debug_assert!(i < self.nr_rows());

        if self.data.row(i).len() == 1 {
            let (column, value) = self.data.row(i).nth(0).unwrap();
            match (self.row_info[i], value) {
                (ConstraintType::Greater, v) if *v > 0f64 => Some((*column, self.b.get_value(i) / v)),
                (ConstraintType::Less, v) if *v < 0f64 => Some((*column, self.b.get_value(i) / v)),
                _ => return None,
            }
        } else {
            None
        }
    }
    /// Multiply the constraints such that the constraint value is >= 0.
    fn make_b_non_negative(&mut self) {
        for i in 0..self.nr_rows() {
            let bound_value = self.b.get_value(i);
            if bound_value < 0f64 {
                self.b.set_value(i, -bound_value);
                self.data.multiply_row(i, -1f64);
            }
        }

        debug_assert_consistent!(self);
    }
    /// Introduce slack variables and in doing so, transform '>=' and '=<' into '==' constraints.
    fn introduce_slack_variables(&mut self) {
        for i in 0..self.nr_rows() {
            match self.row_info[i] {
                ConstraintType::Less => self.add_slack(i, 1f64),
                ConstraintType::Greater => self.add_slack(i, -1f64),
                ConstraintType::Equal => continue,
            }
        }

        debug_assert_consistent!(self);
    }
    /// Add a slack in the specified row. Coefficient should be either 1f64 or -1f64.
    fn add_slack(&mut self, row: usize, coefficient: f64) {
        debug_assert!(coefficient == 1f64 || coefficient == -1f64);

        let column = self.data.nr_columns();
        self.data.push_zero_column();
        self.data.set_value(row, column, coefficient);
        self.cost.push_zero();
        let name = GeneralForm::generate_slack_name(row);
        self.column_info.push(Variable::new(name, VariableType::Continuous, 0f64));
        self.row_info[row] = ConstraintType::Equal;

        debug_assert_consistent!(self);
    }
    /// Generate a name for this slack.
    fn generate_slack_name(row: usize) -> String {
        let mut name = String::from("SLACK");
        name.push_str(&row.to_string());
        name
    }
    /// The number of constraints in this linear program.
    pub fn nr_rows(&self) -> usize {
        self.data.nr_rows()
    }
    /// The number of variables in this linear program.
    pub fn nr_columns(&self) -> usize {
        self.data.nr_columns()
    }
}

/// All types implementing this type can be converted into a `GeneralForm` linear program.
pub trait GeneralFormConvertable: Debug {
    fn to_general_lp(&self) -> GeneralForm;
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_substitute_fixed() {
        let data = vec![vec![1f64, 0f64],
                        vec![1f64, 1f64]];
        let data = SparseMatrix::from_data(data);
        let b = DenseVector::from_data(vec![3f64, 8f64]);
        let cost = SparseVector::from_data(vec![1f64, 1f64]);
        let column_info = vec![Variable::new(String::from("XONE"), VariableType::Continuous, 0f64),
                               Variable::new(String::from("XTWO"), VariableType::Continuous, 0f64)];
        let row_info = vec![ConstraintType::Equal,
                            ConstraintType::Less];
        let mut initial = GeneralForm::new(data, b, cost, 0f64, column_info, Vec::new(), row_info);
        initial.substitute_fixed();

        let data = vec![vec![1f64]];
        let data = SparseMatrix::from_data(data);
        let b = DenseVector::from_data(vec![5f64]);
        let cost = SparseVector::from_data(vec![1f64]);
        let fixed_cost = 3f64;
        let column_info = vec![Variable::new(String::from("XTWO"), VariableType::Continuous, 0f64)];
        let solution_values = vec![(String::from("XONE"), 3f64)];
        let row_info = vec![ConstraintType::Less];
        let expected = GeneralForm::new(data, b, cost, fixed_cost, column_info, solution_values, row_info);

        assert_eq!(initial, expected);
    }

    #[test]
    fn test_shift_variables() {
        let data = vec![vec![3f64, 0f64],
                        vec![1f64, 0f64],
                        vec![0f64, 1f64],
                        vec![2f64, 1f64]];
        let data = SparseMatrix::from_data(data);
        let b = DenseVector::from_data(vec![-3f64, 3f64, 2f64, 8f64]);
        let cost = SparseVector::from_data(vec![1f64, 1f64]);
        let column_info = vec![Variable::new(String::from("XONE"), VariableType::Continuous, 0f64),
                               Variable::new(String::from("XTWO"), VariableType::Continuous, 0f64)];
        let row_info = vec![ConstraintType::Greater,
                            ConstraintType::Less,
                            ConstraintType::Less,
                            ConstraintType::Less];
        let mut general_form = GeneralForm::new(data, b, cost, 0f64, column_info, Vec::new(), row_info);
        general_form.shift_variables();

        let data = vec![vec![1f64, 0f64],
                        vec![0f64, 1f64],
                        vec![2f64, 1f64]];
        let data = SparseMatrix::from_data(data);
        let b = DenseVector::from_data(vec![3f64 + 1f64, 2f64, 8f64 + 2f64 * 1f64]);
        let cost = SparseVector::from_data(vec![1f64, 1f64]);
        let column_info = vec![Variable::new(String::from("XONE"), VariableType::Continuous, -1f64),
                               Variable::new(String::from("XTWO"), VariableType::Continuous, 0f64)];
        let row_info = vec![ConstraintType::Less,
                            ConstraintType::Less,
                            ConstraintType::Less];
        let expected = GeneralForm::new(data, b, cost, -1f64, column_info, Vec::new(), row_info);

        assert_eq!(general_form, expected);
    }

    #[test]
    fn test_introduce_slack_variables() {
        // Positive slack
        let data = SparseMatrix::from_data(vec![vec![1f64]]);
        let b = DenseVector::from_data(vec![8f64]);
        let cost = SparseVector::from_data(vec![1f64]);
        let column_info = vec![Variable::new(String::from("X"), VariableType::Continuous, 0f64)];
        let row_info = vec![ConstraintType::Less];
        let mut result = GeneralForm::new(data, b, cost, 0f64, column_info, Vec::new(), row_info);
        result.introduce_slack_variables();

        let data = SparseMatrix::from_data(vec![vec![1f64, 1f64]]);
        let b = DenseVector::from_data(vec![8f64]);
        let cost = SparseVector::from_data(vec![1f64, 0f64]);
        let column_info = vec![Variable::new(String::from("X"), VariableType::Continuous, 0f64),
                               Variable::new(GeneralForm::generate_slack_name(0), VariableType::Continuous, 0f64)];
        let row_info = vec![ConstraintType::Equal];
        let mut expected = GeneralForm::new(data, b, cost, 0f64, column_info, Vec::new(), row_info);

        assert_eq!(result, expected);

        // Negative slack
        let data = SparseMatrix::from_data(vec![vec![1f64]]);
        let b = DenseVector::from_data(vec![8f64]);
        let cost = SparseVector::from_data(vec![1f64]);
        let column_info = vec![Variable::new(String::from("X"), VariableType::Continuous, 0f64)];
        let row_info = vec![ConstraintType::Greater];
        let mut result = GeneralForm::new(data, b, cost, 0f64, column_info, Vec::new(), row_info);
        result.introduce_slack_variables();

        let data = SparseMatrix::from_data(vec![vec![1f64, -1f64]]);
        let b = DenseVector::from_data(vec![8f64]);
        let cost = SparseVector::from_data(vec![1f64, 0f64]);
        let column_info = vec![Variable::new(String::from("X"), VariableType::Continuous, 0f64),
                               Variable::new(GeneralForm::generate_slack_name(0), VariableType::Continuous, 0f64)];
        let row_info = vec![ConstraintType::Equal];
        let mut expected = GeneralForm::new(data, b, cost, 0f64, column_info, Vec::new(), row_info);

        assert_eq!(result, expected);
    }

    fn lp_general() -> GeneralForm {
        let data = vec![vec![1f64, 1f64, 0f64],
                        vec![1f64, 0f64, 1f64],
                        vec![0f64, -1f64, 1f64],
                        vec![1f64, 0f64, 0f64],
                        vec![0f64, 1f64, 0f64]];
        let data = SparseMatrix::from_data(data);

        let b = DenseVector::from_data(vec![5f64,
                                            10f64,
                                            7f64,
                                            4f64,
                                            1f64]);

        let cost = SparseVector::from_data(vec![1f64, 4f64, 9f64]);

        let column_info = vec![Variable::new(String::from("XONE"), VariableType::Continuous, 0f64),
                               Variable::new(String::from("YTWO"), VariableType::Continuous, 0f64),
                               Variable::new(String::from("ZTHREE"), VariableType::Continuous, 0f64)];

        let row_info = vec![ConstraintType::Less,
                            ConstraintType::Greater,
                            ConstraintType::Equal,
                            ConstraintType::Less,
                            ConstraintType::Less];

        GeneralForm::new(data, b, cost, 0f64, column_info, Vec::new(), row_info)
    }

    #[test]
    fn test_make_b_non_negative() {
        let data = SparseMatrix::from_data(vec![vec![2f64]]);
        let b = DenseVector::from_data(vec![-1f64]);
        let cost = SparseVector::from_data(vec![1f64]);
        let column_info = vec![Variable::new(String::from("X"), VariableType::Continuous, 0f64)];
        let row_info = vec![ConstraintType::Equal];
        let mut result = GeneralForm::new(data, b, cost, 0f64, column_info, Vec::new(), row_info);

        result.make_b_non_negative();

        let data = SparseMatrix::from_data(vec![vec![-2f64]]);
        let b = DenseVector::from_data(vec![1f64]);
        let cost = SparseVector::from_data(vec![1f64]);
        let column_info = vec![Variable::new(String::from("X"), VariableType::Continuous, 0f64)];
        let row_info = vec![ConstraintType::Equal];
        let mut expected = GeneralForm::new(data, b, cost, 0f64, column_info, Vec::new(), row_info);

        assert_eq!(result, expected);
    }

    fn lp_canonical() -> CanonicalForm {
        let data = vec![vec![1f64, 1f64, 0f64, 1f64, 0f64, 0f64, 0f64],
                        vec![1f64, 0f64, 1f64, 0f64, -1f64, 0f64, 0f64],
                        vec![0f64, -1f64, 1f64, 0f64, 0f64, 0f64, 0f64],
                        vec![1f64, 0f64, 0f64, 0f64, 0f64, 1f64, 0f64],
                        vec![0f64, 1f64, 0f64, 0f64, 0f64, 0f64, 1f64]];
        let data = SparseMatrix::from_data(data);

        let b = DenseVector::from_data(vec![5f64,
                                            10f64,
                                            7f64,
                                            4f64,
                                            1f64]);

        let cost = SparseVector::from_data(vec![1f64, 4f64, 9f64, 0f64, 0f64, 0f64, 0f64]);

        let column_info = vec![Variable::new(String::from("XONE"), VariableType::Continuous, 0f64),
                               Variable::new(String::from("YTWO"), VariableType::Continuous, 0f64),
                               Variable::new(String::from("ZTHREE"), VariableType::Continuous, 0f64),
                               Variable::new(String::from("SLACK0"), VariableType::Continuous, 0f64),
                               Variable::new(String::from("SLACK1"), VariableType::Continuous, 0f64),
                               Variable::new(String::from("SLACK3"), VariableType::Continuous, 0f64),
                               Variable::new(String::from("SLACK4"), VariableType::Continuous, 0f64)];

        CanonicalForm::new(data, b, cost, 0f64, column_info, Vec::new())
    }

    #[test]
    fn to_canonical() {
        let result = lp_general().to_canonical();
        let expected = lp_canonical();

        assert_eq!(result, expected);
    }
}
