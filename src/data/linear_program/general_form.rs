use std::fmt::Debug;

use data::linear_algebra::matrix::{Matrix, SparseMatrix};
use data::linear_algebra::vector::{DenseVector, SparseVector, Vector};
use data::linear_program::canonical_form::CanonicalForm;
use data::linear_program::elements::{ConstraintType, RowType, VariableType};


/// A linear program in general form.
#[derive(Debug)]
pub struct GeneralForm {
    /// All coefficients
    data: SparseMatrix,
    /// All right-hands sides of equations
    b: DenseVector,
    /// The linear combination of costs
    cost: SparseVector,

    /// The names of all variables and their type, ordered by index
    column_info: Vec<(String, VariableType)>,
    /// The equation type of all rows, ordered by index
    row_info: Vec<ConstraintType>,
}

impl GeneralForm {
    /// Create a new linear program in general form.
    pub fn new(data: SparseMatrix, b: DenseVector, cost: SparseVector, column_info: Vec<(String, VariableType)>, row_info: Vec<ConstraintType>) -> GeneralForm {
        GeneralForm {
            data,
            b,
            cost,
            column_info,
            row_info,
        }
    }
    /// Convert this linear program into canonical form.
    pub fn to_canonical(&self) -> CanonicalForm {
        let nr_slacks = self.row_info.iter().filter(|row_type| match row_type {
            ConstraintType::Greater | ConstraintType::Less => true,
            ConstraintType::Equal => false,
        }).count();
        let mut data = SparseMatrix::zeros(self.nr_rows(), self.nr_columns() + nr_slacks);
        let mut column_info = self.column_info.clone();

        for (index, row_type) in self.row_info.iter().enumerate() {
            let new_column_index = column_info.len();
            data.set_row(index, self.data.row(index));
            match row_type {
                ConstraintType::Equal => (),
                ConstraintType::Greater => {
                    data.set_value(index, new_column_index, -1f64);
                    column_info.push((format!("SLACK{}", new_column_index - self.column_info.len()),
                                      VariableType::Continuous));
                }
                ConstraintType::Less => {
                    data.set_value(index, new_column_index, 1f64);
                    column_info.push((format!("SLACK{}", new_column_index - self.column_info.len()),
                                      VariableType::Continuous));
                }
            }
        }

        // Make sure b >= 0
        // TODO: We need b > 0. So when a bunch of variables are equal, make the problem smaller.
        let mut b = DenseVector::zeros(self.b.len());
        for (row, value) in self.b.iter().enumerate() {
            if *value < 0f64 {
                b.set_value(row, -*value);
                data.multiply_row(row, -1f64);
            } else {
                b.set_value(row, *value);
            }
        }

        let cost = SparseVector::from_tuples(self.cost.values().map(|&v| v).collect(),
                                             self.nr_columns() + nr_slacks);

        CanonicalForm::new(data, b, cost, column_info)
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

        let column_info = vec![(String::from("XONE"), VariableType::Continuous),
                               (String::from("YTWO"), VariableType::Continuous),
                               (String::from("ZTHREE"), VariableType::Continuous)];

        let row_info = vec![ConstraintType::Less,
                            ConstraintType::Greater,
                            ConstraintType::Equal,
                            ConstraintType::Less,
                            ConstraintType::Less];

        GeneralForm::new(data, b, cost, column_info, row_info)
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

        let column_info = vec![(String::from("XONE"), VariableType::Continuous),
                               (String::from("YTWO"), VariableType::Continuous),
                               (String::from("ZTHREE"), VariableType::Continuous),
                               (String::from("SLACK0"), VariableType::Continuous),
                               (String::from("SLACK1"), VariableType::Continuous),
                               (String::from("SLACK2"), VariableType::Continuous),
                               (String::from("SLACK3"), VariableType::Continuous)];

        CanonicalForm::new(data, b, cost, column_info)
    }

    #[test]
    fn to_canonical() {
        let result = lp_general().to_canonical();
        let expected = lp_canonical();

        assert_eq!(result, expected);
    }
}
