//! Organizing data read in the `parsing` module, and checking the linear program for
//! consistency.
use std::collections::HashMap;
use std::convert::{identity, TryInto};
use std::convert::TryFrom;

use approx::AbsDiffEq;

use crate::data::linear_algebra::matrix::{ColumnMajorOrdering, MatrixOrder, RowMajorOrdering, SparseMatrix};
use crate::data::linear_algebra::SparseTuples;
use crate::data::linear_algebra::vector::{DenseVector, Vector};
use crate::data::linear_program::elements::{ConstraintType, VariableType};
use crate::data::linear_program::elements::Objective;
use crate::data::linear_program::general_form::GeneralForm;
use crate::data::linear_program::general_form::Variable as ShiftedVariable;
use crate::data::number_types::traits::RealField;
use crate::io::EPSILON;
use crate::io::error::InconsistencyError;
use crate::io::mps::Bound;
use crate::io::mps::BoundType;
use crate::io::mps::Constraint;
use crate::io::mps::parsing::UnstructuredBound;
use crate::io::mps::parsing::UnstructuredColumn;
use crate::io::mps::parsing::UnstructuredMPS;
use crate::io::mps::parsing::UnstructuredRhs;
use crate::io::mps::parsing::UnstructuredRow;
use crate::io::mps::Rhs;
use crate::io::mps::Variable;

impl<'a> TryFrom<UnstructuredMPS<'a>> for MPS {
    type Error = InconsistencyError;

    /// Try to convert into an `MPS` instance.
    ///
    /// This method organizes and structures the unstructured information contained in the
    /// `UnstructuredMPS` instance.
    ///
    /// # Arguments
    ///
    /// * `unstructured_mps` - A possibly inconsistent MPS instance.
    ///
    /// # Return value
    ///
    /// A "structured" MPS on success, a `LinearProgramError` on failure.
    fn try_from(mut unstructured_mps: UnstructuredMPS<'a>) -> Result<MPS, Self::Error> {
        let (row_names, row_index) = build_row_index(&unstructured_mps.rows);
        let rows = order_rows(&unstructured_mps.rows, &row_index)?;
        let (cost_values, columns, column_names) = build_columns(
            &mut unstructured_mps.columns,
            &unstructured_mps.cost_row_name,
            &row_index,
        )?;
        let column_index = build_column_index(&column_names);
        let rhss = build_rhss(&mut unstructured_mps.rhss, &row_index)?;
        let bounds = build_bounds(&mut unstructured_mps.bounds, &column_index)?;
        let name = unstructured_mps.name.to_string();
        let cost_row_name = unstructured_mps.cost_row_name.to_string();
        let row_names = row_names.into_iter().map(|name| name.to_string()).collect();

        Ok(MPS::new(
            name,
            cost_row_name,
            cost_values,
            row_names,
            rows,
            column_names,
            columns,
            rhss,
            bounds,
        ))
    }
}

/// Extract all row names, and assign to each row a fixed row index.
///
/// This index will be used throughout building the `MPS`, and isn't ordered in a specific way.
///
/// # Arguments
///
/// * `unstructured_rows` - Collection of unstructured rows.
///
/// # Return value
///
/// A tuple consisting of the names of all rows (this includes the cost row) and
fn build_row_index<'a>(
    unstructured_rows: &Vec<UnstructuredRow<'a>>
) -> (Vec<&'a str>, HashMap<&'a str, usize>) {
    let row_names = unstructured_rows.iter()
        .map(|&UnstructuredRow { name, .. }| name)
        .collect::<Vec<_>>();
    let row_index = row_names.iter()
        .enumerate()
        .map(|(index, &name)| (name, index))
        .collect();
    (row_names, row_index)
}

/// Order the rows according the provided index.
///
/// The name of the row is match against the index.
///
/// # Arguments
///
/// * `unstructured_rows` - Collection of unstructured rows.
/// * `index` - Assigns to each row an index
///
/// # Return value
///
/// `Constraint`s if successful, a `LinearProgramError` if not.
fn order_rows<'a>(
    unstructured_rows: &Vec<UnstructuredRow<'a>>,
    index: &HashMap<&'a str, usize>,
) -> Result<Vec<Constraint>, InconsistencyError> {
    let mut rows: Vec<Constraint> = Vec::with_capacity(unstructured_rows.len());

    for &UnstructuredRow { name, constraint_type, } in unstructured_rows.iter() {
        rows.push(Constraint {
            name: match index.get(name) {
                Some(&index) => index,
                None => return Err(InconsistencyError::new(format!("Unknown row: {}", name))),
            },
            constraint_type,
        });
    }

    Ok(rows)
}

/// Collect the column values.
///
/// # Arguments
///
/// * `unstructured_columns` - Collection of `UnstructuredColumn`s.
/// * `cost_row_name` - Name of the cost row, used to identify which values belong to the objective
/// function.
/// * `row_index` - Index providing an ordering of the row names.
///
/// # Return value
///
/// If successful, the column ind
///
/// # Note
///
/// Assumes that the unstructured columns are sorted together. In the MPS file, all data for a
/// column should be provided together.
///
/// TODO: Generalize the method to relax the above assumption.
fn build_columns<'a>(
    unstructured_columns: &mut Vec<UnstructuredColumn<'a>>,
    cost_row_name: &'a str,
    row_index: &HashMap<&'a str, usize>,
) -> Result<(Vec<(usize, f64)>, Vec<Variable>, Vec<String>), InconsistencyError> {
    unstructured_columns.sort_by_key(|&UnstructuredColumn { name, .. }| name);

    let mut cost_values = Vec::new();
    let mut columns = Vec::new();
    let mut column_names = Vec::new();
    let UnstructuredColumn { name: mut current_name, variable_type: mut current_type, .. } = unstructured_columns[0];
    let mut values = Vec::new();
    for &UnstructuredColumn { name, variable_type, row_name, value, } in unstructured_columns.iter() {
        if name != current_name {
            columns.push(Variable {
                name: column_names.len(),
                variable_type: current_type,
                values,
            });
            column_names.push(current_name.to_string());

            values = Vec::new();
            current_name = name;
            current_type = variable_type;
        }

        if row_name == cost_row_name {
            cost_values.push((column_names.len(), value));
        } else {
            let index = match row_index.get(row_name) {
                Some(&index) => index,
                None => return Err(InconsistencyError::new(format!("Row name \"{}\" not known.", row_name))),
            };
            values.push((index, value));
        }
    }
    columns.push(Variable { name: column_names.len(), variable_type: current_type, values, });
    column_names.push(current_name.to_string());

    Ok((cost_values, columns, column_names))
}

/// Assign to each column a fixed column index.
///
/// This index will be used throughout building the `MPS`.
///
/// # Arguments
///
/// * `column_names` - Collection of names of the columns
///
/// # Return value
///
/// A map assigning to each column name a value.
///
/// # Note
///
/// This assignment is random.
fn build_column_index(column_names: &Vec<String>) -> HashMap<String, usize> {
    column_names.iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect()
}

/// Structure the right-hand side data
///
/// # Arguments
///
/// * `unstructured_rhss` - Collection of unstructured right-hand side data.
/// * `row_index` - Assignment of rows (by name) to an index.
///
/// # Return value
///
/// Vector of right-hands side constraints if successful, `LinearProgramError` if not.
///
/// # Note
///
/// Requires that right-hand side data is sorted by the name of the right-hand side.
///
/// TODO: Generalize the method as to relax the above requirement.
fn build_rhss<'a>(
    unstructured_rhss: &mut Vec<UnstructuredRhs<'a>>,
    row_index: &HashMap<&'a str, usize>,
) -> Result<Vec<Rhs>, InconsistencyError> {
    unstructured_rhss.sort_by_key(|&UnstructuredRhs { name, .. }| name);

    let mut rhss = Vec::new();
    let UnstructuredRhs { name: mut current_name, .. } = unstructured_rhss[0];
    let mut values = Vec::new();
    for &UnstructuredRhs { name, row_name, value, } in unstructured_rhss.iter() {
        if name != current_name {
            rhss.push(Rhs { name: current_name.to_string(), values, });

            current_name = name;
            values = Vec::new()
        }
        values.push((match row_index.get(row_name) {
            Some(&index) => index,
            None => return Err(InconsistencyError::new(format!("Row name \"{}\" not known.", row_name))),
        }, value));
    }
    rhss.push(Rhs { name: current_name.to_string(), values, });

    Ok(rhss)
}

/// Structure the bound data
///
/// # Arguments
///
/// * `unstructured_bounds` - Collection of unstructured bound data.
/// * `column_index` - Assignment of columns/variables (by name) to an index.
///
/// # Return value
///
/// Vector of bound constraints if successful, `LinearProgramError` if not.
///
/// # Note
///
/// Requires that bound data is sorted.
///
/// TODO: Generalize the method as to relax the above requirement.
fn build_bounds<'a>(
    unstructured_bounds: &mut Vec<UnstructuredBound<'a>>,
    column_index: &HashMap<String, usize>,
) -> Result<Vec<Bound>, InconsistencyError> {
    unstructured_bounds.sort_by_key(|&UnstructuredBound { name, .. }| name);

    let mut bounds = Vec::new();
    let UnstructuredBound { name: mut bound_name, .. } = unstructured_bounds[0];
    let mut values = Vec::new();
    for &UnstructuredBound { name, bound_type, column_name, value, } in unstructured_bounds.iter() {
        if name != bound_name {
            bounds.push(Bound { name: bound_name.to_string(), values, });

            bound_name = name;
            values = Vec::new();
        }
        values.push((
            bound_type,
            match column_index.get(column_name) {
                Some(&index) => index,
                None => return Err(InconsistencyError::new(format!("Variable \"{}\" not known", column_name))),
            },
            value,
        ));
    }
    bounds.push(Bound { name: bound_name.to_string(), values, });

    Ok(bounds)
}


/// Represents the contents of a MPS file in a structured manner.
///
/// `usize` variables in contained structs refer to the index of the cost and row names.
#[derive(Debug, PartialEq)]
pub struct MPS {
    /// Name of the linear program
    name: String,
    /// Name of the cost row
    cost_row_name: String,
    /// Variable index and value tuples
    cost_values: Vec<(usize, f64)>,
    /// Name of every constraint row
    row_names: Vec<String>,
    /// All named constraints
    rows: Vec<Constraint>,
    /// Name of every variable
    column_names: Vec<String>,
    /// Constraint name and Variable name combinations
    columns: Vec<Variable>,
    /// Right-hand side constraint values
    rhss: Vec<Rhs>,
    /// Extra bounds on variables
    bounds: Vec<Bound>,
}

impl MPS {
    /// Collect structured information into a `MPS` instance.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the linear program.
    /// * `cost_row_name` - Name of the cost row / objective function.
    /// * `cost_values` - Column (by index) and coefficient combinations for the objective function.
    /// * `row_names` - Names of all rows. The ordering corresponds with the order of the data in
    /// the `rows` argument.
    /// * `rows` - Constraint types. Names of the constraints are in the `row_names` argument, with
    /// corresponding order.
    /// * `column_names` - Names of all columns / variables. The ordering corresponds with the order
    /// of the data in the `columns` argument.
    /// * `columns` - Constraint data by column. Names of the variables are in `column_names`, with
    /// corresponding order.
    /// * `rhss` - Constraint values.
    /// * `bounds` - Separate set of constraints, applying to the variables.
    ///
    /// # Return value
    ///
    /// All data collected in the `MPS` type.
    pub(crate) fn new(
        name: String,
        cost_row_name: String,
        cost_values: Vec<(usize, f64)>,
        row_names: Vec<String>,
        rows: Vec<Constraint>,
        column_names: Vec<String>,
        columns: Vec<Variable>,
        rhss: Vec<Rhs>,
        bounds: Vec<Bound>,
    ) -> MPS {
        MPS {
            name,
            cost_row_name,
            cost_values,
            row_names,
            rows,
            column_names,
            columns,
            rhss,
            bounds,
        }
    }
}

impl<RF: RealField> TryInto<GeneralForm<RF>> for MPS {
    type Error = InconsistencyError;

    /// Convert an `MPS` into a `GeneralForm` linear program.
    ///
    /// # Arguments
    ///
    /// * `self` - `MPS` instance.
    ///
    /// # Return value
    ///
    /// A linear program in general form.
    fn try_into(self) -> Result<GeneralForm<RF>, Self::Error> {
        let rows = compute_rows(&self.rows, &self.columns)?;
        let constraint_types = self.rows.iter()
            .map(|ref row| row.constraint_type)
            .collect::<Vec<_>>();
        let b = compute_b(&self.rhss, &self.rows, &constraint_types)?;
        let cost_values = compute_cost_values(&self.cost_values)?;
        let mut variable_info = compute_variable_info(&self.columns, &self.column_names, &cost_values);
        process_bounds(&mut variable_info, &self.bounds)?;

        Ok(GeneralForm::new(
            Objective::Minimize,
            rows,
            constraint_types,
            b,
            variable_info,
            RF::zero(),
        ))
    }
}

fn compute_rows<RF: RealField>(
    rows: &Vec<Constraint>,
    columns: &Vec<Variable>,
) -> Result<SparseMatrix<RF, RowMajorOrdering>, InconsistencyError> {
    let column_major_data = columns.iter().map(|column| {
        column.values.clone().into_iter()
            .map(|(i, v)| {
                if let Some(number) = RF::from_f64(v) {
                    Ok((i, number))
                } else {
                    Err(InconsistencyError::new(format!("Couldn't convert number: {}", v)))
                }
            })
            .collect::<Result<SparseTuples<_>, _>>()
    }).collect::<Result<Vec<SparseTuples<_>>, _>>()?;
    let columns = ColumnMajorOrdering::new(
        column_major_data,
        rows.len(),
        columns.len(),
    );
    let rows = SparseMatrix::from_column_major_ordered_matrix_although_this_is_expensive(&columns);

    Ok(rows)
}

fn compute_b<RF: RealField>(
    rhss: &Vec<Rhs>,
    rows: &Vec<Constraint>,
    constraints: &Vec<ConstraintType>,
) -> Result<DenseVector<RF>, InconsistencyError> {
    let mut b = DenseVector::constant(RF::additive_identity(), rows.len());
    for rhs in rhss.iter() {
        for &(index, value) in &rhs.values {
            if let Some(value) = RF::from_f64(value) {
                let current = b.get_value(index);
                if current == RF::zero() || match constraints[index] {
                    ConstraintType::Equal if value != current => {
                        return Err(
                            InconsistencyError::new(
                                format!("A constraint can't equal both {} and {}", current, value),
                            )
                        )
                    },
                    ConstraintType::Less if value < current => true,
                    ConstraintType::Greater if value > current => true,
                    _ => false,
                } {
                    b.set_value(index, value);
                }
            } else {
                return Err(InconsistencyError::new(format!("Couldn't convert f64: {}", value)));
            }
        }
    }

    Ok(b)
}

fn compute_cost_values<RF: RealField>(
    cost_values: &Vec<(usize, f64)>,
) -> Result<HashMap<usize, RF>, InconsistencyError> {
    cost_values.iter()
        .map(|&(i, cost)| match RF::from_f64(cost) {
            Some(value) => Ok((i, value)),
            None => Err(InconsistencyError::new(format!("Couldn't convert f64: {}", cost))),
        })
        .collect()
}

fn compute_variable_info<RF: RealField>(
    columns: &Vec<Variable>,
    column_names: &Vec<String>,
    cost_values: &HashMap<usize, RF>,
) -> Vec<ShiftedVariable<RF>> {
    columns.iter().enumerate().map(|(j, variable)| {
        ShiftedVariable {
            name: column_names[variable.name].clone(),
            variable_type: variable.variable_type,
            cost: (&cost_values.get(&j)).map_or(RF::zero(), |&v| v),
            upper_bound: None,
            lower_bound: None,
            shift: RF::zero(),
            flipped: false
        }
    }).collect()
}

fn process_bounds<RF: RealField>(
    variable_info: &mut Vec<ShiftedVariable<RF>>,
    bounds: &Vec<Bound>,
) -> Result<(), InconsistencyError> {
    let mut variable_is_touched = vec![false; variable_info.len()];
    for ref bound in bounds.iter() {
        for &(bound_type, variable_index, raw_value) in bound.values.iter() {
            variable_is_touched[variable_index] = true;
            if let Some(value) = RF::from_f64(raw_value) {
                let variable = &mut variable_info[variable_index];
                match bound_type {
                    BoundType::LowerContinuous    => replace_existing_with(&mut variable.lower_bound, value, RF::max),
                    BoundType::UpperContinuous    => {
                        if variable.lower_bound.is_none() {
                            variable.lower_bound = Some(RF::zero());
                        }
                        replace_existing_with(&mut variable.upper_bound, value, RF::min);
                    },
                    BoundType::Fixed              => {
                        // If there already is a known bound value for this variable, there
                        // won't be any feasible values left after the below two statements.
                        replace_existing_with(&mut variable.lower_bound, value, RF::max);
                        replace_existing_with(&mut variable.upper_bound, value, RF::min);
                    }
                    BoundType::Free               => {
                        if variable.lower_bound.or(variable.upper_bound.clone()).is_some() {
                            return Err(InconsistencyError::new(
                                format!("Variable {} can't be bounded and free", variable.name),
                            ))
                        }
                    },
                    BoundType::LowerMinusInfinity => replace_existing_with(&mut variable.upper_bound, RF::zero(), RF::min),
                    BoundType::UpperInfinity      => replace_existing_with(&mut variable.lower_bound, RF::zero(), RF::max),
                    BoundType::Binary             => {
                        replace_existing_with(&mut variable.lower_bound, RF::zero(), RF::max);
                        replace_existing_with(&mut variable.upper_bound, RF::one(), RF::min);
                        variable.variable_type = VariableType::Integer;
                    }
                    BoundType::LowerInteger       => {
                        replace_existing_with(&mut variable.lower_bound, value, RF::max);
                        variable.variable_type = VariableType::Integer;
                    },
                    BoundType::UpperInteger       => {
                        if variable.lower_bound.is_none() {
                            variable.lower_bound = Some(RF::zero());
                        }
                        replace_existing_with(&mut variable.upper_bound, value, RF::min);
                        variable.variable_type = VariableType::Integer;
                    },
                    BoundType::SemiContinuous     => unimplemented!(),
                }
            } else {
                return Err(InconsistencyError::new(format!("Couldn't convert f64: {}", raw_value)))
            }
        }
    }

    fill_in_default_bounds(variable_info, variable_is_touched);

    Ok(())
}

fn replace_existing_with<RF: RealField, F: Fn(RF, RF) -> RF>(option: &mut Option<RF>, new_value: RF, f: F) {
    if let Some(ref mut existing_value) = option {
        *existing_value = f(*existing_value, new_value);
    } else {
        *option = Some(new_value);
    }
}

fn fill_in_default_bounds<RF: RealField>(variables: &mut Vec<ShiftedVariable<RF>>, touched: Vec<bool>) {
    for (j, touched) in touched.into_iter().enumerate() {
        if !touched {
            variables[j].lower_bound = Some(RF::zero());
        }
    }
}

impl AbsDiffEq for MPS {
    type Epsilon = f64;

    fn default_epsilon() -> <Self as AbsDiffEq>::Epsilon {
        EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, _epsilon: <Self as AbsDiffEq>::Epsilon) -> bool {
        self.name == other.name &&
            self.cost_row_name == other.cost_row_name &&
            self.cost_values.iter().zip(other.cost_values.iter())
                .map(|((index1, value1), (index2, value2))|
                    index1 == index2 && abs_diff_eq!(value1, value2))
                .all(identity) &&
            self.row_names == other.row_names &&
            self.rows == other.rows &&
            self.column_names == other.column_names &&
            self.columns.iter().zip(other.columns.iter())
                .map(|(column1, column2)| abs_diff_eq!(column1, column2))
                .all(identity) &&
            self.rhss.iter().zip(other.rhss.iter())
                .map(|(column1, column2)| abs_diff_eq!(column1, column2))
                .all(identity) &&
            self.bounds.iter().zip(other.bounds.iter())
                .map(|(column1, column2)| abs_diff_eq!(column1, column2))
                .all(identity)
    }
}
