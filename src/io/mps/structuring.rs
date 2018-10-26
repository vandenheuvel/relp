//! Organizing data read in the `parsing` module, and checking the linear program for
//! consistency.
use std::convert::TryFrom;
use std::collections::HashMap;
use io::mps::Constraint;
use io::error::LinearProgramError;
use io::mps::Rhs;
use io::mps::Bound;
use io::mps::BoundType;
use data::linear_program::general_form::GeneralForm;
use data::linear_program::elements::ConstraintType;
use data::linear_algebra::vector::{DenseVector, SparseVector, Vector};
use data::linear_algebra::matrix::{Matrix, SparseMatrix};
use data::linear_program::elements::Variable as ShiftedVariable;
use io::mps::parsing::UnstructuredMPS;
use io::mps::Variable;
use io::mps::parsing::UnstructuredRow;
use io::mps::parsing::UnstructuredColumn;
use io::mps::parsing::UnstructuredRhs;
use io::mps::parsing::UnstructuredBound;

impl<'a> TryFrom<UnstructuredMPS<'a>> for MPS {
    type Error = LinearProgramError;

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
        let (cost_values, columns, column_names) =
            build_columns(&mut unstructured_mps.columns, &unstructured_mps.cost_row_name, &row_index)?;
        let column_index = build_column_index(&column_names);
        let rhss = build_rhss(&mut unstructured_mps.rhss, &row_index)?;
        let bounds = build_bounds(&mut unstructured_mps.bounds, &column_index)?;
        let name = unstructured_mps.name.to_string();
        let cost_row_name = unstructured_mps.cost_row_name.to_string();
        let row_names = row_names.into_iter()
            .map(|name| name.to_string())
            .collect();

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
    let row_names = unstructured_rows.iter().map(|&(name, _)| name).collect::<Vec<_>>();
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
    index: &HashMap<&'a str, usize>
) -> Result<Vec<Constraint>, LinearProgramError> {
    let mut rows: Vec<Constraint> = Vec::with_capacity(unstructured_rows.len());

    for &(name, constraint_type) in unstructured_rows.iter() {
        rows.push(Constraint {
            name: match index.get(name) {
                Some(&index) => index,
                None => return Err(LinearProgramError::new(format!("Unknown row: {}", name))),
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
) -> Result<(Vec<(usize, f64)>, Vec<Variable>, Vec<String>), LinearProgramError> {
    unstructured_columns.sort_by_key(|&(name, _, _, _)| name);

    let mut cost_values = Vec::new();
    let mut columns = Vec::new();
    let mut column_names = Vec::new();
    let (mut current_name, mut current_type, _, _) = unstructured_columns[0];
    let mut values = Vec::new();
    for &(name, variable_type, row_name, value) in unstructured_columns.iter() {
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
                None => return Err(LinearProgramError::new(format!("Row name \"{}\" not known.", row_name))),
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
) -> Result<Vec<Rhs>, LinearProgramError> {
    unstructured_rhss.sort_by_key(|&(name, _, _)| name);

    let mut rhss = Vec::new();
    let (mut current_name, _, _) = unstructured_rhss[0];
    let mut values = Vec::new();
    for &(name, row_name, value) in unstructured_rhss.iter() {
        if name != current_name {
            rhss.push(Rhs { name: current_name.to_string(), values, });

            current_name = name;
            values = Vec::new()
        }
        values.push((match row_index.get(row_name) {
            Some(&index) => index,
            None => return Err(LinearProgramError::new(format!("Row name \"{}\" not known.", row_name))),
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
) -> Result<Vec<Bound>, LinearProgramError> {
    unstructured_bounds.sort_by_key(|&(name, _, _, _)| name);

    let mut bounds = Vec::new();
    let (mut bound_name , _, _, _)= unstructured_bounds[0];
    let mut values = Vec::new();
    for &(name, bound_type, column_name, value) in unstructured_bounds.iter() {
        if name != bound_name {
            bounds.push(Bound { name: bound_name.to_string(), values, });

            bound_name = name;
            values = Vec::new();
        }
        values.push((bound_type, match column_index.get(column_name) {
            Some(&index) => index,
            None => return Err(LinearProgramError::new(format!("Variable \"{}\" not known", column_name))),
        }, value));
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
    pub(super) fn new(name: String,
                      cost_row_name: String,
                      cost_values: Vec<(usize, f64)>,
                      row_names: Vec<String>,
                      rows: Vec<Constraint>,
                      column_names: Vec<String>,
                      columns: Vec<Variable>,
                      rhss: Vec<Rhs>,
                      bounds: Vec<Bound>) -> MPS {
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

impl Into<GeneralForm> for MPS {
    /// Convert an `MPS` into a `GeneralForm` linear program.
    ///
    /// # Arguments
    ///
    /// * `self` - `MPS` instance.
    ///
    /// # Return value
    ///
    /// A linear program in general form.
    fn into(self) -> GeneralForm {
        let (m, n) = (self.rows.len() + self.bounds[0].values.len(), self.columns.len());

        let mut data = SparseMatrix::zeros(m, n);
        for (column_index, column) in self.columns.iter().enumerate() {
            for &(row_index, value) in &column.values {
                data.set_value(row_index, column_index, value);
            }
        }

        let cost = SparseVector::from_tuples(self.cost_values.clone(), n);

        let mut row_info: Vec<ConstraintType> = self.rows.iter()
            .map(|ref row| row.constraint_type).collect();

        let mut b = DenseVector::zeros(m);
        // TODO: Use all RHSs
        if self.rhss.len() > 1 {
            unimplemented!("Only one RHS is supported.");
        }
        for &(index, value) in self.rhss[0].values.iter() {
            b.set_value(index, value);
        }
        let mut nr_bounds_added = 0;
        for ref bound in self.bounds.iter() {
            for &(bound_type, variable_index, value) in bound.values.iter() {
                let row_index = self.rows.len() + nr_bounds_added;
                b.set_value(row_index, value);
                data.set_value(row_index, variable_index, 1f64);
                row_info.push(match bound_type {
                    BoundType::LowerContinuous      => ConstraintType::Greater,
                    BoundType::UpperContinuous      => ConstraintType::Less,
                    BoundType::Fixed                => ConstraintType::Equal,
                    BoundType::Free                 => unimplemented!(),
                    BoundType::LowerMinusInfinity   => ConstraintType::Greater,
                    BoundType::UpperInfinity        => ConstraintType::Less,
                    BoundType::Binary               => unimplemented!(),
                    BoundType::LowerInteger         => ConstraintType::Greater,
                    BoundType::UpperInteger         => ConstraintType::Less,
                    BoundType::SemiContinuous       => unimplemented!(),
                });
                nr_bounds_added += 1;
            }
        }

        let column_info = self.columns.iter()
            .map(|ref variable| ShiftedVariable {
                name: self.column_names[variable.name].clone(),
                variable_type: variable.variable_type,
                offset: 0f64,
            }).collect();

        GeneralForm::new(data, b, cost, 0f64, column_info, Vec::new(), row_info)
    }
}

/// Testing the structuring of parsed information.
#[cfg(test)]
mod test {
    use data::linear_program::general_form::GeneralForm;
    use io::mps::test::lp_mps;
    use io::mps::test::lp_general;

    /// Convert a `MPS` instance into `GeneralForm`.
    #[test]
    fn convert_to_general_lp() {
        let result: GeneralForm = lp_mps().into();
        let expected = lp_general();

        assert_eq!(format!("{:?}", result), format!("{:?}", expected));
    }
}