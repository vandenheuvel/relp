//! Organizing data read in the `parsing` module, and checking the linear program for
//! consistency.
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::convert::{identity, TryInto};
use std::convert::TryFrom;
use std::mem;

use crate::data::linear_algebra::matrix::{ColumnMajor, Order as MatrixOrder, RowMajor, Sparse};
use crate::data::linear_algebra::vector::{DenseVector, Vector};
use crate::data::linear_program::elements::{ConstraintType, VariableType};
use crate::data::linear_program::elements::Objective;
use crate::data::linear_program::general_form::GeneralForm;
use crate::data::linear_program::general_form::Variable as ShiftedVariable;
use crate::data::number_types::traits::{Field, OrderedField};
use crate::io::error::Inconsistency;
use crate::io::mps::{Bound, Range};
use crate::io::mps::BoundType;
use crate::io::mps::Constraint;
use crate::io::mps::parsing::{UnstructuredBound, UnstructuredRange};
use crate::io::mps::parsing::UnstructuredColumn;
use crate::io::mps::parsing::UnstructuredMPS;
use crate::io::mps::parsing::UnstructuredRhs;
use crate::io::mps::parsing::UnstructuredRow;
use crate::io::mps::Rhs;
use crate::io::mps::Variable;

impl<'a, 'b, F> TryFrom<UnstructuredMPS<'a, F>> for MPS<F> {
    type Error = Inconsistency;

    /// Try to convert into an `MPS` instance.
    ///
    /// This method organizes and structures the unstructured information contained in the
    /// `UnstructuredMPS` instance.
    ///
    /// # Arguments
    ///
    /// * `unstructured_mps`: A possibly inconsistent MPS instance.
    ///
    /// # Return value
    ///
    /// A "structured" MPS on success, a `LinearProgramError` on failure.
    fn try_from(unstructured_mps: UnstructuredMPS<'a, F>) -> Result<Self, Self::Error> {
        let (row_names, row_index) = build_row_index(&unstructured_mps.rows)?;
        let rows = order_rows(unstructured_mps.rows, &row_index);
        let (cost_values, columns, column_names) = build_columns(
            unstructured_mps.columns,
            &unstructured_mps.cost_row_name,
            &row_index,
        )?;
        let column_index = build_column_index(&column_names);
        let rhss = build_rhss(unstructured_mps.rhss, &row_index)?;
        let ranges = build_ranges(unstructured_mps.ranges, &row_index)?;
        let bounds = build_bounds(unstructured_mps.bounds, &column_index)?;
        let name = unstructured_mps.name.to_string();
        let cost_row_name = unstructured_mps.cost_row_name.to_string();

        Ok(MPS::new(
            name,
            cost_row_name,
            cost_values,
            row_names,
            rows,
            column_names,
            columns,
            rhss,
            ranges,
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
/// * `unstructured_rows`: Collection of unstructured rows.
///
/// # Return value
///
/// A tuple consisting of the names of all rows (this includes the cost row) and
fn build_row_index<'a>(
    unstructured_rows: &Vec<UnstructuredRow<'a>>
) -> Result<(Vec<String>, HashMap<&'a str, usize>), Inconsistency> {
    let mut row_index = HashMap::with_capacity(unstructured_rows.len());
    let unique = unstructured_rows.iter().enumerate()
        .all(|(i, row)| row_index.insert(row.name, i).is_none());
    if !unique {
        return Err(Inconsistency::new("Not all row names are unique"));
    }

    let row_names = unstructured_rows.iter()
        .map(|&UnstructuredRow { name, .. }| name.to_string())
        .collect::<Vec<_>>();

    Ok((row_names, row_index))
}

/// Order the rows according the provided index.
///
/// The name of the row is match against the index.
///
/// # Arguments
///
/// * `unstructured_rows`: Collection of unstructured rows.
/// * `index`: Assigns to each row an index
///
/// # Return value
///
/// `Constraint`s if successful, a `LinearProgramError` if not.
fn order_rows<'a>(
    unstructured_rows: Vec<UnstructuredRow<'a>>,
    _index: &HashMap<&'a str, usize>,
) -> Vec<Constraint> {
    debug_assert!(unstructured_rows.iter().enumerate().all(|(i, ur)| {
        _index.get(ur.name).map_or(false, |&ii| ii == i)
    }));

    unstructured_rows.iter().enumerate()
        .map(|(i, row)| {
            Constraint {
                name_index: i,
                constraint_type: row.constraint_type,
            }
        })
        .collect()
}

/// Collect the column values.
///
/// # Arguments
///
/// * `unstructured_columns`: Collection of `UnstructuredColumn`s.
/// * `cost_row_name`: Name of the cost row, used to identify which values belong to the objective
/// function.
/// * `row_index`: Index providing an ordering of the row names.
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
fn build_columns<'a, 'b, F>(
    mut unstructured_columns: Vec<UnstructuredColumn<'a, F>>,
    cost_row_name: &'a str,
    row_index: &HashMap<&'a str, usize>,
) -> Result<(Vec<(usize, F)>, Vec<Variable<F>>, Vec<String>), Inconsistency> {
    unstructured_columns.sort_by_key(|&UnstructuredColumn { name, .. }| name);

    let mut cost_values = Vec::new();
    let mut columns = Vec::new();
    let mut column_names = Vec::new();
    if !unstructured_columns.is_empty() {
        let UnstructuredColumn { name: mut current_name, variable_type: mut current_type, .. } = unstructured_columns[0];
        let mut values = Vec::new();
        for UnstructuredColumn { name, variable_type, row_name, value, } in unstructured_columns {
            if name != current_name {
                values.sort_by_key(|&(i, _)| i);
                columns.push(Variable {
                    name_index: column_names.len(),
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
                    None => return Err(Inconsistency::new(format!("Row name \"{}\" not known.", row_name))),
                };
                values.push((index, value));
            }
        }
        values.sort_by_key(|&(i, _)| i);
        columns.push(Variable { name_index: column_names.len(), variable_type: current_type, values, });
        column_names.push(current_name.to_string());
    }

    Ok((cost_values, columns, column_names))
}

/// Assign to each column a fixed column index.
///
/// This index will be used throughout building the `MPS`.
///
/// # Arguments
///
/// * `column_names`: Collection of names of the columns
///
/// # Return value
///
/// A map assigning to each column name a value.
///
/// # Note
///
/// This assignment is random.
fn build_column_index(column_names: &Vec<String>) -> HashMap<&str, usize> {
    column_names.iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect()
}

/// Structure the right-hand side data
///
/// # Arguments
///
/// * `unstructured_rhss`: Collection of unstructured right-hand side data.
/// * `row_index`: Assignment of rows (by name) to an index.
///
/// # Return value
///
/// Vector of right-hands side constraints .
///
/// # Errors
///
/// If a constraint name has not already been encountered in the ROWS section, a
/// `LinearProgramError`.
///
/// # Note
///
/// Requires that right-hand side data is sorted by the name of the right-hand side.
///
/// TODO: Generalize the method as to relax the above requirement.
fn build_rhss<'a, 'b, F>(
    mut unstructured_rhss: Vec<UnstructuredRhs<'a, F>>,
    row_index: &HashMap<&'a str, usize>,
) -> Result<Vec<Rhs<F>>, Inconsistency> {
    unstructured_rhss.sort_by_key(|&UnstructuredRhs { name, .. }| name);

    let mut rhss = Vec::new();
    if !unstructured_rhss.is_empty() {
        let UnstructuredRhs { name: mut current_name, .. } = unstructured_rhss[0];
        let mut values = Vec::new();
        for UnstructuredRhs { name, row_name, value, } in unstructured_rhss {
            if name != current_name {
                rhss.push(Rhs { name: current_name.to_string(), values, });

                current_name = name;
                values = Vec::new()
            }
            values.push((match row_index.get(row_name) {
                Some(&index) => index,
                None => return Err(Inconsistency::new(format!("Row name \"{}\" not known.", row_name))),
            }, value));
        }
        values.sort_by_key(|&(i, _)| i);
        rhss.push(Rhs { name: current_name.to_string(), values, });
    }

    Ok(rhss)
}

/// Structure the ranges data
///
/// # Arguments
///
/// * `unstructured_ranges`: Collection of unstructured ranges.
///
/// # Return value
///
/// Vector of ranges is.
///
/// # Errors
///
/// If a constraint name has not already been encountered in the ROWS section, a
/// `LinearProgramError`.
///
/// # Note
///
/// Requires that right-hand side data is sorted by the name of the right-hand side.
///
/// TODO: Generalize the method as to relax the above requirement.
fn build_ranges<'a, F>(
    mut unstructured_ranges: Vec<UnstructuredRange<'a, F>>,
    row_index: &HashMap<&'a str, usize>,
) -> Result<Vec<Range<F>>, Inconsistency> {
    let names_ok = {
        let mut unique = HashSet::with_capacity(unstructured_ranges.len());
        unstructured_ranges.iter()
            .all(|range| {
                row_index.get(range.row_name).map_or(false, |index| unique.insert(index))
            })
    };
    if !names_ok {
        return Err(Inconsistency::new("A row name is unknown or a row has multiple ranges."));
    }

    unstructured_ranges.sort_by_key(|&UnstructuredRange { name, .. }| name);

    let mut ranges = Vec::new();
    if !unstructured_ranges.is_empty() {
        let UnstructuredRange { name: mut current_name, .. } = unstructured_ranges[0];
        let mut values = Vec::new();
        for UnstructuredRange { name, row_name, value, } in unstructured_ranges {
            if name != current_name {
                ranges.push(Range { name: current_name.to_string(), values, });

                current_name = name;
                values = Vec::new()
            }
            values.push((match row_index.get(row_name) {
                Some(&index) => index,
                None => return Err(Inconsistency::new(format!("Row name \"{}\" not known.", row_name))),
            }, value));
        }
        values.sort_by_key(|&(i, _)| i);
        ranges.push(Range { name: current_name.to_string(), values, });
    }

    Ok(ranges)
}

/// Structure the bound data
///
/// # Arguments
///
/// * `unstructured_bounds`: Collection of unstructured bound data.
/// * `column_index`: Assignment of columns/variables (by name) to an index.
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
fn build_bounds<F>(
    mut unstructured_bounds: Vec<UnstructuredBound<F>>,
    column_index: &HashMap<&str, usize>,
) -> Result<Vec<Bound<F>>, Inconsistency> {
    unstructured_bounds.sort_by_key(|&UnstructuredBound { name, .. }| name);

    let mut bounds = Vec::new();
    if !unstructured_bounds.is_empty() {
        let UnstructuredBound { name: mut bound_name, .. } = unstructured_bounds[0];
        let mut values = Vec::new();
        // TODO: Get the innver bound type value fro m &f to f
        for UnstructuredBound { name, bound_type, column_name, } in unstructured_bounds {
            if name != bound_name {
                bounds.push(Bound { name: bound_name.to_string(), values, });

                bound_name = name;
                values = Vec::new();
            }
            values.push((
                bound_type,
                match column_index.get(column_name) {
                    Some(&index) => index,
                    None => return Err(Inconsistency::new(format!("Variable \"{}\" not known", column_name))),
                },
            ));
        }
        bounds.push(Bound { name: bound_name.to_string(), values, });
    }

    Ok(bounds)
}


/// Represents the contents of a MPS file in a structured manner.
///
/// `usize` variables in contained structs refer to the index of the cost and row names.
#[derive(Debug, PartialEq)]
pub struct MPS<F> {
    /// Name of the linear program
    name: String,
    /// Name of the cost row
    cost_row_name: String,
    /// Variable index and value tuples
    cost_values: Vec<(usize, F)>,
    /// Name of every constraint row
    row_names: Vec<String>,
    /// All named constraints
    rows: Vec<Constraint>,
    /// Name of every variable
    column_names: Vec<String>,
    /// Constraint name and Variable name combinations
    columns: Vec<Variable<F>>,
    /// Right-hand side constraint values
    rhss: Vec<Rhs<F>>,
    /// Limiting constraint activations two-sidedly.
    ranges: Vec<Range<F>>,
    /// Extra bounds on variables
    bounds: Vec<Bound<F>>,
}

impl<F> MPS<F> {
    /// Collect structured information into a `MPS` instance.
    ///
    /// # Arguments
    ///
    /// * `name`: Name of the linear program.
    /// * `cost_row_name`: Name of the cost row / objective function.
    /// * `cost_values`: Column (by index) and coefficient combinations for the objective function.
    /// * `row_names`: Names of all rows. The ordering corresponds with the order of the data in
    /// the `rows` argument.
    /// * `rows`: Constraint types. Names of the constraints are in the `row_names` argument, with
    /// corresponding order.
    /// * `column_names`: Names of all columns / variables. The ordering corresponds with the order
    /// of the data in the `columns` argument.
    /// * `columns`: Constraint data by column. Names of the variables are in `column_names`, with
    /// corresponding order.
    /// * `rhss`: Constraint values.
    /// * `ranges`: Flexibility on constraint activations.
    /// * `bounds`: Separate set of constraints, applying to the variables.
    ///
    /// # Return value
    ///
    /// All data collected in the `MPS` type.
    pub(crate) fn new(
        name: String,
        cost_row_name: String,
        cost_values: Vec<(usize, F)>,
        row_names: Vec<String>,
        rows: Vec<Constraint>,
        column_names: Vec<String>,
        columns: Vec<Variable<F>>,
        rhss: Vec<Rhs<F>>,
        ranges: Vec<Range<F>>,
        bounds: Vec<Bound<F>>,
    ) -> Self {
        Self {
            name,
            cost_row_name,
            cost_values,
            row_names,
            rows,
            column_names,
            columns,
            rhss,
            ranges,
            bounds,
        }
    }
}

impl<F: OrderedField> TryInto<GeneralForm<F>> for MPS<F> {
    type Error = Inconsistency;

    /// Convert an `MPS` into a `GeneralForm` linear program.
    ///
    /// # Arguments
    ///
    /// * `self`: `MPS` instance.
    ///
    /// # Return value
    ///
    /// A linear program in general form.
    fn try_into(self) -> Result<GeneralForm<F>, Self::Error> {
        let (mut variable_info, variable_names) = compute_variable_info(
            &self.columns,
            self.column_names,
            self.cost_values,
        );
        let (columns, constraint_types, b) = compute_constraint_info(
            self.rhss,
            self.columns,
            self.rows,
            self.ranges,
        )?;
        process_bounds(&mut variable_info, self.bounds)?;

        Ok(GeneralForm::new(
            Objective::Minimize,
            columns,
            constraint_types,
            b,
            variable_info,
            variable_names,
            F::additive_identity(),
        ))
    }
}

fn compute_variable_info<F: Field>(
    columns: &Vec<Variable<F>>,
    mut column_names: Vec<String>,
    cost_values: Vec<(usize, F)>,
) -> (Vec<ShiftedVariable<F>>, Vec<String>) {
    let variable_names = columns.iter().map(|variable| {
        mem::take(&mut column_names[variable.name_index])
    }).collect::<Vec<_>>();
    drop(column_names);
    debug_assert!(variable_names.iter().all(|name| name != &String::default()));

    // TODO: Where is this checked for duplicate row indices?
    let cost_values = cost_values.into_iter().collect::<HashMap<_, _>>();
    let variable_info = columns.iter().enumerate().map(|(j, variable)| {
        ShiftedVariable {
            variable_type: variable.variable_type,
            cost: (&cost_values.get(&j)).map_or(F::additive_identity(), |&v| v),
            upper_bound: None,
            lower_bound: None,
            shift: F::additive_identity(),
            flipped: false
        }
    }).collect();

    (variable_info, variable_names)
}

fn compute_constraint_info<OF: OrderedField>(
    rhss: Vec<Rhs<OF>>,
    columns: Vec<Variable<OF>>,
    rows: Vec<Constraint>,
    ranges: Vec<Range<OF>>,
) -> Result<(Sparse<OF, ColumnMajor>, Vec<ConstraintType>, DenseVector<OF>), Inconsistency> {
    let mut range_rows = ranges.iter()
        .flat_map(|range| range.values.iter().copied())
        .collect::<Vec<_>>();
    range_rows.sort_unstable_by_key(|&(i, _)| i);
    let mut from_original_to_extended = Vec::with_capacity(rows.len());
    let mut i = 0;
    for ii in 0..rows.len() {
        from_original_to_extended.push(ii + i);
        if i < range_rows.len() && range_rows[i].0 == ii {
            i += 1;
        }
    }

    let columns = compute_columns(columns, rows.len(), &range_rows);
    let constraint_types = compute_constraint_types(&rows, &range_rows);
    let b = compute_b(rhss, &constraint_types, rows.len(), &range_rows)?;

    Ok((columns, constraint_types, b))
}

fn compute_columns<F: Field>(
    columns: Vec<Variable<F>>,
    original_nr_rows: usize,
    ranges: &Vec<(usize, F)>,
) -> Sparse<F, ColumnMajor> {
    debug_assert!(ranges.is_sorted_by_key(|&(i, _)| i));
    debug_assert!(columns.iter().all(|variable| variable.values.is_sorted_by_key(|&(i, _)| i)));
    let nr_columns = columns.len();

    let mut extra_done = 0;
    let mut new_columns = vec![Vec::new(); columns.len()];
    for (j, column) in columns.into_iter().enumerate() {
        extra_done = 0;
        for (i, value) in column.values {
            new_columns[j].push((i + extra_done, value));
            while extra_done < ranges.len() && ranges[extra_done].0 < i {
                extra_done += 1;
            }
            if extra_done < ranges.len() && ranges[extra_done].0 == i {
                extra_done += 1;
                new_columns[j].push((i + extra_done, value));
            }
        }
    }

    ColumnMajor::new(
        new_columns,
        original_nr_rows + ranges.len(),
        nr_columns,
    )
}

fn compute_constraint_types<F: Field>(
    rows: &Vec<Constraint>,
    ranges: &Vec<(usize, F)>,
) -> Vec<ConstraintType> {
    let mut constraint_types = Vec::with_capacity(rows.len() + ranges.len());
    let mut extra_done = 0;
    for (i, &constraint) in rows.iter().enumerate() {
        if extra_done < ranges.len() && ranges[extra_done].0 == i {
            while extra_done < ranges.len() && ranges[extra_done].0 == i {
                constraint_types.push(ConstraintType::Greater);
                constraint_types.push(ConstraintType::Less);
                extra_done += 1;
            }
        } else {
            constraint_types.push(constraint.constraint_type);
        }
    }

    constraint_types
}

fn compute_b<OF: OrderedField>(
    rhss: Vec<Rhs<OF>>,
    constraints: &Vec<ConstraintType>,
    original_nr_rows: usize,
    ranges: &Vec<(usize, OF)>,
) -> Result<DenseVector<OF>, Inconsistency> {
    debug_assert_eq!(original_nr_rows + ranges.len(), constraints.len());
    debug_assert!(rhss.iter().all(|rhs| rhs.values.is_sorted_by_key(|&(i, _)| i)));

    let mut b = vec![None; constraints.len()];
    let mut extra_done = 0;
    for rhs in rhss {
        extra_done = 0;
        for (i, value) in rhs.values {
            while extra_done < ranges.len() && ranges[extra_done].0 < i {
                extra_done += 1;
            }
            if extra_done < ranges.len() && ranges[extra_done].0 == i {
                let r = ranges[extra_done].1;
                let (h, u) = match (constraints[i + extra_done], r.cmp(&OF::zero())) {
                    (ConstraintType::Greater, _) => (value, value + r.abs()),
                    (ConstraintType::Less, _) => (value - r.abs(), value),
                    (ConstraintType::Equal, Ordering::Greater | Ordering::Equal) => (value, value + r.abs()),
                    (ConstraintType::Equal, Ordering::Less | Ordering::Equal) => (value - r.abs(), value),
                };

                b[i + extra_done] = Some(h);
                extra_done += 1;
                b[i + extra_done] = Some(u);
            } else if let Some(current) = b[i + extra_done] {
                match constraints[i + extra_done] {
                    ConstraintType::Equal => if value != current {
                        return Err(
                            Inconsistency::new(
                                format!("Trivial infeasibility: a constraint can't equal both {} and {}", current, value),
                            )
                        )
                    },
                    ConstraintType::Greater => b[i + extra_done] = Some(current.max(value)),
                    ConstraintType::Less => b[i + extra_done] = Some(current.min(value)),
                }
            } else {
                b[i + extra_done] = Some(value);
            }
        }
    }

    Ok(DenseVector::new(
        b.into_iter().map(|value| value.unwrap_or(OF::zero())).collect(),
        original_nr_rows + ranges.len(),
    ))
}

fn process_bounds<OF: OrderedField>(
    variable_info: &mut Vec<ShiftedVariable<OF>>,
    bounds: Vec<Bound<OF>>,
) -> Result<(), Inconsistency> {
    let mut variable_is_touched = vec![false; variable_info.len()];
    for ref bound in bounds.into_iter() {
        for &(bound_type, variable_index) in bound.values.iter() {
            variable_is_touched[variable_index] = true;
            process_bound(bound_type, &mut variable_info[variable_index])?;
        }
    }
    fill_in_default_bounds(variable_info, variable_is_touched);

    Ok(())
}

fn process_bound<OF: OrderedField>(
    bound_type: BoundType<OF>,
    variable: &mut ShiftedVariable<OF>,
) -> Result<(), Inconsistency> {
    match bound_type {
        BoundType::LowerContinuous(value) => replace_existing_with(&mut variable.lower_bound, value, OF::max),
        BoundType::UpperContinuous(value) => {
            // TODO: Why the below statement?
            if variable.lower_bound.is_none() {
                variable.lower_bound = Some(OF::additive_identity());
            }
            replace_existing_with(&mut variable.upper_bound, value, OF::min);
        },
        BoundType::Fixed(value) => {
            // If there already is a known bound value for this variable, there
            // won't be any feasible values left after the below two statements.
            replace_existing_with(&mut variable.lower_bound, value, OF::max);
            replace_existing_with(&mut variable.upper_bound, value, OF::min);
        }
        BoundType::Free => {
            if variable.lower_bound.or(variable.upper_bound.clone()).is_some() {
                return Err(Inconsistency::new(format!("Variable can't be bounded and free")))
            }
        },
        BoundType::LowerMinusInfinity => replace_existing_with(&mut variable.upper_bound, OF::additive_identity(), OF::min),
        BoundType::UpperInfinity => replace_existing_with(&mut variable.lower_bound, OF::additive_identity(), OF::max),
        BoundType::Binary => {
            replace_existing_with(&mut variable.lower_bound, OF::additive_identity(), OF::max);
            replace_existing_with(&mut variable.upper_bound, OF::multiplicative_identity(), OF::min);
            variable.variable_type = VariableType::Integer;
        }
        BoundType::LowerInteger(value) => {
            replace_existing_with(&mut variable.lower_bound, value, OF::max);
            variable.variable_type = VariableType::Integer;
        },
        BoundType::UpperInteger(value) => {
            if variable.lower_bound.is_none() {
                variable.lower_bound = Some(OF::additive_identity());
            }
            replace_existing_with(&mut variable.upper_bound, value, OF::min);
            variable.variable_type = VariableType::Integer;
        },
        BoundType::SemiContinuous(_, _) => unimplemented!(),
    }

    Ok(())
}

fn replace_existing_with<OF: OrderedField, F: Fn(OF, OF) -> OF>(option: &mut Option<OF>, new_value: OF, f: F) {
    if let Some(ref mut existing_value) = option {
        *existing_value = f(*existing_value, new_value);
    } else {
        *option = Some(new_value);
    }
}

fn fill_in_default_bounds<OF: OrderedField>(variables: &mut Vec<ShiftedVariable<OF>>, touched: Vec<bool>) {
    for (j, touched) in touched.into_iter().enumerate() {
        if !touched {
            variables[j].lower_bound = Some(OF::additive_identity());
        }
    }
}
