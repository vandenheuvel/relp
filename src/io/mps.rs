//! # Reading MPS files
//!
//! Reading of `.mps` files, or files of the Mathematical Programming System format.

use std::collections::HashMap;

use data::linear_program::elements::{RowType, VariableType};
use data::linear_program::general_form::{GeneralForm, GeneralFormConvertable};
use data::linear_algebra::matrix::{Matrix, SparseMatrix};
use data::linear_algebra::vector::{DenseVector, SparseVector, Vector};

/// Takes as argument a string in which the linear program is encoded following the
/// [MPS format](https://en.wikipedia.org/wiki/MPS_(format)).
pub fn parse(program: String) -> Result<Box<GeneralFormConvertable>, String> {
    let mut mps = MPS::new();
    let mut current_section = MPSSection::Name;

    for line in program.lines() {
        // There should be no data after the ENDATA line
        if current_section == MPSSection::ENDATA {
            return Err(String::from("Data after ENDATA line"));
        }

        // Update the current section if a new one is reached, else parse the line
        match parse_section(line) {
            Some(MPSSection::Name) => {
                current_section = MPSSection::Name;
                if let Err(message) = read_and_set(&mut mps, current_section, line) {
                    return Err(message);
                }
            },
            None => if let Err(message) = read_and_set(&mut mps, current_section, line) {
                return Err(message);
            },
            Some(MPSSection::IntegerColumns) => {
                if current_section == MPSSection::RealColumns {
                    current_section = MPSSection::IntegerColumns;
                } else if current_section == MPSSection::IntegerColumns {
                    current_section = MPSSection::RealColumns;
                } else {
                    return Err(format!("Need to be in column section to start integer column section.\
                 Currently in section \"{:?}\"", current_section));
                }
            },
            Some(section) => current_section = section,
        }
    }

    Ok(Box::new(mps))
}

/// Determines, it possible, which section of the MPS file the line announces.
///
/// If no new section is found, `None` is returned.
fn parse_section(line: &str) -> Option<MPSSection> {
    if line.starts_with(" ") {
        // Check for a nested section
        if let Some(word) = line.split_whitespace().nth(1) {
            // Check for a MARKER section
            if word.starts_with("'") && word.ends_with("'") {
                Some(MPSSection::IntegerColumns)
            } else {
                None
            }
        } else {
            None
        }
    }
    else if line.starts_with("NAME") { Some(MPSSection::Name) }
    else if line.starts_with("ROWS") { Some(MPSSection::Rows) }
    else if line.starts_with("COLUMNS") { Some(MPSSection::RealColumns) }
    else if line.starts_with("RHS") { Some(MPSSection::RHS) }
    else if line.starts_with("BOUNDS") { Some(MPSSection::Bounds) }
    else if line.starts_with("ENDATA") { Some(MPSSection::ENDATA) }
    else { None }
}

/// Parses a line and adds collects the result in an `MPS`.
fn read_and_set(mps: &mut MPS, section: MPSSection, line: &str) -> Result<(), String> {
    match section {
        MPSSection::Name => match read_name(line) {
            Ok(name) => Ok(mps.set_name(name)),
            Err(message) => Err(message),
        },
        MPSSection::Rows => match read_rows(line) {
            Ok(row) => Ok(mps.add_row(row)),
            Err(message) => Err(message),
        },
        MPSSection::RealColumns => match read_real_columns(line) {
            Ok(columns) => Ok(for column in columns.into_iter() { mps.add_real_column(column) }),
            Err(message) => Err(message),
        },
        MPSSection::IntegerColumns => match read_integer_columns(line) {
            Ok(column) => Ok(mps.add_integer_column(column)),
            Err(message) => Err(message),
        },
        MPSSection::RHS => match read_rhs(line) {
            Ok(rhs) => Ok(mps.add_rhs(rhs)),
            Err(message) => Err(message),
        },
        MPSSection::Bounds => match read_bound(line) {
            Ok(bound) => Ok(mps.add_bound(bound)),
            Err(message) => Err(message),
        },
        MPSSection::ENDATA => Ok(()),
    }
}

/// Represents the contents of an MPS file.
///
/// Information on variables is spread throughout the `rows`, `real_columns`, `integer_columns`,
/// `rhs` and `bounds` fields.
#[derive(Debug)]
struct MPS {
    // Name of the linear program
    name: String,
    // Name
    rows: Vec<Row>,
    real_columns: Vec<RealColumn>,
    integer_columns: Vec<IntegerColumn>,
    rhs: Vec<RHS>,
    bounds: Vec<Bound>,
}

impl MPS {
    fn new() -> MPS {
        MPS {
            name: String::new(),
            rows: Vec::new(),
            real_columns: Vec::new(),
            integer_columns: Vec::new(),
            rhs: Vec::new(),
            bounds: Vec::new(),
        }
    }
    fn set_name(&mut self, name: String) {
        self.name = name;
    }
    fn add_row(&mut self, row: Row) {
        self.rows.push(row);
    }
    fn add_real_column(&mut self, column: RealColumn) {
        self.real_columns.push(column);
    }
    fn add_integer_column(&mut self, column: IntegerColumn) {
        self.integer_columns.push(column);
    }
    fn add_rhs(&mut self, new_rhs: RHS) {
        for mut rhs in &mut self.rhs {
            if rhs.name == new_rhs.name {
                for (row, value) in new_rhs.values() {
                    rhs.add_value(row, value);
                }
                return;
            }
        }
        self.rhs.push(new_rhs);
    }
    fn add_bound(&mut self, bound: Bound) {
        self.bounds.push(bound);
    }
}

/// Read the name of a linear program from a string.
fn read_name(line: &str) -> Result<String, String> {
    let parts = line.split_whitespace().collect::<Vec<&str>>();
    if parts.len() != 2 || parts[0] != "NAME" {
        Err(format!("This is not a \"NAME\" line: {}", line))
    } else {
        Ok(String::from(parts[1]))
    }
}

/// Read a row of a linear program from a string.
fn read_rows(line: &str) -> Result<Row, String> {
    let mut parts = line.split_whitespace();

    // Parse the type of the row
    let row_type = match parts.next() {
        Some("N") => RowType::Cost,
        Some("L") => RowType::Less,
        Some("E") => RowType::Equal,
        Some("G") => RowType::Greater,
        Some(word) => return Err(format!("Row type not recognised: type {} on line {}", word, line)),
        None => return Err(format!("No row type: {}", line)),
    };

    let row_name = match parts.next() {
        Some(name) => name,
        None => return Err(format!("Now row name: {}", line)),
    };

    Ok(Row::new(row_type, String::from(row_name)))
}

/// Read a continuous variable name, row name and coefficient from a string.
fn read_real_columns(line: &str) -> Result<Vec<RealColumn>, String> {
    let mut parts = line.split_whitespace();
    // Make sure at least one column is read
    let mut columns = Vec::new();

    // Parse the variable name
    let variable_name = match parts.next() {
        Some(name) => String::from(name),
        None => return Err(format!("Column name not found: {}", line)),
    };

    // Try to read two pairs of row-coefficient combinations
    for _ in 0..2 {
        let row_name = match parts.next() {
            Some(name) => String::from(name),
            None => continue,
        };
        let coefficient = match parts.next() {
            Some(number) => match number.parse::<f64>() {
                Ok(f) => f,
                Err(message) => return Err(format!("Column coefficient \"{}\"\
                    could not be parsed from line \"{}\" : {}", number, line, message)),
            },
            None => return Err(format!("Column coefficient not found: {}", line)),
        };

        columns.push(RealColumn::new(variable_name.clone(), row_name, coefficient));
    }

    if columns.len() > 0 {
        Ok(columns)
    } else {
        Err(format!("Failed to read at least one column: {}", line))
    }
}

/// Read an integer variable name, row name and coefficient from a string.
fn read_integer_columns(line: &str) -> Result<IntegerColumn, String> {
    let mut parts = line.split_whitespace();

    // Read the variable name
    let variable_name = match parts.next() {
        Some(name) => String::from(name),
        None => return Err(format!("Column name not found: {}", line)),
    };

    // Read the row name
    let row_name = match parts.next() {
        Some(name) => String::from(name),
        None => return Err(format!("Failed to read row name: {}", line)),
    };

    // Read the value
    let coefficient = match parts.next() {
        Some(number) => match number.parse::<f64>() {
            Ok(f) => f,
            Err(message) => return Err(format!("Column coefficient \"{}\"\
                    could not be parsed from line \"{}\" : {}", number, line, message)),
        },
        None => return Err(format!("Column coefficient not found: {}", line)),
    };

    Ok(IntegerColumn::new(variable_name, row_name, coefficient))
}

/// Read a `RHS` or right-hand side from a string.
fn read_rhs(line: &str) -> Result<RHS, String> {
    let mut parts = line.split_whitespace();

    // Parse the RHS vector name
    let vector_name = match parts.next() {
        Some(name) => String::from(name),
        None => return Err(format!("RHS vector name not found: {}", line)),
    };

    let mut rhs_vector = RHS::new(vector_name);
    // Try to read two pairs of row-coefficient combinations
    for _ in 0..2 {
        let row_name = match parts.next() {
            Some(name) => String::from(name),
            None => continue,
        };
        let coefficient = match parts.next() {
            Some(number) => match number.parse::<f64>() {
                Ok(f) => f,
                Err(message) => return Err(format!("RHS vector coefficient \"{}\"\
                    could not be parsed from line \"{}\": {}", number, line, message)),
            },
            None => return Err(format!("RHS vector coefficient not found: {}", line)),
        };

        rhs_vector.add_value(row_name, coefficient);
    }

    // Make sure at least column is read
    if rhs_vector.len() > 0 {
        Ok(rhs_vector)
    } else {
        Err(format!("Failed to add at least one RHS value: {}", line))
    }
}

/// Read a `Bound` from a string
fn read_bound(line: &str) -> Result<Bound, String> {
    let mut parts = line.split_whitespace();

    let (direction, variable_type) = match parts.next() {
        Some("UP") => (BoundDirection::Upper, VariableType::Continuous),
        Some("LO") => (BoundDirection::Lower, VariableType::Continuous),
        Some("UI") => (BoundDirection::Upper, VariableType::Integer),
        Some("LI") => (BoundDirection::Lower, VariableType::Integer),
        Some("FX") => (BoundDirection::Fixed, VariableType::Continuous),
        Some(word) => return Err(format!("Bound type \"{}\" not recognised: {}", word, line)),
        None => return Err(format!("Bound direction and type not found: {}", line)),
    };

    let name = match parts.next() {
        Some(name) => String::from(name),
        None => return Err(format!("Bound name not found: {}", line)),
    };

    let variable_name = match parts.next() {
        Some(name) => String::from(name),
        None => return Err(format!("Column name not found: {}", line)),
    };

    let value = match parts.next() {
        Some(number) => match number.parse::<f64>() {
            Ok(f) => f,
            Err(message) => return Err(format!("Bound value \"{}\" could not be parsed \
                from line \"{}\": {}", number, line, message)),
        },
        None => return Err(format!("Bound value not found: {}", line)),
    };

    Ok(Bound::new(name, direction, variable_name, variable_type, value))
}

/// Describes how to convert an `MPS` to a `GeneralForm`, or linear program in general form.
impl GeneralFormConvertable for MPS {
    fn to_general_lp(&self) -> GeneralForm {
        // TODO: Split up and test the sections this method separately
        let mut row_names_index = HashMap::new();
        let mut cost_row_name = None;
        for row in &self.rows {
            if row.row_type == RowType::Cost {
                cost_row_name = Some(row.name.clone());
            } else if !row_names_index.contains_key(&row.name) {
                let size = row_names_index.len();
                row_names_index.insert(row.name.clone(), size);
            }
        }
        let cost_row_name = cost_row_name.unwrap();

        let mut column_names_index = HashMap::new();
        for column in &self.real_columns {
            if !column_names_index.contains_key(&column.variable_name) {
                let size = column_names_index.len();
                column_names_index.insert(column.variable_name.clone(), size);
            }
        }
        for column in &self.integer_columns {
            if !column_names_index.contains_key(&column.variable_name) {
                let size = column_names_index.len();
                column_names_index.insert(column.variable_name.clone(), size);
            }
        }

        // Fill the coefficient matrix
        let mut data = SparseMatrix::zeros(row_names_index.len() + self.bounds.len(), column_names_index.len());
        let mut cost = SparseVector::zeros(column_names_index.len());
        for column in &self.real_columns {
            if column.row_name == cost_row_name {
                cost.set_value(*column_names_index.get(&column.variable_name).unwrap(), column.coefficient);
            } else {

                data.set_value(*row_names_index.get(&column.row_name).unwrap(), *column_names_index.get(&column.variable_name).unwrap(), column.coefficient);
            }
        }
        for column in &self.integer_columns {
            if column.row_name == cost_row_name {
                cost.set_value(*column_names_index.get(&column.variable_name).unwrap(), column.coefficient);
            } else {
                data.set_value(*row_names_index.get(&column.row_name).unwrap(), *column_names_index.get(&column.variable_name).unwrap(), column.coefficient);
            }
        }

        // TODO: Don't just get the first RHS
        let b_tuples: Vec<(usize, f64)> = self.rhs[0].values.iter()
            .map(|&(ref row_name, value)| (*row_names_index.get(&row_name.to_owned()).unwrap(), value)).collect();
        let mut b = DenseVector::zeros(row_names_index.len() + self.bounds.len());
        for (index, value) in b_tuples {
            b.set_value(index, value);
        }

        // Column info
        let mut column_info_map = HashMap::new();
        for column in &self.real_columns {
            column_info_map.insert(*column_names_index.get(&column.variable_name).unwrap(), (column.variable_name.clone(), VariableType::Continuous));
        }
        for column in &self.integer_columns {
            column_info_map.insert(*column_names_index.get(&column.variable_name).unwrap(), (column.variable_name.clone(), VariableType::Integer));
        }
        let mut column_info_tuples: Vec<(usize, (String, VariableType))> = column_info_map.into_iter().collect();
        column_info_tuples.sort_by(|ref a, b| a.0.cmp(&b.0));
        let column_info = column_info_tuples.into_iter().map(|t| t.1).collect();

        // Row info
        let mut row_info_map = HashMap::new();
        for row in &self.rows {
            if row.name != cost_row_name {
                row_info_map.insert(*row_names_index.get(&row.name).unwrap(), row.row_type);
            }
        }
        let mut row_info_tuples: Vec<(usize, RowType)> = row_info_map.into_iter().collect();
        row_info_tuples.sort_by(|a, b| a.0.cmp(&b.0));
        let mut row_info: Vec<RowType> = row_info_tuples.into_iter().map(|t| t.1).collect();

        // Bounds as constraints
        for i in 0..self.bounds.len() {
            let bound = &self.bounds[i];
            b.set_value(row_info.len(), bound.value);
            row_info.push(match bound.direction {
                BoundDirection::Lower => RowType::Greater,
                BoundDirection::Upper => RowType::Less,
                BoundDirection::Fixed => RowType::Equal,
            });
            data.set_value(row_names_index.len() + i, *column_names_index.get(&bound.variable_name).unwrap(), 1f64);
        }

        GeneralForm::new(data, b, cost, column_info, row_info)
    }
}

/// All sections in a MPS file.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum MPSSection {
    Name,
    Rows,
    RealColumns,
    IntegerColumns,
    RHS,
    Bounds,
    ENDATA,
}

/// Every `Row` has a name and a `RowType`.
#[derive(Debug, Eq, PartialEq)]
struct Row {
    name: String,
    row_type: RowType,
}

impl Row {
    /// Create a new `Row`.
    pub fn new(row_type: RowType, name: String) -> Row {
        Row { name, row_type, }
    }
}

/// Describes with which coefficient a variable appears in a row. This variable is be a real number.
#[derive(Debug, PartialEq)]
struct RealColumn {
    variable_name: String,
    row_name: String,
    coefficient: f64,
}

impl RealColumn {
    pub fn new(variable_name: String, row_name: String, coefficient: f64) -> RealColumn {
        RealColumn { variable_name, row_name, coefficient, }
    }
}

/// Describes with which coefficient a variable appears in a row. This variable is an integer.
#[derive(Debug, PartialEq)]
struct IntegerColumn {
    variable_name: String,
    row_name: String,
    coefficient: f64,
}

impl IntegerColumn {
    pub fn new(variable_name: String, row_name: String, coefficient: f64) -> IntegerColumn {
        IntegerColumn { variable_name, row_name, coefficient, }
    }
}

/// A `RHS` is a constraint. A single linear program defined in MPS can have multiple right-hand
/// sides. It relates a row name to a real constant.
#[derive(Debug, PartialEq)]
struct RHS {
    name: String,
    values: Vec<(String, f64)>,
}

impl RHS {
    pub fn new(name: String) -> RHS {
        RHS { name, values: Vec::new() }
    }
    pub fn add_value(&mut self, name: String, value: f64) {
        self.values.push((name, value));
    }
    pub fn values(&self) -> Vec<(String, f64)>  {
        self.values.clone()
    }
    pub fn len(&self) -> usize {
        self.values.len()
    }
}

/// A `Bound` gives a bound on a variable. The variable can either be continuous or integer, while
/// the bound can have any direction.
#[derive(Debug, PartialEq)]
struct Bound {
    name: String,
    direction: BoundDirection,
    variable_name: String,
    variable_type: VariableType,
    value: f64,
}

impl Bound {
    pub fn new(name: String, direction: BoundDirection, variable_name: String, variable_type: VariableType, value: f64) -> Bound {
        Bound { name, direction, variable_name, variable_type, value, }
    }
}

/// A `BoundDirection` denotes the type of a bound on a variable. This can either be an upper bound,
/// a lower bound or describe that the variable is a fixed number.
#[derive(Debug, PartialEq)]
enum BoundDirection {
    Upper,
    Lower,
    Fixed,
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_parse_section() {
        let line = "NAME          TESTPROB";
        let result = parse_section(line);
        assert_eq!(result, Some(MPSSection::Name));

        let line = "ROWS";
        let result = parse_section(line);
        assert_eq!(result, Some(MPSSection::Rows));

        let line = "COLUMNS";
        let result = parse_section(line);
        assert_eq!(result, Some(MPSSection::RealColumns));

        let line = "RHS     ";
        let result = parse_section(line);
        assert_eq!(result, Some(MPSSection::RHS));

        let line = "BOUNDS";
        let result = parse_section(line);
        assert_eq!(result, Some(MPSSection::Bounds));

        let line = "ENDATA";
        let result = parse_section(line);
        assert_eq!(result, Some(MPSSection::ENDATA));

        let line = "    MARK0000  'MARKER'                 'INTORG'";
        let result = parse_section(line);
        assert_eq!(result, Some(MPSSection::IntegerColumns));

        let line = "DIFFERENT";
        let result = parse_section(line);
        assert_eq!(result, None);
    }

    #[test]
    fn test_read_name() {
        // Correct line
        let line = "NAME          TESTPROB";
        let result = read_name(line);
        assert_eq!(result, Ok(String::from("TESTPROB")));

        // Incorrect indicator
        let line = "X TESTPROB";
        let result = read_name(line);
        assert!(result.is_err());

        // Missing name
        let line = "NAME          ";
        let result = read_name(line);
        assert!(result.is_err());

        // Empty line
        let line = "";
        let result = read_name(line);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_rows() {
        // Cost row
        let line = " N  COST";
        let result = read_rows(line);
        assert_eq!(result, Ok(Row::new(RowType::Cost, String::from("COST"))));

        // Less-or-equal row
        let line = " L  LIM1";
        let result = read_rows(line);
        assert_eq!(result, Ok(Row::new(RowType::Less, String::from("LIM1"))));

        // Equal row
        let line = " E  MYEQN";
        let result = read_rows(line);
        assert_eq!(result, Ok(Row::new(RowType::Equal, String::from("MYEQN"))));

        // Greater row
        let line = " G  LIM2";
        let result = read_rows(line);
        assert_eq!(result, Ok(Row::new(RowType::Greater, String::from("LIM2"))));

        // Missing row name
        let line = " E ";
        let result = read_rows(line);
        assert!(result.is_err());

        // Unknown row type
        let line = " X  ROWNAME";
        let result = read_rows(line);
        assert!(result.is_err());

        // Empty line
        let line = "";
        let result = read_rows(line);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_columns() {
        // Correct line, two coefficients
        let line = "    XONE      COST                 1   LIM1                 1";
        let result = read_real_columns(line);
        assert_eq!(result, Ok(vec![RealColumn {
            variable_name: String::from("XONE"),
            row_name: String::from("COST"),
            coefficient: 1f64,
        }, RealColumn {
            variable_name: String::from("XONE"),
            row_name: String::from("LIM1"),
            coefficient: 1f64,
        }]));

        // Correct line, one coefficient
        let line = "    ZTHREE    MYEQN                1";
        let result = read_real_columns(line);
        assert_eq!(result, Ok(vec![RealColumn {
            variable_name: String::from("ZTHREE"),
            row_name: String::from("MYEQN"),
            coefficient: 1f64,
        }]));

        // Missing coefficient
        let line = "    XONE      COST                 1   LIM1                 ";
        let result = read_real_columns(line);
        assert!(result.is_err());

        // Missing row name
        let line = "    XONE      COST                 1           1";
        let result = read_real_columns(line);

        assert!(result.is_err());

        // Missing variable name
        let line = "      COST                 1   LIM1                 1";
        let result = read_real_columns(line);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_integer_columns() {
        // Correct line
        let line = "    XONE      COST                 1";
        let result = read_integer_columns(line);
        assert_eq!(result, Ok(IntegerColumn {
            variable_name: String::from("XONE"),
            row_name: String::from("COST"),
            coefficient: 1f64,
        }));

        // Missing coefficient
        let line = "    XONE      COST            ";
        let result = read_integer_columns(line);
        assert!(result.is_err());

        // Missing row name
        let line = "    XONE                     1       ";
        let result = read_integer_columns(line);

        assert!(result.is_err());

        // Missing variable name
        let line = "      COST                 1";
        let result = read_integer_columns(line);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_rhs() {
        // Correct line, two coefficients
        let line = "    RHS1      LIM1                 5   LIM2                10";
        let result = read_rhs(line);
        assert_eq!(result, Ok(RHS {
            name: String::from("RHS1"),
            values: vec![(String::from("LIM1"), 5f64),
                         (String::from("LIM2"), 10f64)],
        }));

        // Correct line, one coefficient
        let line = "    RHS1      MYEQN                7";
        let result = read_rhs(line);
        assert_eq!(result, Ok(RHS {
            name: String::from("RHS1"),
            values: vec![(String::from("MYEQN"), 7f64)],
        }));

        // Missing coefficient
        let line = "    RHS1      LIM1                 5   LIM2           ";
        let result = read_rhs(line);
        assert!(result.is_err());

        // Missing row name
        let line = "    RHS1      LIM1                 5                  10";
        let result = read_rhs(line);
        assert!(result.is_err());

        // Missing rhs name
        let line = "         LIM1                 5   LIM2                10";
        let result = read_rhs(line);
        assert!(result.is_err());

        // Missing coefficient
        let line = "    RHS1      LIM1   ";
        let result = read_rhs(line);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_bounds() {
        // Correct bound
        let line = " UP BND1      XONE                 4";
        let result = read_bound(line);
        assert_eq!(result, Ok(Bound {
            name: String::from("BND1"),
            direction: BoundDirection::Upper,
            variable_name: String::from("XONE"),
            variable_type: VariableType::Continuous,
            value: 4f64,
        }));

        // Missing coefficient
        let line = " UP BND1      XONE            ";
        let result = read_bound(line);
        assert!(result.is_err());

        // Missing variable name
        let line = " UP BND1                       4";
        let result = read_bound(line);
        assert!(result.is_err());

        // Missing bound name
        let line = " UP     XONE                 4";
        let result = read_bound(line);
        assert!(result.is_err());

        // Missing bound direction
        let line = "  BND1      XONE                 4";
        let result = read_bound(line);
        assert!(result.is_err());
    }

    fn lp_string() -> String {
        let mps = "NAME          TESTPROB
ROWS
 N  COST
 L  LIM1
 G  LIM2
 E  MYEQN
COLUMNS
    XONE      COST                 1   LIM1                 1
    XONE      LIM2                 1
    YTWO      COST                 4   LIM1                 1
    YTWO      MYEQN               -1
    ZTHREE    COST                 9   LIM2                 1
    ZTHREE    MYEQN                1
RHS
    RHS1      LIM1                 5   LIM2                10
    RHS1      MYEQN                7
BOUNDS
 UP BND1      XONE                 4
 LO BND1      YTWO                -1
 UP BND1      YTWO                 1
ENDATA";
        String::from(mps)
    }

    #[allow(unused)]
    fn lp_mps() -> MPS {
        let mut mps = MPS::new();
        mps.set_name(String::from("TESTPROB"));

        mps.add_row(Row::new(RowType::Cost, String::from("COST")));
        mps.add_row(Row::new(RowType::Less, String::from("LIM1")));
        mps.add_row(Row::new(RowType::Greater, String::from("LIM2")));
        mps.add_row(Row::new(RowType::Equal, String::from("MYEQN")));

        mps.add_real_column(RealColumn::new(String::from("XONE"), String::from("COST"), 1f64));
        mps.add_real_column(RealColumn::new(String::from("XONE"), String::from("LIM1"), 1f64));
        mps.add_real_column(RealColumn::new(String::from("XONE"), String::from("LIM2"), 1f64));
        mps.add_real_column(RealColumn::new(String::from("YTWO"), String::from("COST"), 4f64));
        mps.add_real_column(RealColumn::new(String::from("YTWO"), String::from("LIM1"), 1f64));
        mps.add_real_column(RealColumn::new(String::from("YTWO"), String::from("MYEQN"), -1f64));
        mps.add_real_column(RealColumn::new(String::from("ZTHREE"), String::from("COST"), 9f64));
        mps.add_real_column(RealColumn::new(String::from("ZTHREE"), String::from("LIM2"), 1f64));
        mps.add_real_column(RealColumn::new(String::from("ZTHREE"), String::from("MYEQN"), 1f64));

        let mut rhs = RHS::new(String::from("RHS1"));
        rhs.add_value(String::from("LIM1"), 5f64);
        rhs.add_value(String::from("LIM2"), 10f64);
        rhs.add_value(String::from("MYEQN"), 7f64);
        mps.add_rhs(rhs);

        mps.add_bound(Bound::new(String::from("BND1"), BoundDirection::Upper, String::from("XONE"), VariableType::Continuous, 4f64));
        mps.add_bound(Bound::new(String::from("BND1"), BoundDirection::Lower, String::from("YTWO"), VariableType::Continuous, -1f64));
        mps.add_bound(Bound::new(String::from("BND1"), BoundDirection::Upper, String::from("YTWO"), VariableType::Continuous, 1f64));

        mps
    }

    #[test]
    fn read() {
        let result = parse(lp_string());
        let expected = lp_mps();

        assert!(result.is_ok());
        assert_eq!(format!("{:?}", result.ok().unwrap()), format!("{:?}", expected));
    }

    fn lp_general() -> GeneralForm {
        let data = vec![vec![1f64, 1f64, 0f64],
                        vec![1f64, 0f64, 1f64],
                        vec![0f64, -1f64, 1f64],
                        vec![1f64, 0f64, 0f64],
                        vec![0f64, 1f64, 0f64],
                        vec![0f64, 1f64, 0f64]];
        let data = SparseMatrix::from_data(data);

        let b = DenseVector::from_data(vec![5f64,
                                            10f64,
                                            7f64,
                                            4f64,
                                            -1f64,
                                            1f64]);

        let cost = SparseVector::from_data(vec![1f64, 4f64, 9f64]);

        let column_info = vec![(String::from("XONE"), VariableType::Continuous),
                               (String::from("YTWO"), VariableType::Continuous),
                               (String::from("ZTHREE"), VariableType::Continuous)];

        let row_info = vec![RowType::Less,
                            RowType::Greater,
                            RowType::Equal,
                            RowType::Less,
                            RowType::Greater,
                            RowType::Less];

        GeneralForm::new(data, b, cost, column_info, row_info)
    }

    #[test]
    fn convert_to_general_lp() {
        let result = lp_mps().to_general_lp();
        let expected = lp_general();

        assert_eq!(format!("{:?}", result), format!("{:?}", expected));
    }

    #[test]
    fn parse_and_convert() {
        let result = parse(lp_string()).ok().unwrap().to_general_lp();
        let expected = lp_general();

        assert_eq!(format!("{:?}", result), format!("{:?}", expected));
    }
}
