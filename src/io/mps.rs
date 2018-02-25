//! # Reading MPS files
//!
//! Reading of `.mps` files, or files of the Mathematical Programming System format.

use std::collections::HashMap;
use std::iter::Iterator;
use std::str::Lines;

use data::linear_program::elements::{ConstraintType, Variable as ShiftedVariable, VariableType};
use data::linear_program::general_form::{GeneralForm, GeneralFormConvertable};
use data::linear_algebra::matrix::{Matrix, SparseMatrix};
use data::linear_algebra::vector::{DenseVector, SparseVector, Vector};


/// Takes as argument a string in which the linear program is encoded following the
/// [MPS format](https://en.wikipedia.org/wiki/MPS_(format)).
pub fn parse(mut lines: Lines) -> Result<Box<GeneralFormConvertable>, String> {
    let name = read_name(lines.next())?;
    lines.next();
    let (cost_row_name, constraints) = parse_rows(&mut lines)?;
    let columns = parse_columns(&mut lines)?;
    let rhs = parse_rhs(&mut lines)?;
    let bounds = match parse_bounds(&mut lines) {
        Ok(bounds) => bounds,
        Err(e) => Vec::new(),
    };

    Ok(Box::new(MPS { name, cost_row_name, rows: constraints, columns, rhss: rhs, bounds, }))
}

/// Read the name
fn read_name(line: Option<&str>) -> Result<String, String> {
    match line {
        Some(string) => {
            let parts = string.split_whitespace().collect::<Vec<&str>>();
            if parts.len() != 2 || parts[0] != "NAME" {
                Err(format!("This is not a \"NAME\" line: {}", string))
            } else {
                Ok(String::from(parts[1]))
            }
        },
        None => Err(String::from("Empty line")),
    }
}

/// Read the ROWS section
fn parse_rows(lines: &mut Lines) -> Result<(String, Vec<Constraint>), String> {
    let mut constraints: Vec<Constraint> = Vec::new();
    let mut cost_row_name = None;

    for line in lines {
        match parse_section(line) {
            Some(MPSSection::Columns(VariableType::Continuous)) => break,
            Some(section) => return Err(format!("Didn't expect section \"{:?}\"", section)),
            None => {
                let mut parts = line.split_whitespace();
                let constraint_type = match parts.next() {
                    Some("N") => {
                        cost_row_name = match parts.next() {
                            Some(name) => Some(String::from(name)),
                            None => return Err(format!("No row name: {}", line)),
                        };
                        continue;
                    },
                    Some("L") => ConstraintType::Less,
                    Some("E") => ConstraintType::Equal,
                    Some("G") => ConstraintType::Greater,
                    _ => return Err(format!("No row type: {}", line)),
                };
                let row_name = match parts.next() {
                    Some(name) => String::from(name),
                    None => return Err(format!("No row name: {}", line)),
                };

                let constraint = Constraint::new(row_name, constraint_type);
                constraints.push(constraint);
            }
        }
    }

    match cost_row_name {
        Some(name) => Ok((name, constraints)),
        None => Err(format!("Did not read cost row name")),
    }
}

/// Read the column values
fn parse_columns(lines: &mut Lines) -> Result<Vec<Variable>, String> {
    let mut columns = Vec::new();
    let mut variable_type = VariableType::Continuous;

    let mut previous_variable_name = None;
    let mut coefficients = Vec::new();
    for line in lines {
        match parse_section(line) {
            Some(MPSSection::Columns(new_type)) => {
                variable_type = new_type;
                continue;
            },
            Some(MPSSection::RHS) => break,
            Some(section) => return Err(format!("Section \"{:?}\" not expected", section)),
            None => (),
        }

        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("RHS") => {
                break;
            },
            None => return Err(format!("Empty line")),
            variable_name => {
                if variable_name != previous_variable_name {
                    if previous_variable_name != None {
                        columns.push(Variable {
                            variable_name: String::from(previous_variable_name.unwrap()),
                            variable_type,
                            coefficients,
                        });
                        coefficients = Vec::new();
                    }
                    previous_variable_name = variable_name;
                }

                for _ in 0..2 {
                    let row_name = match parts.next() {
                        Some(name) => String::from(name),
                        None => continue,
                    };
                    let value = match parts.next() {
                        Some(number) => match number.parse::<f64>() {
                            Ok(f) => f,
                            Err(message) => return Err(format!("Column value \"{}\" could not be \
                        parsed from line \"{}\" : {}", number, line, message)),
                        },
                        None => return Err(format!("Column value not found: {}", line)),
                    };
                    coefficients.push((row_name, value));
                }
            },
        }
    }

    if coefficients.len() > 0 {
        columns.push(Variable {
            variable_name: match previous_variable_name {
                Some(name) => String::from(name),
                None => return Err(format!("No column values read")),
            },
            variable_type,
            coefficients,
        });
    }

    Ok(columns)
}

/// Read a `RHS` or right-hand side
fn parse_rhs(lines: &mut Lines) -> Result<Vec<Rhs>, String> {
    let mut rhss = Vec::new();

    let mut previous_rhs_name = None;
    let mut values = Vec::new();
    for line in lines {
        match parse_section(line) {
            // The next section is starting
            Some(MPSSection::Bounds) | Some(MPSSection::ENDATA) => {
                match previous_rhs_name {
                    Some(name) => rhss.push(Rhs {
                        name: String::from(name),
                        values: values.clone(),
                    }),
                    None => return Err(format!("Reached RHS section, no RHS found")),
                };
                break;
            },
            // An unexpected section
            Some(other_section) => {
                return Err(format!("Didn't expect section {:?}", other_section));
            },
            // No new section, we expect one or two RHS values
            None => {
                let mut parts = line.split_whitespace();
                match parts.next() {
                    None => return Err(format!("No RHS name on line \"{}\"", line)),
                    rhs_name => {
                        if rhs_name != previous_rhs_name {
                            if previous_rhs_name != None {
                                rhss.push(Rhs {
                                    name: String::from(rhs_name.unwrap()),
                                    values: values.clone(),
                                });
                                values = Vec::new();
                            }
                            previous_rhs_name = rhs_name;
                        }

                        for _ in 0..2 {
                            let row_name = match parts.next() {
                                Some(name) => String::from(name),
                                None => continue,
                            };
                            let value = match parts.next() {
                                Some(number) => match number.parse::<f64>() {
                                    Ok(f) => f,
                                    Err(message) => return Err(format!("Column value \"{}\" could not be \
                        parsed from line \"{}\" : {}", number, line, message)),
                                },
                                None => return Err(format!("Column value not found: {}", line)),
                            };
                            values.push((row_name, value));
                        }
                    },
                }
            },
        }
    }

    Ok(rhss)
}

/// Read a `Bound` from a string
fn parse_bounds(lines: &mut Lines) -> Result<Vec<Bound>, String> {
    let mut bounds = Vec::new();

    for line in lines {
        match parse_section(&line) {
            Some(MPSSection::ENDATA) => break,
            Some(section) => return Err(format!("Did not expect section \"{:?}\"", section)),
            None => {
                let mut parts = line.split_whitespace();

                let direction = match parts.next() {
                    Some("UP") => BoundDirection::Upper,
                    Some("LO") => BoundDirection::Lower,
                    Some("FX") => BoundDirection::Fixed,
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
                bounds.push(Bound { name, direction, variable_name, value, });
            }
        }
    }

    Ok(bounds)
}


/// Determines, it possible, which section of the MPS file the line announces.
///
/// If no new section is found, `None` is returned.
fn parse_section(line: &str) -> Option<MPSSection> {
    if line.starts_with(" ") {
        // Check for a nested section
        let mut parts = line.split_whitespace();
        parts.next();
        if let Some("'MARKER'") = parts.next() {
            match parts.next() {
                Some("'INTORG'") => Some(MPSSection::Columns(VariableType::Integer)),
                Some("'INTEND'") => Some(MPSSection::Columns(VariableType::Continuous)),
                _ => panic!("Marker not recognized on line {}", line),
            }
        } else { None }
    }
    else if line.starts_with("NAME") { Some(MPSSection::Name) }
    else if line.starts_with("ROWS") { Some(MPSSection::Rows) }
    else if line.starts_with("COLUMNS") { Some(MPSSection::Columns(VariableType::Continuous)) }
    else if line.starts_with("RHS") { Some(MPSSection::RHS) }
    else if line.starts_with("BOUNDS") { Some(MPSSection::Bounds) }
    else if line.starts_with("ENDATA") { Some(MPSSection::ENDATA) }
    else { None }
}

/// Represents the contents of an MPS file.
///
/// Information on variables is spread throughout the `rows`, `real_columns`, `integer_columns`,
/// `rhs` and `bounds` fields.
#[derive(Debug)]
struct MPS {
    // Name of the linear program
    name: String,
    // Name of the cost row
    cost_row_name: String,
    // All named constraints
    rows: Vec<Constraint>,
    // Constraint name and Variable name combinations
    columns: Vec<Variable>,
    // Right-hand side constraint values
    rhss: Vec<Rhs>,
    // Extra bounds on variables
    bounds: Vec<Bound>,
}

impl MPS {
    fn get_rows_map(&self) -> (HashMap<String, usize>, usize) {
        let rows_map = self.rows
            .iter()
            .map(|row| row.name.clone())
            .enumerate()
            .map(|(index, name)| (name, index))
            .collect::<HashMap<String, usize>>();
        let nr_rows = rows_map.len();
        (rows_map, nr_rows)
    }
    fn get_columns_map(&self) -> (HashMap<String, usize>, usize) {
        let columns_map = self.columns
            .iter()
            .map(|variable| variable.variable_name.clone())
            .enumerate()
            .map(|(index, name)| (name, index))
            .collect::<HashMap<String, usize>>();
        let nr_columns = columns_map.len();
        (columns_map, nr_columns)
    }
    fn get_data_and_cost(&self,
                         rows_map: &HashMap<String, usize>,
                         columns_map: &HashMap<String, usize>) -> (SparseMatrix, SparseVector) {
        let (nr_rows, nr_columns) = (rows_map.len(), columns_map.len());
        let mut data = SparseMatrix::zeros(nr_rows + self.bounds.len(), nr_columns);
        let mut cost = SparseVector::zeros(nr_columns);

        for column in &self.columns {
            let column_index = *columns_map.get(&column.variable_name).unwrap();
            for coefficient in &column.coefficients {
                let value = coefficient.1;

                if coefficient.0 == self.cost_row_name {
                    cost.set_value(column_index, value);
                } else {
                    let row_index = *rows_map.get(&coefficient.0).unwrap();
                    data.set_value(row_index, column_index, value);
                }
            }
        }

        (data, cost)
    }
    fn get_b(&self, rows_map: &HashMap<String, usize>) -> DenseVector {
        // TODO: Don't just get the first RHS
        let rhs = &self.rhss[0];
        let nr_rows = rows_map.len();
        let b_tuples: Vec<(usize, f64)> = rhs.values.iter()
            .map(|&(ref row_name, value)| {
                let row_nr = *rows_map.get(row_name).unwrap();
                (row_nr, value)
            }).collect();

        let mut b = DenseVector::zeros(nr_rows + self.bounds.len());
        for (index, value) in b_tuples {
            b.set_value(index, value);
        }
        b
    }
    fn get_row_info(&self, rows_map: &HashMap<String, usize>) -> Vec<ConstraintType> {
        let mut row_info = self.rows.clone();
        row_info.sort_by_key(|row| *rows_map.get(&row.name).unwrap());
        let row_info = row_info.into_iter()
            .map(|row| row.constraint_type)
            .collect::<Vec<ConstraintType>>();
        row_info
    }
    fn get_column_info(&self, columns_map: &HashMap<String, usize>) -> Vec<ShiftedVariable> {
        let mut column_info = self.columns.iter()
            .map(|column| (column.variable_name.clone(), column.variable_type))
            .collect::<Vec<(String, VariableType)>>();
        column_info.sort_by_key(|(name, _)| *columns_map.get(&name.to_owned()).unwrap());
        column_info.into_iter()
            .map(|(name, variable_type)| ShiftedVariable::new(name, variable_type, 0f64))
            .collect()
    }
}

/// Describes how to convert an `MPS` to a `GeneralForm`, or linear program in general form.
impl GeneralFormConvertable for MPS {
    fn to_general_lp(&self) -> GeneralForm {
        let (rows_map, nr_rows) = self.get_rows_map();
        let (columns_map, _) = self.get_columns_map();
        let (mut data, cost) = self.get_data_and_cost(&rows_map, &columns_map);
        let mut b = self.get_b(&rows_map);
        let mut row_info = self.get_row_info(&rows_map);
        let column_info = self.get_column_info(&columns_map);

        let nr_ordinary_constraints = nr_rows;
        for (bound_number, bound) in self.bounds.iter().enumerate() {
            let row_index = nr_ordinary_constraints + bound_number;
            b.set_value(row_index, bound.value);
            row_info.push(match bound.direction {
                BoundDirection::Lower => ConstraintType::Greater,
                BoundDirection::Upper => ConstraintType::Less,
                BoundDirection::Fixed => ConstraintType::Equal,
            });
            let column_index = *columns_map.get(&bound.variable_name).unwrap();
            data.set_value(row_index, column_index, 1f64);
        }

        GeneralForm::new(data, b, cost, 0f64, column_info, Vec::new(), row_info)
    }
}

/// All sections in a MPS file.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum MPSSection {
    Name,
    Rows,
    Columns(VariableType),
    RHS,
    Bounds,
    ENDATA,
}

/// Every `Row` has a name and a `RowType`.
#[derive(Clone, Debug, Eq, PartialEq)]
struct Constraint {
    name: String,
    constraint_type: ConstraintType,
}

impl Constraint {
    /// Create a new `Constraint`.
    pub fn new(name: String, constraint_type: ConstraintType) -> Constraint {
        Constraint { name, constraint_type, }
    }
}

/// A `Variable` is either continuous or integer, and has for some rows a coefficient.
#[derive(Debug, PartialEq)]
struct Variable {
    variable_name: String,
    variable_type: VariableType,
    coefficients: Vec<(String, f64)>,
}

impl Variable {
    /// Create a new `Column`
    pub fn new(variable_name: String, variable_type: VariableType, coefficients: Vec<(String, f64)>) -> Variable {
        Variable { variable_name, variable_type, coefficients, }
    }
}

/// A `RHS` is a constraint. A single linear program defined in MPS can have multiple right-hand
/// sides. It relates a row name to a real constant.
#[derive(Debug, PartialEq)]
struct Rhs {
    name: String,
    values: Vec<(String, f64)>,
}

/// A `Bound` gives a bound on a variable. The variable can either be continuous or integer, while
/// the bound can have any direction.
#[derive(Debug, PartialEq)]
struct Bound {
    name: String,
    direction: BoundDirection,
    variable_name: String,
    value: f64,
}

impl Bound {
    pub fn new(name: String, direction: BoundDirection, variable_name: String, value: f64) -> Bound {
        Bound { name, direction, variable_name, value, }
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
        assert_eq!(result, Some(MPSSection::Columns(VariableType::Continuous)));

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
        assert_eq!(result, Some(MPSSection::Columns(VariableType::Integer)));

        let line = "DIFFERENT";
        let result = parse_section(line);
        assert_eq!(result, None);
    }

    #[test]
    fn test_read_name() {
        // Correct line
        let name = Some("NAME          TESTPROB");
        let result = read_name(name);
        assert_eq!(result, Ok(String::from("TESTPROB")));

        // Incorrect indicator
        let name = Some("X TESTPROB");
        let result = read_name(name);
        assert!(result.is_err());

        // Missing name
        let name = Some("NAME          ");
        let result = read_name(name);
        assert!(result.is_err());

        // Empty line
        let name = Some("");
        let result = read_name(name);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_rows() {
        // Cost row
        let rows = " N  COST\n L  LIM1\n E  MYEQN\n G  LIM2";
        let result = parse_rows(&mut rows.lines());
        let expected = Ok((String::from("COST"),
                           vec![Constraint::new(String::from("LIM1"), ConstraintType::Less),
                                Constraint::new(String::from("MYEQN"), ConstraintType::Equal),
                                Constraint::new(String::from("LIM2"), ConstraintType::Greater)]));
        assert_eq!(result, expected);

        // Missing row name
        let rows = " E ";
        let result = parse_rows(&mut rows.lines());
        assert!(result.is_err());

        // Unknown row type
        let rows = " X  ROWNAME";
        let result = parse_rows(&mut rows.lines());
        assert!(result.is_err());

        // Empty line
        let rows = "";
        let result = parse_rows(&mut rows.lines());
        assert!(result.is_err());
    }

    #[test]
    fn test_read_columns() {
        // Correct line, two coefficients
        let columns = "    XONE      COST                 1   LIM1                 1";
        let result = parse_columns(&mut columns.lines());
        let expected = Ok(vec![Variable::new(String::from("XONE"),
                               VariableType::Continuous,
                               vec![(String::from("COST"), 1f64),
                                    (String::from("LIM1"), 1f64)])]);
        assert_eq!(result, expected);

        // Correct line, one coefficient
        let columns = "    ZTHREE    MYEQN                1";
        let result = parse_columns(&mut columns.lines());
        let expected = Ok(vec![Variable::new(String::from("ZTHREE"),
                               VariableType::Continuous,
                               vec![(String::from("MYEQN"), 1f64)])]);
        assert_eq!(result, expected);

        // Missing coefficient
        let columns = "    XONE      COST                 1   LIM1                 ";
        let result = parse_columns(&mut columns.lines());
        assert!(result.is_err());

        // Missing row name
        let columns = "    XONE      COST                 1           1";
        let result = parse_columns(&mut columns.lines());

        assert!(result.is_err());

        // Missing variable name
        let columns = "      COST                 1   LIM1                 1";
        let result = parse_columns(&mut columns.lines());
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_rhs() {
        // Correct line, two coefficients
        let rhs = "    RHS1      LIM1                 5   LIM2                10\nBOUNDS";
        let result = parse_rhs(&mut rhs.lines());
        let expected = Ok(vec![Rhs {
            name: String::from("RHS1"),
            values: vec![(String::from("LIM1"), 5f64),
                         (String::from("LIM2"), 10f64)],
        }]);
        assert_eq!(result, expected);

        // Correct line, one coefficient
        let rhs = "    RHS1      MYEQN                7\nBOUNDS";
        let result = parse_rhs(&mut rhs.lines());
        let expected = Ok(vec![Rhs {
            name: String::from("RHS1"),
            values: vec![(String::from("MYEQN"), 7f64)],
        }]);
        assert_eq!(result, expected);

        // Missing coefficient
        let rhs = "    RHS1      LIM1                 5   LIM2           ";
        let result = parse_rhs(&mut rhs.lines());
        assert!(result.is_err());

        // Missing row name
        let rhs = "    RHS1      LIM1                 5                  10";
        let result = parse_rhs(&mut rhs.lines());
        assert!(result.is_err());

        // Missing rhs name
        let rhs = "         LIM1                 5   LIM2                10";
        let result = parse_rhs(&mut rhs.lines());
        assert!(result.is_err());

        // Missing coefficient
        let rhs = "    RHS1      LIM1   ";
        let result = parse_rhs(&mut rhs.lines());
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_bounds() {
        // Correct bound
        let bound = " UP BND1      XONE                 4\nENDATA";
        let result = parse_bounds(&mut bound.lines());
        assert_eq!(result, Ok(vec![Bound {
            name: String::from("BND1"),
            direction: BoundDirection::Upper,
            variable_name: String::from("XONE"),
            value: 4f64,
        }]));

        // Missing coefficient
        let bound = " UP BND1      XONE            ";
        let result = parse_bounds(&mut bound.lines());
        assert!(result.is_err());

        // Missing variable name
        let bound = " UP BND1                       4";
        let result = parse_bounds(&mut bound.lines());
        assert!(result.is_err());

        // Missing bound name
        let bound = " UP     XONE                 4";
        let result = parse_bounds(&mut bound.lines());
        assert!(result.is_err());

        // Missing bound direction
        let bound = "  BND1      XONE                 4";
        let result = parse_bounds(&mut bound.lines());
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
        let name = String::from("TESTPROB");
        let cost_row_name = String::from("COST");
        let rows = vec![Constraint::new(String::from("LIM1"), ConstraintType::Less),
                        Constraint::new(String::from("LIM2"), ConstraintType::Greater),
                        Constraint::new(String::from("MYEQN"), ConstraintType::Equal)];

        let columns = vec![Variable::new(String::from("XONE"),
                                         VariableType::Continuous,
                                         vec![(String::from("COST"), 1f64),
                                              (String::from("LIM1"), 1f64),
                                              (String::from("LIM2"), 1f64)]),
                           Variable::new(String::from("YTWO"),
                                         VariableType::Continuous,
                                         vec![(String::from("COST"), 4f64),
                                              (String::from("LIM1"), 1f64),
                                              (String::from("MYEQN"), -1f64)]),
                           Variable::new(String::from("ZTHREE"),
                                         VariableType::Continuous,
                                         vec![(String::from("COST"), 9f64),
                                              (String::from("LIM2"), 1f64),
                                              (String::from("MYEQN"), 1f64)])];


        let rhss = vec![Rhs {
            name: String::from("RHS1"),
            values: vec![(String::from("LIM1"), 5f64),
                         (String::from("LIM2"), 10f64),
                         (String::from("MYEQN"), 7f64)],
        }];


        let bounds = vec![Bound::new(String::from("BND1"), BoundDirection::Upper, String::from("XONE"), 4f64),
                          Bound::new(String::from("BND1"), BoundDirection::Lower, String::from("YTWO"), -1f64),
                          Bound::new(String::from("BND1"), BoundDirection::Upper, String::from("YTWO"), 1f64)];

        MPS { name, cost_row_name, rows, columns, rhss, bounds, }
    }

    #[test]
    fn read() {
        let result = parse(lp_string().lines());
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

        let column_info = vec![ShiftedVariable::new(String::from("XONE"), VariableType::Continuous, 0f64),
                               ShiftedVariable::new(String::from("YTWO"), VariableType::Continuous, 0f64),
                               ShiftedVariable::new(String::from("ZTHREE"), VariableType::Continuous, 0f64)];

        let row_info = vec![ConstraintType::Less,
                            ConstraintType::Greater,
                            ConstraintType::Equal,
                            ConstraintType::Less,
                            ConstraintType::Greater,
                            ConstraintType::Less];

        GeneralForm::new(data, b, cost, 0f64, column_info, Vec::new(), row_info)
    }

    #[test]
    fn convert_to_general_lp() {
        let result = lp_mps().to_general_lp();
        let expected = lp_general();

        assert_eq!(format!("{:?}", result), format!("{:?}", expected));
    }

    #[test]
    fn parse_and_convert() {
        let result = parse(lp_string().lines()).ok().unwrap().to_general_lp();
        let expected = lp_general();

        assert_eq!(format!("{:?}", result), format!("{:?}", expected));
    }
}
