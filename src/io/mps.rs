//! # Reading MPS files
//!
//! Reading of `.mps` files, or files of the Mathematical Programming System format.
//!
//! TODO:
//!     * Support all `BoundType` variants

use std::convert::{TryFrom, TryInto};
use std::collections::HashMap;

use data::linear_program::elements::{ConstraintType, Variable as ShiftedVariable, VariableType};
use data::linear_algebra::matrix::{Matrix, SparseMatrix};
use data::linear_algebra::vector::{DenseVector, SparseVector, Vector};
use data::linear_program::general_form::GeneralForm;
use io::ParseError;

/// Parse an MPS program, in string form, to a MPS.
///
/// # Arguments
///
/// * `program` - The input string in [MPS format](https://en.wikipedia.org/wiki/MPS_(format)).
///
/// # Return value
///
/// If successful, an `MPS` instance.
pub fn parse(program: String) -> Result<MPS, ParseError> {
    let atom_lines = into_atom_lines(&program);
    let unstructured_mps = UnstructuredMPS::try_from(atom_lines)?;
    let mps = unstructured_mps.try_into();

    mps
}

/// Most fundamental element in an MPS text file.
///
/// Every part of the input string to end up in the final `MPS` struct is parsed as either a `Word`
/// or a `Number`.
#[derive(Clone, Debug, PartialEq)]
enum Atom<'a> {
    Word(&'a str),
    Number(f64),
}

/// Convert a line into `Atom`s by testing whether an atom is a number.
///
/// # Arguments
///
/// * `line` - The input string slice
///
/// # Return value
///
/// A `Vec` of words and numbers.
fn into_atoms(line: &str) -> Vec<Atom> {
    line.split_whitespace()
        .map(|atom| match atom.parse::<f64>() {
            Ok(value) => Atom::Number(value),
            Err(_) => Atom::Word(atom),
        })
        .collect()
}

/// Convert an MPS program in string form to structured data.
///
/// # Arguments
///
/// * `program` - The input string
///
/// # Return value
///
/// All lines stored in a `Vec`. The first `(usize, &str)` tuple is used for creating of errors, it
/// contains the line number and line.
fn into_atom_lines<'a>(program: &'a String) -> Vec<((u64, &'a str), Vec<Atom<'a>>)> {
    program.lines()
        .enumerate()
        .map(|(number, line)| (number as u64 + 1u64, line))
        .filter(|(_, line)| !line.trim_left().starts_with("*"))
        .filter(|(_, line)| !line.is_empty())
        .map(|(number, line)| ((number, line), into_atoms(line)))
        .collect()
}

/// Every row is either a cost row or some constraint.
#[derive(Debug, Eq, PartialEq)]
enum RowType {
    Cost,
    Constraint(ConstraintType),
}

impl<'a> TryFrom<&'a str> for RowType {
    type Error = ();

    /// Try to read a `RowType` from a string slice.
    ///
    /// The type of a row is denoted by `N` if it's the cost row; this row is often unique. There is
    /// no defined behaviour for multiple cost rows. Constraint rows are indicated by `L`, `E` or
    /// `G`.
    ///
    /// # Arguments
    ///
    /// * `word` - The input `String` slice.
    ///
    /// # Return value
    ///
    /// A `RowType` variant if the `String` slice matches either `N`, `L`, `E` or `G`.
    ///
    /// # Errors
    ///
    /// Any `String` slices not equal to either `N`, `L`, `E` or `G` will fair to be parsed.
    fn try_from(word: &str) -> Result<RowType, ()> {
        match word {
            "N" => Ok(RowType::Cost),
            "L" => Ok(RowType::Constraint(ConstraintType::Less)),
            "E" => Ok(RowType::Constraint(ConstraintType::Equal)),
            "G" => Ok(RowType::Constraint(ConstraintType::Greater)),
            _ => Err(()),
        }
    }
}

/// MPS defines the `BoundType`s described in this enum.
///
/// # Note
///
/// Not all `BoundType` variants are currently supported.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum BoundType {
    /// b <- x (< +inf)
    LowerContinuous,
    /// (0 <=) x <= b
    UpperContinuous,
    /// x = b
    Fixed,
    /// -inf < x < +inf
    Free,
    /// -inf < x (<= 0)
    LowerMinusInfinity,
    /// (0 <=) x < +inf
    UpperInfinity,
    /// x = 0 or 1
    Binary,
    /// b <= x ( +inf)
    LowerInteger,
    /// (0 <=) x <= b
    UpperInteger,
    /// x = 0 or l =< x =< b
    SemiContinuous,
}

impl<'a> TryFrom<&'a str> for BoundType {
    type Error = ();

    /// Try to read a `BoundType` from a `String` slice.
    ///
    /// # Arguments
    ///
    /// * `word` - The input `String` slice.
    ///
    /// # Return value
    ///
    /// A `BoundType` variant describing the bound, if the type is known.
    ///
    /// # Errors
    ///
    /// A `()` error if the bound type is not known.
    fn try_from(word: &str) -> Result<BoundType, ()> {
        match word {
            "LO" => Ok(BoundType::LowerContinuous),
            "UP" => Ok(BoundType::UpperContinuous),
            "FX" => Ok(BoundType::Fixed),
            "FR" => Ok(BoundType::Free),
            "MI" => Ok(BoundType::LowerMinusInfinity),
            "PL" => Ok(BoundType::UpperInfinity),
            "BV" => Ok(BoundType::Binary),
            "LI" => Ok(BoundType::LowerInteger),
            "UI" => Ok(BoundType::UpperInteger),
            "SC" => Ok(BoundType::SemiContinuous),
            _ => Err(()),
        }
    }
}

/// MPS files are divided into sections.
///
/// # Note
///
/// The `Endata` variant (notice the odd spelling) denotes the end of the file.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Section<'a> {
    Name(&'a str),
    Rows,
    Columns(VariableType),
    Rhs,
    Bounds,
    Ranges,
    Endata,
}

impl<'a, 'b,> TryFrom<&'b Vec<Atom<'a>>> for Section<'a> {
    type Error = ();

    /// Try to read a `Section` from a `Vec` slice of `Atom`s.
    ///
    /// # Arguments
    ///
    /// * `line` - The input line consisting of a sequence of `Atom`s.
    ///
    /// # Return value
    ///
    /// A `Section` variant describing the section this line announces, if one is recognized.
    ///
    /// # Errors
    ///
    /// A `()` error if no `Section` is recognized.
    fn try_from(line: &Vec<Atom<'a>>) -> Result<Section<'a>, ()> {
        match line.as_slice() {
            &[Atom::Word("NAME"), Atom::Word(name)] => Ok(Section::Name(name)),
            &[Atom::Word(name)] => match name {
                "ROWS"     => Ok(Section::Rows),
                "COLUMNS"  => Ok(Section::Columns(VariableType::Continuous)),
                "RHS"      => Ok(Section::Rhs),
                "BOUNDS"   => Ok(Section::Bounds),
                "RANGES"   => Ok(Section::Ranges),
                "ENDATA"   => Ok(Section::Endata),
                _  => Err(()),
            },
            _ => Err(()),
        }
    }
}

impl<'a> TryFrom<Vec<((u64, &'a str), Vec<Atom<'a>>)>> for UnstructuredMPS<'a> {
    type Error = ParseError;

    /// Try to read an `UnstructuredMPS` struct from lines of `Atom`s.
    ///
    /// This method attempts to determine the section that is currently being parsed, and remembers
    /// that section. Subsequent lines are then parsed and collected in the `mps_*` variables.
    ///
    /// # Arguments
    ///
    /// * `program` - The input program parsed as lines of `Atom`s.
    ///
    /// # Return value
    ///
    /// An `UnstructuredMPS` instance, if no errors were encountered.
    ///
    /// # Errors
    ///
    /// A `ParseError` at failure.
    fn try_from(program: Vec<((u64, &'a str), Vec<Atom<'a>>)>) -> Result<UnstructuredMPS<'a>, ParseError> {
        let mut current_section: Option<Section> = None;

        let mut mps_name = None;
        let mut mps_cost_row_name = None;
        let mut mps_rows = Vec::new();
        let mut mps_columns = Vec::new();
        let mut mps_rhss = Vec::new();
        let mut mps_bounds = Vec::new();

        for (error_info, line) in program.into_iter() {
            if let Ok(new_section) = Section::try_from(&line) {
                if let Section::Name(name) = new_section {
                    mps_name = Some(name);
                }
                current_section = Some(new_section);
                continue;
            }

            match current_section {
                Some(Section::Endata) => break,
                Some(section) => match (section, line.as_slice()) {
                    (Section::Rows, &[Atom::Word(ty), Atom::Word(row)]) => match RowType::try_from(ty) {
                        Ok(RowType::Cost) => mps_cost_row_name = Some(row),
                        Ok(RowType::Constraint(ty)) => mps_rows.push((error_info, (row, ty))),
                        _ => panic!(),
                    },
                    (Section::Columns(marker), &[Atom::Word(column), Atom::Word(row), Atom::Number(value)]) =>
                        mps_columns.push((error_info, (column, marker, row, value))),
                    (Section::Columns(marker), &[Atom::Word(column), Atom::Word(row1), Atom::Number(value1), Atom::Word(row2), Atom::Number(value2)]) => {
                        mps_columns.push((error_info, (column, marker, row1, value1)));
                        mps_columns.push((error_info, (column, marker, row2, value2)));
                    },
                    (Section::Columns(current_marker), &[Atom::Word(_name), Atom::Word("'MARKER'"), Atom::Word(marker)]) =>
                        current_section = Some(Section::Columns(match (current_marker, marker) {
                            (VariableType::Continuous, "'INTORG'")   => VariableType::Integer,
                            (VariableType::Integer, "'INTEND'") => VariableType::Continuous,
                            _ => return Err(ParseError::new(format!("Didn't expect marker \
                            \"{}\": currently parsing variables which are {:?}", marker, current_marker), Some(error_info))),
                        })),
                    (Section::Rhs, &[Atom::Word(name), Atom::Word(row), Atom::Number(value)]) =>
                        mps_rhss.push((error_info, (name, row, value))),
                    (Section::Rhs, &[Atom::Word(name), Atom::Word(row1), Atom::Number(value1), Atom::Word(row2), Atom::Number(value2)]) => {
                        mps_rhss.push((error_info, (name, row1, value1)));
                        mps_rhss.push((error_info, (name, row2, value2)));
                    },
                    (Section::Bounds, &[Atom::Word(ty), Atom::Word(name), Atom::Word(column), Atom::Number(value)]) => match BoundType::try_from(ty) {
                        Ok(bound_type) => mps_bounds.push((error_info, (name, bound_type, column, value))),
                        _ => panic!(),
                    },
                    _ => unimplemented!(),
                },
                None => panic!(),
            };
        }

        if mps_rows.len() == 0 {
            return Err(ParseError::new("No row names read.".to_string(), None));
        }
        if mps_columns.len() == 0 {
            return Err(ParseError::new("No variables read.".to_string(), None));
        }
        if mps_rhss.len() == 0 {
            return Err(ParseError::new("No RHSs read.".to_string(), None));
        }
        if let Some(name) = mps_name {
            if let Some(cost_row_name) = mps_cost_row_name {
                Ok(UnstructuredMPS {
                    name,
                    cost_row_name,
                    rows: mps_rows,
                    columns: mps_columns,
                    rhss: mps_rhss,
                    bounds: mps_bounds,
                })
            } else {
                Err(ParseError::new("No cost row name read.".to_string(), None))
            }
        } else {
            Err(ParseError::new("No MPS name read.".to_string(), None))
        }
    }
}

/// An `UnstructedMPS` instance is used to gather all data of the `MPS`.
///
/// The struct holds all `MPS` data in an intermediate parse phase.
struct UnstructuredMPS<'a> {
    name: &'a str,
    cost_row_name: &'a str,
    rows: Vec<((u64, &'a str), (&'a str, ConstraintType))>,
    columns: Vec<((u64, &'a str), (&'a str, VariableType, &'a str, f64))>,
    rhss: Vec<((u64, &'a str), (&'a str, &'a str, f64))>,
    bounds: Vec<((u64, &'a str), (&'a str, BoundType, &'a str, f64))>,
}

impl<'a> TryInto<MPS> for UnstructuredMPS<'a> {
    type Error = ParseError;

    /// Try to convert this `UnstructedMPS` into an `MPS` instance.
    ///
    /// This method organizes and structures the unstructured information containted in the
    /// `UnstructuredMPS` instance.
    fn try_into(mut self) -> Result<MPS, ParseError> {
        let row_names = self.rows.iter().map(|&(_, (name, _))| name).collect::<Vec<_>>();
        let row_index = row_names.iter()
            .enumerate()
            .map(|(index, &name)| (name, index))
            .collect::<HashMap<&str, usize>>();

        let mut rows = Vec::new();
        for &(error_info, (name, constraint_type)) in self.rows.iter() {
            rows.push(Constraint {
                name: match row_index.get(name) {
                    Some(&index) => index,
                    None => return Err(ParseError::new(format!("Unknown row: {}", name),
                    Some(error_info))),
                },
                constraint_type,
            });
        }

        self.columns.sort_by_key(|&(_, (name, _, _, _))| name);
        let mut cost_values = Vec::new();
        let mut columns = Vec::new();
        let mut column_names = Vec::new();
        let (_, (mut current_name, mut current_type, _, _)) = self.columns[0];
        let mut values = Vec::new();
        for &(error_info, (name, variable_type, row_name, value)) in self.columns.iter() {
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

            if row_name == self.cost_row_name {
                cost_values.push((column_names.len(), value));
            } else {
                let index = match row_index.get(row_name) {
                    Some(&index) => index,
                    None => return Err(ParseError::new(format!("Row name \"{}\" not known.", row_name),
                        Some(error_info))),
                };
                values.push((index, value));
            }
        }
        columns.push(Variable { name: column_names.len(), variable_type: current_type, values, });
        column_names.push(current_name.to_string());
        let column_index = column_names.iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), index))
            .collect::<HashMap<String, usize>>();

            self.rhss.sort_by_key(|&(_, (name, _, _))| name);
        let mut rhss = Vec::new();
        let (_, (mut current_name, _, _))= self.rhss[0];
        let mut values = Vec::new();
        for &(error_info, (name, row_name, value)) in self.rhss.iter() {
            if name != current_name {
                rhss.push(Rhs { name: current_name.to_string(), values, });

                current_name = name;
                values = Vec::new()
            }
            values.push((match row_index.get(row_name) {
                Some(&index) => index,
                None => return Err(ParseError::new(format!("Row name \"{}\" not known.", row_name),
                                                   Some(error_info))),
            }, value));
        }
        rhss.push(Rhs { name: current_name.to_string(), values, });

        self.bounds.sort_by_key(|&(_, (name, _, _, _))| name);
        let mut bounds = Vec::new();
        let (_, (mut bound_name , _, _, _))= self.bounds[0];
        let mut values = Vec::new();
        for &(error_info, (name, bound_type, column_name, value)) in self.bounds.iter() {
            if name != bound_name {
                bounds.push(Bound { name: bound_name.to_string(), values, });

                bound_name = name;
                values = Vec::new();
            }
            values.push((bound_type, match column_index.get(column_name) {
                Some(&index) => index,
                None => return Err(ParseError::new(format!("Variable \"{}\" not known",
                                                           column_name), Some(error_info))),
            }, value));
        }
        bounds.push(Bound { name: bound_name.to_string(), values, });

        let name = self.name.to_string();
        let cost_row_name = self.cost_row_name.to_string();
        let row_names = row_names.into_iter()
            .map(|name| name.to_string())
            .collect();

        Ok(MPS { name, cost_row_name, cost_values, row_names, rows, column_names, columns, rhss, bounds, })
    }
}

/// Represents the contents of an MPS file in a structured manner.
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

impl Into<GeneralForm> for MPS {
    /// Convert an `MPS` into a `GeneralForm` linear program.
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

/// Every `Row` has a name and a `RowType`.
#[derive(Debug, Eq, PartialEq)]
struct Constraint {
    name: usize,
    constraint_type: ConstraintType,
}

/// A `Variable` is either continuous or integer, and has for some rows a coefficient.
#[derive(Debug, PartialEq)]
struct Variable {
    name: usize,
    variable_type: VariableType,
    values: Vec<(usize, f64)>,
}

/// A `RHS` is a constraint. A single linear program defined in MPS can have multiple right-hand
/// sides. It relates a row name to a real constant.
#[derive(Debug, PartialEq)]
struct Rhs {
    name: String,
    values: Vec<(usize, f64)>,
}

/// A `Bound` gives a bound on a variable. The variable can either be continuous or integer, while
/// the bound can have any direction.
#[derive(Debug, PartialEq)]
struct Bound {
    name: String,
    values: Vec<(BoundType, usize, f64)>,
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_into_atoms() {
        use super::Atom::*;

        macro_rules! test {
            ($line:expr, [$($words:expr), *]) => {
                let result = into_atoms($line);
                let expected = vec![$($words), *];
                assert_eq!(result, expected);
            }
        }

        test!("NAME TESTPROB", [Word("NAME"), Word("TESTPROB")]);
        test!("ROWS", [Word("ROWS")]);
        test!("RHS     ", [Word("RHS")]);
        test!("NUMBER 134", [Word("NUMBER"), Number(134f64)]);
        test!("NUMBER 1.6734", [Word("NUMBER"), Number(1.6734f64)]);
        test!("    MARK0000  'MARKER'                 'INTORG'",
              [Word("MARK0000"), Word("'MARKER'"), Word("'INTORG'")]);
    }

    #[test]
    fn test_into_atom_lines() {
        use io::mps::Atom::*;

        let program = "SOMETEXT 1.2\n*comment\n\n   \t line before is empty".to_string();
        let result = into_atom_lines(&program);
        let expected = vec![((1, "SOMETEXT 1.2"), vec![Word("SOMETEXT"), Number(1.2f64)]),
                            ((4, "   \t line before is empty"),
                                vec![Word("line"), Word("before"), Word("is"), Word("empty")])];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_try_from_row_type() {
        macro_rules! test {
            ($word:expr, $expected:expr) => {
                let result = RowType::try_from($word);
                assert_eq!(result, $expected);
            }
        }

        test!("N", Ok(RowType::Cost));
        test!("L", Ok(RowType::Constraint(ConstraintType::Less)));
        test!("E", Ok(RowType::Constraint(ConstraintType::Equal)));
        test!("G", Ok(RowType::Constraint(ConstraintType::Greater)));
        test!("X", Err(()));
        test!("", Err(()));
        test!("\t", Err(()));
    }

    #[test]
    fn test_try_from_bound_type() {
        macro_rules! test {
            ($word:expr, $expected:expr) => {
                let result = BoundType::try_from($word);
                assert_eq!(result, $expected);
            }
        }

        test!("LO", Ok(BoundType::LowerContinuous));
        test!("UP", Ok(BoundType::UpperContinuous));
        test!("FX", Ok(BoundType::Fixed));
        test!("FR", Ok(BoundType::Free));
        test!("MI", Ok(BoundType::LowerMinusInfinity));
        test!("PL", Ok(BoundType::UpperInfinity));
        test!("BV", Ok(BoundType::Binary));
        test!("LI", Ok(BoundType::LowerInteger));
        test!("UI", Ok(BoundType::UpperInteger));
        test!("SC", Ok(BoundType::SemiContinuous));
        test!("X", Err(()));
        test!("", Err(()));
        test!("\t", Err(()));
    }

    #[test]
    fn test_try_from_section() {
        use super::Atom::*;

        macro_rules! test {
            ([$($words:expr), *], $expected:expr) => {
                let result = Section::try_from(&vec![$($words), *]);
                assert_eq!(result, $expected);
            }
        }

        test!([Word("NAME"), Word("THENAME")], Ok(Section::Name("THENAME")));
        test!([Word("ROWS")], Ok(Section::Rows));
        test!([Word("COLUMNS")], Ok(Section::Columns(VariableType::Continuous)));
        test!([Word("RHS")], Ok(Section::Rhs));
        test!([Word("BOUNDS")], Ok(Section::Bounds));
        test!([Word("RANGES")], Ok(Section::Ranges));
        test!([Word("ENDATA")], Ok(Section::Endata));
        test!([Word("X")], Err(()));
        test!([Number(1.556f64)], Err(()));
        test!([], Err(()));
    }

    const MPS_STR: &str = "NAME          TESTPROB
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

    fn mps_string() -> String {
        MPS_STR.to_string()
    }

    fn lp_mps() -> MPS {
        let name = "TESTPROB".to_string();
        let cost_row_name = "COST".to_string();
        let cost_values = vec![(0, 1f64), (1, 4f64), (2, 9f64)];
        let row_names = vec!["LIM1", "LIM2", "MYEQN"].into_iter().map(String::from).collect();
        let rows = vec![Constraint { name: 0, constraint_type: ConstraintType::Less, },
                        Constraint { name: 1, constraint_type: ConstraintType::Greater, },
                        Constraint { name: 2, constraint_type: ConstraintType::Equal, }];
        let column_names = vec!["XONE", "YTWO", "ZTHREE"].into_iter().map(String::from).collect();
        let columns = vec![Variable { name: 0, variable_type: VariableType::Continuous, values: vec![(0, 1f64),
                                                                                                     (1, 1f64)], },
                           Variable { name: 1, variable_type: VariableType::Continuous, values: vec![(0, 1f64),
                                                                                                     (2, -1f64)], },
                           Variable { name: 2, variable_type: VariableType::Continuous, values: vec![(1, 1f64),
                                                                                                     (2, 1f64)], }];
        let rhss = vec![Rhs { name: "RHS1".to_string(), values: vec![(0, 5f64), (1, 10f64), (2, 7f64)], }];
        let bounds = vec![Bound { name: "BND1".to_string(), values: vec![(BoundType::UpperContinuous, 0, 4f64),
                                                                         (BoundType::LowerContinuous, 1, -1f64),
                                                                         (BoundType::UpperContinuous, 1, 1f64)], }];

        MPS { name, cost_row_name, cost_values, row_names, rows, column_names, columns, rhss, bounds, }
    }

    #[test]
    fn read() {
        let result = parse(mps_string());
        let expected = lp_mps();

        assert_eq!(result.unwrap(), expected);
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

        let column_info = vec![ShiftedVariable { name: "XONE".to_string(), variable_type: VariableType::Continuous, offset: 0f64, },
                               ShiftedVariable { name: "YTWO".to_string(), variable_type: VariableType::Continuous, offset: 0f64, },
                               ShiftedVariable { name: "ZTHREE".to_string(), variable_type: VariableType::Continuous, offset: 0f64, }];

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
        let result: GeneralForm = lp_mps().into();
        let expected = lp_general();

        assert_eq!(format!("{:?}", result), format!("{:?}", expected));
    }

    #[test]
    fn parse_and_convert() {
        let result: GeneralForm = parse(mps_string()).unwrap().into();
        let expected = lp_general();

        assert_eq!(format!("{:?}", result), format!("{:?}", expected));
    }
}
