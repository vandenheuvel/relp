//! # Parsing MPS files
//!
//! First stage of importing linear programs. Checks if the MPS file is syntactically correct, but
//! doesn't do any consistency checks for e.g. undefined row names in the column section.
use std::convert::TryFrom;
use std::error::Error;

use data::linear_program::elements::{ConstraintType, VariableType};
use io::error::{ParseError, FileLocation};
use io::mps::token::{COMMENT_INDICATOR, COLUMN_SECTION_MARKER, END_OF_INTEGER, START_OF_INTEGER};
use io::mps::BoundType;
use io::mps::RowType;
use io::mps::Section;

use self::Atom::{Word, Number};
use approx::AbsDiffEq;
use io::EPSILON;
use std::convert::identity;


/// Most fundamental element in an MPS text file.
///
/// Every part of the input string to end up in the final `MPS` struct is parsed as either a `Word`
/// or a `Number`.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Atom<'a> {
    /// A token consisting of text which should not contain whitespace.
    Word(&'a str),
    /// A token representing a number.
    Number(f64),
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
pub(super) fn into_atom_lines(program: &impl AsRef<str>) -> Vec<((u64, &str), Vec<Atom>)> {
    program.as_ref()
        .lines()
        .enumerate()
        .map(|(number, line)| (number as u64 + 1u64, line))
        .filter(|(_, line)| !line.trim_left().starts_with(COMMENT_INDICATOR))
        .filter(|(_, line)| !line.is_empty())
        .map(|(number, line)| ((number, line), into_atoms(line)))
        .collect()
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
            Ok(value) => Number(value),
            Err(_) => Word(atom),
        })
        .collect()
}

/// Used to gather all data of the `MPS`.
///
/// The struct holds all `MPS` data in an intermediate parse phase.
///
/// # Note
///
/// The information contained in this struct is not necessarily consistent, e.g. some columns
/// might reference rows which are not declared.
#[derive(Debug, PartialEq)]
pub(super) struct UnstructuredMPS<'a> {
    /// Name of the linear program
    pub name: &'a str,
    /// Name of the cost row, or objective function
    pub cost_row_name: &'a str,
    /// Names of all constraint rows and their constraint type
    pub rows: Vec<UnstructuredRow<'a>>,
    /// Names and type (continuous, discrete) of all variables
    pub columns: Vec<UnstructuredColumn<'a>>,
    /// Right-hand sides giving a numerical value to the constraints
    pub rhss: Vec<UnstructuredRhs<'a>>,
    /// Bounds of the problem (does not includes elements of the "RANGES" section.
    pub bounds: Vec<UnstructuredBound<'a>>,
}

/// Name and constraint type of a row
#[derive(Debug, PartialEq, Eq)]
pub(super) struct UnstructuredRow<'a> {
    pub name: &'a str,
    pub constraint_type: ConstraintType,
}
/// Name of a column, variable type of a column, name of a row and a value (a constraint matrix
/// entry in sparse description)
#[derive(Debug, PartialEq)]
pub(super) struct UnstructuredColumn<'a> {
    pub name: &'a str,
    pub variable_type: VariableType,
    pub row_name: &'a str,
    pub value: f64,
}
/// Name of the right-hand side, name of the variable and constraint value
#[derive(Debug, PartialEq)]
pub(super) struct UnstructuredRhs<'a> {
    pub name: &'a str,
    pub row_name: &'a str,
    pub value: f64,
}
/// Name of the bound, bound type, name of the variable and constraint value
#[derive(Debug, PartialEq)]
pub(super) struct UnstructuredBound<'a> {
    pub name: &'a str,
    pub bound_type: BoundType,
    pub column_name: &'a str,
    pub value: f64,
}
/// TODO: Support the RANGES section
#[derive(Debug, PartialEq)]
pub(super) struct UnstructuredRange {
}

impl<'a> TryFrom<Vec<(FileLocation<'a>, Vec<Atom<'a>>)>> for UnstructuredMPS<'a> {
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
    fn try_from(atom_lines: Vec<(FileLocation<'a>, Vec<Atom<'a>>)>) -> Result<UnstructuredMPS<'a>, Self::Error> {
        let mut current_section: Option<Section> = None;

        let mut name = None;
        let mut cost_row_name = None;
        let mut rows = Vec::new();
        let mut columns = Vec::new();
        let mut rhss = Vec::new();
        let mut bounds = Vec::new();
        let mut ranges = Vec::new();

        for (file_location, line) in atom_lines.into_iter() {
            if let Ok(new_section) = Section::try_from(&line) {
                match new_section {
                    Section::Name(new_program_name) => {
                        name = Some(new_program_name);
                        current_section = None;
                    },
                    Section::Endata => break,
                    _other_section => current_section = Some(new_section),
                }
                continue;
            }

            match current_section {
                None => Err(ParseError::new("Section unknown, can't parse line")),
                Some(Section::Name(_)) => panic!("The name should be updated in section parsing"),
                Some(Section::Rows) => parse_row_line(line, &mut cost_row_name, &mut rows),
                Some(Section::Columns(ref mut marker)) => parse_column_line(line, marker, &mut columns),
                Some(Section::Rhs) => parse_rhs_line(line, &mut rhss),
                Some(Section::Bounds) => parse_bound_line(line, &mut bounds),
                Some(Section::Ranges) => parse_range_line(line, &mut ranges),
                Some(Section::Endata) => panic!("Row parsing should have been aborted after Endata detection"),
            }.map_err(|parse_error|
                ParseError::with_file_location(parse_error.description(), file_location))?;
        }

        UnstructuredMPS::all_required_fields_present(&name, &cost_row_name, &rows, &columns, &rhss)
            .map(|_| UnstructuredMPS {
                name: name.unwrap(),
                cost_row_name: cost_row_name.unwrap(),
                rows,
                columns,
                rhss,
                bounds,
            })
    }
}

/// Parse a line of atoms which describes a row entry.
///
/// # Arguments
///
/// * `line` - The tokens on the line.
/// * `cost_row_name` - The previously parsed name of the cost row.
/// * `row_collector` - Data structure in which to collect the parsed rows.
///
/// # Return value
///
/// A ParseError if parsing fails.
fn parse_row_line<'a>(line: Vec<Atom<'a>>,
                      cost_row_name: &mut Option<&'a str>,
                      row_collector: &mut Vec<UnstructuredRow<'a>>) -> Result<(), ParseError> {
    match line.as_slice() {
        &[Word(ty), Word(row_name)] => match RowType::try_from(ty)? {
            RowType::Cost => *cost_row_name = Some(row_name),
            RowType::Constraint(ty) => row_collector.push(UnstructuredRow {
                name: row_name, constraint_type: ty,
            }),
        },
        _ => return Err(ParseError::new("Line can't be parsed as part of the row section."))
    }

    Ok(())
}

/// Parse a line of atoms which describes a column entry.
///
/// # Arguments
///
/// * `line` - The tokens on the line.
/// * `marker` - Indicator for the continuous or discrete part of the column section.
/// * `column_collector` - Data structure in which to collect the parsed values.
///
/// # Return value
///
/// A new variable type, indicating that the integer section of the columns section has started or
/// ended.
fn parse_column_line<'a>(line: Vec<Atom<'a>>,
                         marker: &mut VariableType,
                         column_collector: &mut Vec<UnstructuredColumn<'a>>,
) -> Result<(), ParseError> {
    match line.as_slice() {
        &[Word(column_name), Word(row_name), Number(value)] =>
            column_collector.push(UnstructuredColumn {
                name: column_name, variable_type: *marker, row_name, value,
            }),
        &[Word(column_name), Word(row1), Number(value1), Word(row2), Number(value2)] => {
            column_collector.push(UnstructuredColumn {
                name: column_name, variable_type: *marker, row_name: row1, value: value1,
            });
            column_collector.push(UnstructuredColumn {
                name: column_name, variable_type: *marker, row_name: row2, value: value2,
            });
        },
        &[Word(_name), Word(COLUMN_SECTION_MARKER), Word(new_marker)] => {
            *marker = match (&marker, new_marker) {
                (VariableType::Continuous, START_OF_INTEGER) => VariableType::Integer,
                (VariableType::Integer, END_OF_INTEGER) => VariableType::Continuous,
                _ => return Err(ParseError::new(format!("Didn't expect marker \
                            \"{}\": currently parsing variables which are {:?}", new_marker, marker))),
            };
        },
        _ => return Err(ParseError::new("Line can't be parsed as part of the column section.")),
    }

    Ok(())
}

/// Parse a line of atoms which describes a right hand side entry.
///
/// # Arguments
///
/// * `line` - The tokens on the line.
/// * `rhs_collector` - Data structure in which to collect the parsed rhs values.
///
/// # Return value
///
/// A `ParseError` in case of an unknown format.
fn parse_rhs_line<'a>(line: Vec<Atom<'a>>, rhs_collector: &mut Vec<UnstructuredRhs<'a>>) -> Result<(), ParseError> {
    match line.as_slice() {
        &[Word(name), Word(row_name), Number(value)] =>
            rhs_collector.push(UnstructuredRhs { name, row_name, value, }),
        &[Word(name), Word(row1), Number(value1), Word(row2), Number(value2)] => {
            rhs_collector.push(UnstructuredRhs { name, row_name: row1, value: value1, });
            rhs_collector.push(UnstructuredRhs { name, row_name: row2, value: value2, });
        },
        _ => return Err(ParseError::new("Can't parse this row as right-hand side (RHS).")),
    }

    Ok(())
}

/// Parse a line of atoms which describes a bound.
///
/// # Arguments
///
/// * `line` - The tokens on the line.
/// * `bound_collector` - Data structure in which to collect the parsed bounds.
///
/// # Return value
///
/// A `ParseError` in case of an unknown format.
fn parse_bound_line<'a>(line: Vec<Atom<'a>>, bound_collector: &mut Vec<UnstructuredBound<'a>>) -> Result<(), ParseError> {
    match line.as_slice() {
        &[Word(ty), Word(name), Word(column_name), Number(value)] =>
            bound_collector.push(UnstructuredBound {
                name, bound_type: BoundType::try_from(ty)?, column_name, value,
            }),
        _ => return Err(ParseError::new("Can't parse this row as bound.")),
    }

    Ok(())
}

/// Parse a line of atoms which describes a range.
///
/// # Arguments
///
/// * `line` - The tokens on the line.
/// * `range_collector` - Data structure in which to collect the parsed ranges.
///
/// # Return value
///
/// A `ParseError` in case of an unknown format.
///
/// # Note
///
/// TODO: Ranges are currently not supported.
fn parse_range_line<'a>(
    line: Vec<Atom<'a>>,
    range_collector: &mut Vec<UnstructuredRange>
) -> Result<(), ParseError> {
    match line.as_slice() {
        _ => unimplemented!(),
    }

    Ok(())
}

impl<'a> UnstructuredMPS<'a> {
    /// Checks that a minimal set of sections is present.
    ///
    /// # Note
    ///
    /// This is not a check for consistency, this method is a helper method for the TryFrom
    /// constructor for this type.
    fn all_required_fields_present(mps_name: &Option<&str>,
                                   mps_cost_row_name: &Option<&str>,
                                   mps_rows: &Vec<UnstructuredRow>,
                                   mps_columns: &Vec<UnstructuredColumn>,
                                   mps_rhss: &Vec<UnstructuredRhs>) -> Result<(), ParseError> {
        if mps_name.is_none() {
            return Err(ParseError::new("No MPS name read."));
        }
        if mps_cost_row_name.is_none() {
            return Err(ParseError::new("No cost row name read."));
        }
        if mps_rows.is_empty() {
            return Err(ParseError::new("No row names read."));
        }
        if  mps_columns.is_empty() {
            return Err(ParseError::new("No variables read."));
        }
        if mps_rhss.is_empty() {
            return Err(ParseError::new("No RHSs read."));
        }

        Ok(())
    }
}

impl<'a> AbsDiffEq for UnstructuredMPS<'a> {
    type Epsilon = f64;

    fn default_epsilon() -> <Self as AbsDiffEq>::Epsilon {
        EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: <Self as AbsDiffEq>::Epsilon) -> bool {
        self.name == other.name &&
            self.cost_row_name == other.cost_row_name &&
            self.rows == other.rows &&
            self.columns.iter().zip(other.columns.iter())
                .map(|(column1, column2)| abs_diff_eq!(column1, column2))
                .all(identity) &&
            self.rhss.iter().zip(other.rhss.iter())
                .map(|(rhs1, rhs2)| abs_diff_eq!(rhs1, rhs2))
                .all(identity) &&
            self.bounds.iter().zip(other.bounds.iter())
                .map(|(bound1, bound2)| abs_diff_eq!(bound1, bound2))
                .all(identity)
    }
}

impl<'a> AbsDiffEq for UnstructuredColumn<'a> {
    type Epsilon = f64;

    fn default_epsilon() -> <Self as AbsDiffEq>::Epsilon {
        EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: <Self as AbsDiffEq>::Epsilon) -> bool {
        self.name == other.name &&
            self.variable_type == other.variable_type &&
            self.row_name == other.row_name &&
            abs_diff_eq!(self.value, other.value)
    }
}

impl<'a> AbsDiffEq for UnstructuredRhs<'a> {
    type Epsilon = f64;

    fn default_epsilon() -> <Self as AbsDiffEq>::Epsilon {
        EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: <Self as AbsDiffEq>::Epsilon) -> bool {
        self.name == other.name &&
            self.row_name == other.row_name &&
            abs_diff_eq!(self.value, other.value)
    }
}

impl<'a> AbsDiffEq for UnstructuredBound<'a> {
    type Epsilon = f64;

    fn default_epsilon() -> <Self as AbsDiffEq>::Epsilon {
        EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: <Self as AbsDiffEq>::Epsilon) -> bool {
        self.name == other.name &&
            self.bound_type == other.bound_type &&
            self.column_name == other.column_name &&
            abs_diff_eq!(self.value, other.value)
    }
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
    fn try_from(line: &Vec<Atom<'a>>) -> Result<Section<'a>, Self::Error> {
        match line.as_slice() {
            &[Word("NAME"), Word(name)] => Ok(Section::Name(name)),
            &[Word(name)] => match name {
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

impl<'a> TryFrom<&'a str> for RowType {
    type Error = ParseError;

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
    fn try_from(word: &str) -> Result<RowType, Self::Error> {
        match word {
            "N" => Ok(RowType::Cost),
            "L" => Ok(RowType::Constraint(ConstraintType::Less)),
            "E" => Ok(RowType::Constraint(ConstraintType::Equal)),
            "G" => Ok(RowType::Constraint(ConstraintType::Greater)),
            _ => Err(ParseError::new(format!("Row type \"{}\" unknown.", word))),
        }
    }
}

impl<'a> TryFrom<&'a str> for BoundType {
    type Error = ParseError;

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
    fn try_from(word: &str) -> Result<BoundType, Self::Error> {
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
            _ => Err(ParseError::new(format!("Cant' parse \"{}\" as bound type.", word))),
        }
    }
}


/// Testing the parsing functionality
#[cfg(test)]
mod test {

    use std::convert::TryFrom;

    use data::linear_program::elements::{ConstraintType, VariableType};
    use io::mps::test::lp_unstructured_mps;
    use io::mps::test::MPS_STRING;
    use io::mps::parsing::Atom::*;
    use io::mps::parsing::into_atom_lines;
    use io::mps::parsing::into_atoms;
    use io::mps::parsing::{UnstructuredColumn, UnstructuredMPS, UnstructuredRow};
    use io::mps::BoundType;
    use io::mps::RowType;
    use io::mps::Section;
    use io::mps::parsing::parse_row_line;
    use io::mps::parsing::parse_column_line;
    use io::mps::token::{COLUMN_SECTION_MARKER, END_OF_INTEGER, START_OF_INTEGER};


    #[test]
    fn test_into_atom_lines() {
        let program = "SOMETEXT 1.2\n*comment\n\n   \t line before is empty".to_string();
        let result = into_atom_lines(&program);
        let expected = vec![((1, "SOMETEXT 1.2"), vec![Word("SOMETEXT"), Number(1.2f64)]),
                            ((4, "   \t line before is empty"),
                             vec![Word("line"), Word("before"), Word("is"), Word("empty")])];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_into_atoms() {
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
    fn test_parse_row_line() {
        let initial_cost_row_name = Some("COST_ROW_NAME");

        macro_rules! test_positive {
            ([$($words:expr), *], [$($expected_row:expr), *], $expected_cost_row_name:expr) => {
                let line = vec![$($words), *];
                let mut cost_row_name = initial_cost_row_name.clone();
                let mut collector = Vec::new();

                let result = parse_row_line(line, &mut cost_row_name, &mut collector);

                assert!(result.is_ok());
                assert_eq!(cost_row_name, $expected_cost_row_name);
                assert_eq!(collector, vec![$($expected_row), *]);
            }
        }

        test_positive!([Word("E"), Word("ROW_NAME")],
                       [UnstructuredRow { name: "ROW_NAME", constraint_type: ConstraintType::Equal, }],
                       Some("COST_ROW_NAME"));
        test_positive!([Word("N"), Word("NEW_COST_ROW_NAME")],
                       [], Some("NEW_COST_ROW_NAME"));

        macro_rules! test_negative {
            ([$($words:expr), *]) => {
                let line = vec![$($words), *];
                let mut cost_row_name = initial_cost_row_name.clone();
                let mut collector = Vec::new();

                let result = parse_row_line(line, &mut cost_row_name, &mut collector);

                assert!(result.is_err());
            }
        }

        test_negative!([Word("UNKNOWN_ROW_TYPE"), Word("ROW_NAME")]);
        test_negative!([Word("JUST_ONE_WORD")]);
        test_negative!([Word("ONE"), Word("TWO"), Word("THREE")]);
        test_negative!([Word("ONE"), Word("TWO"), Word("THREE"), Word("FOUR")]);
    }

    #[test]
    fn test_parse_column_line() {
        macro_rules! test_positive {
            (
                [$($words:expr), *],
                [$($expected_data:expr, ) *],
                $initial_marker:expr,
                $expected_marker:expr
            ) => {
                let line = vec![$($words), *];
                let mut marker = $initial_marker;
                let mut collector = Vec::new();

                let result = parse_column_line(line, &mut marker, &mut collector);

                assert!(result.is_ok());
                assert_eq!(collector, vec![$($expected_data), *]);
                assert_eq!(marker, $expected_marker);
            }
        }

        test_positive!([Word("CNAME"), Word("RNAME"), Number(5f64)],
            [
                UnstructuredColumn {
                    name: "CNAME",
                    variable_type: VariableType::Continuous,
                    row_name: "RNAME",
                     value: 5f64,
                 },
            ],
            VariableType::Continuous, VariableType::Continuous);
        test_positive!([Word("CNAME"), Word("RNAME"), Number(5f64)],
            [
                UnstructuredColumn {
                    name: "CNAME",
                    variable_type: VariableType::Integer,
                    row_name: "RNAME",
                     value: 5f64,
                },
            ],
            VariableType::Integer, VariableType::Integer);
        test_positive!([Word("CNAME1"), Word("RNAME1"), Number(1f64), Word("RNAME2"), Number(2f64)],
            [
                UnstructuredColumn {
                    name: "CNAME1",
                    variable_type: VariableType::Continuous,
                    row_name: "RNAME1",
                     value: 1f64,
                 },
                 UnstructuredColumn {
                     name: "CNAME1",
                     variable_type: VariableType::Continuous,
                     row_name: "RNAME2",
                      value: 2f64,
                  },
            ],
            VariableType::Continuous, VariableType::Continuous);
        test_positive!([Word("MARKER_NAME"), Word(COLUMN_SECTION_MARKER), Word(START_OF_INTEGER)],
            [], VariableType::Continuous, VariableType::Integer);
        test_positive!([Word("MARKER_NAME"), Word(COLUMN_SECTION_MARKER), Word(END_OF_INTEGER)],
            [], VariableType::Integer, VariableType::Continuous);
    }

    #[test]
    fn test_try_from_unstructured_mps() {
        let lines = into_atom_lines(&MPS_STRING);

        let result = UnstructuredMPS::try_from(lines);
        assert!(result.is_ok());
        assert_abs_diff_eq!(result.unwrap(), lp_unstructured_mps());
    }

    #[test]
    fn test_try_from_section() {
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

    #[test]
    fn test_try_from_row_type() {
        macro_rules! test_positive {
            ($word:expr, $expected:expr) => {
                let result = RowType::try_from($word);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), $expected);
            }
        }
        test_positive!("N", RowType::Cost);
        test_positive!("L", RowType::Constraint(ConstraintType::Less));
        test_positive!("E", RowType::Constraint(ConstraintType::Equal));
        test_positive!("G", RowType::Constraint(ConstraintType::Greater));

        macro_rules! test_negative {
            ($word:expr) => {
                let result = RowType::try_from($word);
                assert!(result.is_err());
            }
        }
        test_negative!("X");
        test_negative!("");
        test_negative!("\t");
    }

    #[test]
    fn test_try_from_bound_type() {
        macro_rules! test {
            ($word:expr, $expected:ident) => {
                let result = BoundType::try_from($word);
                assert!(result.is_ok());
                assert_eq!(result.unwrap(), BoundType::$expected);
            }
        }

        test!("LO", LowerContinuous);
        test!("UP", UpperContinuous);
        test!("FX", Fixed);
        test!("FR", Free);
        test!("MI", LowerMinusInfinity);
        test!("PL", UpperInfinity);
        test!("BV", Binary);
        test!("LI", LowerInteger);
        test!("UI", UpperInteger);
        test!("SC", SemiContinuous);
        assert!(BoundType::try_from("X").is_err());
        assert!(BoundType::try_from("").is_err());
        assert!(BoundType::try_from("\t").is_err());
    }

}
