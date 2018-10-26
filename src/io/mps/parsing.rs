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


/// Most fundamental element in an MPS text file.
///
/// Every part of the input string to end up in the final `MPS` struct is parsed as either a `Word`
/// or a `Number`.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Atom<'a> {
    Word(&'a str),
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
            Ok(value) => Atom::Number(value),
            Err(_) => Atom::Word(atom),
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
pub(super) type UnstructuredRow<'a> = (&'a str, ConstraintType);
/// Name of a column, variable type of a column, name of a row and a value (a constraint matrix
/// entry in sparse description)
pub(super) type UnstructuredColumn<'a> = (&'a str, VariableType, &'a str, f64);
/// Name of the right-hand side, column name and constraint value
pub(super) type UnstructuredRhs<'a> = (&'a str, &'a str, f64);
/// Name of the bound, bound type, name of the variable and constraint value
pub(super) type UnstructuredBound<'a> = (&'a str, BoundType, &'a str, f64);
/// TODO: Support the RANGES section
pub(super) type UnstructuredRange<'a> = ();

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
                    other_section => current_section = Some(new_section),
                }
                continue;
            }

            match current_section {
                None => Err(ParseError::new("Section unknown, can't parse line")),
                Some(Section::Name(_)) => panic!("The name should be updated in section parsing"),
                Some(Section::Rows) => parse_row_line(line, &mut cost_row_name, &mut rows),
                Some(Section::Columns(marker)) => parse_column_line(line, marker, &mut columns).map(|maybe_marker| {
                    if let Some(new_marker) = maybe_marker {
                        current_section = Some(Section::Columns(new_marker));
                    }
                }),
                Some(Section::Rhs) => parse_rhs_line(line, &mut rhss),
                Some(Section::Bounds) => parse_bound_line(line, &mut bounds),
                Some(Section::Ranges) => parse_range_line(line, &mut ranges),
                Some(Section::Endata) => panic!("Row parsing should have been aborted after Endata detection"),
            }.map_err(|parse_error| ParseError::with_file_location(parse_error.description(), file_location))?;
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
        &[Atom::Word(ty), Atom::Word(row)] => match RowType::try_from(ty) {
            Ok(RowType::Cost) => *cost_row_name = Some(row),
            Ok(RowType::Constraint(ty)) => row_collector.push((row, ty)),
            _ => unimplemented!(),
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
                         marker: VariableType,
                         column_collector: &mut Vec<UnstructuredColumn<'a>>
) -> Result<Option<VariableType>, ParseError> {
    match line.as_slice() {
        &[Atom::Word(column), Atom::Word(row), Atom::Number(value)] =>
            column_collector.push((column, marker, row, value)),
        &[Atom::Word(column), Atom::Word(row1), Atom::Number(value1), Atom::Word(row2), Atom::Number(value2)] => {
            column_collector.push((column, marker, row1, value1));
            column_collector.push((column, marker, row2, value2));
        },
        &[Atom::Word(_name), Atom::Word(COLUMN_SECTION_MARKER), Atom::Word(new_marker)] => {
            let new_marker = match (&marker, new_marker) {
                (VariableType::Continuous, START_OF_INTEGER) => VariableType::Integer,
                (VariableType::Integer, END_OF_INTEGER) => VariableType::Continuous,
                _ => return Err(ParseError::new(format!("Didn't expect marker \
                            \"{}\": currently parsing variables which are {:?}", new_marker, marker))),
            };

            return Ok(Some(new_marker));
        },
        _ => return Err(ParseError::new("Line can't be parsed as part of the column section.")),
    }

    Ok(None)
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
        &[Atom::Word(name), Atom::Word(row), Atom::Number(value)] =>
            rhs_collector.push((name, row, value)),
        &[Atom::Word(name), Atom::Word(row1), Atom::Number(value1), Atom::Word(row2), Atom::Number(value2)] => {
            rhs_collector.push((name, row1, value1));
            rhs_collector.push((name, row2, value2));
        },
        _ => unimplemented!(),
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
        &[Atom::Word(ty), Atom::Word(name), Atom::Word(column), Atom::Number(value)] => match BoundType::try_from(ty) {
            Ok(bound_type) => bound_collector.push((name, bound_type, column, value)),
            _ => panic!(),
        },
        _ => unimplemented!(),
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
    range_collector: &mut Vec<UnstructuredRange<'a>>
) -> Result<(), ParseError> {
    match line.as_slice() {
        _ => unimplemented!(),
    }

    Ok(())
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


/// Testing the parsing functionality
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
        use io::mps::parsing::Atom::*;

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

}
