//! # Parsing MPS files
//!
//! First stage of importing linear programs. Checks if the MPS file is syntactically correct, but
//! doesn't do any consistency checks for e.g. undefined row names in the column section.
use std::convert::TryFrom;
use std::str::FromStr;

use crate::data::linear_program::elements::{ConstraintType, VariableType};
use crate::io::error::{FileLocation, Parse};
use crate::io::mps::BoundType;
use crate::io::mps::RowType;
use crate::io::mps::Section;
use crate::io::mps::token::{
    COLUMN_SECTION_MARKER, COMMENT_INDICATOR, END_OF_INTEGER, START_OF_INTEGER,
};

use self::Atom::{Number, Word};
use num::{Num, FromPrimitive};

/// Most fundamental element in an MPS text file.
///
/// Every part of the input string to end up in the final `MPS` struct is parsed as either a `Word`
/// or a `Number`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Atom<'a, F> {
    /// A token consisting of text which should not contain whitespace.
    Word(&'a str),
    /// A token representing a number.
    Number(F),
}

/// Convert an MPS program in string form to structured data.
///
/// # Arguments
///
/// * `program`: The input string
///
/// # Return value
///
/// All lines stored in a `Vec`. The first `(usize, &str)` tuple is used for creating of errors, it
/// contains the line number and line.
pub(crate) fn into_atom_lines<F: FromPrimitive>(
    program: &impl AsRef<str>,
) -> Vec<((u64, &str), Vec<Atom<F>>)> {
    program.as_ref()
        .lines()
        .enumerate()
        .map(|(number, line)| (number as u64 + 1_u64, line))
        .filter(|(_, line)| !line.trim_start().starts_with(COMMENT_INDICATOR))
        .filter(|(_, line)| !line.is_empty())
        .map(|(number, line)| ((number, line), into_atoms(line)))
        .collect()
}

/// Convert a line into `Atom`s by testing whether an atom is a number.
///
/// # Arguments
///
/// * `line`: The input string slice
///
/// # Return value
///
/// A `Vec` of words and numbers.
fn into_atoms<F: FromPrimitive>(line: &str) -> Vec<Atom<F>> {
    line.split_whitespace()
        .map(|atom| match atom.parse().map(F::from_f64) {
            Ok(Some(value)) => Number(value),
            _ => Word(atom),
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
pub(crate) struct UnstructuredMPS<'a, F> {
    /// Name of the linear program
    pub name: &'a str,
    /// Name of the cost row, or objective function
    pub cost_row_name: &'a str,
    /// Names of all constraint rows and their constraint type
    pub rows: Vec<UnstructuredRow<'a>>,
    /// Names and type (continuous, discrete) of all variables
    pub columns: Vec<UnstructuredColumn<'a, F>>,
    /// Right-hand sides giving a numerical value to the constraints
    pub rhss: Vec<UnstructuredRhs<'a, F>>,
    /// Activation of constraints has a range
    pub ranges: Vec<UnstructuredRange<'a, F>>,
    /// Bounds of the problem (does not includes elements of the "RANGES" section.
    pub bounds: Vec<UnstructuredBound<'a, F>>,
}

/// Name and constraint type of a row
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct UnstructuredRow<'a> {
    pub name: &'a str,
    pub constraint_type: ConstraintType,
}
/// Name of a column, variable type of a column, name of a row and a value (a constraint matrix
/// entry in sparse description)
#[derive(Debug, PartialEq)]
pub(crate) struct UnstructuredColumn<'a, F> {
    pub name: &'a str,
    pub variable_type: VariableType,
    pub row_name: &'a str,
    pub value: F,
}
/// Name of the right-hand side, name of the variable and constraint value
#[derive(Debug, PartialEq)]
pub(crate) struct UnstructuredRhs<'a, F> {
    pub name: &'a str,
    pub row_name: &'a str,
    pub value: F,
}
/// Name of the range, name of the variable and range value.
///
/// Overview of how the range of a constraint is defined, depending on the constraint type:
///
/// row type | sign of r |    h    |    u
/// ---------|-----------|---------|---------
/// G        |  + or -   |    b    | b + |r|
/// L        |  + or -   | b - |r| |   b
/// E        |     +     |    b    | b + |r|
/// E        |     -     | b - |r| |   b
#[derive(Debug, PartialEq)]
pub(crate) struct UnstructuredRange<'a, F> {
    pub name: &'a str,
    pub row_name: &'a str,
    pub value: F,
}
/// Name of the bound, bound type, name of the variable and constraint value
#[derive(Debug, PartialEq)]
pub(crate) struct UnstructuredBound<'a, F> {
    pub name: &'a str,
    pub bound_type: BoundType<F>,
    pub column_name: &'a str,
}

impl<'a, F: Clone> TryFrom<Vec<(FileLocation<'a>, Vec<Atom<'a, F>>)>> for UnstructuredMPS<'a, F> {
    type Error = Parse;

    /// Try to read an `UnstructuredMPS` struct from lines of `Atom`s.
    ///
    /// This method attempts to determine the section that is currently being parsed, and remembers
    /// that section. Subsequent lines are then parsed and collected in the `mps_*` variables.
    ///
    /// # Arguments
    ///
    /// * `program`: The input program parsed as lines of `Atom`s.
    ///
    /// # Return value
    ///
    /// An `UnstructuredMPS` instance, if no errors were encountered.
    ///
    /// # Errors
    ///
    /// A `ParseError` at failure.
    fn try_from(
        atom_lines: Vec<(FileLocation<'a>, Vec<Atom<'a, F>>)>,
    ) -> Result<Self, Self::Error> {
        let mut current_section: Option<Section> = None;

        let mut name = None;
        let mut cost_row_name = None;
        let mut rows = Vec::new();
        let mut columns = Vec::new();
        let mut rhss = Vec::new();
        let mut bounds = Vec::new();
        let mut ranges = Vec::new();

        for (file_location, line) in atom_lines {
            if let Ok(new_section) = Section::try_from(&line) {
                match new_section {
                    Section::Name(new_program_name) => {
                        name = Some(new_program_name);
                        current_section = None;
                    }
                    Section::Endata => break,
                    _other_section => current_section = Some(new_section),
                }
                continue;
            }

            match current_section {
                None => Err(Parse::new("Section unknown, can't parse line")),
                Some(Section::Name(_)) => panic!("The name should be updated in section parsing"),
                Some(Section::Rows) => parse_row_line(line, &mut cost_row_name, &mut rows),
                Some(Section::Columns(ref mut marker)) => parse_column_line(line, marker, &mut columns),
                Some(Section::Rhs) => parse_rhs_line(line, &mut rhss),
                Some(Section::Bounds) => parse_bound_line(line, &mut bounds),
                Some(Section::Ranges) => parse_range_line(line, &mut ranges),
                Some(Section::Endata) => panic!("Row parsing should have been aborted after Endata detection"),
            }.map_err(|parse_error|
                Parse::with_file_location(parse_error.to_string(), file_location)
            )?;
        }

        Self::all_required_fields_present(&name, &cost_row_name, &rows, &columns, &rhss)
            .map(|_| Self {
                name: name.unwrap(),
                cost_row_name: cost_row_name.unwrap(),
                rows,
                columns,
                rhss,
                ranges,
                bounds,
            })
    }
}

/// Parse a line of atoms which describes a row entry.
///
/// # Arguments
///
/// * `line`: The tokens on the line.
/// * `cost_row_name`: The previously parsed name of the cost row.
/// * `row_collector`: Data structure in which to collect the parsed rows.
///
/// # Return value
///
/// A Parse error if parsing fails.
fn parse_row_line<'a, F>(
    line: Vec<Atom<'a, F>>,
    cost_row_name: &mut Option<&'a str>,
    row_collector: &mut Vec<UnstructuredRow<'a>>,
) -> Result<(), Parse> {
    match *line.as_slice() {
        [Word(ty), Word(row_name)] => match RowType::try_from(ty)? {
            RowType::Cost => *cost_row_name = Some(row_name),
            RowType::Constraint(ty) => row_collector.push(UnstructuredRow {
                name: row_name,
                constraint_type: ty,
            }),
        },
        _ => return Err(Parse::new("Line can't be parsed as part of the row section."))
    }

    Ok(())
}

/// Parse a line of atoms which describes a column entry.
///
/// # Arguments
///
/// * `line`: The tokens on the line.
/// * `marker`: Indicator for the continuous or discrete part of the column section.
/// * `column_collector`: Data structure in which to collect the parsed values.
///
/// # Return value
///
/// A new variable type, indicating that the integer section of the columns section has started or
/// ended.
fn parse_column_line<'a, F>(
    line: Vec<Atom<'a, F>>,
    marker: &mut VariableType,
    column_collector: &mut Vec<UnstructuredColumn<'a, F>>,
) -> Result<(), Parse> {
    let mut line = line.into_iter();
    match [line.next(), line.next(), line.next(), line.next(), line.next(), line.next()] {
        [Some(Word(column_name)), Some(Word(row_name)), Some(Number(value)), None, None, None] =>
            column_collector.push(UnstructuredColumn {
                name: column_name, variable_type: *marker, row_name, value,
            }),
        [Some(Word(column_name)), Some(Word(row1)), Some(Number(value1)), Some(Word(row2)), Some(Number(value2)), None] => {
            column_collector.push(UnstructuredColumn {
                name: column_name, variable_type: *marker, row_name: row1, value: value1,
            });
            column_collector.push(UnstructuredColumn {
                name: column_name, variable_type: *marker, row_name: row2, value: value2,
            });
        },
        [Some(Word(_name)), Some(Word(COLUMN_SECTION_MARKER)), Some(Word(new_marker)), None, None, None] => {
            *marker = match (&marker, new_marker) {
                (VariableType::Continuous, START_OF_INTEGER) => VariableType::Integer,
                (VariableType::Integer, END_OF_INTEGER) => VariableType::Continuous,
                _ => return Err(Parse::new(format!("Didn't expect marker \
                            \"{}\": currently parsing variables which are {:?}", new_marker, marker))),
            };
        },
        _ => return Err(Parse::new("Line can't be parsed as part of the column section.")),
    }

    Ok(())
}

/// Parse a line of atoms which describes a right hand side entry.
///
/// # Arguments
///
/// * `line`: The tokens on the line.
/// * `rhs_collector`: Data structure in which to collect the parsed rhs values.
///
/// # Return value
///
/// A `ParseError` in case of an unknown format.
fn parse_rhs_line<'a, F>(
    line: Vec<Atom<'a, F>>,
    rhs_collector: &mut Vec<UnstructuredRhs<'a, F>>,
) -> Result<(), Parse> {
    let mut line = line.into_iter();
    match [line.next(), line.next(), line.next(), line.next(), line.next(), line.next()] {
        [Some(Word(name)), Some(Word(row_name)), Some(Number(value)), None, None, None] =>
            rhs_collector.push(UnstructuredRhs { name, row_name, value, }),
        [Some(Word(name)), Some(Word(row1)), Some(Number(value1)), Some(Word(row2)), Some(Number(value2)), None] => {
            rhs_collector.push(UnstructuredRhs { name, row_name: row1, value: value1, });
            rhs_collector.push(UnstructuredRhs { name, row_name: row2, value: value2, });
        },
        _ => return Err(Parse::new("Can't parse this row as right-hand side (RHS).")),
    }

    Ok(())
}

/// Parse a line of atoms which describes a bound.
///
/// # Arguments
///
/// * `line`: The tokens on the line.
/// * `bound_collector`: Data structure in which to collect the parsed bounds.
///
/// # Return value
///
/// A `ParseError` in case of an unknown format.
fn parse_bound_line<'a, F>(
    line: Vec<Atom<'a, F>>,
    bound_collector: &mut Vec<UnstructuredBound<'a, F>>,
) -> Result<(), Parse> {
    let mut line = line.into_iter();
    let bound = match [line.next(), line.next(), line.next(), line.next(), line.next(), line.next()] {
        [Some(Word(ty)), Some(Word(name)), Some(Word(column_name)), None, None, None] =>
            UnstructuredBound {
                name,
                bound_type: BoundType::try_from(ty)?,
                column_name,
            },
        [Some(Word(ty)), Some(Word(name)), Some(Word(column_name)), Some(Number(value)), None, None] =>
            UnstructuredBound {
                name,
                bound_type: BoundType::try_from((ty, value))?,
                column_name,
            },
        [Some(Word(ty)), Some(Word(name)), Some(Word(column_name)), Some(Number(value)), Some(Number(value2)), None] =>
            UnstructuredBound {
                name,
                bound_type: BoundType::try_from((ty, value, value2))?,
                column_name,
            },
        _ => return Err(Parse::new("Can't parse this row as bound.")),
    };
    bound_collector.push(bound);

    Ok(())
}

/// Parse a line of atoms which describes a range.
///
/// # Arguments
///
/// * `line`: The tokens on the line.
/// * `range_collector`: Data structure in which to collect the parsed ranges.
///
/// # Return value
///
/// A `ParseError` in case of an unknown format.
fn parse_range_line<'a, F>(
    line: Vec<Atom<'a, F>>,
    range_collector: &mut Vec<UnstructuredRange<'a, F>>,
) -> Result<(), Parse> {
    let mut line = line.into_iter();
    let range = match [line.next(), line.next(), line.next(), line.next()] {
        [Some(Word(name)), Some(Word(column_name)), Some(Number(value)), None] =>
            UnstructuredRange {
                name,
                row_name: column_name,
                value
            },
        _ => return Err(Parse::new("Can't parse this row as range.")),
    };
    range_collector.push(range);

    Ok(())
}

impl<'a, F> UnstructuredMPS<'a, F> {
    /// Checks that a minimal set of sections is present.
    ///
    /// # Note
    ///
    /// This is not a check for consistency, this method is a helper method for the TryFrom
    /// constructor for this type.
    fn all_required_fields_present(
        mps_name: &Option<&str>,
        mps_cost_row_name: &Option<&str>,
        mps_rows: &Vec<UnstructuredRow>,
        mps_columns: &Vec<UnstructuredColumn<F>>,
        mps_rhss: &Vec<UnstructuredRhs<F>>,
    ) -> Result<(), Parse> {
        if mps_name.is_none() {
            return Err(Parse::new("No MPS name read."));
        }
        if mps_cost_row_name.is_none() {
            return Err(Parse::new("No cost row name read."));
        }
        if mps_rows.is_empty() {
            return Err(Parse::new("No row names read."));
        }
        if mps_columns.is_empty() {
            return Err(Parse::new("No variables read."));
        }
        if mps_rhss.is_empty() {
            return Err(Parse::new("No RHSs read."));
        }

        Ok(())
    }
}


impl<'a, 'b, F> TryFrom<&'b Vec<Atom<'a, F>>> for Section<'a> {
    type Error = ();

    /// Try to read a `Section` from a `Vec` slice of `Atom`s.
    ///
    /// # Arguments
    ///
    /// * `line`: The input line consisting of a sequence of `Atom`s.
    ///
    /// # Return value
    ///
    /// A `Section` variant describing the section this line announces, if one is recognized.
    ///
    /// # Errors
    ///
    /// A `()` error if no `Section` is recognized.
    fn try_from(line: &Vec<Atom<'a, F>>) -> Result<Self, Self::Error> {
        let mut line = line.into_iter();
        match [line.next(), line.next(), line.next()] {
            [Some(Word("NAME")), Some(Word(name)), None] => Ok(Section::Name(name)),
            [Some(Word(name)), None, None] => match *name {
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
    type Error = Parse;

    /// Try to read a `RowType` from a string slice.
    ///
    /// The type of a row is denoted by `N` if it's the cost row; this row is often unique. There is
    /// no defined behaviour for multiple cost rows. Constraint rows are indicated by `L`, `E` or
    /// `G`.
    ///
    /// # Arguments
    ///
    /// * `word`: The input `String` slice.
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
            _ => Err(Parse::new(format!("Row type \"{}\" unknown.", word))),
        }
    }
}

impl<'a, F> TryFrom<&'a str> for BoundType<F> {
    type Error = Parse;

    /// Try to read a `BoundType` from a `String` slice.
    ///
    /// # Arguments
    ///
    /// * `word`: The input `String` slice.
    ///
    /// # Return value
    ///
    /// A `BoundType` variant describing the bound, if the type is known.
    ///
    /// # Errors
    ///
    /// A `()` error if the bound type is not known.
    fn try_from(word: &str) -> Result<Self, Self::Error> {
        match word {
            "FR" => Ok(BoundType::Free),
            "MI" => Ok(BoundType::LowerMinusInfinity),
            "PL" => Ok(BoundType::UpperInfinity),
            "BV" => Ok(BoundType::Binary),
            _ => Err(Parse::new(format!("Cant' parse \"{}\" as bound type.", word))),
        }
    }
}

impl<'a, F> TryFrom<(&'a str, F)> for BoundType<F> {
    type Error = Parse;

    /// Try to read a `BoundType` from a `String` slice.
    ///
    /// # Arguments
    ///
    /// * `word`: The input `String` slice.
    ///
    /// # Return value
    ///
    /// A `BoundType` variant describing the bound, if the type is known.
    ///
    /// # Errors
    ///
    /// A `()` error if the bound type is not known.
    fn try_from((word, value): (&str, F)) -> Result<Self, Self::Error> {
        match word {
            "LO" => Ok(BoundType::LowerContinuous(value)),
            "UP" => Ok(BoundType::UpperContinuous(value)),
            "FX" => Ok(BoundType::Fixed(value)),
            "LI" => Ok(BoundType::LowerInteger(value)),
            "UI" => Ok(BoundType::UpperInteger(value)),
            _ => Err(Parse::new(format!("Cant' parse \"{}\" as bound type.", word))),
        }
    }
}

impl<'a, F> TryFrom<(&'a str, F, F)> for BoundType<F> {
    type Error = Parse;

    /// Try to read a `BoundType` from a `String` slice.
    ///
    /// # Arguments
    ///
    /// * `word`: The input `String` slice.
    /// *
    ///
    /// # Return value
    ///
    /// A `BoundType` variant describing the bound, if the type is known.
    ///
    /// # Errors
    ///
    /// A `()` error if the bound type is not known.
    fn try_from((word, lower, upper): (&str, F, F)) -> Result<Self, Self::Error> {
        match word {
            "SC" => Ok(BoundType::SemiContinuous(lower, upper)),
            _ => Err(Parse::new(format!("Cant' parse \"{}\" as bound type.", word))),
        }
    }
}

#[cfg(test)]
mod test {
    use std::convert::TryFrom;

    use crate::data::linear_program::elements::{ConstraintType, VariableType};
    use crate::io::mps::BoundType;
    use crate::io::mps::parsing::{UnstructuredColumn, UnstructuredRow};
    use crate::io::mps::parsing::Atom::*;
    use crate::io::mps::parsing::Atom;
    use crate::io::mps::parsing::into_atom_lines;
    use crate::io::mps::parsing::into_atoms;
    use crate::io::mps::parsing::parse_column_line;
    use crate::io::mps::parsing::parse_row_line;
    use crate::io::mps::RowType;
    use crate::io::mps::Section;
    use crate::io::mps::token::{COLUMN_SECTION_MARKER, END_OF_INTEGER, START_OF_INTEGER};

    #[test]
    fn test_into_atom_lines() {
        let program = "SOMETEXT 1.2\n*comment\n\n   \t line before is empty".to_string();
        let result = into_atom_lines(&program);
        let expected = vec![
            ((1, "SOMETEXT 1.2"), vec![Word("SOMETEXT"), Number(1.2f64)]),
            ((4, "   \t line before is empty"), vec![Word("line"), Word("before"), Word("is"), Word("empty")]),
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_into_atoms() {
        macro_rules! test {
            ($line:expr, [$($words:expr), *]) => {
                let result = into_atoms::<f64>($line);
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

                let result = parse_row_line::<f64>(line, &mut cost_row_name, &mut collector);

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

                let result = parse_row_line::<f64>(line, &mut cost_row_name, &mut collector);

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

                let result = parse_column_line::<i32>(line, &mut marker, &mut collector);

                assert!(result.is_ok());
                assert_eq!(collector, vec![$($expected_data), *]);
                assert_eq!(marker, $expected_marker);
            }
        }

        test_positive!([Word("CNAME"), Word("RNAME"), Number(5)],
            [
                UnstructuredColumn {
                    name: "CNAME",
                    variable_type: VariableType::Continuous,
                    row_name: "RNAME",
                     value: 5,
                 },
            ],
            VariableType::Continuous, VariableType::Continuous);
        test_positive!([Word("CNAME"), Word("RNAME"), Number(5)],
            [
                UnstructuredColumn {
                    name: "CNAME",
                    variable_type: VariableType::Integer,
                    row_name: "RNAME",
                     value: 5,
                },
            ],
            VariableType::Integer, VariableType::Integer);
        test_positive!([Word("CNAME1"), Word("RNAME1"), Number(1), Word("RNAME2"), Number(2)],
            [
                UnstructuredColumn {
                    name: "CNAME1",
                    variable_type: VariableType::Continuous,
                    row_name: "RNAME1",
                     value: 1,
                 },
                 UnstructuredColumn {
                     name: "CNAME1",
                     variable_type: VariableType::Continuous,
                     row_name: "RNAME2",
                      value: 2,
                  },
            ],
            VariableType::Continuous, VariableType::Continuous);
        test_positive!([Word("MARKER_NAME"), Word(COLUMN_SECTION_MARKER), Word(START_OF_INTEGER)],
            [], VariableType::Continuous, VariableType::Integer);
        test_positive!([Word("MARKER_NAME"), Word(COLUMN_SECTION_MARKER), Word(END_OF_INTEGER)],
            [], VariableType::Integer, VariableType::Continuous);
    }

    #[test]
    fn test_try_from_section() {
        macro_rules! test {
            ([$($words:expr), *], $expected:expr) => {
                let input: Vec<Atom<f64>> = vec![$($words), *];
                let result = Section::try_from(&input);
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
            };
        }
        test_positive!("N", RowType::Cost);
        test_positive!("L", RowType::Constraint(ConstraintType::Less));
        test_positive!("E", RowType::Constraint(ConstraintType::Equal));
        test_positive!("G", RowType::Constraint(ConstraintType::Greater));

        macro_rules! test_negative {
            ($word:expr) => {
                let result = RowType::try_from($word);
                assert!(result.is_err());
            };
        }
        test_negative!("X");
        test_negative!("");
        test_negative!("\t");
    }

    #[test]
    fn test_try_from_bound_type() {
        let result: Result<BoundType<f64>, _> = BoundType::try_from("FR");
        assert_eq!(result, Ok(BoundType::Free));

        macro_rules! test {
            ($word:expr, $expected:ident) => {
                let result: Result<BoundType<f64>, _> = BoundType::try_from($word);
                assert_eq!(result, Ok(BoundType::$expected));
            };
        }

        test!("MI", LowerMinusInfinity);
        test!("PL", UpperInfinity);
        test!("BV", Binary);
        assert!(BoundType::<f64>::try_from("X").is_err());
        assert!(BoundType::<f64>::try_from("").is_err());
        assert!(BoundType::<i32>::try_from("\t").is_err());

        macro_rules! test {
            ($word:expr, $expected:ident) => {
                let result = BoundType::try_from(($word, 4));
                assert_eq!(result, Ok(BoundType::$expected(4)));
            };
        }
        test!("LO", LowerContinuous);
        test!("UP", UpperContinuous);
        test!("FX", Fixed);
        test!("LI", LowerInteger);
        test!("UI", UpperInteger);
    }
}
