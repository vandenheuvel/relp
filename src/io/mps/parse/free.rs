use std::str::SplitWhitespace;

use crate::io::error::Import as ImportError;
use crate::io::error::Parse as ParseError;
use crate::io::error::ParseResult;
use crate::io::mps::MPS;
use crate::io::mps::number::parse::Parse;
use crate::io::mps::parse::{ColumnLineContent, ColumnRetriever};
use crate::io::mps::token::COLUMN_SECTION_MARKER;


/// Parse an MPS program, in string form, to a MPS.
///
/// This is the flexible mode, which doesn't require columns to be in exactly the right place.
///
/// # Arguments
///
/// * `program`: The input in [MPS format](https://en.wikipedia.org/wiki/MPS_(format)).
///
/// # Return value
///
/// A `Result<MPS, ImportError>` instance.
///
/// # Errors
///
/// An Import error, wrapping either a parse error indicating that the file was syntactically
/// incorrect, or an Inconsistency error indicating that the file is "logically" incorrect.
pub fn parse<F: Parse>(program_string: &impl AsRef<str>) -> Result<MPS<F>, ImportError> {
    super::parse::<_, Free>(program_string.as_ref())
}

pub(super) struct Free;
impl<'a> ColumnRetriever<'a> for Free {
    type RestOfLine = SplitWhitespace<'a>;

    fn two_or_three(line_after_name: &str) -> ParseResult<[&str; 1]> {
        let mut splitted = line_after_name.split_whitespace();
        splitted.next().map(|v| [v])
            .ok_or_else(|| ParseError::new("No name found."))
    }

    fn one_and_two(line: &str) -> ParseResult<[&str; 2]> {
        let mut parts = line.split_whitespace();
        let one = parts.next()
            .ok_or_else(|| ParseError::new("Could not read first field"))?;
        let two = parts.next()
            .ok_or_else(|| ParseError::new("Could not read second field"))?;

        Ok([one, two])
    }

    fn is_column_marker_line(line: &'a str) -> ParseResult<ColumnLineContent<'a, Self::RestOfLine>> {
        let mut parts = line.split_whitespace();

        let two = parts.next()
            .ok_or_else(|| ParseError::new("Could not read second field"))?;
        let three = parts.next()
            .ok_or_else(|| ParseError::new("Could not read third field"))?;
        let four = parts.next()
            .ok_or_else(|| ParseError::new("Could not read fourth field"))?;

        Ok(if three == COLUMN_SECTION_MARKER {
            ColumnLineContent::Marker([four])
        } else {
            ColumnLineContent::Data([two, three, four], parts)
        })
    }

    fn two_through_four(line: &'a str) -> ParseResult<([&'a str; 3], Self::RestOfLine)> {
        let mut parts = line.split_whitespace();

        let two = parts.next()
            .ok_or_else(|| ParseError::new("Could not read second field"))?;
        let three = parts.next()
            .ok_or_else(|| ParseError::new("Could not read third field"))?;
        let four = parts.next()
            .ok_or_else(|| ParseError::new("Could not read fourth field"))?;

        Ok(([two, three, four], parts))
    }

    fn five_and_six(line_after_first_four: Self::RestOfLine) -> ParseResult<Option<[&'a str; 2]>> {
        let mut parts = line_after_first_four;

        let five = parts.next();
        let six = parts.next();

        if parts.next().is_some() {
            Err(ParseError::new("Line has more than 6 elements"))
        } else {
            match (five, six) {
                (Some(five), Some(six)) => Ok(Some([five, six])),
                (None, None) => Ok(None),
                (Some(_), None) | (None, Some(_)) => Err(
                    ParseError::new("Line has a fifth element, but no sixth")
                )
            }
        }
    }

    fn one_through_three(line: &'a str) -> ParseResult<([&'a str; 3], Self::RestOfLine)> {
        let mut parts = line.split_whitespace();

        let one = parts.next()
            .ok_or_else(|| ParseError::new("Could not read first field"))?;
        let two = parts.next()
            .ok_or_else(|| ParseError::new("Could not read second field"))?;
        let three = parts.next()
            .ok_or_else(|| ParseError::new("Could not read third field"))?;

        Ok(([one, two, three], parts))
    }

    fn four(rest_of_line: Self::RestOfLine) -> ParseResult<[&'a str; 1]> {
        let mut parts = rest_of_line;

        parts.next().map(|v| [v])
            .ok_or_else(|| ParseError::new("Could not read value for bound."))
    }
}
