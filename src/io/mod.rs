//! # Reading and writing of linear programs
//!
//! This module provides read and write functionality for linear program formats.

use std::error::Error;
use std::fmt::{Display, Formatter, Result as FormatResult};
use std::fs::File;
use std::io::{Error as IOError, Read};
use std::path::Path;

use data::linear_program::general_form::GeneralForm;

mod mps;

/// The `read` function takes a file path and returns, if successful, a struct which can be
/// converted to a linear program in general form.
pub fn read(file_path: &Path) -> Result<impl Into<GeneralForm>, ReadError> {
    let mut program = String::new();
    File::open(&file_path)?
        .read_to_string(&mut program)?;

    // Choose the right parser
    match file_path.extension() {
        Some(extension) => match extension.to_str() {
            Some("mps") => mps::parse(program).map_err(ReadError::from),
            Some(extension_string) => Err(FileExtensionError {
                description: format!("Could not recognise file extension \"{}\" of file {:?}",
                                     extension_string, file_path),
            }).map_err(ReadError::from),
            None => Err(FileExtensionError {
                description: format!("Could not convert OsStr to &str, probably invalid unicode: \
                {:?}", extension),
            }).map_err(ReadError::from),
        },
        None => Err(FileExtensionError {
            description: format!("Could not read extension from file path: {:?}", file_path),
        }).map_err(ReadError::from),
    }
}

/// A `ReadError` is created when an error was encountered during IO and parsing.
///
/// # Note
///
/// It is the highest error in the error hierarchy.
#[derive(Debug)]
pub enum ReadError {
    FileExtensionError(FileExtensionError),
    IOError(IOError),
    ParseError(ParseError),
}

impl From<FileExtensionError> for ReadError {
    fn from(error: FileExtensionError) -> ReadError {
        ReadError::FileExtensionError(error)
    }
}

impl From<IOError> for ReadError {
    fn from(error: IOError) -> ReadError {
        ReadError::IOError(error)
    }
}

impl From<ParseError> for ReadError {
    fn from(error: ParseError) -> ReadError {
        ReadError::ParseError(error)
    }
}

impl Display for ReadError {
    fn fmt(&self, f: &mut Formatter) -> FormatResult {
        match self {
            ReadError::FileExtensionError(err) => err.fmt(f),
            ReadError::IOError(err) => err.fmt(f),
            ReadError::ParseError(err) => err.fmt(f),
        }
    }
}

impl Error for ReadError {
    fn description(&self) -> &str {
        match self {
            ReadError::FileExtensionError(err) => err.description(),
            ReadError::IOError(err) => err.description(),
            ReadError::ParseError(err) => err.description(),
        }
    }
}

#[derive(Debug)]
pub struct FileExtensionError {
    description: String,
}

impl Display for FileExtensionError {
    fn fmt(&self, f: &mut Formatter) -> FormatResult {
        write!(f, "{}", self.description)
    }
}

impl Error for FileExtensionError {
    fn description(&self) -> &str {
        self.description.as_str()
    }
}

/// A `ParseError` represents all errors encountered during parsing.
///
/// It may recursively hold more ParseErrors to provide more detail. At the end of this chain, there
/// may be a `ParseErrorSource` instance containing a line number and line.
#[derive(Debug)]
pub struct ParseError {
    description: String,
    cause: Option<ParseErrorCause>,
}

impl ParseError {
    /// Create a new `ParseError` instance.
    ///
    /// # Arguments
    ///
    /// * `description` - A `String` describing in a readable way what went wrong.
    /// * `source` - May contain a reference to a line number and line that caused the error.
    ///
    /// # Return value
    ///
    /// A new `ParseError` instance.
    fn new(description: String, source: Option<(u64, &str)>) -> ParseError {
        ParseError { description, cause: match source {
            Some((line_number, line)) => Some(ParseErrorCause::Source(ParseErrorSource {
                line_number,
                line: line.to_string(),
            })),
            None => None,
        }, }
    }
    /// Get all errors in the chain, leading up to this one.
    ///
    /// # Return value
    ///
    /// A new `ParseError` instance.
    fn chain_description(&self) -> Vec<String> {
        let mut descriptions = vec![self.description().to_string()];
        if let Some(ref cause) = self.cause {
            match cause {
                ParseErrorCause::Chain(error) => descriptions.append(&mut error.chain_description()),
                ParseErrorCause::Source(error) => descriptions.push(error.description()),
            }
        }
        descriptions
    }
    /// Get the source of this error.
    ///
    /// # Return value
    ///
    /// The source of this error containing a line number and line.
    fn source(&self) -> Option<&ParseErrorSource> {
        if let Some(ref cause) = self.cause {
            match cause {
                ParseErrorCause::Chain(cause) => cause.source(),
                ParseErrorCause::Source(source) => return Some(&source),
            }
        } else { None }
    }
}

impl Display for ParseError {
    fn fmt(&self, f: &mut Formatter) -> FormatResult {
        write!(f, "ParseError: {}", self.description())
    }
}

impl Error for ParseError {
    fn description(&self) -> &str {
        self.description.as_str()
    }
}

#[derive(Debug)]
enum ParseErrorCause {
    Chain(Box<ParseError>),
    Source(ParseErrorSource)
}

/// The source of a `ParseError`.
///
/// Contains a line number and line, of the place where the error originated.
#[derive(Debug)]
struct ParseErrorSource {
    line_number: u64,
    line: String,
}

impl ParseErrorSource {
    /// Get a readable string containing the data.
    ///
    /// # Return value
    ///
    /// A formatted string to be read by human being.
    fn description(&self) -> String {
        format!("\tCaused at line {}: \"{}\"", self.line_number, self.line)
    }
}

impl Display for ParseErrorSource {
    fn fmt(&self, f: &mut Formatter) -> FormatResult {
        f.write_str(&self.description())
    }
}
