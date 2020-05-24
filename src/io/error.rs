//! # Error reporting for reading of linear program files
//!
//! A collection of enums and structures describing any problems encountered during reading and
//! parsing.
use core::fmt::Display;
use std::convert::Into;
use std::error::Error;
use std::fmt;
use std::io;
use std::ops::Deref;

/// An `ImportError` is created when an error was encountered during IO or parsing.
///
/// It is the highest error in the io error hierarchy.
#[derive(Debug)]
pub enum ImportError {
    /// The file extension of the provided file path is not known or supported.
    ///
    /// The contained `String` is a message for the end user.
    FileExtension(String),
    /// The file to read isn't found, or the reading of file couldn't start or was interrupted.
    IO(io::Error),
    /// Contents of the file could not be parsed into a linear program.
    ///
    /// # Note
    ///
    /// If the linear program is inconsistent, that will not be represented with this error. This
    /// variant should only be created for syntactically incorrect files.
    Parse(ParseError),
    /// There is a logical inconsistency in the linear program described by a file.
    ///
    /// For example, a bound might be given for a variable which is not known.
    LinearProgram(InconsistencyError),
}

impl Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ImportError::FileExtension(err) => err.fmt(f),
            ImportError::IO(error) => error.fmt(f),
            ImportError::Parse(error) => error.fmt(f),
            ImportError::LinearProgram(error) => error.fmt(f),
        }
    }
}

impl Error for ImportError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ImportError::FileExtension(_) => None,
            ImportError::IO(error) => error.source(),
            ImportError::Parse(error) => error.source(),
            ImportError::LinearProgram(_error) => None,
        }
    }
}

/// A `ParseError` represents all errors encountered during parsing.
///
/// It may recursively hold more ParseErrors to provide more detail. At the end of this chain, there
/// may be a file location containing a line number and line, at which the error was caused.
#[derive(Debug)]
pub struct ParseError {
    description: String,
    source: Option<ParseErrorSource>,
}

/// A `ParseErrorCause` can be used with a `ParseError` to describe its cause.
///
/// It can be either a file line number and line contents, or another `ParseError` with its own
/// description and optionally, a cause.
#[derive(Debug)]
enum ParseErrorSource {
    FileLocation(u64, String),
    Nested(Box<ParseError>),
}

impl Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ParseError: {}", self.to_string())
    }
}

impl Error for ParseError {
    /// Describe this error.
    ///
    /// # Return value
    ///
    /// The description as a `&str`.
    fn description(&self) -> &str {
        self.description.as_str()
    }

    /// Find out what caused this error.
    ///
    /// # Return value
    ///
    /// Option<&Error> which may be a `ParseErrorCause`.
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        if let Some(ParseErrorSource::Nested(ref error)) = self.source {
            Some(error)
        } else { None }
    }
}

impl ParseError {
    /// Create a new `ParseError` with only a description.
    ///
    /// # Arguments
    ///
    /// * `description`: What's wrong at the moment of creation.
    ///
    /// # Return value
    ///
    /// * A `ParseError` instance without a cause.
    pub fn new(description: impl Into<String>) -> ParseError {
        ParseError { description: description.into(), source: None, }
    }

    /// Create a new `ParseError` instance with a `FileLocation` as a cause.
    ///
    /// # Arguments
    ///
    /// * `description`: What's wrong at the moment of creation.
    /// * `file_location`: A reference to a line number and line that caused the error.
    ///
    /// # Return value
    ///
    /// A new `ParseError` instance with a `FileLocation` cause.
    pub fn with_file_location(
        description: impl Into<String>,
        file_location: FileLocation,
    ) -> ParseError {
        let (line_number, line) = file_location;
        ParseError {
            description: description.into(),
            source: Some(ParseErrorSource::FileLocation(line_number, line.to_string())),
        }
    }

    /// Wrap a new `ParseError` around an existing one.
    ///
    /// # Arguments
    ///
    /// * `description`: What's wrong at the moment of creation.
    /// * `parse_error`: What caused this `ParseError`.
    ///
    /// # Return value
    ///
    /// A new `ParseError` instance with a `ParseError` cause.
    pub fn with_cause(description: impl Into<String>, parse_error: ParseError) -> ParseError {
        ParseError {
            description: description.into(),
            source: Some(ParseErrorSource::Nested(Box::new(parse_error))),
        }
    }

    /// Get all errors in the chain, leading up to this one.
    ///
    /// # Return value
    ///
    /// A new `ParseError` instance.
    fn chain_description(&self) -> Vec<String> {
        let mut descriptions = vec![self.to_string()];

        if let Some(ref source) = self.source {
            match source {
                ParseErrorSource::FileLocation(line_number, line) => {
                    descriptions.push(format!("\tCaused at line\t{}:\t{}", line_number, line));
                }
                ParseErrorSource::Nested(error) => {
                    descriptions.append(&mut error.chain_description());
                }
            }
        }

        descriptions
    }

    /// Get the source of this error.
    ///
    /// # Return value
    ///
    /// The source of this error, or self.
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        if let Some(ParseErrorSource::Nested(ref error)) = self.source {
            Some(error.deref() as &dyn Error)
        } else {
            Some(self as &dyn Error)
        }
    }
}

/// A `FileLocation` references a line in the file by the line number of the file as originally
/// read from the disk. It contains a reference to the line itself.
pub(super) type FileLocation<'a> = (u64, &'a str);

/// A `LinearProgramError` is thrown when the linear program is inconsistently represented in the
/// file.
///
/// This `Error` is not returned when the linear program is unfeasible or unbounded. It is meant
/// only for descriptions of linear programs, and should not be used after the importing process.
#[derive(Debug)]
pub struct InconsistencyError {
    description: String,
}

impl InconsistencyError {
    /// Wrap a text in a `LinearProgramError`.
    ///
    /// # Arguments
    ///
    /// * `description`: A human-readable text mean for the end user.
    ///
    /// # Returns
    ///
    /// A `LinearProgramError`.
    pub fn new(description: impl Into<String>) -> InconsistencyError {
        InconsistencyError { description: description.into(), }
    }
}

impl Display for InconsistencyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ProgramError: {}", self.to_string())
    }
}

impl Error for InconsistencyError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self)
    }
}
