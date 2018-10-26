use std::convert::Into;
use std::fmt;
use std::error::Error;
use std::io;
use core::fmt::Display;

/// A `ReadError` is created when an error was encountered during IO or parsing.
///
/// # Note
///
/// It is the highest error in the io error hierarchy.
#[derive(Debug)]
pub enum ImportError {
    FileExtension(String),
    IO(io::Error),
    Parse(ParseError),
    LinearProgram(LinearProgramError),
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
    fn description(&self) -> &str {
        match self {
            ImportError::FileExtension(description) => description,
            ImportError::IO(error) => error.description(),
            ImportError::Parse(error) => error.description(),
            ImportError::LinearProgram(error) => error.description(),
        }
    }

    fn cause(&self) -> Option<&Error> {
        match self {
            ImportError::FileExtension(_) => None,
            ImportError::IO(error) => error.cause(),
            ImportError::Parse(error) => error.cause(),
            ImportError::LinearProgram(error) => None,
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
    cause: Option<ParseErrorCause>,
}

/// A `ParseErrorCause` can be used with a `ParseError` to describe its cause.
///
/// It can be either a file line number and line contents, or another `ParseError` with its own
/// description and optionally, a cause.
#[derive(Debug)]
enum ParseErrorCause {
    FileLocation(u64, String),
    Nested(Box<ParseError>),
}

impl Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ParseError: {}", self.description())
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
    fn cause(&self) -> Option<&Error> {
        if let Some(ParseErrorCause::Nested(ref error)) = self.cause {
            Some(error)
        } else { None }
    }
}

impl ParseError {
    /// Create a new `ParseError` with only a description.
    ///
    /// # Arguments
    ///
    /// * `description` - What's wrong at the moment of creation.
    ///
    /// # Return value
    ///
    /// * A `ParseError` instance without a cause.
    pub fn new(description: impl Into<String>) -> ParseError {
        ParseError { description: description.into(), cause: None, }
    }

    /// Create a new `ParseError` instance with a `FileLocation` as a cause.
    ///
    /// # Arguments
    ///
    /// * `description` - What's wrong at the moment of creation.
    /// * `file_location` - A reference to a line number and line that caused the error.
    ///
    /// # Return value
    ///
    /// A new `ParseError` instance with a `FileLocation` cause.
    pub fn with_file_location(description: impl Into<String>, file_location: FileLocation) -> ParseError {
        let (line_number, line) = file_location;
        ParseError {
            description: description.into(),
            cause: Some(ParseErrorCause::FileLocation(line_number, line.to_string())),
        }
    }

    /// Wrap a new `ParseError` around an existing one.
    ///
    /// # Arguments
    ///
    /// * `description` - What's wrong at the moment of creation.
    /// * `parse_error` - What caused this `ParseError`.
    ///
    /// # Return value
    ///
    /// A new `ParseError` instance with a `ParseError` cause.
    pub fn with_cause(description: impl Into<String>, parse_error: ParseError) -> ParseError {
        ParseError {
            description: description.into(),
            cause: Some(ParseErrorCause::Nested(Box::new(parse_error))),
        }
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
                ParseErrorCause::FileLocation(line_number, line) => {
                    descriptions.push(format!("\tCaused at line\t{}:\t{}", line_number, line));
                },
                ParseErrorCause::Nested(error) => {
                    descriptions.append(&mut error.chain_description());
                },
            }
        }

        descriptions
    }
    /// Get the source of this error.
    ///
    /// # Return value
    ///
    /// The source of this error, or self.
    fn source(&self) -> &ParseError {
        if let Some(ParseErrorCause::Nested(ref error)) = self.cause {
            error
        } else {
            &self
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
pub struct LinearProgramError {
    description: String,
}

impl Display for LinearProgramError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ProgramError: {}", self.description())
    }
}

impl Error for LinearProgramError {
    fn description(&self) -> &str {
        self.description.as_str()
    }

    fn cause(&self) -> Option<&Error> {
        Some(self)
    }
}

impl LinearProgramError {
    pub fn new(description: impl Into<String>) -> LinearProgramError {
        LinearProgramError { description: description.into(), }
    }
}
