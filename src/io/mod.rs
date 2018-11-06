//! # Reading and writing of linear programs
//!
//! This module provides read and write functionality for linear program formats.
use std::f64;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use data::linear_program::general_form::GeneralForm;
use io::error::ImportError;


mod mps;
pub mod error;

pub const EPSILON: f64 = f64::EPSILON;


/// The `import` function takes a file path and returns, if successful, a struct which can be
/// converted to a linear program in general form.
pub fn import(file_path: &Path) -> Result<impl Into<GeneralForm>, ImportError> {
    let mut program = String::new();
    File::open(&file_path)
        .map_err(ImportError::IO)?
        .read_to_string(&mut program)
        .map_err(ImportError::IO)?;

    // Choose the right parser
    match file_path.extension() {
        Some(extension) => match extension.to_str() {
            Some("mps") => mps::import(&program),
            Some(extension_string) => Err(ImportError::FileExtension(
                format!("Could not recognise file extension \"{}\" of file: {:?}",
                        extension_string, file_path))),
            None => Err(ImportError::FileExtension(
                format!("Could not convert OsStr to &str, probably invalid unicode: {:?}",
                        extension))),
        },
        None => Err(ImportError::FileExtension(format!("Could not read extension from file path: {:?}",
                                                       file_path))),
    }
}
