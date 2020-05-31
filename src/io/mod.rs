//! # Reading and writing of linear programs
//!
//! This module provides read and write functionality for linear program formats.
use std::convert::TryInto;
use std::f64;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use num::FromPrimitive;

use crate::data::linear_program::general_form::GeneralForm;
use crate::data::number_types::traits::{OrderedField, RealField};
use crate::io::error::Import;

pub mod error;
pub mod mps;

/// Allowable numerical error used for equality testing in this module.
///
/// Because none of the values that are being read or written are modified or calculated with, this
/// tolerance allows only for rounding from values with a higher precision than f64.
const EPSILON: f64 = f64::EPSILON;

/// The `import` function takes a file path and returns, if successful, a struct which can be
/// converted to a linear program in general form.
pub fn import<F: OrderedField + FromPrimitive>(
    file_path: &Path
) -> Result<impl TryInto<GeneralForm<F>>, Import> {
    // Open and read the file
    let mut program = String::new();
    File::open(&file_path)
        .map_err(Import::IO)?
        .read_to_string(&mut program)
        .map_err(Import::IO)?;

    // Choose the right parser
    match file_path.extension() {
        Some(extension) => match extension.to_str() {
            Some("mps") => mps::import(&program),
            Some(extension_string) => Err(Import::FileExtension(format!(
                "Could not recognise file extension \"{}\" of file: {:?}",
                extension_string, file_path
            ))),
            None => Err(Import::FileExtension(format!(
                "Could not convert OsStr to &str, probably invalid unicode: {:?}",
                extension
            ))),
        },
        None => Err(Import::FileExtension(format!(
            "Could not read extension from file path: {:?}",
            file_path
        ))),
    }
}
