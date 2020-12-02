//! # Reading and writing of linear programs
//!
//! This module provides read and write functionality for linear program formats.
use std::convert::TryInto;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use num::FromPrimitive;

use crate::data::linear_algebra::traits::SparseElementZero;
use crate::data::linear_program::general_form::GeneralForm;
use crate::data::number_types::traits::{OrderedField, OrderedFieldRef};
use crate::io::error::Import;

pub mod error;
pub mod mps;

/// Import a problem from a file.
///
/// Currently only supports the MPS filetype.
///
/// The `import` function takes a file path and returns, if successful, a struct which can be
/// converted to a linear program in general form.
///
/// # Errors
///
/// When a file extension is unknown, a file cannot be found or read, there is an inconsistency in
/// the problem file, etc. an error type is returned.
pub fn import<OF: OrderedField + FromPrimitive + 'static, OFZ: SparseElementZero<OF>>(
    file_path: &Path
) -> Result<impl TryInto<GeneralForm<OF, OFZ>>, Import>
where
    for<'r> &'r OF: OrderedFieldRef<OF>,
{
    // Open and read the file
    let mut program = String::new();
    File::open(&file_path)
        .map_err(Import::IO)?
        .read_to_string(&mut program)
        .map_err(Import::IO)?;

    // Choose the right parser
    match file_path.extension() {
        Some(extension) => match extension.to_str() {
            Some("mps" | "SIF") => mps::import(&program),
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
