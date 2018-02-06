//! # Reading and writing of linear programs
//!
//! This module provides read and write functionality for linear program formats.

mod mps;

use std::fs::File;
use std::io::Read;
use std::path::Path;
use data::linear_program::general_form::GeneralFormConvertable;

/// The `read` function takes a file path and returns, if successful, a struct which can be
/// converted to a linear program in general form.
pub fn read(file_path: &Path) -> Result<Box<GeneralFormConvertable>, String> {
    // Open the file
    let mut file = match File::open(&file_path) {
        Ok(file) => file,
        Err(_) => return Err(String::from("Failed to open data file")),
    };

    // Read the file
    let mut program = String::new();
    if let Err(_) = file.read_to_string(&mut program) {
        return Err(String::from("Error reading data file"));
    }
    let lines = program.lines();

    // Choose the right parser
    match file_path.extension() {
        Some(extension) => match extension.to_str() {
            Some("mps") => mps::parse(lines),
            Some(extension_string) => Err(format!("Could not recognise file extension \"{}\" of \
            file {:?}", extension_string, file_path)),
            None => Err(format!("Could not convert OsStr to &str, probably invalid unicode: {:?}",
                                extension)),
        },
        None => Err(format!("Could not read extension from file path: {:?}", file_path)),
    }
}
