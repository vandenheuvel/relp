//! # Importing MPS files
//!
//! Reading of `.mps` files, or files of the Mathematical Programming System format.
//!
//! See http://lpsolve.sourceforge.net/5.5/mps-format.htm for a specification.
//!
//! TODO:
//!     * Support all `BoundType` variants
use std::convert::{TryFrom, TryInto};

use num::FromPrimitive;

use crate::data::linear_program::elements::ConstraintType;
use crate::data::linear_program::elements::VariableType;
use crate::io::error::Import;
use crate::io::mps::parsing::into_atom_lines;
use crate::io::mps::parsing::UnstructuredMPS;
use crate::io::mps::structuring::MPS;

pub mod parsing;
pub mod structuring;
mod token;

/// Parse an MPS program, in string form, to a MPS.
///
/// # Arguments
///
/// * `program`: The input in [MPS format](https://en.wikipedia.org/wiki/MPS_(format)).
///
/// # Return value
///
/// A `Result<MPS, ImportError>` instance.
pub fn import<F: FromPrimitive + Clone>(program: &impl AsRef<str>) -> Result<MPS<F>, Import> {
    let atom_lines = into_atom_lines(program);
    let unstructured_mps = UnstructuredMPS::try_from(atom_lines)
        .map_err(|e| Import::Parse(e))?;
    unstructured_mps.try_into()
        .map_err(|e| Import::LinearProgram(e))
}

/// MPS files are divided into sections.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Section<'a> {
    Name(&'a str),
    Rows,
    Columns(VariableType),
    Rhs,
    Bounds,
    /// This section is not used in the MIPLIB 2010 benchmark.
    Ranges,
    /// The `Endata` variant (notice the odd spelling) denotes the end of the file.
    Endata,
}

/// Every row is either a cost row or some constraint.
#[derive(Debug, Eq, PartialEq)]
enum RowType {
    Cost,
    Constraint(ConstraintType),
}

/// The MPS format defines the `BoundType`s described in this enum.
///
/// # Note
///
/// Not all `BoundType` variants are currently supported.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum BoundType<F> {
    /// b <- x (< +inf)
    LowerContinuous(F),
    /// (0 <=) x <= b
    UpperContinuous(F),
    /// x = b
    Fixed(F),
    /// -inf < x < +inf
    Free,
    /// -inf < x (<= 0)
    LowerMinusInfinity,
    /// (0 <=) x < +inf
    UpperInfinity,
    /// x = 0 or 1
    Binary,
    /// b <= x (< +inf)
    LowerInteger(F),
    /// (0 <=) x <= b
    UpperInteger(F),
    /// x = 0 or l =< x =< b
    ///
    /// Note: appears only very rarely in the MIPLIB benchmark set.
    SemiContinuous(F, F),
}

/// Every `Row` has a name and a `RowType`.
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub(crate) struct Constraint {
    pub name_index: usize,
    pub constraint_type: ConstraintType,
}

/// Is either continuous or integer, and has for some rows a coefficient.
#[derive(Debug, PartialEq)]
pub(crate) struct Variable<F> {
    pub name_index: usize,
    pub variable_type: VariableType,
    pub values: Vec<(usize, F)>,
}

/// The right-hand side of Ax = b.
///
/// A single linear program defined in MPS can have multiple right-hand sides. It relates a row name
/// to a real constant.
#[derive(Debug, PartialEq)]
pub(crate) struct Rhs<F> {
    pub name: String,
    pub values: Vec<(usize, F)>,
}

/// Specifies a bound on constraint activation.
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
pub(crate) struct Range<F> {
    pub name: String,
    /// Sorted constraint indices and their 'r' value.
    pub values: Vec<(usize, F)>,
}

/// Specifies a bound on a variable. The variable can either be continuous or integer, while the
/// bound can have any direction.
#[derive(Debug, PartialEq)]
pub(crate) struct Bound<F> {
    pub name: String,
    pub values: Vec<(BoundType<F>, usize)>,
}
