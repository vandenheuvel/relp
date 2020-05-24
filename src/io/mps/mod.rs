//! # Importing MPS files
//!
//! Reading of `.mps` files, or files of the Mathematical Programming System format.
//!
//! TODO:
//!     * Support all `BoundType` variants
use std::convert::identity;
use std::convert::TryFrom;
use std::convert::TryInto;

use approx::AbsDiffEq;

use crate::data::linear_program::elements::ConstraintType;
use crate::data::linear_program::elements::VariableType;
use crate::io::EPSILON;
use crate::io::error::ImportError;
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
pub fn import(program: &impl AsRef<str>) -> Result<MPS, ImportError> {
    let atom_lines = into_atom_lines(program);
    let unstructured_mps = UnstructuredMPS::try_from(atom_lines).map_err(|e| ImportError::Parse(e))?;
    unstructured_mps.try_into().map_err(|e| ImportError::LinearProgram(e))
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
pub(crate) enum BoundType {
    /// b <- x (< +inf)
    LowerContinuous,
    /// (0 <=) x <= b
    UpperContinuous,
    /// x = b
    Fixed,
    /// -inf < x < +inf
    Free,
    /// -inf < x (<= 0)
    LowerMinusInfinity,
    /// (0 <=) x < +inf
    UpperInfinity,
    /// x = 0 or 1
    Binary,
    /// b <= x (< +inf)
    LowerInteger,
    /// (0 <=) x <= b
    UpperInteger,
    /// x = 0 or l =< x =< b
    ///
    /// Note: appears only very rarely in the MIPLIB benchmark set.
    SemiContinuous,
}

/// Every `Row` has a name and a `RowType`.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Constraint {
    pub name: usize,
    pub constraint_type: ConstraintType,
}

/// Is either continuous or integer, and has for some rows a coefficient.
#[derive(Debug, PartialEq)]
pub(crate) struct Variable {
    pub name: usize,
    pub variable_type: VariableType,
    pub values: Vec<(usize, f64)>,
}

impl AbsDiffEq for Variable {
    type Epsilon = f64;

    fn default_epsilon() -> <Self as AbsDiffEq>::Epsilon {
        EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, _epsilon: <Self as AbsDiffEq>::Epsilon) -> bool {
        self.name == other.name &&
            self.variable_type == other.variable_type &&
            self.values.iter().zip(other.values.iter())
                .map(|((index1, value1), (index2, value2))|
                    index1 == index2 && abs_diff_eq!(value1, value2))
                .all(identity)
    }
}

/// The right-hand side of Ax = b.
///
/// A single linear program defined in MPS can have multiple right-hand sides. It relates a row name
/// to a real constant.
#[derive(Debug, PartialEq)]
pub(crate) struct Rhs {
    pub name: String,
    pub values: Vec<(usize, f64)>,
}

impl AbsDiffEq for Rhs {
    type Epsilon = f64;

    fn default_epsilon() -> <Self as AbsDiffEq>::Epsilon {
        EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, _epsilon: <Self as AbsDiffEq>::Epsilon) -> bool {
        self.name == other.name &&
            self.values.iter().zip(other.values.iter())
                .map(|((index1, value1), (index2, value2))|
                    index1 == index2 && abs_diff_eq!(value1, value2))
                .all(identity)
    }
}

/// Specifies a bound on a variable. The variable can either be continuous or integer, while the
/// bound can have any direction.
#[derive(Debug, PartialEq)]
pub(crate) struct Bound {
    pub name: String,
    pub values: Vec<(BoundType, usize, f64)>,
}

impl AbsDiffEq for Bound {
    type Epsilon = f64;

    fn default_epsilon() -> <Self as AbsDiffEq>::Epsilon {
        EPSILON
    }

    fn abs_diff_eq(&self, other: &Self, _epsilon: <Self as AbsDiffEq>::Epsilon) -> bool {
        self.name == other.name &&
            self.values.iter().zip(other.values.iter())
                .map(|((bound_type1, index1, value1), (bound_type2, index2, value2))|
                    bound_type1 == bound_type2 &&
                        index1 == index2 &&
                        abs_diff_eq!(value1, value2))
                .all(identity)
    }
}
