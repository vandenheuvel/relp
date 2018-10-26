//! # Importing MPS files
//!
//! Reading of `.mps` files, or files of the Mathematical Programming System format.
//!
//! TODO:
//!     * Support all `BoundType` variants
use std::convert::TryFrom;
use std::convert::TryInto;

use data::linear_program::elements::ConstraintType;
use data::linear_program::elements::VariableType;
use io::mps::parsing::into_atom_lines;
use io::mps::parsing::UnstructuredMPS;
use io::mps::structuring::MPS;
use io::error::ImportError;

mod parsing;
mod structuring;
mod token;


/// Parse an MPS program, in string form, to a MPS.
///
/// # Arguments
///
/// * `program` - The input in [MPS format](https://en.wikipedia.org/wiki/MPS_(format)).
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
///
/// # Note
///
/// The `Endata` variant (notice the odd spelling) denotes the end of the file.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Section<'a> {
    Name(&'a str),
    Rows,
    Columns(VariableType),
    Rhs,
    Bounds,
    Ranges,
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
enum BoundType {
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
    /// b <= x ( +inf)
    LowerInteger,
    /// (0 <=) x <= b
    UpperInteger,
    /// x = 0 or l =< x =< b
    SemiContinuous,
}

/// Every `Row` has a name and a `RowType`.
#[derive(Debug, Eq, PartialEq)]
struct Constraint {
    name: usize,
    constraint_type: ConstraintType,
}

/// Is either continuous or integer, and has for some rows a coefficient.
#[derive(Debug, PartialEq)]
struct Variable {
    name: usize,
    variable_type: VariableType,
    values: Vec<(usize, f64)>,
}

/// The right-hand side of Ax = b.
///
/// A single linear program defined in MPS can have multiple right-hand sides. It relates a row name
/// to a real constant.
#[derive(Debug, PartialEq)]
struct Rhs {
    name: String,
    values: Vec<(usize, f64)>,
}

/// Specifies a bound on a variable. The variable can either be continuous or integer, while the
/// bound can have any direction.
#[derive(Debug, PartialEq)]
struct Bound {
    name: String,
    values: Vec<(BoundType, usize, f64)>,
}

/// Integration testing the `io::mps` module.
#[cfg(test)]
pub(super) mod test {
    use io::mps::Constraint;
    use io::mps::structuring::MPS;
    use data::linear_program::elements::VariableType;
    use data::linear_program::elements::ConstraintType;
    use io::mps::BoundType;
    use io::mps::Bound;
    use io::mps::Rhs;
    use io::mps::Variable;
    use data::linear_program::general_form::GeneralForm;
    use data::linear_algebra::matrix::SparseMatrix;
    use data::linear_algebra::matrix::Matrix;
    use data::linear_algebra::vector::DenseVector;
    use data::linear_algebra::vector::SparseVector;
    use data::linear_algebra::vector::Vector;
    use data::linear_program::elements::Variable as ShiftedVariable;
    use io::mps::import;

    /// A complete MPS file, in a static &str.
    pub const MPS_STRING: &str =
"NAME          TESTPROB
ROWS
 N  COST
 L  LIM1
 G  LIM2
 E  MYEQN
COLUMNS
    XONE      COST                 1   LIM1                 1
    XONE      LIM2                 1
    YTWO      COST                 4   LIM1                 1
    YTWO      MYEQN               -1
    ZTHREE    COST                 9   LIM2                 1
    ZTHREE    MYEQN                1
RHS
    RHS1      LIM1                 5   LIM2                10
    RHS1      MYEQN                7
BOUNDS
 UP BND1      XONE                 4
 LO BND1      YTWO                -1
 UP BND1      YTWO                 1
ENDATA";

    /// Build the expected `MPS` instance, corresponding to the MPS file string.
    pub fn lp_mps() -> MPS {
        let name = "TESTPROB".to_string();
        let cost_row_name = "COST".to_string();
        let cost_values = vec![(0, 1f64), (1, 4f64), (2, 9f64)];
        let row_names = vec!["LIM1", "LIM2", "MYEQN"].into_iter().map(String::from).collect();
        let rows = vec![Constraint { name: 0, constraint_type: ConstraintType::Less, },
                        Constraint { name: 1, constraint_type: ConstraintType::Greater, },
                        Constraint { name: 2, constraint_type: ConstraintType::Equal, }];
        let column_names = vec!["XONE", "YTWO", "ZTHREE"].into_iter().map(String::from).collect();
        let columns = vec![
            Variable {
                name: 0,
                variable_type: VariableType::Continuous,
                values: vec![(0, 1f64), (1, 1f64)],
            },
            Variable {
                name: 1,
                variable_type: VariableType::Continuous,
                values: vec![(0, 1f64), (2, -1f64)],
            },
            Variable {
                name: 2,
                variable_type: VariableType::Continuous,
                values: vec![(1, 1f64), (2, 1f64)],
            },
        ];
        let rhss = vec![
            Rhs {
                name: "RHS1".to_string(),
                values: vec![(0, 5f64), (1, 10f64), (2, 7f64)],
            }
        ];
        let bounds = vec![
            Bound {
                name: "BND1".to_string(),
                values: vec![(BoundType::UpperContinuous, 0, 4f64),
                             (BoundType::LowerContinuous, 1, -1f64),
                             (BoundType::UpperContinuous, 1, 1f64)],
            }
        ];

        MPS::new(name,
                 cost_row_name,
                 cost_values,
                 row_names,
                 rows,
                 column_names,
                 columns,
                 rhss,
                 bounds)
    }

    /// Test parsing and structuring
    #[test]
    fn test_import() {
        let result = import(&MPS_STRING);
        let expected = lp_mps();

        assert_eq!(result.unwrap(), expected);
    }

    /// The linear program in expected `GeneralForm`.
    pub fn lp_general() -> GeneralForm {
        let data = vec![vec![1f64, 1f64, 0f64],
                        vec![1f64, 0f64, 1f64],
                        vec![0f64, -1f64, 1f64],
                        vec![1f64, 0f64, 0f64],
                        vec![0f64, 1f64, 0f64],
                        vec![0f64, 1f64, 0f64]];
        let data = SparseMatrix::from_data(data);

        let b = DenseVector::from_data(vec![5f64,
                                            10f64,
                                            7f64,
                                            4f64,
                                            -1f64,
                                            1f64]);

        let cost = SparseVector::from_data(vec![1f64, 4f64, 9f64]);

        let column_info = vec![
            ShiftedVariable {
                name: "XONE".to_string(),
                variable_type: VariableType::Continuous,
                offset: 0f64,
            },
            ShiftedVariable {
                name: "YTWO".to_string(),
                variable_type: VariableType::Continuous,
                offset: 0f64,
            },
            ShiftedVariable {
                name: "ZTHREE".to_string(),
                variable_type: VariableType::Continuous,
                offset: 0f64,
            }
        ];

        let row_info = vec![ConstraintType::Less,
                            ConstraintType::Greater,
                            ConstraintType::Equal,
                            ConstraintType::Less,
                            ConstraintType::Greater,
                            ConstraintType::Less];

        GeneralForm::new(data, b, cost, 0f64, column_info, Vec::new(), row_info)
    }

    /// Test parsing, structuring and conversion to `GeneralForm` linear program.
    #[test]
    fn import_and_convert() {
        let result: GeneralForm = import(&MPS_STRING).unwrap().into();
        let expected = lp_general();

        assert_eq!(format!("{:?}", result), format!("{:?}", expected));
    }
}
