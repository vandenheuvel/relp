use std::ops::Not;

use data::linear_algebra::vector::SparseVector;

/// A `Constraint` is a type of (in)equality.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ConstraintType {
    Equal,
    Greater,
    Less,
}

/// A variable is either continuous or integer.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum VariableType {
    Continuous,
    Integer,
}

impl Not for VariableType {
    type Output = VariableType;

    fn not(self) -> VariableType {
        match self {
            VariableType::Continuous => VariableType::Integer,
            VariableType::Integer => VariableType::Continuous,
        }
    }
}

/// After the second phase, either an optimum is found or the problem is determined to be unbounded.
#[derive(Debug)]
pub enum LinearProgramType {
    FiniteOptimum(SparseVector, f64),
    Infeasible,
    Unbounded,
}

/// A variable is named, of continuous or integer type and may be shifted.
#[derive(Clone, Debug, PartialEq)]
pub struct Variable {
    pub name: String,
    pub variable_type: VariableType,
    pub offset: f64,
}
