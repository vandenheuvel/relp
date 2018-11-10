//! # Building blocks to describe linear programs.
#![allow(missing_docs)]
use std::ops::Not;

use crate::data::number_types::traits::OrderedField;

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
#[derive(Debug, Eq, PartialEq)]
pub enum LinearProgramType<OF: OrderedField> {
    FiniteOptimum(OF),
    Infeasible,
    Unbounded,
}

/// Direction of optimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Objective {
    Maximize,
    Minimize,
}
