//! # Building blocks to describe linear programs.
#![allow(missing_docs)]
use std::ops::Not;

use crate::data::linear_program::solution::Solution;

/// A `Constraint` is a type of (in)equality.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ConstraintType {
    Equal,
    Greater,
    Less,
}

/// Direction of a bound.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum BoundDirection {
    Lower,
    Upper,
}
impl Not for BoundDirection {
    type Output = Self;

    fn not(self) -> Self {
        match self {
            Self::Lower => Self::Upper,
            Self::Upper => Self::Lower,
        }
    }
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
pub enum LinearProgramType<F> {
    FiniteOptimum(Solution<F>),
    Infeasible,
    Unbounded,
}

/// Direction of optimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Objective {
    Maximize,
    Minimize,
}
impl Default for Objective {
    fn default() -> Self {
        Objective::Minimize
    }
}
