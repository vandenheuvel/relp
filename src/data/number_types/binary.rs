//! # Binary data
//!
//! A number type that is either zero or one.
use std::fmt;
use std::ops::{Add, Mul};

use num::{One, Zero};

/// A binary data type.
#[derive(Eq, PartialEq, Copy, Clone, Debug)]
#[allow(missing_docs)]
pub enum Binary {
    Zero,
    One,
}

impl Zero for Binary {
    fn zero() -> Self {
        Self::Zero
    }

    fn is_zero(&self) -> bool {
        self == &Binary::Zero
    }
}

impl One for Binary {
    fn one() -> Self {
        Self::One
    }
}

impl fmt::Display for Binary {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Binary::Zero => "0",
            Binary::One => "1",
        })
    }
}

impl Add<Binary> for Binary {
    type Output = Self;

    fn add(self, _rhs: Binary) -> Self::Output {
        panic!(
            "You should probably not be adding this type to itself, this implementation exists \
            only to satisfy the trait bound on the Zero trait."
        )
    }
}

impl Mul<Binary> for Binary {
    type Output = Self;

    fn mul(self, rhs: Binary) -> Self::Output {
        match (self, rhs) {
            (Binary::One, Binary::One) => Binary::One,
            _ => Binary::Zero,
        }
    }
}
