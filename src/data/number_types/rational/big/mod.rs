//! # An arbitrary precision rational type
//!
//! At the moment, this is just wrapping the `num::BigRational` type, following the newtype pattern.
//! This is needed because some of the impl's in this module are not provided by `num`. Methods on
//! this type can be modified and specialized as needed.
use std::ops::{Add, AddAssign, Mul};
use std::str::FromStr;

use num::Zero;

use crate::algorithm::two_phase::tableau::kind::artificial::Cost;
use crate::data::number_types::rational::Rational64;

mod wrapping;
#[cfg(test)]
mod test;

/// A big rational type that currently completely relies on the methods of `num::BigRational`.
#[derive(
    Clone,
    Ord, PartialOrd, Eq, PartialEq,
    Debug,
)]
pub struct Big(num::BigRational);

impl Big {
    /// Wrap an inner `num::BigRational`.
    fn wrap(inner: num::BigRational) -> Self {
        Self(inner)
    }
    /// Create a new instance by converting the two provided numbers into arbitrary size ints.
    pub fn new(numer: i64, denom: i64) -> Self {
        Self(num::BigRational::new(numer.into(), denom.into()))
    }
}

impl From<&str> for Big {
    fn from(input: &str) -> Self {
        Self(num::BigRational::from_str(input).unwrap())
    }
}

impl From<Rational64> for Big {
    fn from(value: Rational64) -> Self {
        let (numer, denom) = value.into();
        Self((numer.into(), denom.into()).into())
    }
}

impl PartialEq<Rational64> for Big {
    fn eq(&self, other: &Rational64) -> bool {
        // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
        let (numer, denom) = (*other.numer(), *other.denom());
        let uptyped = num::BigRational::new(numer.into(), denom.into());
        self.0.eq(&uptyped)
    }
}

impl Add<&Rational64> for Big {
    type Output = Big;

    fn add(mut self, rhs: &Rational64) -> Self::Output {
        // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
        let (numer, denom) = (*rhs.numer(), *rhs.denom());
        let uptyped = num::BigRational::new(numer.into(), denom.into());
        self.0 += uptyped;
        self
    }
}

impl Add<Option<&Rational64>> for Big {
    type Output = Big;

    fn add(self, rhs: Option<&Rational64>) -> Self::Output {
        // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
        match rhs {
            None => self,
            Some(rhs) => Add::add(self, rhs),
        }
    }
}

impl Add<&Rational64> for &Big {
    type Output = Big;

    fn add(self, rhs: &Rational64) -> Self::Output {
        // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
        self.clone().add(rhs)
    }
}

impl Add<Cost> for Big {
    type Output = Big;

    fn add(self, rhs: Cost) -> Self::Output {
        match rhs {
            Cost::Zero => self,
            Cost::One => {
                let (numer, denom): (num::BigInt, num::BigInt) = self.0.into();
                Self(num::BigRational::new(numer + &denom, denom))
            },
        }
    }
}

impl Add<Option<&Big>> for Big {
    type Output = Big;

    fn add(self, rhs: Option<&Big>) -> Self::Output {
        // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
        match rhs {
            None => self,
            Some(rhs) => Add::add(self, rhs),
        }
    }
}

impl Add<Option<&Rational64>> for &Big {
    type Output = Big;

    fn add(self, rhs: Option<&Rational64>) -> Self::Output {
        // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
        let copy = self.clone();
        match rhs {
            None => copy,
            Some(rhs) => Add::add(copy, rhs),
        }
    }
}

impl Add<Option<&Big>> for &Big {
    type Output = Big;

    fn add(self, rhs: Option<&Big>) -> Self::Output {
        // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
        let copy = self.clone();
        match rhs {
            None => copy,
            Some(rhs) => Add::add(copy, rhs),
        }
    }
}

impl AddAssign<Rational64> for Big {
    fn add_assign(&mut self, rhs: Rational64) {
        // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
        let (numer, denom): (i64, i64) = rhs.into();
        let uptyped = num::BigRational::new(numer.into(), denom.into());
        self.0 += uptyped;
    }
}

impl AddAssign<&Rational64> for Big {
    fn add_assign(&mut self, rhs: &Rational64) {
        // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
        let (numer, denom) = (*rhs.numer(), *rhs.denom());
        let uptyped = num::BigRational::new(numer.into(), denom.into());
        self.0 += uptyped;
    }
}

impl Mul<&Rational64> for Big {
    type Output = Big;

    fn mul(self, rhs: &Rational64) -> Self::Output {
        // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
        let (numer, denom) = (rhs.numer(), rhs.denom());
        let uptyped = num::BigRational::new((*numer).into(), (*denom).into());
        Self(self.0 * uptyped)
    }
}

impl Mul<&Rational64> for &Big {
    type Output = Big;

    fn mul(self, rhs: &Rational64) -> Self::Output {
        // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
        self.clone() * rhs
    }
}

impl Mul<Cost> for Big {
    type Output = Big;

    fn mul(self, rhs: Cost) -> Self::Output {
        match rhs {
            Cost::Zero => Self::Output::zero(),
            Cost::One => self,
        }
    }
}

impl Mul<Cost> for &Big {
    type Output = Big;

    fn mul(self, rhs: Cost) -> Self::Output {
        match rhs {
            Cost::Zero => Self::Output::zero(),
            Cost::One => self.clone(),
        }
    }
}

impl Mul<Option<&Rational64>> for &Big {
    type Output = Big;

    fn mul(self, rhs: Option<&Rational64>) -> Self::Output {
        // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
        match rhs {
            None => Big::zero(),
            Some(rhs) => Mul::mul(self, rhs),
        }
    }
}

impl Mul<Option<&Big>> for &Big {
    type Output = Big;

    fn mul(self, rhs: Option<&Big>) -> Self::Output {
        match rhs {
            None => Big::zero(),
            Some(rhs) => Mul::mul(self, rhs)
        }
    }
}
