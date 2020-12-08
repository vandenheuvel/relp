//! # An arbitrary precision rational type
//!
//! At the moment, this is just wrapping the `num::BigRational` type, following the newtype pattern.
//! This is needed because some of the impl's in this module are not provided by `num`. Methods on
//! this type can be modified and specialized as needed.
use std::ops::{Add, AddAssign, Mul};

use crate::data::number_types::rational::Rational64;
use std::str::FromStr;

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

impl Add<&Rational64> for &Big {
    type Output = Big;

    fn add(self, rhs: &Rational64) -> Self::Output {
        // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
        self.clone().add(rhs)
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
