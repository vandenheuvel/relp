//! # Rational types of fixed size
use std::ops::Add;

use crate::algorithm::two_phase::tableau::kind::artificial::Cost;

/// Aliased type to ease a possible transition to own variant in the future.
pub type Rational32 = num::rational::Rational32;
/// Aliased type to ease a possible transition to own variant in the future.
pub type Rational64 = num::rational::Rational64;
/// Aliased type to ease a possible transition to own variant in the future.
pub type Rational128 = num::rational::Ratio<i128>;

impl Add<Cost> for Rational64 {
    type Output = Self;

    fn add(self, rhs: Cost) -> Self::Output {
        // TODO(PERFORMANCE): Ensure that this is fast
        match rhs {
            Cost::Zero => self,
            Cost::One => {
                Self::new(self.numer() + self.denom(), *self.denom())
            }
        }
    }
}
