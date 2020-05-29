//! # Traits
//!
//! A hierarchy of number types is defined. The hierarchy is "mathematically exact", but the
//! implementations aren't. That is, the contracts that these traits define, or their names imply,
//! may not be kept precisely. This is due to finite representation of these numbers and is a
//! fundamental problem that cannot be avoided, but perhaps be dealt with differently.
use std::fmt::{Debug, Display};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num::{FromPrimitive, One, Zero, Num};
use std::cmp::Ordering;

/// Trait representing the unique field of real numbers.
///
/// While the real numbers are unique, the finite representations that imitate them aren't. That's
/// why the real numbers are represented in a trait, rather than a type.
pub trait RealField:
// Fundamental types
        OrderedField + DedekindComplete
// Convenience types
        + Zero + One
        + FromPrimitive
{
    /// Largest integer smaller or equal to this number.
    fn floor(self) -> Self;
    /// Smallest integer larger or equal to this number.
    fn ceil(self) -> Self;
    /// "Closest" integer to this number.
    ///
    /// Exact behavior depends on the implementor.
    fn round(self) -> Self;
}

/// The simplex algorithm is defined over the ordered fields. All methods containing algorithmic
/// logic should be defined to work an ordered field (or a field, if they don't need the ordering).
/// All methods representing a matrix should be defined over a field, because they don't need the
/// additional ordering.
pub trait OrderedField: Ord + Field {
    /// The absolute value of a number.
    ///
    /// Compute the additive inverse if the number is smaller than the additive identity.
    fn abs(self) -> Self {
        if self < Self::additive_identity() {
            -self
        } else {
            self
        }
    }

    /// Whether this number is positive, negative or zero. Represented by 1, -1 and 0 respectively.
    fn signum(self) -> Self {
        match self.cmp(&Self::additive_identity()) {
            Ordering::Less => -Self::multiplicative_identity(),
            Ordering::Equal => Self::additive_identity(),
            Ordering::Greater => Self::multiplicative_identity(),
        }
    }
}

/// An ordered field that is Dedekind complete is the field of real numbers. It only exists to make
/// explicit where we "cheat" the trait hierarchy.
pub trait DedekindComplete {}

pub trait Field:
// The core properties
    Num
// TODO: Remove redundant trait requirements
        + Add<Output = Self> + AddAssign + Sum
        + Sub<Output = Self> + SubAssign
        + Neg<Output = Self>
        + Mul<Output = Self> + MulAssign + Product
// TODO: Conceptually, muladd should be possible
//         + MulAdd
// TODO: Can code be rewritten such that Div is not needed?
        + Div<Output = Self> + DivAssign
// The data properties
// TODO: Would passing by reference be faster? How about unsized big decimals?
        + Copy
        + Sized
// Convenience
        + Display
        + Debug
{
    /// Value such that for all elements of the fields, e + e_0 = e.
    fn additive_identity() -> Self;
    /// Value such that for all elements of the fields, e * e_1 = e.
    fn multiplicative_identity() -> Self;
}

/// Shorthand for creating a `RealField` number in tests.
#[macro_export]
macro_rules! RF {
    ($value:expr) => {
        RF::from_f64($value as f64).unwrap()
    };
}

#[cfg(test)]
mod test {
    use num::rational::Ratio;

    use crate::data::number_types::float::FiniteErrorControlledFloat;
    use crate::data::number_types::traits::RealField;

    /// Testing the `RealField::abs` method.
    #[test]
    fn test_real_field_abs() {
        fn test<F: RealField> () {
            let v = F::one();
            let w = -v;
            assert_eq!(w.abs(), v);

            let v = F::one();
            assert_eq!(v.abs() ,v);

            let v = F::zero();
            assert_eq!(v.abs(), v)
        }

        test::<Ratio<i32>>();
        test::<Ratio<i64>>();
        test::<FiniteErrorControlledFloat<f32, u32>>();
        test::<FiniteErrorControlledFloat<f64, u64>>();
    }
}
