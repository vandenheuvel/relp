//! # Traits
//!
//! A hierarchy of number types is defined. The hierarchy is "mathematically exact", but the
//! implementations aren't. That is, the contracts that these traits define, or their names imply,
//! may not be kept precisely. This is due to finite representation of these numbers and is a
//! fundamental problem that cannot be avoided, but perhaps be dealt with differently.
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num::{One, Zero};

/// Logic for integer linear programs.
///
/// Is a specific kind of feasibility, more generally one can think of arbitrary logic using
/// "closest feasible to the right", "closest feasible to the left", etc.
///
/// Currently not used.
pub trait IntegersEmbedded:
    OrderedField // Concepts like higher and lower require ordering
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
}

/// A reference to an ordered field.
pub trait OrderedFieldRef<Deref>: Ord + FieldRef<Deref> {}

/// Basic field operations with Self and with references to Self.
pub trait Field:
    PartialEq  // Equivalence relation
        + Eq
        + PartialOrd
        + Zero  // Additive identity
        + Neg<Output = Self>  // Additive inverse
        + One  // Multiplicative identity
        // First operation
        + Add<Self, Output = Self>
        + for<'r> Add<&'r Self, Output = Self>
        + AddAssign<Self>
        + for<'r> AddAssign<&'r Self>
        + Sum
        // First operation inverse
        + Sub<Self, Output = Self>
        + for<'r> Sub<&'r Self, Output = Self>
        + SubAssign<Self>
        + for<'r> SubAssign<&'r Self>
        // Second operation
        + Mul<Self, Output = Self>
        + for<'r> Mul<&'r Self, Output = Self>
        + MulAssign<Self>
        + for<'r> MulAssign<&'r Self>
        // Second operation inverse
        + Div<Self, Output = Self>
        + for<'r> Div<&'r Self, Output = Self>
        + DivAssign<Self>
        + for<'r> DivAssign<&'r Self>
        // TODO: MulAdd should be possible. Only in specialization?
        //  + MulAdd

        // Practicalities
        + Clone
        + Display
        + ToString
        + Debug
{
    /// Value such that for all elements of the fields, e + e_0 = e.
    fn additive_identity() -> Self {
        Self::zero()
    }
    /// Value such that for all elements of the fields, e * e_1 = e.
    fn multiplicative_identity() -> Self {
        Self::one()
    }
}

/// A reference to a variable that is in a `Field`.
///
/// TODO: Can less HRTB be used? Can the be written down less often? Can this trait be integrated
///  with the `Field` trait?
pub trait FieldRef<Deref>:
    // Equivalence relation
    PartialEq<Self>
        + Neg<Output = Deref>  // Additive inverse
        // First operation
        + Add<Deref, Output = Deref>
        + for<'s> Add<&'s Deref, Output = Deref>
        // First operation inverse
        + Sub<Deref, Output = Deref>
        + for<'s> Sub<&'s Deref, Output = Deref>
        // Second operation
        + Mul<Deref, Output = Deref>
        + for<'s> Mul<&'s Deref, Output = Deref>
        // Second operation inverse
        + Div<Deref, Output = Deref>
        + for<'s> Div<&'s Deref, Output = Deref>
        // TODO: MulAdd should be possible. Only in specialization?
        //  + MulAdd

        // Practicalities
        + Copy
        + Clone
        + Display
        + Debug
        // Necessary for the Add, Sub, Mul and Div traits. References are sized anyways.
        + Sized
{
}

/// Helper macro for tests.
#[macro_export]
macro_rules! F {
    ($value:expr) => {
        {
            F::from_f64($value as f64).unwrap()
        }
    };
}

#[cfg(test)]
mod test {
    use num::FromPrimitive;
    use num::rational::Ratio;

    use crate::data::number_types::traits::OrderedField;

    /// Testing the `RealField::abs` method.
    #[test]
    fn real_field_abs() {
        fn test<F: OrderedField + FromPrimitive> () {
            let v = F::one();
            let w = -v.clone();
            assert_eq!(w.abs(), v);

            let v = F::one();
            assert_eq!(v.clone().abs() ,v);

            let v = F::zero();
            assert_eq!(v.clone().abs(), v)
        }

        test::<Ratio<i32>>();
        test::<Ratio<i64>>();
    }
}
