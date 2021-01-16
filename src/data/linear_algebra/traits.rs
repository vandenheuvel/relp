//! # Traits for elements in sparse datastructures
//! 
//! Once a sparse data structure contains references to values, it is not obvious what value should
//! be returned for a zero value, that is not stored. It is also not clear, how one should compare
//! the elements contained in the sparse structure with the zero element (mostly for debug 
//! purposes).
//! 
//! One idea, implemented here, is to have three types related to a sparse data structure:
//! 
//! * The first is the type that is stored many times in the data structure
//! * The second is a type that can be zero (and is not a reference), ideally is small (Copy) and 
//! not stored behind a reference (like `RationalBig`).
//! * The third is the type that both can be dereferenced to. This is used to create a row-major
//! copy of the constraint matrix using references, rather than the actual values.
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::ops::Neg;

use num::Zero;

/// Element of a `Vector` of `Matrix` type.
///
/// This is an alias for a traits that are needed to derive a few practical traits for the
/// aforementioned types.
pub trait Element =
    PartialEq +
    Clone +
    Display +
    Debug +
;

/// Element of a sparse data structure.
///
/// Needs to be borrowable as a type that can be used for comparison with the zero type, as well as
/// reference types.
pub trait SparseElement<Comparator> =
    Borrow<Comparator> +
    Element +
;

/// Element to do comparisons between vectors of different "reference levels".
///
/// We might have an inner product between a `SparseVector<F>` and `SparseVector<&F>`. Then this
/// comparator type would be `F`, such that the values can be compared as `&F`'s.
pub trait SparseComparator =
    PartialEq +
    Element +
    NotZero +
;

/// Implementors can be nonzero.
///
/// This trait is used for debug asserts. Values in sparse data structures should never be zero, and
/// requiring that they implement `num::Zero` prohibits writing number types that can't represent
/// the value 0.
pub trait NotZero {
    /// Whether the value is not equal to zero.
    fn is_not_zero(&self) -> bool;
}

impl<T: Zero> NotZero for T {
    fn is_not_zero(&self) -> bool {
        !self.is_zero()
    }
}

/// A signed number that can have a nonzero value.
pub trait NotZeroSigned: NotZero + Neg<Output=Self> + Clone {
    /// Absolute value of x, |x|.
    fn abs(&self) -> Self {
        let cloned = self.clone();
        match self.signum() {
            Sign::Positive => cloned,
            Sign::Negative => -cloned,
        }
    }
    /// Whether the value is positive or negative.
    fn signum(&self) -> Sign;
    /// Whether `x > 0`.
    fn is_positive(&self) -> bool {
        self.signum() == Sign::Positive
    }
    /// Whether `x < 0`.
    fn is_negative(&self) -> bool {
        self.signum() == Sign::Negative
    }
}
impl<T: Zero + NotZero + Neg<Output=Self> + PartialOrd<Self> + Clone> NotZeroSigned for T {
    default fn signum(&self) -> Sign {
        debug_assert!(self.is_not_zero());

        match self.partial_cmp(&Self::zero()) {
            Some(Ordering::Less) => Sign::Negative,
            Some(Ordering::Greater) => Sign::Positive,
            Some(Ordering::Equal) | None => unreachable!("\
                Should only be used on nonzero values, and those should always be comparable with \
                the zero value of the type.\
            "),
        }
    }
}

/// Sign of a nonzero value.
///
/// Existing `Sign` traits, such in `num`, typically have a third value for the sign of 0. Working
/// with that trait creates many branches or match cases that should never be possible.
#[derive(Eq, PartialEq, Copy, Clone)]
pub enum Sign {
    /// `x > 0`
    Positive,
    /// `x < 0`
    Negative,
}

impl PartialOrd for Sign {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Sign::Positive, Sign::Positive) => None,
            (Sign::Positive, Sign::Negative) => Some(Ordering::Greater),
            (Sign::Negative, Sign::Positive) => Some(Ordering::Less),
            (Sign::Negative, Sign::Negative) => None,
        }
    }
}
