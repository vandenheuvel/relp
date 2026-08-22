//! # Number operations
//!
//! During the interaction with each of a `MatrixProvider`'s associated types it should be possible
//! to do certain operations.
//!
//! A matrix provider stores its values in a narrow type: the incidence matrix of a network holds
//! nothing but `1`, `-1` and `0`. Those values act on the wide accumulator maintained here through
//! [`Absorb`](relp_num::Absorb), a bound that is stated on this type rather than on the narrow one.
use std::ops::{Div, Mul, Neg};

use crate::data::linear_algebra::traits::SparseElement;

/// Operations done by the number type within the inverse maintenance algorithm.
pub trait Field = relp_num::OrderedField
    + SparseElement<Self>
    // The accumulator acts on itself: values already in this type are "narrow" too.
    + relp_num::Absorb<Self>
    // The algorithm builds identity columns of its own, whatever the provider stores, so being
    // able to read a `One` is a property of the accumulator rather than of any one caller.
    + relp_num::Absorb<relp_num::fixed::One>
    + for<'r> From<&'r Self>;

// TODO(ARCHITECTURE): Once HRTB are propagated like normal associated type trait bounds, remove
//  this trait by integrating the requirements into `InverseMaintenance::F`'s trait bounds.
/// Reference operations done by the number type within the inverse maintenance algorithm.
#[allow(clippy::type_repetition_in_bounds)]
pub trait FieldHR = where
    for<'r> &'r Self: Neg<Output = Self>,
    for<'r> &'r Self: Mul<&'r Self, Output = Self>,
    for<'r> &'r Self: Div<&'r Self, Output = Self>;
