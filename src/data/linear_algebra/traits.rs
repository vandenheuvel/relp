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
//! not stored behind a reference (like Ratio<BigInt>).
//! * The third is the type that both can be dereferenced to, and that can do equality equality 
//! comparisons
//! 
//! Practically, the third is pretty much redundant, because there is no implementation in the
//! `num::rational` crate to deref a small Copy type to any larger type. 
//! TODO: Remove the third type because of the above?
use std::borrow::Borrow;
use std::fmt::{Debug, Display};

use num::Zero;

/// Element of a `Vector` of `Matrix` type.
///
/// This is an alias for a traits that are needed to derive a few practical traits for the
/// aforementioned types.
pub trait Element:
    PartialEq
        + Clone
        + Display
        + Debug
{}
impl<T:
    PartialEq
        + Clone
        + Display
        + Debug
> Element for T {}

/// Element of a sparse data structure.
///
/// Needs to be borrowable as a type that can be used for comparison with the zero type, as well as
/// reference types.
pub trait SparseElement<Comparator>:
    Borrow<Comparator>
        + Element
{}
impl<Comparator, T:
    Borrow<Comparator>
        + Element
> SparseElement<Comparator> for T {}

/// Zero element of a sparse data structure.
///
/// A separate type from the comparison element, because it needs to be `Copy` such that is not
/// (unnecessarily) behind a reference.
///
/// TODO: Does this even matter? Zero values should never really be constructed anyways, right?
///  Could this allow the elimination of all these traits?
pub trait SparseElementZero<Comparator>:
    Zero
        + Borrow<Comparator>
        // + Copy  // TODO: It should be a small type and not one that's behind a reference
        + Element
{}
impl<Comparator, T:
    Zero
        + Borrow<Comparator>
        // + Copy
        + Element
> SparseElementZero<Comparator> for T {}

/// Element to do comparisons between vectors of different "reference levels".
///
/// We might have an inner product between a `SparseVector<F>` and `SparseVector<&F>`. Then this
/// comparator type would be `F`, such that the values can be compared as `&F`'s.
pub trait SparseComparator:
    PartialEq
        + Element
{}
impl<T:
    PartialEq
        + Element
> SparseComparator for T {}
