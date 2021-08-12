//! # Identity column
//!
//! The trait is useful for artificial tableaus: if the first phase is to be used, one needs to
//! represent the artificial columns, which are identity columns, somehow.
//!
//! Mostly useful for debugging, this module also contains a trivial identity column implementation
//! using a custom type for the constant `1`. As computing an identity column w.r.t. the current
//! basis is equivalent to just inverting the basis explicitly.
use std::iter;
use std::iter::Once;

use relp_num::One;

use crate::algorithm::two_phase::matrix_provider::column::Column;
use crate::algorithm::two_phase::matrix_provider::column::ColumnIterator;
use crate::data::linear_algebra::SparseTuple;

/// Identity columns are needed for artificial matrices.
///
/// When a matrix provider is to be used in the first phase, it should be possible to represent
/// identity columns.
pub trait Identity: Column {
    /// Create an identity column, placing a "1" at a certain index and "0"'s otherwise.
    ///
    /// # Arguments
    ///
    /// * `i`: Index at which the "1" should be placed.
    /// * `len`: Length of the column. Might not be used in an actual implementation.
    fn identity(i: usize, len: usize) -> Self;
}

const ONE: One = One;

/// A simple identity column useful for debugging, mostly.
#[derive(Debug, Copy, Clone)]
pub struct IdentityColumn {
    index: usize,
}

impl IdentityColumn {
    pub fn new(index: usize) -> Self {
        Self {
            index,
        }
    }
}

impl Column for IdentityColumn {
    type F = One;
    type Iter<'a> = impl ColumnIterator<'a, F=Self::F>;

    fn iter(&self) -> Self::Iter<'_> {
        iter::once((self.index, &ONE))
    }

    fn index_to_string(&self, i: usize) -> String {
        if i == self.index {
            "1"
        } else {
            "0"
        }.to_string()
    }
}

impl IntoIterator for IdentityColumn {
    type Item = SparseTuple<One>;
    type IntoIter = Once<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        iter::once((self.index, One))
    }
}
