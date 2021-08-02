//! # Columns
//!
//! Items yielded by a matrix provider. Used to iterate over when computing what they look like
//! w.r.t. the latest basis.
use std::fmt::{Debug, Display};

use relp_num::NonZero;

use crate::data::linear_algebra::SparseTuple;

pub mod identity;

/// Columns represent part of a (virtual) data matrix.
///
/// This column is sparse.
///
/// A column might be expensive to compute. It can store computed values and once GATs work
/// (better), also references to items stored in the matrix provider that yields instances of this
/// trait.
///
/// It can't necessarily be iterated over directly. That needs to happen many times, for example
/// when computing many inner products with a data matrix. The trait has an associated type to be
/// used for iteration, that should be cheaply cloneable and probably not store any values itself.
/// Rather, it should describe how this column should be iterated over.
///
/// TODO(ARCHITECTURE): Many basis inverse maintenance algorithms require reallocation of the
///  column. Is this more complex set-up worth it?
// TODO(ARCHITECTURE): Once GATs work, consider giving this trait a lifetime parameter.
pub trait Column: Clone {
    /// Input data type.
    ///
    /// Items of this type get read and used in additions and multiplications often.
    // TODO(ENHANCEMENT): Don't work with a field type directly, but an `Into<F>` type to separate.
    type F: 'static + ColumnNumber;

    /// Type of struct to iterate over this column.
    ///
    /// It should be somewhat cheaply cloneable and as such not be too large.
    type Iter<'a>: Iterator<Item=SparseTuple<&'a Self::F>> + Clone;

    /// Derive the iterator object.
    ///
    /// Because this column might need to be iterated over many times, it doesn't consume the
    /// column but instead produces a struct that might keep references to this column.
    fn iter(&self) -> Self::Iter<'_>;

    /// Format an index of the column.
    ///
    /// Note that this index might not be explicitly stored due to the column being sparse.
    fn index_to_string(&self, i: usize) -> String;
}

/// Basic operations that should be possible with the type of the values of the column.
///
/// This trait was introduced to avoid some verbose trait bounds throughout the codebase.
pub trait ColumnNumber =
    NonZero +

    Eq +
    PartialEq +

    Display +
    Clone +
    Debug +
;

/// Column that can be iterated over in-order.
///
/// This trait is simply a marker trait to be used in specialization.
///
/// TODO(ENHANCEMENT): At the time of writing, it is not possible to specialize the generic
///  arguments of trait methods. That is why this trait and the standard `Column` trait are
///  currently both needed.
///
// TODO(ARCHITECTURE): Once GATs work, consider giving this trait a lifetime parameter.
pub trait OrderedColumn: Column {
}

/// Wrapping a sparse vector of tuples.
#[derive(Clone)]
pub struct SparseColumn<F> {
    inner: Vec<SparseTuple<F>>
}

impl<F> SparseColumn<F> {
    pub fn new(data: Vec<SparseTuple<F>>) -> Self {
        Self {
            inner: data,
        }
    }
}

impl<F: 'static + ColumnNumber> Column for SparseColumn<F> {
    type F = F;
    type Iter<'a> = impl Iterator<Item=SparseTuple<&'a Self::F>> + Clone;

    fn iter(&self) -> Self::Iter<'_> {
        SparseSliceIterator::new(&self.inner)
    }

    fn index_to_string(&self, i: usize) -> String {
        match self.inner.iter().find(|&&(ii, _)| ii == i) {
            None => "0".to_string(),
            Some((_, v)) => v.to_string(),
        }
    }
}

#[derive(Clone)]
pub struct SparseSliceIterator<'a, F> {
    creator: &'a [SparseTuple<F>],
    data_index: usize,
}

impl<'a, F: ColumnNumber> SparseSliceIterator<'a, F> {
    pub fn new(slice: &'a [SparseTuple<F>]) -> Self {
        debug_assert!(slice.iter().all(|(_, v)| v.is_not_zero()));

        Self {
            creator: slice,
            data_index: 0,
        }
    }
}

impl<'a, F> Iterator for SparseSliceIterator<'a, F> {
    type Item = SparseTuple<&'a F>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.data_index < self.creator.len() {
            let (index, value) = &self.creator[self.data_index];
            self.data_index += 1;

            Some((*index, value))
        } else {
            None
        }
    }
}

impl<F: 'static + ColumnNumber> OrderedColumn for SparseColumn<F> {
}

#[derive(Clone)]
struct DenseColumn<F> {
    inner: Vec<F>,
}

impl<F: ColumnNumber> DenseColumn<F> {
    pub fn new(data: Vec<F>) -> Self {
        debug_assert!(data.iter().any(|v| v.is_not_zero()));

        Self {
            inner: data,
        }
    }
}

impl<F: 'static + ColumnNumber> Column for DenseColumn<F> {
    type F = F;
    type Iter<'a> = impl Iterator<Item=SparseTuple<&'a Self::F>> + Clone;

    fn iter(&self) -> Self::Iter<'_> {
        DenseSliceIterator::new(&self.inner)
    }

    fn index_to_string(&self, i: usize) -> String {
        debug_assert!(i < self.inner.len());

        self.inner[i].to_string()
    }
}

#[derive(Clone)]
pub struct DenseSliceIterator<'a, F> {
    creator: &'a [F],
    data_index: usize,
}

impl<'a, F> DenseSliceIterator<'a, F> {
    pub fn new(slice: &'a [F]) -> Self {
        Self {
            creator: slice,
            data_index: 0,
        }
    }
}

impl<'a, F: ColumnNumber> Iterator for DenseSliceIterator<'a, F> {
    type Item = SparseTuple<&'a F>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.data_index < self.creator.len() {
            let value = &self.creator[self.data_index];
            if value.is_not_zero() {
                let index = self.data_index;
                self.data_index += 1;
                return Some((index, value));
            }
            self.data_index += 1;
        }

        None
    }
}

impl<F: 'static + ColumnNumber> OrderedColumn for DenseColumn<F> {
}

#[cfg(test)]
mod test {
    use crate::algorithm::two_phase::matrix_provider::column::{Column, DenseColumn};

    #[test]
    fn test_dense() {
        let data = vec![1, 2, 3, 0, 5];
        let column = DenseColumn::new(data);
        let mut column_iter = column.iter();
        assert_eq!(column_iter.next(), Some((0, &1)));
        assert_eq!(column_iter.next(), Some((1, &2)));
        assert_eq!(column_iter.next(), Some((2, &3)));
        assert_eq!(column_iter.next(), Some((4, &5)));
        assert_eq!(column_iter.next(), None);
    }
}
