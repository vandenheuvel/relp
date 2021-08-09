//! # Sparse vector
//!
//! Wrapping a `Vec<(usize, _)>`, fixed size.
use std::{fmt, mem};
use std::borrow::Borrow;
use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::iter::{FromIterator, Sum};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Deref, DivAssign, Mul, MulAssign, Neg};
use std::slice::{Iter, IterMut};
use std::vec::IntoIter;

use index_utils::{inner_product_slice_iter, remove_sparse_indices};
use num_traits::{One, Zero};
use relp_num::NonZero;

use crate::algorithm::two_phase::matrix_provider::column::{Column, ColumnNumber, SparseSliceIterator};
use crate::algorithm::two_phase::matrix_provider::column::ColumnIterator;
use crate::data::linear_algebra::SparseTuple;
use crate::data::linear_algebra::traits::{SparseComparator, SparseElement};
use crate::data::linear_algebra::vector::{DenseVector, Vector};

/// A sparse vector using a `Vec` with (row, value) combinations as back-end. Indices start at
/// `0`.
///
/// TODO(ENHANCEMENT): Consider making this backed by a `HashMap`.
#[derive(Eq, PartialEq, Clone, Debug)]
pub struct Sparse<F, C> {
    data: Vec<SparseTuple<F>>,
    len: usize,

    /// The level that comparison is done at: a single reference to the underlying data.
    phantom_comparison_type: PhantomData<C>,
}

impl<F, C> Sparse<F, C> {
    fn get_data_index(&self, i: usize) -> Result<usize, usize> {
        self.data.binary_search_by_key(&i, |&(index, _)| index)
    }

    fn set_zero(&mut self, i: usize) {
        if let Ok(index) = self.get_data_index(i) {
            self.data.remove(index);
        }
    }

    /// Increase the length of the vector by passing with zeros.
    pub fn extend(&mut self, extra_len: usize) {
        self.len += extra_len;
    }
}

impl<F, C> Sparse<F, C>
where
    F: ColumnNumber,
{
    pub fn iter(&self) -> SparseSliceIterator<F> {
        SparseSliceIterator::new(&self.data)
    }
}

impl<F, C> IntoIterator for Sparse<F, C> {
    type Item = SparseTuple<F>;
    type IntoIter = IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<F, C> Deref for Sparse<F, C> {
    type Target = [(usize, F)];

    fn deref(&self) -> &Self::Target {
        self.data.deref()
    }
}

impl<F, C> Vector<F> for Sparse<F, C>
where
    F: NonZero + SparseElement<C>,
    C: SparseComparator,
{
    type Inner = SparseTuple<F>;

    /// Create a vector of length `len` from `data`.
    ///
    /// Requires that values close to zero are already filtered.
    fn new(data: Vec<Self::Inner>, len: usize) -> Self {
        debug_assert!(data.iter().all(|&(i, _)| i < len));
        debug_assert!(data.is_sorted_by_key(|&(i, _)| i));
        debug_assert!(data.iter().all(|(_, v)| v.borrow().is_not_zero()));
        debug_assert_ne!(len, 0);
        debug_assert!(data.len() <= len);

        Self {
            data,
            len,

            phantom_comparison_type: PhantomData,
        }
    }

    fn sparse_inner_product<'a, 'b, I: ColumnIterator<'b>, O>(&'a self, column: I) -> O
    where
        &'a F: Mul<&'b I::F, Output=O>,
        O: Zero + AddAssign,
    {
        inner_product_slice_iter(&self.data, column)
    }

    /// Append a non-zero value.
    fn push_value(&mut self, value: F) {
        debug_assert!(value.borrow().is_not_zero());
        self.data.push((self.len, value));
        self.len += 1;
    }

    /// Set the value at index `i` to `value`.
    ///
    /// # Arguments
    ///
    /// * `i`: Index of the value. New tuple will be inserted, potentially causing many values to
    /// be shifted.
    /// * `value`: Value to be taken at index `i`. Should not be very close to zero to avoid
    /// memory usage and numerical error build-up.
    fn set(&mut self, i: usize, value: F) {
        debug_assert!(i < self.len);
        debug_assert!(value.borrow().is_not_zero());

        match self.get_data_index(i) {
            Ok(index) => self.data[index].1 = value,
            Err(index) => self.data.insert(index, (i, value)),
        }
    }

    fn get(&self, index: usize) -> Option<&F> {
        debug_assert!(index < self.len);

        self.get_data_index(index).ok().map(|i| &self.data[i].1)
    }

    /// Remove elements.
    ///
    /// # Arguments
    ///
    /// * `indices` is assumed sorted.
    fn remove_indices(&mut self, indices: &[usize]) {
        debug_assert!(indices.is_sorted());
        // All values are unique
        debug_assert!(indices.iter().collect::<HashSet<_>>().len() == indices.len());
        debug_assert!(indices.iter().all(|&i| i < self.len));
        debug_assert!(indices.len() < self.len);

        remove_sparse_indices(&mut self.data, indices);
        self.len -= indices.len();
    }

    fn iter(&self) -> Iter<Self::Inner> {
        self.data.iter()
    }

    fn iter_mut(&mut self) -> IterMut<Self::Inner> {
        self.data.iter_mut()
    }

    /// The length of this vector.
    fn len(&self) -> usize {
        self.len
    }

    /// Whether this vector has zero size.
    fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The size of this vector in memory.
    fn size(&self) -> usize {
        self.data.len()
    }
}

impl<F: NonZero + SparseElement<C>, C: SparseComparator> FromIterator<F> for Sparse<F, C> {
    fn from_iter<I: IntoIterator<Item=F>>(iter: I) -> Self {
        let mut data = Vec::new();
        let mut counter = 0;

        for item in iter.into_iter() {
            if item.is_not_zero() {
                data.push((counter, item));
            }
            counter += 1;
        }

        Self::new(data, counter)
    }
}

impl<F: 'static, C> Column for Sparse<F, C>
where
    F: ColumnNumber,
    C: SparseComparator,
{
    type F = F;
    type Iter<'a> = SparseSliceIterator<'a, F>;

    fn iter(&self) -> Self::Iter<'_> {
        SparseSliceIterator::new(&self.data)
    }

    fn index_to_string(&self, i: usize) -> String {
        match self.data.binary_search_by_key(&i, |&(ii, _)| ii) {
            Ok(index) => self.data[index].1.to_string(),
            Err(_) => "0".to_string(),
        }
    }
}

impl<F, C> Sparse<F, C>
where
    F: SparseElement<C>,
    C: SparseComparator,
{
    /// Create a `SparseVector` representation of standard basis unit vector e_i.
    ///
    /// # Arguments
    ///
    /// * `i`: Only index where there should be a 1. Note that indexing starts at zero, and runs
    /// until (not through) `len`.
    /// * `len`: Size of the `SparseVector`.
    #[must_use]
    pub fn standard_basis_vector(i: usize, len: usize) -> Self
    where
        F: One + NonZero + Clone,
    {
        debug_assert!(i < len);

        Self::new(vec![(i, F::one())], len)
    }

    /// Add the multiple of another row to this row.
    ///
    /// # Arguments
    ///
    /// * `multiple`: Non-zero constant that all elements of the `other` vector are multiplied with.
    /// * `other`: Vector to add a multiple of to this vector.
    ///
    /// # Return value
    ///
    /// A new `SparseVector`.
    ///
    /// # Note
    ///
    /// The implementation of this method doesn't look pretty, but it seems to be reasonably fast.
    /// If this method is too slow, it might be wise to consider the switching of the `SparseVector`
    /// storage backend from a `Vec` to a `HashMap`.
    pub fn add_multiple_of_row<'a, G>(&mut self, multiple: G, other: &'a Sparse<F, C>)
    where
        F: Add<F, Output=F> + NonZero,
        G: Copy,
        &'a F: Mul<G, Output=F>,
    {
        debug_assert_eq!(other.len(), self.len());

        let mut new_tuples = Vec::new();

        let mut j = 0;  // data index
        let old_data = mem::replace(&mut self.data, Vec::with_capacity(0));
        for (i, value) in old_data {
            while j < other.data.len() && other.data[j].0 < i {
                new_tuples.push((other.data[j].0, &other.data[j].1 * multiple));
                j += 1;
            }

            if j < other.data.len() && i == other.data[j].0 {
                let new_value = value + &other.data[j].1 * multiple;
                if new_value.is_not_zero() {
                    new_tuples.push((i, new_value.into()));
                }
                j += 1;
            } else {
                new_tuples.push((i, value));
            }
        }
        for (j, value) in &other.data[j..] {
            new_tuples.push((*j, value * multiple));
        }

        self.data = new_tuples;
    }

    /// Set the value at index `i` to `value`.
    ///
    /// # Arguments
    ///
    /// * `i`: Index of the value. New tuple will be inserted, potentially causing many values to
    /// be shifted.
    /// * `value`: Value to be taken at index `i`.
    pub fn shift_value<G>(&mut self, i: usize, value: G)
    where
        F: PartialEq<G> + AddAssign<G> + From<G>,
        for<'r> &'r G: Neg<Output=G>,
        G: NonZero,
    {
        debug_assert!(i < self.len);

        if value.is_not_zero() {
            match self.get_data_index(i) {
                Ok(index) => {
                    if self.data[index].1 == -&value {
                        self.set_zero(i);
                    } else {
                        self.data[index].1 += value;
                    }
                },
                Err(index) => self.data.insert(index, (i, From::from(value))),
            }
        }
    }

    /// Multiply each element of the vector by a value.
    pub fn element_wise_multiply(&mut self, value: &F)
    where
        for<'r> F: NonZero + MulAssign<&'r F>,
    {
        debug_assert!(value.is_not_zero());

        for (_, v) in &mut self.data {
            *v *= value;
        }
    }

    /// Divide each element of the vector by a value.
    pub fn element_wise_divide(&mut self, value: &F)
    where
       for<'r> F: NonZero + DivAssign<&'r F>,
    {
        debug_assert!(value.is_not_zero());

        for (_, v) in &mut self.data {
            *v /= value;
        }
    }
}

impl<F, C> Sparse<F, C>
where
    F: SparseElement<C> + NonZero,
    C: SparseComparator,
{
    /// Calculate the inner product between two vectors.
    ///
    /// # Arguments
    ///
    /// * `other`: Vector to calculate inner product with.
    ///
    /// # Return value
    ///
    /// The inner product.
    #[must_use]
    pub fn inner_product_with_dense<'a, F2, O: 'a>(&'a self, other: &'a DenseVector<F2>) -> O
    where
        F: Borrow<O>,
        F2: Borrow<O> + PartialEq + Display + Debug,
        O: Sum,
        &'a O: Mul<&'a O, Output=O>,
    {
        debug_assert_eq!(other.len(), self.len());

        self.data.iter().map(|(i, value)| other[*i].borrow() * value.borrow()).sum()
    }
}

impl<F, C> Sparse<F, C>
where
    for<'r> &'r F: Mul<&'r F, Output=F>,
    F: Sum,
{
    pub fn squared_norm(&self) -> F {
        self.data.iter()
            .map(|(_, value)| value * value)
            .sum()
    }
}

impl<F: SparseElement<C>, C: SparseComparator> Display for Sparse<F, C> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        for (index, value) in &self.data {
            write!(f, "({} {})", index, value)?;
            if *index < self.data.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod test {
    use crate::algorithm::two_phase::matrix_provider::column::SparseSliceIterator;
    use crate::data::linear_algebra::vector::{SparseVector, Vector};

    #[test]
    fn test_inner_product_iter() {
        assert_eq!(
            SparseVector::new(vec![(0, 1)], 2).sparse_inner_product(std::iter::empty::<(_, &i32)>()),
            0,
        );

        assert_eq!(
            SparseVector::new(vec![(0, 1)], 2).sparse_inner_product(SparseSliceIterator::new(&[(0, 1)])),
            1,
        );

        assert_eq!(
            SparseVector::new(vec![(0, 1)], 2).sparse_inner_product(SparseSliceIterator::new(&[(2, 1)])),
            0,
        );

        assert_eq!(
            SparseVector::new(vec![(0, 1), (3, 1), (12, 1), (13, 1)], 15)
                .sparse_inner_product(
                    SparseSliceIterator::new(&[(0, -1), (1, 1), (2, 1), (12, 1)]),
                ),
            0,
        );
    }
}
