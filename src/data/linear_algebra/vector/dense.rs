//! # Dense vector
//!
//! Wrapping a `Vec` such that it has a fixed size and can interact with sparse vectors.
use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::fmt;
use std::iter::FromIterator;
use std::ops::{AddAssign, Deref, Index, IndexMut, Mul};
use std::slice::{Iter, IterMut};
use std::vec::IntoIter;

use index_utils::remove_indices;
use num_traits::Zero;

use crate::algorithm::two_phase::matrix_provider::column::ColumnIterator;
use crate::data::linear_algebra::vector::Vector;

/// Uses a `Vec` as underlying data a structure. Length is fixed at creation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Dense<F> {
    #[allow(missing_docs)]
    data: Vec<F>,
}

impl<F> Dense<F> {
    /// Create a vector with all values being equal to a given value.
    ///
    /// # Arguments
    ///
    /// * `value`: The value which all elements of this vector are equal to.
    /// * `len`: Length of the vector, number of elements.
    ///
    /// # Return value
    ///
    /// A constant `DenseVector`.
    pub fn constant(value: F, len: usize) -> Self
    where
        F: Clone,
    {
        debug_assert_ne!(len, 0);

        Self { data: vec![value; len], }
    }

    /// Append multiple values to this vector.
    ///
    /// # Arguments
    ///
    /// * `new_values`: An ordered collections of values to append.
    pub fn extend_with_values(&mut self, new_values: Vec<F>) {
        self.data.extend(new_values);
    }

    /// Slice view of the inner data.
    pub fn inner(&self) -> &[F] {
        &self.data
    }

    /// Slice view of the inner data.
    pub fn inner_mut(&mut self) -> &mut [F] {
        &mut self.data
    }
}

impl<F: PartialEq + Display + Debug> Index<usize> for Dense<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < self.len());

        &self.data[index]
    }
}

impl<F: PartialEq + Display + Debug> IndexMut<usize> for Dense<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < self.len());

        &mut self.data[index]
    }
}

impl<F> Deref for Dense<F> {
    type Target = [F];

    fn deref(&self) -> &Self::Target {
        self.data.deref()
    }
}

impl<F: PartialEq + Display + Debug> Vector<F> for Dense<F> {
    type Inner = F;

    /// Create a `DenseVector` from the provided data.
    fn new(data: Vec<Self::Inner>, len: usize) -> Self {
        debug_assert_eq!(data.len(), len);

        Self { data, }
    }

    fn sparse_inner_product<'a, 'b, I: ColumnIterator<'b>, O>(&'a self, column: I) -> O
    where
        &'a F: Mul<&'b I::F, Output=O>,
        O: Zero + AddAssign,
    {
        let mut total = O::zero();
        for (i, v) in column {
            total += &self.data[i] * v;
        }

        total
    }

    /// Append a value to this vector.
    ///
    /// # Arguments
    ///
    /// * `value`: The value to append.
    fn push_value(&mut self, value: F) {
        self.data.push(value);
    }

    /// Set the value at index `i` to `value`.
    fn set(&mut self, i: usize, value: F) {
        debug_assert!(i < self.len());

        self.data[i] = value;
    }

    fn get(&self, i: usize) -> Option<&F> {
        debug_assert!(i < self.len());

        Some(&self.data[i])
    }

    /// Reduce the size of the vector by removing values.
    ///
    /// # Arguments
    ///
    /// * `indices`: A set of indices to remove from the vector, assumed sorted.
    fn remove_indices(&mut self, indices: &[usize]) {
        debug_assert!(indices.len() <= self.len());
        debug_assert!(indices.is_sorted());
        // All values are unique
        debug_assert!(indices.iter().collect::<HashSet<_>>().len() == indices.len());
        debug_assert!(indices.iter().all(|&i| i < self.len()));

        remove_indices(&mut self.data, indices);
    }

    /// Iterate over the values of this vector.
    fn iter(&self) -> Iter<Self::Inner> {
        self.data.iter()
    }

    /// Iterate over the values of this vector mutably.
    fn iter_mut(&mut self) -> IterMut<Self::Inner> {
        self.data.iter_mut()
    }

    /// The length of this vector.
    fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether this vector is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The size of this vector in memory.
    fn size(&self) -> usize {
        self.data.len()
    }
}

impl<T> FromIterator<T> for Dense<T> {
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> Self {
        Self {
            data: iter.into_iter().collect()
        }
    }
}

impl<T> IntoIterator for Dense<T> {
    type Item = T;
    type IntoIter = IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<F: Display> Display for Dense<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for value in &self.data {
            writeln!(f, "{}", value)?;
        }
        writeln!(f)
    }
}
