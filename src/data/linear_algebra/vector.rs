//! # Vector types for linear programs
//!
//! Sparse and dense vectors. These were written by hand, because a certain specific set of
//! operations needs to be done quickly with these types.
use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::fmt::Error;
use std::fmt::Formatter;
use std::slice::Iter;

use crate::algorithm::utilities::remove_indices;
use crate::data::linear_algebra::matrix::{ColumnMajorOrdering, SparseMatrix};
use crate::data::linear_algebra::SparseTuples;
use crate::data::linear_algebra::utilities::remove_sparse_indices;
use crate::data::number_types::traits::Field;
use std::ops::{Index, IndexMut};

/// Defines basic ways to create or change a vector, regardless of back-end.
pub trait Vector<F>: Index<usize> + Eq + Display + Debug {
    /// Items stored internally.
    type Inner;

    /// Create a new instance.
    ///
    /// # Arguments
    ///
    /// * `data`: Internal data values. Will not be changed and directly used for creation.
    /// * `len`: Length of the vector represented (and not necessarily of the internal data struct
    /// ure).
    ///
    /// # Return value
    ///
    /// Input data wrapped inside a vector.
    fn new(data: Vec<Self::Inner>, len: usize) -> Self;
    /// Remove the items at the specified indices.
    fn remove_indices(&mut self, indices: &Vec<usize>);
    /// Make a vector longer by one, by adding an extra value at the end of this vector.
    fn push_value(&mut self, value: F);
    /// Iterate over the internal values.
    fn iter_values(&self) -> Iter<Self::Inner>;
    /// Get the value at an index.
    ///
    /// Depending on internal representation, this can be an expensive operation (for `SparseVector`
    /// 's, the cost depends on the (lack of) sparsity.
    fn get_value(&self, index: usize) -> F;
    /// Set the value at an index.
    ///
    /// Depending on internal representation, this can be an expensive operation (for `SparseVector`
    /// 's, the cost depends on the (lack of) sparsity.
    fn set_value(&mut self, index: usize, value: F);
    /// Shift the value at an index.
    fn shift_value(&mut self, index: usize, value: F);
    /// Number of items represented by the vector.
    fn len(&self) -> usize;
    /// Get the size of the internal data structure (and not of the represented vector).
    fn size(&self) -> usize;
}

/// Uses a Vec<f64> as underlying data a structure. Length is fixed at creation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DenseVector<F> {
    data: Vec<F>,
    len: usize,
}

impl<F: Field> DenseVector<F> {
    /// Create a vector with all values being equal to a given value.
    ///
    /// # Arguments
    ///
    /// * `value`: The value which all elements of this vector are equal to.
    /// * `len`: Length of the vector, number of elements.
    ///
    /// # Return value
    ///
    /// A constant `DenseVector`
    pub fn constant(value: F, len: usize) -> Self {
        debug_assert_ne!(len, 0);

        Self { data: vec![value; len], len, }
    }

    /// Append multiple values to this vector.
    ///
    /// # Arguments
    ///
    /// * `new_values`: An ordered collections of values to append.
    pub fn extend_with_values(&mut self, new_values: Vec<F>) {
        self.len += new_values.len();
        self.data.extend(new_values);
    }
}

impl<F> Index<usize> for DenseVector<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < self.len);

        &self.data[index]
    }
}

impl<F> IndexMut<usize> for DenseVector<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < self.len);

        &mut self.data[index]
    }
}

impl<F: Field> Vector<F> for DenseVector<F> {
    type Inner = F;

    /// Create a `DenseVector` from the provided data.
    fn new(data: Vec<Self::Inner>, len: usize) -> Self {
        debug_assert_ne!(len, 0);
        debug_assert_eq!(data.len(), len);

        Self { data, len, }
    }

    /// Reduce the size of the vector by removing values.
    ///
    /// # Arguments
    ///
    /// * `indices`: A set of indices to remove from the vector, assumed sorted.
    fn remove_indices(&mut self, indices: &Vec<usize>) {
        debug_assert!(indices.len() <= self.len);
        debug_assert!(indices.is_sorted());
        // All values are unique
        debug_assert!(indices.clone().into_iter().collect::<HashSet<_>>().len() == indices.len());
        debug_assert!(indices.iter().all(|&i| i < self.len));

        remove_indices(&mut self.data, indices);
        self.len -= indices.len();
    }

    /// Append a value to this vector.
    ///
    /// # Arguments
    ///
    /// * `value`: The value to append.
    fn push_value(&mut self, value: F) {
        self.data.push(value);
        self.len += 1;
    }

    /// Iterate over the values of this vector.
    fn iter_values(&self) -> Iter<Self::Inner> {
        self.data.iter()
    }

    /// Get the value at index `i`.
    fn get_value(&self, i: usize) -> F {
        debug_assert!(i < self.len);

        self.data[i]
    }

    /// Set the value at index `i` to `value`.
    fn set_value(&mut self, i: usize, value: F) {
        debug_assert!(i < self.len);

        self.data[i] = value;
    }

    /// Shift the value at index `i` by `shift`.
    fn shift_value(&mut self, i: usize, value: F) {
        debug_assert!(i < self.len);

        self.data[i] += value;
    }

    /// The length of this vector.
    fn len(&self) -> usize {
        self.len
    }

    /// The size of this vector in memory.
    fn size(&self) -> usize {
        self.data.len()
    }
}

impl<F: Display> Display for DenseVector<F> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        for value in self.data.iter() {
            writeln!(f, "{}", value)?;
        }
        writeln!(f, "")
    }
}

/// A sparse vector using a `Vec<>` with (row, value) combinations as back-end. Indices start at
/// `0`.
///
/// TODO: Consider making this backed by a `HashMap`.
#[derive(Eq, PartialEq, Clone, Debug)]
pub struct SparseVector<F> {
    data: SparseTuples<F>,
    len: usize,

    /// Used to implement the `Index` trait. Should never be modified and never be read except from
    /// that trait impl.
    ///
    /// TODO(OPTIMIZATION: Does this get compiled away, or does it cost memory / performance?
    constant_zero: F,
}

impl<F: Field> SparseVector<F> {
    fn set_zero(&mut self, i: usize) {
        match self.get_data_index(i) {
            Ok(index) => { self.data.remove(index); },
            Err(_) => (),
        }
    }

    fn get_data_index(&self, i: usize) -> Result<usize, usize> {
        self.data.binary_search_by_key(&i, |&(index, _)| index)
    }

    /// Calculate the inner product between two vectors.
    ///
    /// # Arguments
    ///
    /// * `other`: Vector to calculate inner product with.
    ///
    /// # Return value
    ///
    /// The inner product.
    pub fn inner_product(&self, other: &Self) -> F {
        debug_assert_eq!(other.len(), self.len());

        let mut self_tuple_index = 0;
        let mut other_tuple_index = 0;

        let mut total = F::additive_identity();
        while self_tuple_index < self.data.len() && other_tuple_index < other.data.len() {
            let self_tuple = self.data[self_tuple_index];
            let other_tuple = other.data[other_tuple_index];

            if self_tuple.0 < other_tuple.0 {
                self_tuple_index += 1;
            } else if self_tuple.0 > other_tuple.0 {
                other_tuple_index += 1;
            } else {
                // self_tuple.0 == other_tuple.0
                total += self_tuple.1 * other_tuple.1;
                self_tuple_index += 1;
                other_tuple_index += 1;
            }
        }

        total
    }

    /// Add the multiple of another row to this row.
    ///
    /// # Arguments
    ///
    /// * `multiple`: Constant that all elements of the `other` vector are multiplied with.
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
    pub fn add_multiple_of_row(&self, multiple: F, other: &Self) -> Self {
        debug_assert_eq!(other.len(), self.len());

        let mut new_tuples = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < self.data.len() && j < other.data.len() {
            if self.data[i].0 < other.data[j].0 {
                new_tuples.push(self.data[i]);
                i += 1;
            } else if self.data[i].0 == other.data[j].0 {
                let new_value = self.data[i].1 + multiple * other.data[j].1;
                if new_value != F::additive_identity() {
                    new_tuples.push((self.data[i].0, new_value));
                }
                i += 1;
                j += 1;
            } else {
                // self.data[i].0 > other.data[j].0
                let new_value = multiple * other.data[j].1;
                if new_value != F::additive_identity() {
                    new_tuples.push((other.data[j].0, multiple * other.data[j].1));
                }
                j += 1;
            }
        }
        new_tuples.extend_from_slice(&self.data[i..]);
        new_tuples.extend(other.data[j..].iter()
            .map(|&(i, v)| (i, multiple * v))
            .filter(|&(_, v)| v != F::additive_identity()));

        Self::new(new_tuples, self.len)
    }

    /// TODO: Consider moving this to the matrix module
    pub fn multiply(&self, matrix: SparseMatrix<F, ColumnMajorOrdering>) -> Self {
        debug_assert_eq!(matrix.nr_rows(), self.len());

        Self::new(
            (0..matrix.nr_columns())
                .map(|j| (j, self.inner_product(&matrix.clone_column(j))))
                .filter(|&(_, v)| v != F::additive_identity())
                .collect(),
            matrix.nr_columns(),
        )
    }

    /// Consume this `SparseVector` by iterating over it's inner `SparseTuple`'s.
    pub fn values_into_iter(self) -> impl IntoIterator<Item = (usize, F)> {
        self.data.into_iter()
    }

    /// Create a `SparseVector` representation of standard basis unit vector e_i.
    ///
    /// # Arguments
    ///
    /// * `i`: Only index where there should be a 1. Note that indexing starts at zero, and runs
    /// until (not through) `len`.
    /// * `len`: Size of the `SparseVector`.
    pub fn standard_basis_vector(i: usize, len: usize) -> Self {
        debug_assert!(i < len);

        Self::new(vec![(i, F::multiplicative_identity())], len)
    }

    /// Multiply each element of the vector by a value.
    pub fn element_wise_multiply(&mut self, value: F) {
        for (_, v) in self.data.iter_mut() {
            *v *= value;
        }
        self.data.retain(|&(_, v)| v != F::additive_identity());
    }

    /// Divide each element of the vector by a value.
    pub fn element_wise_divide(&mut self, value: F) {
        for (_, v) in self.data.iter_mut() {
            *v /= value;
        }
        self.data.retain(|&(_, v)| v != F::additive_identity());
    }
}

impl<F: Field> Index<usize> for SparseVector<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < self.len);

        match self.get_data_index(index) {
            Ok(data_index) => &self.data[data_index].1,
            Err(_) => &self.constant_zero,
        }
    }
}

impl<F: Field> Vector<F> for SparseVector<F> {
    type Inner = (usize, F);

    /// Create a vector of length `len` from `data`.
    ///
    /// Requires that values close to zero are already filtered.
    fn new(data: Vec<Self::Inner>, len: usize) -> Self {
        debug_assert!(data.iter().all(|&(i, _)| i < len));
        debug_assert!(data.is_sorted_by_key(|&(i, _)| i));
        debug_assert_ne!(len, 0);
        debug_assert!(data.len() <= len);
        if !data.iter().all(|&(_, v)| v != F::additive_identity()) {
            print!("x");
        }
        debug_assert!(data.iter().all(|&(_, v)| v != F::additive_identity()));

        Self { data, len, constant_zero: F::additive_identity(), }
    }

    /// Remove elements.
    ///
    /// # Arguments
    ///
    /// * `indices` is assumed sorted.
    fn remove_indices(&mut self, indices: &Vec<usize>) {
        debug_assert!(indices.is_sorted());
        // All values are unique
        debug_assert!(indices.clone().into_iter().collect::<HashSet<_>>().len() == indices.len());
        debug_assert!(indices.iter().all(|&i| i < self.len));
        debug_assert!(indices.len() < self.len);

        remove_sparse_indices(&mut self.data, indices);
        self.len -= indices.len();
    }

    /// Append a zero value
    fn push_value(&mut self, value: F) {
        debug_assert!(value != F::additive_identity());

        self.data.push((self.len, value));
        self.len += 1;
    }

    fn iter_values(&self) -> Iter<Self::Inner> {
        self.data.iter()
    }

    /// Get the value at index `i`.
    fn get_value(&self, i: usize) -> F {
        debug_assert!(i < self.len);

        match self.get_data_index(i) {
            Ok(index) => self.data[index].1,
            Err(_) => F::additive_identity(),
        }
    }

    /// Set the value at index `i` to `value`.
    ///
    /// # Arguments
    ///
    /// * `i`: Index of the value. New tuple will be inserted, potentially causing many values to
    /// be shifted.
    /// * `value`: Value to be taken at index `i`. Should not be very close to zero to avoid
    /// memory usage and numerical error build-up.
    fn set_value(&mut self, i: usize, value: F) {
        debug_assert!(i < self.len);

        if value == F::additive_identity() {
            self.set_zero(i);
        } else {
            match self.get_data_index(i) {
                Ok(index) => self.data[index].1 = value,
                Err(index) => self.data.insert(index, (i, value)),
            }
        }
    }

    /// Set the value at index `i` to `value`.
    ///
    /// # Arguments
    ///
    /// * `i`: Index of the value. New tuple will be inserted, potentially causing many values to
    /// be shifted.
    /// * `value`: Value to be taken at index `i`. Should not be very close to zero to avoid
    /// memory usage and numerical error build-up.
    fn shift_value(&mut self, i: usize, value: F) {
        debug_assert!(i < self.len);

        match self.get_data_index(i) {
            Ok(index) => {
                if self.data[index].1 == -value {
                    self.set_zero(i);
                } else {
                    self.data[index].1 += value;
                }
            },
            Err(index) => self.data.insert(index, (i, value)),
        }
    }

    /// The length of this vector.
    fn len(&self) -> usize {
        self.len
    }

    /// The size of this vector in memory.
    fn size(&self) -> usize {
        self.data.len()
    }
}

impl<F: Display> Display for SparseVector<F> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        for (index, value) in self.data.iter() {
            writeln!(f, "({} {}), ", index, value)?;
        }
        writeln!(f, "")
    }
}

#[cfg(test)]
#[allow(dead_code)]
pub mod test {
    //! Contains also some helper methods to be used in other test modules.

    use std::f64::EPSILON;

    use crate::data::linear_algebra::vector::{DenseVector, SparseVector, Vector};
    use crate::data::number_types::traits::RealField;
    use crate::RF;

    pub trait TestVector<F>: Vector<F> {
        fn from_test_data(data: Vec<f64>) -> Self;
    }
    impl<RF: RealField> TestVector<RF> for DenseVector<RF> {
        /// Create a `DenseVector` from the provided data.
        fn from_test_data(data: Vec<f64>) -> Self {
            let size = data.len();
            Self { data: data.into_iter().map(|v| RF!(v)).collect(), len: size, }
        }
    }
    impl<RF: RealField> TestVector<RF> for SparseVector<RF> {
        /// Create a `SparseVector` from the provided data.
        fn from_test_data(data: Vec<f64>) -> Self {
            debug_assert_ne!(data.len(), 0);

            let size = data.len();
            Self {
                data: data.into_iter()
                    .enumerate()
                    .map(|(i, v)| (i, RF!(v)))
                    .filter(|&(_, v)| v != RF::additive_identity())
                    .collect(),
                len: size,

                constant_zero: RF::zero(),
            }
        }
    }

    impl<RF: RealField> SparseVector<RF> {
        /// Create a `SparseVector` from the provided data.
        pub fn from_test_tuples(data: Vec<(usize, f64)>, len: usize) -> Self {
            debug_assert!(data.iter().all(|&(i, _)| i < len));
            debug_assert!(data.is_sorted_by_key(|&(i, _)| i));
            debug_assert_ne!(len, 0);
            debug_assert!(data.len() <= len);
            debug_assert!(data.iter().all(|&(_, v)| v.abs() > EPSILON));

            Self {
                data: data.into_iter().map(|(i, v)| (i, RF!(v))).collect(),
                len,

                constant_zero: RF::zero(),
            }
        }
    }

    /// Test data used in tests.
    fn test_data() -> Vec<f64> {
        vec![0, 5, 6].into_iter().map(|v| v as f64).collect()
    }

    /// A test vector used in tests.
    fn test_vector<RF: RealField, V: TestVector<RF>>() -> V {
        return V::from_test_data(test_data())
    }

    /// Test
    fn push_value<RF: RealField, V: TestVector<RF>>() {
        let mut v = test_vector::<RF, V>();
        let len = v.len();
        let new_v = RF!(1);
        v.push_value(new_v);
        assert_eq!(v.len(), len + 1);
        assert_eq!(v.get_value(v.len() - 1), new_v);
    }

    /// Test
    fn get_set<RF: RealField, V: TestVector<RF>>() {
        let mut v = test_vector::<RF, V>();

        // Getting a zero value
        assert_eq!(v.get_value(0), RF!(0));

        // Getting a nonzero value
        assert_eq!(v.get_value(1), RF!(5));

        // Setting to the same value doesn't change
        let value = v.get_value(2);
        v.set_value(2, value);
        assert_eq!(v.get_value(2), value);

        // Changing a value
        let value = RF!(3);
        v.set_value(1, value);
        assert_eq!(v.get_value(1), value);

        // Changing a value
        let value = RF!(3);
        v.set_value(0, value);
        assert_eq!(v.get_value(1), value);
    }

    /// Test
    fn out_of_bounds_get<RF: RealField, V: TestVector<RF>>() {
        let v = test_vector::<RF, V>();

        v.get_value(400);
    }

    /// Test
    fn out_of_bounds_set<RF: RealField, V: TestVector<RF>>() {
        let mut v = test_vector::<RF, V>();

        v.set_value(400, RF!(45));
    }

    /// Test
    fn len<F: RealField, V: TestVector<F>>() {
        let v = test_vector::<F, V>();

        assert_eq!(v.len(), 3);
    }

    #[cfg(test)]
    mod dense_vector {
        use num::rational::Ratio;
        use num::traits::FromPrimitive;

        use crate::{R32, RF};
        use crate::data::linear_algebra::vector::{DenseVector, Vector};
        use crate::data::linear_algebra::vector::test::{get_set, len, out_of_bounds_get, out_of_bounds_set, push_value, TestVector};
        use crate::data::number_types::traits::RealField;

        type T = Ratio<i32>;

        fn new<RF: RealField>() {
            let d = vec![0, 5, 6].into_iter().map(|v| RF!(v)).collect::<Vec<_>>();
            let len = d.len();
            let v = DenseVector::<RF>::new(d, len);

            assert_eq!(v.get_value(0), RF::zero());
        }

        #[test]
        fn test_new() {
            new::<Ratio<i32>>();
            new::<Ratio<i64>>();
        }

        #[test]
        fn test_push_value() {
            push_value::<T, DenseVector<T>>();
        }

        #[test]
        fn test_get_set() {
            get_set::<T, DenseVector<T>>();
        }

        #[test]
        #[should_panic]
        fn test_out_of_bounds_get() {
            out_of_bounds_get::<T, DenseVector<T>>();
        }

        #[test]
        #[should_panic]
        fn test_out_of_bounds_set() {
            out_of_bounds_set::<T, DenseVector<T>>();
        }

        #[test]
        fn test_remove_indices() {
            let mut v = DenseVector::<T>::from_test_data(vec![0f64, 1f64, 2f64]);
            v.remove_indices(&vec![1]);
            assert_eq!(v, DenseVector::<T>::from_test_data(vec![0f64, 2f64]));

            let mut v = DenseVector::<T>::from_test_data(vec![3f64, 0f64, 0f64]);
            v.remove_indices(&vec![0]);
            assert_eq!(v, DenseVector::<T>::from_test_data(vec![0f64, 0f64]));

            let vs = vec![0f64, 0f64, 2f64, 3f64, 0f64, 5f64, 0f64, 0f64, 0f64, 9f64];
            let mut v = DenseVector::<T>::from_test_data(vs);
            v.remove_indices(&vec![3, 4, 6]);
            let vs = vec![0f64, 0f64, 2f64, 5f64, 0f64, 0f64, 9f64];
            assert_eq!(v, DenseVector::<T>::from_test_data(vs));
        }

        #[test]
        fn test_len() {
            len::<T, DenseVector<T>>()
        }

        #[test]
        fn test_extend_with_values() {
            let mut v = DenseVector::<T>::from_test_data(vec![0f64, 1f64, 2f64]);
            v.extend_with_values(vec![]);
            assert_eq!(v, DenseVector::<T>::from_test_data(vec![0f64, 1f64, 2f64]));

            let mut v = DenseVector::<T>::from_test_data(vec![0f64, 1f64, 2f64]);
            v.extend_with_values(vec![R32!(3)]);
            assert_eq!(v, DenseVector::<T>::from_test_data(vec![0f64, 1f64, 2f64, 3f64]));

            let mut v = DenseVector::<T>::from_test_data(vec![0f64, 1f64, 2f64]);
            v.extend_with_values(vec![R32!(3), R32!(4)]);
            assert_eq!(v, DenseVector::<T>::from_test_data(vec![0f64, 1f64, 2f64, 3f64, 4f64]));
        }
    }

    #[cfg(test)]
    mod sparse_vector {
        use num::{FromPrimitive, Zero};
        use num::rational::Ratio;

        use crate::data::linear_algebra::matrix::{ColumnMajorOrdering, MatrixOrder};
        use crate::data::linear_algebra::vector::{SparseVector, Vector};
        use crate::data::linear_algebra::vector::test::{get_set, len, out_of_bounds_get, out_of_bounds_set, push_value, test_vector, TestVector};
        use crate::R32;

        type T = Ratio<i32>;

        #[test]
        fn test_new() {
            let d = vec![(1, T::from_i32(5).unwrap()), (2, T::from_i32(6).unwrap())];
            let len = 3;
            let v = SparseVector::new(d, len);

            assert_eq!(v.get_value(0), T::zero());
        }

        #[test]
        fn test_push_value() {
            push_value::<T, SparseVector<T>>();
        }

        #[test]
        fn test_get_set() {
            get_set::<T, SparseVector<T>>();
        }

        #[test]
        #[should_panic]
        fn test_out_of_bounds_get() {
            out_of_bounds_get::<T, SparseVector<T>>();
        }

        #[test]
        #[should_panic]
        fn test_out_of_bounds_set() {
            out_of_bounds_set::<T, SparseVector<T>>();
        }

        #[test]
        fn test_inner_product() {
            let v = test_vector::<T, SparseVector<T>>();
            let u = test_vector::<T, SparseVector<T>>();
            assert_eq!(v.inner_product(&u), R32!(5 * 5 + 6 * 6));

            let v = SparseVector::<T>::from_test_data(vec![3f64]);
            let w = SparseVector::from_test_data(vec![5f64]);
            assert_eq!(v.inner_product(&w), R32!(15));

            let v = SparseVector::<T>::from_test_data(vec![0f64]);
            let w = SparseVector::from_test_data(vec![0f64]);
            assert_eq!(v.inner_product(&w), R32!(0));

            let v = SparseVector::<T>::from_test_data(vec![0f64, 2f64]);
            let w = SparseVector::from_test_data(vec![0f64, 3f64]);
            assert_eq!(v.inner_product(&w), R32!(6));

            let v = SparseVector::<T>::from_test_data(vec![2f64, 0f64]);
            let w = SparseVector::from_test_data(vec![0f64, 3f64]);
            assert_eq!(v.inner_product(&w), R32!(0));

            let v = SparseVector::<T>::from_test_data(vec![0f64, 0f64, 0f64]);
            let w = SparseVector::from_test_data(vec![0f64, 3f64, 7f64]);
            assert_eq!(v.inner_product(&w), R32!(0));

            let v = SparseVector::<T>::from_test_data(vec![2f64, 3f64]);
            let w = SparseVector::from_test_data(vec![5f64, 7f64]);
            assert_eq!(v.inner_product(&w), R32!(31));
        }

        #[test]
        fn test_add_multiple_of_row() {
            let v = SparseVector::from_test_data(vec![3f64]);
            let w = SparseVector::from_test_data(vec![5f64]);
            let result = v.add_multiple_of_row(R32!(4), &w);
            assert_eq!(result, SparseVector::from_test_data(vec![23f64]));

            let v = SparseVector::from_test_data(vec![0f64]);
            let w = SparseVector::from_test_data(vec![0f64]);
            let result = v.add_multiple_of_row(R32!(4), &w);
            assert_eq!(result, SparseVector::from_test_data(vec![0f64]));

            let v = SparseVector::from_test_data(vec![0f64, 3f64]);
            let w = SparseVector::from_test_data(vec![5f64, 0f64]);
            let result = v.add_multiple_of_row(R32!(4), &w);
            assert_eq!(result, SparseVector::from_test_data(vec![20f64, 3f64]));

            let v = SparseVector::from_test_data(vec![12f64]);
            let w = SparseVector::from_test_data(vec![3f64]);
            let result = v.add_multiple_of_row(R32!(-4), &w);
            assert_eq!(result, SparseVector::from_test_data(vec![0f64]));

            let v = SparseVector::from_test_data(vec![12f64, 0f64, 0f64]);
            let w = SparseVector::from_test_data(vec![0f64, 0f64, 3f64]);
            let result = v.add_multiple_of_row(R32!(-4), &w);
            assert_eq!(result, SparseVector::from_test_data(vec![12f64, 0f64, -12f64]));
        }

        #[test]
        fn test_multiply() {
            let v = SparseVector::<T>::from_test_data(vec![3f64]);
            let m = ColumnMajorOrdering::from_test_data::<T>(&vec![vec![0f64]], 1);
            assert_eq!(v.multiply(m), SparseVector::from_test_data(vec![0f64]));

            let v = SparseVector::<T>::from_test_data(vec![0f64]);
            let m = ColumnMajorOrdering::from_test_data::<T>(&vec![vec![0f64]], 1);
            assert_eq!(v.multiply(m), SparseVector::from_test_data(vec![0f64]));

            let v = SparseVector::<T>::from_test_data(vec![0f64, 0f64]);
            let m = ColumnMajorOrdering::from_test_data::<T>(&vec![vec![0f64, 0f64], vec![0f64, 0f64]], 2);
            assert_eq!(v.multiply(m), SparseVector::from_test_data(vec![0f64, 0f64]));

            let v = SparseVector::<T>::from_test_data(vec![2f64, 3f64]);
            let m = ColumnMajorOrdering::from_test_data::<T>(&vec![vec![5f64, 7f64], vec![11f64, 13f64]], 2);
            assert_eq!(v.multiply(m), SparseVector::from_test_data(vec![43f64, 53f64]));
        }

        #[test]
        fn test_remove_indices() {
            let mut v = SparseVector::<T>::from_test_tuples(vec![(0, 3f64)], 3);
            v.remove_indices(&vec![1]);
            assert_eq!(v, SparseVector::from_test_tuples(vec![(0, 3f64)], 2));

            let mut v = SparseVector::<T>::from_test_tuples(vec![(0, 3f64)], 3);
            v.remove_indices(&vec![0]);
            assert_eq!(v, SparseVector::from_test_tuples(vec![], 2));

            let vs = vec![(2, 2f64), (3, 3f64), (5, 5f64), (10, 10f64)];
            let mut v = SparseVector::<T>::from_test_tuples(vs, 11);
            v.remove_indices(&vec![3, 4, 6]);
            assert_eq!(v, SparseVector::from_test_tuples(vec![(2, 2f64), (3, 5f64), (7, 10f64)], 8));
        }

        #[test]
        fn test_len() {
            len::<T, SparseVector<T>>()
        }
    }
}
