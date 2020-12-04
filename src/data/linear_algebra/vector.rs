//! # Vector types for linear programs
//!
//! Sparse and dense vectors. These were written by hand, because a certain specific set of
//! operations needs to be done quickly with these types.
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::fmt::Error;
use std::fmt::Formatter;
use std::iter::Sum;
use std::marker::PhantomData;
use std::mem;
use std::ops::{AddAssign, Index, IndexMut, Mul};
use std::slice::Iter;

use num::Zero;

use crate::algorithm::utilities::{remove_indices, remove_sparse_indices};
use crate::data::linear_algebra::{SparseTuple, SparseTupleVec};
use crate::data::linear_algebra::traits::{SparseComparator, SparseElement};
use crate::data::number_types::traits::{Field, FieldRef};

/// Defines basic ways to create or change a vector, regardless of back-end.
pub trait Vector<F>: PartialEq + Display + Debug {
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
    /// Compute the inner product with a column vector from a matrix.
    fn sparse_inner_product<'a, V: Iterator<Item=&'a SparseTuple<F>>>(&self, column: V) -> F
    where
        F: Field + 'a,
        for<'r> &'r F: FieldRef<F>,
    ;
    /// Make a vector longer by one, by adding an extra value at the end of this vector.
    fn push_value(&mut self, value: F) where F: Zero;
    /// Set the value at an index.
    ///
    /// Depending on internal representation, this can be an expensive operation (for `SparseVector`
    /// 's, the cost depends on the (lack of) sparsity.
    fn set(&mut self, index: usize, value: F) where F: Zero;
    /// Retrieve the value at an index.
    ///
    /// # Returns
    ///
    /// `None` if the representation is `Sparse` and the value at the index is zero.
    fn get(&self, index: usize) -> Option<&F>;
    /// Remove the items at the specified indices.
    fn remove_indices(&mut self, indices: &[usize]);
    /// Iterate over the internal values.
    fn iter_values(&self) -> Iter<Self::Inner>;
    /// Number of items represented by the vector.
    fn len(&self) -> usize;
    /// Whether the vector is empty.
    fn is_empty(&self) -> bool;
    /// Get the size of the internal data structure (and not of the represented vector).
    fn size(&self) -> usize;
}

/// Uses a Vec<f64> as underlying data a structure. Length is fixed at creation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Dense<F> {
    #[allow(missing_docs)]
    pub data: Vec<F>,
    /// Length of the vector being represented. Equal to the length of the inner data.
    /// TODO: Consider removing this field.
    len: usize,
}

impl<F: Field> Dense<F> {
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

impl<F> Index<usize> for Dense<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < self.len);

        &self.data[index]
    }
}

impl<F> IndexMut<usize> for Dense<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < self.len);

        &mut self.data[index]
    }
}

impl<F: PartialEq + Display + Debug> Vector<F> for Dense<F> {
    type Inner = F;

    /// Create a `DenseVector` from the provided data.
    fn new(data: Vec<Self::Inner>, len: usize) -> Self {
        debug_assert_eq!(data.len(), len);

        Self { data, len, }
    }

    fn sparse_inner_product<'a, V: Iterator<Item=&'a SparseTuple<F>>>(&self, column: V) -> F
    where
        F: Field + 'a,
        for<'r> &'r F: FieldRef<F>,
    {
        column.map(|(i, v)| v * &self.data[*i]).sum()
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

    /// Set the value at index `i` to `value`.
    fn set(&mut self, i: usize, value: F) {
        debug_assert!(i < self.len);

        self.data[i] = value;
    }

    fn get(&self, i: usize) -> Option<&F> {
        debug_assert!(i < self.len);

        Some(&self.data[i])
    }

    /// Reduce the size of the vector by removing values.
    ///
    /// # Arguments
    ///
    /// * `indices`: A set of indices to remove from the vector, assumed sorted.
    fn remove_indices(&mut self, indices: &[usize]) {
        debug_assert!(indices.len() <= self.len);
        debug_assert!(indices.is_sorted());
        // All values are unique
        debug_assert!(indices.iter().collect::<HashSet<_>>().len() == indices.len());
        debug_assert!(indices.iter().all(|&i| i < self.len));

        remove_indices(&mut self.data, indices);
        self.len -= indices.len();
    }

    /// Iterate over the values of this vector.
    fn iter_values(&self) -> Iter<Self::Inner> {
        self.data.iter()
    }

    /// The length of this vector.
    fn len(&self) -> usize {
        self.len
    }

    /// Whether this vector is empty.
    fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The size of this vector in memory.
    fn size(&self) -> usize {
        self.data.len()
    }
}

impl<F: Display> Display for Dense<F> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        for value in self.data.iter() {
            writeln!(f, "{}", value)?;
        }
        writeln!(f)
    }
}

/// A sparse vector using a `Vec<>` with (row, value) combinations as back-end. Indices start at
/// `0`.
///
/// TODO: Consider making this backed by a `HashMap`.
#[allow(non_snake_case)]
#[derive(Eq, PartialEq, Clone, Debug)]
// TODO: Remove these trait bounds to simplify the codebase.
pub struct Sparse<F, C> {
    data: SparseTupleVec<F>,
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

impl<F, C> Vector<F> for Sparse<F, C>
where
    F: SparseElement<C> + Display + Debug,
    C: SparseComparator,
{
    type Inner = SparseTuple<F>;

    /// Create a vector of length `len` from `data`.
    ///
    /// Requires that values close to zero are already filtered.
    fn new(data: Vec<Self::Inner>, len: usize) -> Self {
        debug_assert!(data.iter().all(|&(i, _)| i < len));
        debug_assert!(data.is_sorted_by_key(|&(i, _)| i));
        debug_assert_ne!(len, 0);
        debug_assert!(data.len() <= len);

        Self {
            data,
            len,
            
            phantom_comparison_type: PhantomData,
        }
    }

    fn sparse_inner_product<'a, I: Iterator<Item=&'a SparseTuple<F>>>(&self, column: I) -> F
    where
        F: Field + 'a,
        for<'r> &'r F: FieldRef<F>,
    {
        let mut total = F::zero();

        let mut i = 0;
        for (index, value) in column {
            while i < self.data.len() && self.data[i].0 < *index {
                i += 1;
            }

            if i < self.data.len() && self.data[i].0 == *index {
                total += &self.data[i].1 * value;
                i += 1;
            }

            if i == self.len {
                break;
            }
        }

        total
    }

    /// Append a zero value.
    fn push_value(&mut self, value: F) {
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

    fn iter_values(&self) -> Iter<Self::Inner> {
        self.data.iter()
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

impl<F: Field, C> Sparse<F, C>
where
    F: SparseElement<C>,
    for<'r> &'r F: FieldRef<F>,
    C: SparseComparator,
{
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
    pub fn add_multiple_of_row(&mut self, multiple: F, other: &Sparse<F, C>) {
        debug_assert_eq!(other.len(), self.len());

        let mut new_tuples = Vec::new();

        let mut j = 0;
        let old_data = mem::replace(&mut self.data, Vec::with_capacity(0));
        for (i, value) in old_data {
            while j < other.data.len() && other.data[j].0 < i {
                let new_value: F = &multiple * &other.data[j].1;
                if !new_value.is_zero() {
                    new_tuples.push((other.data[j].0, new_value));
                }
                j += 1;
            }

            if j < other.data.len() && i == other.data[j].0 {
                let new_value = value + &multiple * &other.data[j].1;
                if new_value != F::zero() {
                    new_tuples.push((i, new_value));
                }
                j += 1;
            } else {
                new_tuples.push((i, value));
            }
        }
        for (j, value) in &other.data[j..] {
            new_tuples.push((*j, &multiple * value));
        }

        self.data = new_tuples;
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

        Self::new(vec![(i, F::one())], len)
    }

    /// Set the value at index `i` to `value`.
    ///
    /// # Arguments
    ///
    /// * `i`: Index of the value. New tuple will be inserted, potentially causing many values to
    /// be shifted.
    /// * `value`: Value to be taken at index `i`. Should not be very close to zero to avoid
    /// memory usage and numerical error build-up.
    pub fn shift_value(&mut self, i: usize, value: &F) {
        debug_assert!(i < self.len);

        match self.get_data_index(i) {
            Ok(index) => {
                if self.data[index].1 == -value {
                    self.set_zero(i);
                } else {
                    self.data[index].1 += value;
                }
            },
            Err(index) => self.data.insert(index, (i, value.clone())),
        }
    }

    /// Multiply each element of the vector by a value.
    pub fn element_wise_multiply(&mut self, value: &F) {
        for (_, v) in self.data.iter_mut() {
            *v *= value;
        }
        self.data.retain(|(_, v)| !v.is_zero());
    }

    /// Divide each element of the vector by a value.
    pub fn element_wise_divide(&mut self, value: &F) {
        for (_, v) in self.data.iter_mut() {
            *v /= value;
        }
        self.data.retain(|(_, v)| !v.is_zero());
    }
}

impl<F, C> Sparse<F, C>
where
    F: SparseElement<C>,
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
    pub fn inner_product_with_dense<'a, F2, O: 'a>(&'a self, other: &'a Dense<F2>) -> O
    where
        F: Borrow<O>,
        F2: Borrow<O>,
        F2: PartialEq + Display + Debug,
        O: Sum,
        &'a O: Mul<&'a O, Output = O>,
    {
        debug_assert_eq!(other.len(), self.len());

        self.data.iter().map(|(i, value)| other[*i].borrow() * value.borrow()).sum()
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
    pub fn inner_product<'a, F2>(&'a self, other: &'a Sparse<F2, C>) -> C
    where
        F2: SparseElement<C>,
        C: Zero + AddAssign,
        &'a C: Mul<&'a C, Output = C>,
    {
        debug_assert_eq!(other.len(), self.len());

        let mut self_lowest = 0;
        let mut other_lowest = 0;

        let mut total = C::zero();
        while self_lowest < self.data.len() && other_lowest < other.data.len() {
            let self_sought = self.data[self_lowest].0;
            let other_sought = other.data[other_lowest].0;
            match self_sought.cmp(&other_sought) {
                Ordering::Less => {
                    match self.data[self_lowest..].binary_search_by_key(&other_sought, |&(i, _)| i) {
                        Err(diff) => {
                            self_lowest += diff;
                            other_lowest += 1;
                        },
                        Ok(diff) => {
                            total += self.data[self_lowest + diff].1.borrow() * other.data[other_lowest].1.borrow();
                            self_lowest += diff + 1;
                            other_lowest += 1;
                        },
                    }
                },
                Ordering::Greater => {
                    match other.data[other_lowest..].binary_search_by_key(&self_sought, |&(i, _)| i) {
                        Err(diff) => {
                            self_lowest += 1;
                            other_lowest += diff;
                        },
                        Ok(diff) => {
                            total += self.data[self_lowest].1.borrow() * other.data[other_lowest + diff].1.borrow();
                            self_lowest += 1;
                            other_lowest += diff + 1;
                        },
                    }
                },
                Ordering::Equal => {
                    total += self.data[self_lowest].1.borrow() * other.data[other_lowest].1.borrow();
                    self_lowest += 1;
                    other_lowest += 1;
                },
            }
        }

        total
    }
}

impl<F: SparseElement<C>, C: SparseComparator> Display for Sparse<F, C> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        for (index, value) in self.data.iter() {
            writeln!(f, "({} {}), ", index, value)?;
        }
        writeln!(f)
    }
}

#[cfg(test)]
#[allow(dead_code)]
pub mod test {
    //! Contains also some helper methods to be used in other test modules.

    use std::f64::EPSILON;
    use std::marker::PhantomData;

    use num::{FromPrimitive, NumCast, ToPrimitive, Zero};

    use crate::data::linear_algebra::traits::{SparseComparator, SparseElement};
    use crate::data::linear_algebra::vector::{Dense, Sparse as SparseVector, Vector};
    use crate::data::number_types::traits::{Field, FieldRef};
    use crate::F;

    pub trait TestVector<F>: Vector<F> {
        fn from_test_data<T: ToPrimitive + Zero>(data: Vec<T>) -> Self;
    }
    impl<F: Field + FromPrimitive> TestVector<F> for Dense<F> {
        /// Create a `DenseVector` from the provided data.
        fn from_test_data<T: ToPrimitive + Zero>(data: Vec<T>) -> Self {
            let size = data.len();
            Self {
                data: data.into_iter()
                    .map(|v| F::from_f64(v.to_f64().unwrap()).unwrap())
                    .collect(),
                len: size,
            }
        }
    }
    impl<F: SparseElement<C>, C: SparseComparator> TestVector<F> for SparseVector<F, C>
    where
        F: Field + FromPrimitive,
    {
        /// Create a `SparseVector` from the provided data.
        fn from_test_data<T: ToPrimitive + Zero>(data: Vec<T>) -> Self {
            debug_assert_ne!(data.len(), 0);

            let size = data.len();
            Self {
                data: data.into_iter()
                    .enumerate()
                    .filter(|(_, v)| !v.is_zero())
                    .map(|(i, v)| (i, F::from_f64(v.to_f64().unwrap()).unwrap()))
                    .collect(),
                len: size,

                phantom_comparison_type: PhantomData,
            }
        }
    }

    impl<F: SparseElement<C>, C: SparseComparator> SparseVector<F, C>
    where
        F: FromPrimitive,
    {
        /// Create a `SparseVector` from the provided data.
        pub fn from_test_tuples<T: NumCast + Copy>(data: Vec<(usize, T)>, len: usize) -> Self {
            debug_assert!(data.iter().all(|&(i, _)| i < len));
            debug_assert!(data.is_sorted_by_key(|&(i, _)| i));
            debug_assert_ne!(len, 0);
            debug_assert!(data.len() <= len);
            debug_assert!(data.iter().all(|&(_, v)| v.to_f64().unwrap().abs() > EPSILON));

            Self {
                data: data.into_iter().map(|(i, v)| {
                    (i, F::from_f64(v.to_f64().unwrap()).unwrap())
                }).collect(),
                len,

                phantom_comparison_type: PhantomData,
            }
        }
    }

    /// A test vector used in tests.
    fn get_test_vector<F: Field + FromPrimitive, V: TestVector<F>>() -> V {
        V::from_test_data(vec![0, 5, 6])
    }

    /// Test
    fn push_value<F: Field + FromPrimitive, V: TestVector<F>>() where for<'r> &'r F: FieldRef<F> + {
        let mut v = get_test_vector::<F, V>();
        let len = v.len();
        let new_v = F!(1);
        v.push_value(new_v.clone());
        assert_eq!(v.len(), len + 1);
        let x: Option<&F> = v.get(v.len() - 1);
        assert_eq!(x, Some(&new_v));
    }

    /// Test
    fn get_set<F: Field + FromPrimitive, V: TestVector<F>>() {
        let mut v = get_test_vector::<F, V>();

        // Getting a nonzero value
        assert_eq!(v.get(1), Some(&F!(5)));

        // Setting to the same value doesn't change
        let value = v.get(2).unwrap().clone();
        v.set(2, value.clone());
        assert_eq!(v.get(2), Some(&value));

        // Changing a value
        let value = F!(3);
        v.set(1, value.clone());
        assert_eq!(v.get(1), Some(&value));

        // Changing a value
        let value = F!(3);
        v.set(0, value.clone());
        assert_eq!(v.get(1), Some(&value));
    }

    /// Test
    fn out_of_bounds_get<F: Field + FromPrimitive, V: TestVector<F>>() {
        let v = get_test_vector::<F, V>();

        &v.get(400);
    }

    /// Test
    fn out_of_bounds_set<F: Field + FromPrimitive, V: TestVector<F>>() {
        let mut v = get_test_vector::<F, V>();

        v.set(400, F!(45));
    }

    /// Test
    fn len<F: Field + FromPrimitive, V: TestVector<F>>() {
        let v = get_test_vector::<F, V>();

        assert_eq!(v.len(), 3);
    }

    #[cfg(test)]
    mod dense_vector {
        use num::rational::Ratio;
        use num::traits::FromPrimitive;
        use num::Zero;

        use crate::{F, R32};
        use crate::data::linear_algebra::vector::{Dense, Vector};
        use crate::data::linear_algebra::vector::test::{get_set, get_test_vector, len, out_of_bounds_get, out_of_bounds_set, push_value, TestVector};
        use crate::data::number_types::traits::Field;

        type T = Ratio<i32>;

        fn new<F: Field + FromPrimitive>() {
            let d = vec![0, 5, 6].into_iter().map(|v| F!(v)).collect::<Vec<_>>();
            let len = d.len();
            let v = Dense::<F>::new(d, len);

            assert_eq!(v[0], F::zero());
        }

        #[test]
        fn test_new() {
            new::<Ratio<i32>>();
            new::<Ratio<i64>>();
        }

        #[test]
        fn test_push_value() {
            push_value::<T, Dense<T>>();
        }

        #[test]
        fn test_get_set() {
            let v = get_test_vector::<T, Dense<_>>();
            // Getting a zero value
            assert_eq!(v.get(0), Some(&T::zero()));

            get_set::<T, Dense<T>>();
        }

        #[test]
        #[should_panic]
        fn test_out_of_bounds_get() {
            out_of_bounds_get::<T, Dense<T>>();
        }

        #[test]
        #[should_panic]
        fn test_out_of_bounds_set() {
            out_of_bounds_set::<T, Dense<T>>();
        }

        #[test]
        fn remove_indices() {
            let mut v = Dense::<T>::from_test_data(vec![0, 1, 2]);
            v.remove_indices(&vec![1]);
            assert_eq!(v, Dense::<T>::from_test_data(vec![0, 2]));

            let mut v = Dense::<T>::from_test_data(vec![3, 0, 0]);
            v.remove_indices(&vec![0]);
            assert_eq!(v, Dense::<T>::from_test_data(vec![0, 0]));

            let vs = vec![0, 0, 2, 3, 0, 5, 0, 0, 0, 9];
            let mut v = Dense::<T>::from_test_data(vs);
            v.remove_indices(&vec![3, 4, 6]);
            let vs = vec![0, 0, 2, 5, 0, 0, 9];
            assert_eq!(v, Dense::<T>::from_test_data(vs));
        }

        #[test]
        fn test_len() {
            len::<T, Dense<T>>()
        }

        #[test]
        fn extend_with_values() {
            let mut v = Dense::<T>::from_test_data(vec![0, 1, 2]);
            v.extend_with_values(vec![]);
            assert_eq!(v, Dense::<T>::from_test_data(vec![0, 1, 2]));

            let mut v = Dense::<T>::from_test_data(vec![0, 1, 2]);
            v.extend_with_values(vec![R32!(3)]);
            assert_eq!(v, Dense::<T>::from_test_data(vec![0, 1, 2, 3]));

            let mut v = Dense::<T>::from_test_data(vec![0, 1, 2]);
            v.extend_with_values(vec![R32!(3), R32!(4)]);
            assert_eq!(v, Dense::<T>::from_test_data(vec![0, 1, 2, 3, 4]));
        }
    }

    #[cfg(test)]
    mod sparse_vector {
        use num::{FromPrimitive, Zero};
        use num::rational::Ratio;

        use crate::{F, R32};
        use crate::data::linear_algebra::vector::{Sparse as SparseVector, Vector};
        use crate::data::linear_algebra::vector::test::{get_set, get_test_vector, len, out_of_bounds_get, out_of_bounds_set, push_value, TestVector};

        type T = Ratio<i32>;

        #[test]
        fn new() {
            let d = vec![(1, T::from_i32(5).unwrap()), (2, T::from_i32(6).unwrap())];
            let len = 3;
            let v = SparseVector::<T, T>::new(d, len);

            assert_eq!(v.get(0), None);
        }

        #[test]
        fn test_push_value() {
            push_value::<T, SparseVector<T, T>>();
        }

        #[test]
        fn test_get_set() {
            let v = get_test_vector::<T, SparseVector<_, _>>();
            // Getting a zero value
            assert_eq!(v.get(0), None);

            get_set::<T, SparseVector<T, T>>();
        }

        #[test]
        #[should_panic]
        fn test_out_of_bounds_get() {
            out_of_bounds_get::<T, SparseVector<T, T>>();
        }

        #[test]
        #[should_panic]
        fn test_out_of_bounds_set() {
            out_of_bounds_set::<T, SparseVector<T, T>>();
        }

        #[test]
        fn inner_product() {
            let v = get_test_vector::<T, SparseVector<T, T>>();
            let u = get_test_vector::<T, SparseVector<T, T>>();
            assert_eq!(v.inner_product(&u), R32!(5 * 5 + 6 * 6));

            let v = SparseVector::<T, T>::from_test_data(vec![3]);
            let w = SparseVector::<T, T>::from_test_data(vec![5]);
            assert_eq!(v.inner_product(&w), R32!(15));

            let v = SparseVector::<T, T>::from_test_data(vec![0]);
            let w = SparseVector::<T, T>::from_test_data(vec![0]);
            assert_eq!(v.inner_product(&w), R32!(0));

            let v = SparseVector::<T, T>::from_test_data(vec![0, 2]);
            let w = SparseVector::<T, T>::from_test_data(vec![0, 3]);
            assert_eq!(v.inner_product(&w), R32!(6));

            let v = SparseVector::<T, T>::from_test_data(vec![2, 0]);
            let w = SparseVector::<T, T>::from_test_data(vec![0, 3]);
            assert_eq!(v.inner_product(&w), R32!(0));

            let v = SparseVector::<T, T>::from_test_data(vec![0, 2]);
            let w = SparseVector::<T, T>::from_test_data(vec![3, 0]);
            assert_eq!(v.inner_product(&w), R32!(0));

            let v = SparseVector::<T, T>::from_test_data(vec![0, 2]);
            let w = SparseVector::<T, T>::from_test_data(vec![0, 3]);
            assert_eq!(v.inner_product(&w), R32!(6));

            let v = SparseVector::<T, T>::from_test_data(vec![2, 3]);
            let w = SparseVector::<T, T>::from_test_data(vec![5, 7]);
            assert_eq!(v.inner_product(&w), R32!(31));

            let v = SparseVector::<T, T>::from_test_data(vec![0, 0, 0]);
            let w = SparseVector::<T, T>::from_test_data(vec![0, 3, 7]);
            assert_eq!(v.inner_product(&w), R32!(0));

            let v = SparseVector::<T, T>::from_test_data(vec![0, 2, 0]);
            let w = SparseVector::<T, T>::from_test_data(vec![0, 3, 0]);
            assert_eq!(v.inner_product(&w), R32!(6));

            let v = SparseVector::<T, T>::from_test_data(vec![0, 0, 0]);
            let w = SparseVector::<T, T>::from_test_data(vec![5, 7, 0]);
            assert_eq!(v.inner_product(&w), R32!(0));

            let v = SparseVector::<T, T>::from_test_data(vec![0, 0, 2]);
            let w = SparseVector::<T, T>::from_test_data(vec![5, 7, 0]);
            assert_eq!(v.inner_product(&w), R32!(0));

            let v = SparseVector::<T, T>::from_test_data(vec![2, 3, 0]);
            let w = SparseVector::<T, T>::from_test_data(vec![5, 7, 0]);
            assert_eq!(v.inner_product(&w), R32!(31));

            let v = SparseVector::<T, T>::from_test_data(vec![2, 0, 0]);
            let w = SparseVector::<T, T>::from_test_data(vec![5, 7, 0]);
            assert_eq!(v.inner_product(&w), R32!(10));

            let v = SparseVector::<T, T>::from_test_data(vec![0, 2, 0]);
            let w = SparseVector::<T, T>::from_test_data(vec![5, 7, 0]);
            assert_eq!(v.inner_product(&w), R32!(14));

            let v = SparseVector::<T, T>::from_test_data(vec![0, 2, 0]);
            let w = SparseVector::<T, T>::from_test_data(vec![5, 0, 7]);
            assert_eq!(v.inner_product(&w), R32!(0));

            let v = SparseVector::<T, T>::from_test_data(vec![1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]);
            let w = SparseVector::<T, T>::from_test_data(vec![-1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]);
            assert_eq!(v.inner_product(&w), R32!(0));
        }

        #[test]
        fn sparse_inner_product() {
            let v = get_test_vector::<T, SparseVector<T, T>>();
            let w = vec![(1, R32!(5)), (2, R32!(6)), ];
            assert_eq!(v.sparse_inner_product(w.iter()), R32!(5 * 5 + 6 * 6));

            let v = SparseVector::<T, T>::from_test_data(vec![3]);
            let w = vec![(0, R32!(5))];
            assert_eq!(v.sparse_inner_product(w.iter()), R32!(15));

            let v = SparseVector::<T, T>::from_test_data(vec![0]);
            let w = vec![(0, R32!(0))];
            assert_eq!(v.sparse_inner_product(w.iter()), R32!(0));

            let v = SparseVector::<T, T>::from_test_data(vec![0, 2]);
            let w = vec![(1, R32!(3))];
            assert_eq!(v.sparse_inner_product(w.iter()), R32!(6));

            let v = SparseVector::<T, T>::from_test_data(vec![2, 0]);
            let w = vec![(1, R32!(3))];
            assert_eq!(v.sparse_inner_product(w.iter()), R32!(0));

            let v = SparseVector::<T, T>::from_test_data(vec![0, 2]);
            let w = vec![(0, R32!(3))];
            assert_eq!(v.sparse_inner_product(w.iter()), R32!(0));

            let v = SparseVector::<T, T>::from_test_data(vec![0, 2]);
            let w = vec![(1, R32!(3))];
            assert_eq!(v.sparse_inner_product(w.iter()), R32!(6));

            let v = SparseVector::<T, T>::from_test_data(vec![0, 0, 0]);
            let w = vec![(1, R32!(3)), (2, R32!(7))];
            assert_eq!(v.sparse_inner_product(w.iter()), R32!(0));

            let v = SparseVector::<T, T>::from_test_data(vec![0, 2, 0]);
            let w = vec![(0, R32!(5)), (1, R32!(7))];
            assert_eq!(v.sparse_inner_product(w.iter()), R32!(14));

            let v = SparseVector::<T, T>::from_test_data(vec![0, 2, 0]);
            let w = vec![(0, R32!(5)), (2, R32!(7))];
            assert_eq!(v.sparse_inner_product(w.iter()), R32!(0));

            let v = SparseVector::<T, T>::from_test_data(vec![1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]);
            let w = vec![(0, R32!(-1)), (1, R32!(1)), (2, R32!(1)), (12, R32!(1))];
            assert_eq!(v.sparse_inner_product(w.iter()), R32!(0));
        }

        #[test]
        fn add_multiple_of_row() {
            let mut v = SparseVector::<T, T>::from_test_data(vec![3]);
            let w = SparseVector::<T, T>::from_test_data(vec![5]);
            v.add_multiple_of_row(R32!(4), &w);
            assert_eq!(v, SparseVector::<T, T>::from_test_data(vec![23]));

            let mut v = SparseVector::<T, T>::from_test_data(vec![0]);
            let w = SparseVector::<T, T>::from_test_data(vec![0]);
            v.add_multiple_of_row(R32!(4), &w);
            assert_eq!(v, SparseVector::<T, T>::from_test_data(vec![0]));

            let mut v = SparseVector::<T, T>::from_test_data(vec![0, 3]);
            let w = SparseVector::<T, T>::from_test_data(vec![5, 0]);
            v.add_multiple_of_row(R32!(4), &w);
            assert_eq!(v, SparseVector::<T, T>::from_test_data(vec![20, 3]));

            let mut v = SparseVector::<T, T>::from_test_data(vec![12]);
            let w = SparseVector::<T, T>::from_test_data(vec![3]);
            v.add_multiple_of_row(R32!(-4), &w);
            assert_eq!(v, SparseVector::<T, T>::from_test_data(vec![0]));

            let mut v = SparseVector::<T, T>::from_test_data(vec![12, 0, 0]);
            let w = SparseVector::<T, T>::from_test_data(vec![0, 0, 3]);
            v.add_multiple_of_row(R32!(-4), &w);
            assert_eq!(v, SparseVector::<T, T>::from_test_data(vec![12, 0, -12]));
        }

        #[test]
        fn remove_indices() {
            let mut v = SparseVector::<T, T>::from_test_tuples(vec![(0, 3)], 3);
            v.remove_indices(&vec![1]);
            assert_eq!(v, SparseVector::<T, T>::from_test_tuples(vec![(0, 3)], 2));

            let mut v = SparseVector::<T, T>::from_test_tuples(vec![(0, 3)], 3);
            v.remove_indices(&vec![0]);
            assert_eq!(v, SparseVector::<T, T>::from_test_tuples::<f64>(vec![], 2));

            let vs = vec![(2, 2), (3, 3), (5, 5), (10, 10)];
            let mut v = SparseVector::<T, T>::from_test_tuples(vs, 11);
            v.remove_indices(&vec![3, 4, 6]);
            assert_eq!(v, SparseVector::<T, T>::from_test_tuples(vec![(2, 2), (3, 5), (7, 10)], 8));
        }

        #[test]
        fn test_len() {
            len::<T, SparseVector<T, T>>()
        }
    }
}
