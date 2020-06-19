//! # Vector types for linear programs
//!
//! Sparse and dense vectors. These were written by hand, because a certain specific set of
//! operations needs to be done quickly with these types.
use std::borrow::Borrow;
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
use crate::data::linear_algebra::traits::{SparseComparator, SparseElement, SparseElementZero};
use crate::data::number_types::traits::{Field, FieldRef};

/// Defines basic ways to create or change a vector, regardless of back-end.
pub trait Vector<F>: Index<usize> + PartialEq + Display + Debug {
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
    /// Make a vector longer by one, by adding an extra value at the end of this vector.
    fn push_value(&mut self, value: F) where F: Zero;
    /// Set the value at an index.
    ///
    /// Depending on internal representation, this can be an expensive operation (for `SparseVector`
    /// 's, the cost depends on the (lack of) sparsity.
    fn set_value(&mut self, index: usize, value: F) where F: Zero;
    /// Remove the items at the specified indices.
    fn remove_indices(&mut self, indices: &Vec<usize>);
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
    pub data: Vec<F>,
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
    fn set_value(&mut self, i: usize, value: F) {
        debug_assert!(i < self.len);

        self.data[i] = value;
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
pub struct Sparse<F: SparseElement<C>, FZ: SparseElementZero<C>, C: SparseComparator> {
    data: SparseTupleVec<F>,
    len: usize,

    /// Used to implement the `Index` trait. Should never be modified and never be read except from
    /// that trait impl.
    ///
    /// TODO(OPTIMIZATION): Does this get compiled away, or does it cost memory / performance?
    ZERO: FZ,
    
    /// The level that comparison is done at: a single reference to the underlying data.
    phantom_comparison_type: PhantomData<C>,
}

impl<F: SparseElement<C>, FZ: SparseElementZero<C>, C: SparseComparator> Index<usize> for Sparse<F, FZ, C> {
    type Output = C;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < self.len);

        match self.get_data_index(index) {
            Ok(data_index) => self.data[data_index].1.borrow(),
            Err(_) => self.ZERO.borrow(),
        }
    }
}

impl<F: SparseElement<C>, FZ: SparseElementZero<C>, C: SparseComparator> Sparse<F, FZ, C> {
    fn get_data_index(&self, i: usize) -> Result<usize, usize> {
        self.data.binary_search_by_key(&i, |&(index, _)| index)
    }

    fn set_zero(&mut self, i: usize) {
        match self.get_data_index(i) {
            Ok(index) => { self.data.remove(index); },
            Err(_) => (),
        }
    }

    /// Consume this `SparseVector` by iterating over it's inner `SparseTuple`'s.
    pub fn values(self) -> impl IntoIterator<Item = SparseTuple<F>> {
        self.data
    }
}

impl<F, FZ, C> Vector<F> for Sparse<F, FZ, C>
where
    F: SparseElement<C> + Display + Debug,
    FZ: SparseElementZero<C>,
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
        debug_assert!(data.iter().all(|(_, v)| v.borrow() != FZ::zero().borrow()));

        Self {
            data,
            len,
            
            ZERO: FZ::zero(),
            
            phantom_comparison_type: PhantomData,
        }
    }

    /// Append a zero value.
    fn push_value(&mut self, value: F) {
        debug_assert_ne!(value.borrow(), FZ::zero().borrow());

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
    fn set_value(&mut self, i: usize, value: F) {
        debug_assert!(i < self.len);

        if value.borrow() == FZ::zero().borrow() {
            self.set_zero(i);
        } else {
            match self.get_data_index(i) {
                Ok(index) => self.data[index].1 = value,
                Err(index) => self.data.insert(index, (i, value)),
            }
        }
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

impl<F: Field, FZ, C> Sparse<F, FZ, C>
where
    F: SparseElement<C>,
    for<'r> &'r F: FieldRef<F>,
    FZ: SparseElementZero<C>,
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
    pub fn add_multiple_of_row(&mut self, multiple: F, other: &Sparse<F, FZ, C>) {
        debug_assert_eq!(other.len(), self.len());

        let mut new_tuples = Vec::new();

        let mut j = 0;
        let old_data = mem::replace(&mut self.data, Vec::with_capacity(0));
        for (i, value) in old_data {
            while j < other.data.len() && other.data[j].0 < i {
                let new_value: F = &multiple * &other.data[j].1;
                if <F as Borrow<C>>::borrow(&new_value) != FZ::zero().borrow() {
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
        self.data.retain(|(_, v)| <F as Borrow<C>>::borrow(v) != FZ::zero().borrow());
    }

    /// Divide each element of the vector by a value.
    pub fn element_wise_divide(&mut self, value: &F) {
        for (_, v) in self.data.iter_mut() {
            *v /= value;
        }
        self.data.retain(|(_, v)| v.borrow() != FZ::zero().borrow());
    }
}

impl<F, FZ, C> Sparse<F, FZ, C>
where
    F: SparseElement<C>,
    FZ: SparseElementZero<C>,
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
    pub fn inner_product<'a, F2>(&'a self, other: &'a Sparse<F2, FZ, C>) -> C
    where
        F2: SparseElement<C>,
        C: Zero + AddAssign,
        &'a C: Mul<&'a C, Output = C>,
    {
        debug_assert_eq!(other.len(), self.len());

        // TODO(OPTIMIZATION): When can binary search-like strategies be used?
        let mut self_tuple_index = 0;
        let mut other_tuple_index = 0;

        let mut total = C::zero();

        while self_tuple_index < self.data.len() && other_tuple_index < other.data.len() {
            if self.data[self_tuple_index].0 < other.data[other_tuple_index].0 {
                self_tuple_index += 1;
            } else if self.data[self_tuple_index].0 > other.data[other_tuple_index].0 {
                other_tuple_index += 1;
            } else {
                // self_tuple.0 == other_tuple.0
                total += other.data[other_tuple_index].1.borrow() * self.data[self_tuple_index].1.borrow();
                self_tuple_index += 1;
                other_tuple_index += 1;
            }
        }

        total
    }
}

impl<F: SparseElement<C>, FZ: SparseElementZero<C>, C: SparseComparator> Display for Sparse<F, FZ, C> {
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
    use std::ops::Index;

    use num::{FromPrimitive, NumCast, ToPrimitive};

    use crate::data::linear_algebra::traits::{SparseComparator, SparseElement, SparseElementZero};
    use crate::data::linear_algebra::vector::{Dense, Sparse as SparseVector, Vector};
    use crate::data::number_types::traits::{Field, FieldRef};
    use crate::F;

    pub trait TestVector<F: Field + FromPrimitive>: Vector<F> {
        fn from_test_data<T: ToPrimitive>(data: Vec<T>) -> Self;
    }
    impl<F: Field + FromPrimitive> TestVector<F> for Dense<F> {
        /// Create a `DenseVector` from the provided data.
        fn from_test_data<T: ToPrimitive>(data: Vec<T>) -> Self {
            let size = data.len();
            Self {
                data: data.into_iter()
                    .map(|v| F::from_f64(v.to_f64().unwrap()).unwrap())
                    .collect(),
                len: size,
            }
        }
    }
    impl<F: SparseElement<C>, FZ: SparseElementZero<C>, C: SparseComparator> TestVector<F> for SparseVector<F, FZ, C>
    where
        F: Field + FromPrimitive,
    {
        /// Create a `SparseVector` from the provided data.
        fn from_test_data<T: ToPrimitive>(data: Vec<T>) -> Self {
            debug_assert_ne!(data.len(), 0);

            let size = data.len();
            Self {
                data: data.into_iter()
                    .enumerate()
                    .map(|(i, v)| (i, F::from_f64(v.to_f64().unwrap()).unwrap()))
                    .filter(|(_, v)| v.borrow() != FZ::zero().borrow())
                    .collect(),
                len: size,

                ZERO: FZ::zero(),
                phantom_comparison_type: PhantomData
            }
        }
    }

    impl<F: SparseElement<C>, FZ: SparseElementZero<C>, C: SparseComparator> SparseVector<F, FZ, C>
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

                ZERO: FZ::zero(),
                phantom_comparison_type: PhantomData,
            }
        }
    }

    /// A test vector used in tests.
    fn get_test_vector<F: Field + FromPrimitive, V: TestVector<F>>() -> V {
        V::from_test_data(vec![0, 5, 6])
    }

    /// Test
    fn push_value<F: Field + FromPrimitive, V: TestVector<F> + Index<usize, Output = F>>() where for<'r> &'r F: FieldRef<F> + {
        let mut v = get_test_vector::<F, V>();
        let len = v.len();
        let new_v = F!(1);
        v.push_value(new_v.clone());
        assert_eq!(v.len(), len + 1);
        let x: &F = &v[v.len() - 1];
        assert_eq!(x, &new_v);
    }

    /// Test
    fn get_set<F: Field + FromPrimitive, V: TestVector<F> + Index<usize, Output = F>>() {
        let mut v = get_test_vector::<F, V>();

        // Getting a zero value
        assert_eq!(&v[0], &F!(0));

        // Getting a nonzero value
        assert_eq!(v[1], F!(5));

        // Setting to the same value doesn't change
        let value = v[2].clone();
        v.set_value(2, value.clone());
        assert_eq!(&v[2], &value);

        // Changing a value
        let value = F!(3);
        v.set_value(1, value.clone());
        assert_eq!(&v[1], &value);

        // Changing a value
        let value = F!(3);
        v.set_value(0, value.clone());
        assert_eq!(&v[1], &value);
    }

    /// Test
    fn out_of_bounds_get<F: Field + FromPrimitive, V: TestVector<F> + Index<usize, Output = F>>() {
        let v = get_test_vector::<F, V>();

        &v[400];
    }

    /// Test
    fn out_of_bounds_set<F: Field + FromPrimitive, V: TestVector<F> + Index<usize, Output = F>>() {
        let mut v = get_test_vector::<F, V>();

        v.set_value(400, F!(45));
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

        use crate::{F, R32};
        use crate::data::linear_algebra::vector::{Dense, Vector};
        use crate::data::linear_algebra::vector::test::{get_set, len, out_of_bounds_get, out_of_bounds_set, push_value, TestVector};
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

        use crate::data::linear_algebra::vector::{Sparse as SparseVector, Vector};
        use crate::data::linear_algebra::vector::test::{get_set, get_test_vector, len, out_of_bounds_get, out_of_bounds_set, push_value, TestVector};
        use crate::R32;

        type T = Ratio<i32>;

        #[test]
        fn new() {
            let d = vec![(1, T::from_i32(5).unwrap()), (2, T::from_i32(6).unwrap())];
            let len = 3;
            let v = SparseVector::<T, T, T>::new(d, len);

            assert_eq!(v[0], T::zero());
        }

        #[test]
        fn test_push_value() {
            push_value::<T, SparseVector<T, T, T>>();
        }

        #[test]
        fn test_get_set() {
            get_set::<T, SparseVector<T, T, T>>();
        }

        #[test]
        #[should_panic]
        fn test_out_of_bounds_get() {
            out_of_bounds_get::<T, SparseVector<T, T, T>>();
        }

        #[test]
        #[should_panic]
        fn test_out_of_bounds_set() {
            out_of_bounds_set::<T, SparseVector<T, T, T>>();
        }

        #[test]
        fn inner_product() {
            let v = get_test_vector::<T, SparseVector<T, T, T>>();
            let u = get_test_vector::<T, SparseVector<T, T, T>>();
            assert_eq!(v.inner_product(&u), R32!(5 * 5 + 6 * 6));

            let v = SparseVector::<T, T, T>::from_test_data(vec![3]);
            let w = SparseVector::<T, T, T>::from_test_data(vec![5]);
            assert_eq!(v.inner_product(&w), R32!(15));

            let v = SparseVector::<T, T, T>::from_test_data(vec![0]);
            let w = SparseVector::<T, T, T>::from_test_data(vec![0]);
            assert_eq!(v.inner_product(&w), R32!(0));

            let v = SparseVector::<T, T, T>::from_test_data(vec![0, 2]);
            let w = SparseVector::<T, T, T>::from_test_data(vec![0, 3]);
            assert_eq!(v.inner_product(&w), R32!(6));

            let v = SparseVector::<T, T, T>::from_test_data(vec![2, 0]);
            let w = SparseVector::<T, T, T>::from_test_data(vec![0, 3]);
            assert_eq!(v.inner_product(&w), R32!(0));

            let v = SparseVector::<T, T, T>::from_test_data(vec![0, 2]);
            let w = SparseVector::<T, T, T>::from_test_data(vec![3, 0]);
            assert_eq!(v.inner_product(&w), R32!(0));

            let v = SparseVector::<T, T, T>::from_test_data(vec![0, 0, 0]);
            let w = SparseVector::<T, T, T>::from_test_data(vec![0, 3, 7]);
            assert_eq!(v.inner_product(&w), R32!(0));

            let v = SparseVector::<T, T, T>::from_test_data(vec![2, 3]);
            let w = SparseVector::<T, T, T>::from_test_data(vec![5, 7]);
            assert_eq!(v.inner_product(&w), R32!(31));

            let v = SparseVector::<T, T, T>::from_test_data(vec![0, 2, 0]);
            let w = SparseVector::<T, T, T>::from_test_data(vec![0, 3, 0]);
            assert_eq!(v.inner_product(&w), R32!(6));

            let v = SparseVector::<T, T, T>::from_test_data(vec![0, 0, 0]);
            let w = SparseVector::<T, T, T>::from_test_data(vec![5, 7, 0]);
            assert_eq!(v.inner_product(&w), R32!(0));

            let v = SparseVector::<T, T, T>::from_test_data(vec![0, 0, 2]);
            let w = SparseVector::<T, T, T>::from_test_data(vec![5, 7, 0]);
            assert_eq!(v.inner_product(&w), R32!(0));

            let v = SparseVector::<T, T, T>::from_test_data(vec![2, 3, 0]);
            let w = SparseVector::<T, T, T>::from_test_data(vec![5, 7, 0]);
            assert_eq!(v.inner_product(&w), R32!(31));

            let v = SparseVector::<T, T, T>::from_test_data(vec![2, 0, 0]);
            let w = SparseVector::<T, T, T>::from_test_data(vec![5, 7, 0]);
            assert_eq!(v.inner_product(&w), R32!(10));

            let v = SparseVector::<T, T, T>::from_test_data(vec![0, 2, 0]);
            let w = SparseVector::<T, T, T>::from_test_data(vec![5, 7, 0]);
            assert_eq!(v.inner_product(&w), R32!(14));
        }

        #[test]
        fn add_multiple_of_row() {
            let mut v = SparseVector::<T, T, T>::from_test_data(vec![3]);
            let w = SparseVector::<T, T, T>::from_test_data(vec![5]);
            v.add_multiple_of_row(R32!(4), &w);
            assert_eq!(v, SparseVector::<T, T, T>::from_test_data(vec![23]));

            let mut v = SparseVector::<T, T, T>::from_test_data(vec![0]);
            let w = SparseVector::<T, T, T>::from_test_data(vec![0]);
            v.add_multiple_of_row(R32!(4), &w);
            assert_eq!(v, SparseVector::<T, T, T>::from_test_data(vec![0]));

            let mut v = SparseVector::<T, T, T>::from_test_data(vec![0, 3]);
            let w = SparseVector::<T, T, T>::from_test_data(vec![5, 0]);
            v.add_multiple_of_row(R32!(4), &w);
            assert_eq!(v, SparseVector::<T, T, T>::from_test_data(vec![20, 3]));

            let mut v = SparseVector::<T, T, T>::from_test_data(vec![12]);
            let w = SparseVector::<T, T, T>::from_test_data(vec![3]);
            v.add_multiple_of_row(R32!(-4), &w);
            assert_eq!(v, SparseVector::<T, T, T>::from_test_data(vec![0]));

            let mut v = SparseVector::<T, T, T>::from_test_data(vec![12, 0, 0]);
            let w = SparseVector::<T, T, T>::from_test_data(vec![0, 0, 3]);
            v.add_multiple_of_row(R32!(-4), &w);
            assert_eq!(v, SparseVector::<T, T, T>::from_test_data(vec![12, 0, -12]));
        }

        #[test]
        fn remove_indices() {
            let mut v = SparseVector::<T, T, T>::from_test_tuples(vec![(0, 3)], 3);
            v.remove_indices(&vec![1]);
            assert_eq!(v, SparseVector::<T, T, T>::from_test_tuples(vec![(0, 3)], 2));

            let mut v = SparseVector::<T, T, T>::from_test_tuples(vec![(0, 3)], 3);
            v.remove_indices(&vec![0]);
            assert_eq!(v, SparseVector::<T, T, T>::from_test_tuples::<f64>(vec![], 2));

            let vs = vec![(2, 2), (3, 3), (5, 5), (10, 10)];
            let mut v = SparseVector::<T, T, T>::from_test_tuples(vs, 11);
            v.remove_indices(&vec![3, 4, 6]);
            assert_eq!(v, SparseVector::<T, T, T>::from_test_tuples(vec![(2, 2), (3, 5), (7, 10)], 8));
        }

        #[test]
        fn test_len() {
            len::<T, SparseVector<T, T, T>>()
        }
    }
}
