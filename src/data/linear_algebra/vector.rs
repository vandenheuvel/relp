use std::slice::Iter;

use data::linear_algebra::matrix::{DenseMatrix, Matrix};
use algorithm::simplex::EPSILON;

/// Defines basic ways to create or change a vector, regardless of back-end.
pub trait Vector {
    fn from_data(data: Vec<f64>) -> Self;
    fn ones(size: usize) -> Self;
    fn zeros(size: usize) -> Self;
    fn get_value(&self, index: usize) -> f64;
    fn set_value(&mut self, index: usize, value: f64);
    fn len(&self) -> usize;
    fn size(&self) -> usize;
}

/// Uses a Vec<f64> as underlying data a structure. Length is fixed at creation.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseVector {
    data: Vec<f64>,
    len: usize,
}

impl DenseVector {
    /// Consider this a column vector and concatenate a `DenseMatrix` to the "right" (high column
    /// indices) of this matrix "horizontally" (number of rows must be equal).
    pub fn hcat(self, other: DenseMatrix) -> DenseMatrix {
        debug_assert_eq!(other.nr_rows(), self.len());

        let mut data = other.data();
        for i in 0..self.len() {
            data[i].insert(0, self.data[i]);
        }
        DenseMatrix::from_data(data)
    }
    /// Consider this a row vector and concatenate a `DenseMatrix` to the "bottom" (high row
    /// indices) of this matrix "vertically" (number of columns must be equal).
    pub fn vcat(self, other: DenseMatrix) -> DenseMatrix {
        debug_assert_eq!(other.nr_columns(), self.len());

        let mut data = other.data();
        data.insert(0, self.data);

        DenseMatrix::from_data(data)
    }
    /// Iterate over the values of this vector.
    pub fn iter(&self) -> Iter<f64> {
        self.data.iter()
    }
    /// Remove the value at index `i`.
    pub fn remove_value(&mut self, i: usize) {
        debug_assert!(i < self.len());

        self.data.remove(i);
        self.len -= 1;
    }
}

impl Vector for DenseVector {
    /// Create a `DenseVector` from the provided data.
    fn from_data(data: Vec<f64>) -> DenseVector {
        debug_assert_ne!(data.len(), 0);

        let size = data.len();
        DenseVector { data, len: size, }
    }
    /// Create a `DenseVector` of length `len` filled with `1f64`.
    fn ones(len: usize) -> DenseVector {
        debug_assert_ne!(len, 0);

        DenseVector { data: vec![1f64; len], len, }
    }
    /// Create a `DenseVector` of length `len` filled with `0f64`.
    fn zeros(len: usize) -> DenseVector {
        debug_assert_ne!(len, 0);

        DenseVector { data: vec![0f64; len], len, }
    }
    /// Get the value at index `i`.
    fn get_value(&self, i: usize) -> f64 {
        debug_assert!(i < self.len);

        self.data[i]
    }
    /// Set the value at index `i` to `value`.
    fn set_value(&mut self, i: usize, value: f64) {
        debug_assert!(i < self.len);

        for _ in 0..(i as i64 - self.data.len() as i64 + 1) {
            self.data.push(0f64);
        }
        self.data[i] = value;
    }
    /// The length of this vector.
    fn len(&self) -> usize {
        self.data.len()
    }
    /// The size of this vector in memory.
    fn size(&self) -> usize {
        self.len
    }
}

/// A sparse vector using a `Vec<>` with (row, value) combinations as back-end. Indices start at
/// `0`.
#[derive(Debug)]
pub struct SparseVector {
    data: Vec<(usize, f64)>,
    len: usize,
}

impl SparseVector {
    /// Create a vector of length `len` from `data`.
    pub fn from_tuples(data: Vec<(usize, f64)>, len: usize) -> SparseVector {
        debug_assert_ne!(len, 0);
        debug_assert!(data.iter().map(|&(i, _)| i).max().unwrap_or(0 as usize) < len);

        SparseVector { data, len, }
    }
    /// Concatenate this vector with another into a third. The length of the new vector is the sum
    /// of the lengths of the vectors it was created from.
    pub fn cat(self, other: SparseVector) -> SparseVector {
        let len = self.len() + other.len();
        let mut data = self.data;
        for (row, value) in other.data {
            data.push((self.len + row, value));
        }
        SparseVector { data, len, }
    }
    /// Get all (row, value) tuples of this vector in an Iter.
    pub fn values(&self) -> Iter<'_, (usize, f64)> {
        self.data.iter()
    }
    /// Calculate the inner product between two vectors.
    pub fn inner_product(&self, other: &SparseVector) -> f64 {
        debug_assert_eq!(other.len(), self.len());

        let mut total = 0f64;
        let mut other_values = other.values();
        let mut next_value = other_values.next();
        for (my_column, my_value) in self.values() {
            if next_value.is_none() {
                break;
            }
            let mut column = next_value.unwrap().0;
            let mut value = next_value.unwrap().1;

            while column < *my_column {
                let next_value = other_values.next();
                if next_value.is_none() {
                    break;
                }
                column = next_value.unwrap().0;
                value = next_value.unwrap().1;
            }

            if column == *my_column {
                total += value * *my_value;
                next_value = other_values.next();
            }
        }

        total
    }
    /// Append a zero value
    pub fn push_zero(&mut self) {
        self.len += 1;
    }
    /// Remove the value at row `i`.
    pub fn remove_value(&mut self, i: usize) {
        debug_assert!(self.len > 1);
        debug_assert!(i < self.len);

        self.len -= 1;

        let mut index = 0;
        let mut removed = false;
        while index < self.data.len() {
            if self.data[index].0 == i && !removed {
                self.data.remove(index);
                removed = true;
            } else if self.data[index].0 >= i {
                self.data[index].0 -= 1;
                index += 1;
            } else {
                index += 1;
            }
        }
    }
}

impl Vector for SparseVector {
    /// Create a `SparseVector` from the provided data.
    fn from_data(data: Vec<f64>) -> SparseVector {
        debug_assert_ne!(data.len(), 0);

        let size = data.len();
        SparseVector {
            data: data.into_iter()
                .enumerate()
                .filter(|(_, v)| *v != 0f64)
                .collect(),
            len: size,
        }
    }
    /// Create a `SparseVector` of length `len` filled with `1f64`.
    fn ones(len: usize) -> SparseVector {
        debug_assert_ne!(len, 0);

        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            data.push((i, 1f64));
        }
        SparseVector { data, len, }
    }
    /// Create a `SparseVector` of length `len` filled with `0f64`.
    fn zeros(len: usize) -> SparseVector {
        debug_assert_ne!(len, 0);

        SparseVector { data: Vec::new(), len, }
    }
    /// Get the value at index `i`.
    fn get_value(&self, i: usize) -> f64 {
        debug_assert!(i < self.len);

        match self.data.iter().find(|&&(index, _)| index == i) {
            Some(&(_, value)) => value,
            None => 0f64,
        }
    }
    /// Set the value at index `i` to `value`.
    fn set_value(&mut self, i: usize, value: f64) {
        debug_assert!(i < self.len);

        if value == 0f64 {
            return;
        }

        for index in 0..self.data.len() {
            if self.data[index].0 < i {
                continue;
            } else if self.data[index].0 == i {
                self.data[index].1 = value;
            } else if self.data[index].0 > i {
                self.data.insert(i, (i, value));
            }
            return;
        }
        self.data.push((i, value));
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

impl Clone for SparseVector {
    fn clone(&self) -> SparseVector {
        SparseVector {
            data: self.data.clone(),
            len: self.len,
        }
    }
}

impl PartialEq for SparseVector {
    fn eq(&self, other: &SparseVector) -> bool {
        if !(self.len() == other.len()) {
            return false;
        }

        for i in 0..self.len() {
            if !((self.get_value(i) - other.get_value(i)).abs() < EPSILON) {
                return false;
            }
        }

        true
    }
}

impl Eq for SparseVector {}


#[cfg(test)]
mod test {

    use super::*;

    fn test_data() -> Vec<f64> {
        vec![0f64, 5f64, 6f64]
    }

    fn test_vector<T>() -> T where T: Vector {
        return T::from_data(test_data())
    }

    fn create<T>() where T: Vector {
        from_data::<T>();
        zeros::<T>();
        ones::<T>();
    }

    fn from_data<T>() where T: Vector {
        let d = test_data();
        let v = T::from_data(d);

        assert_approx_eq!(v.get_value(0), 0f64);
    }

    fn zeros<T>() where T: Vector {
        let size = 533;
        let v= T::zeros(size);

        assert_approx_eq!(v.get_value(0), 0f64);
        assert_approx_eq!(v.get_value(size - 1), 0f64);
    }

    fn ones<T>() where T: Vector {
        let size = 593;
        let v = T::ones(size);

        assert_approx_eq!(v.get_value(0), 1f64);
        assert_approx_eq!(v.get_value(size - 1), 1f64);
    }

    fn get_set<T>() where T: Vector {
        let mut v = test_vector::<T>();


        // Getting a zero value
        assert_approx_eq!(v.get_value(0), 0f64);

        // Getting a nonzero value
        assert_approx_eq!(v.get_value(1), 5f64);

        // Setting to the same value doesn't change
        let value = v.get_value(2);
        v.set_value(2, value);
        assert_approx_eq!(v.get_value(2), value);

        // Changing a value
        let value = 3f64;
        v.set_value(1, value);
        assert_approx_eq!(v.get_value(1), value);
    }

    fn out_of_bounds_get<T>() where T: Vector {
        let v = test_vector::<T>();

        v.get_value(400);
    }

    fn out_of_bounds_set<T>() where T: Vector {
        let mut v = test_vector::<T>();

        v.set_value(400, 45f64);
    }

    #[cfg(test)]
    mod dense_vector {

        use super::*;

        #[test]
        fn test_create() {
            create::<DenseVector>();
        }

        #[test]
        fn test_get_set() {
            get_set::<DenseVector>();
        }

        #[test]
        #[should_panic]
        fn test_out_of_bounds_get() {
            out_of_bounds_get::<DenseVector>();
        }

        #[test]
        #[should_panic]
        fn test_out_of_bounds_set() {
            out_of_bounds_set::<DenseVector>();
        }
    }

    #[cfg(test)]
    mod sparse_vector {

        use super::*;

        #[test]
        fn test_create() {
            create::<SparseVector>();
        }

        #[test]
        fn test_get_set() {
            get_set::<SparseVector>();
        }

        #[test]
        #[should_panic]
        fn test_out_of_bounds_get() {
            out_of_bounds_get::<SparseVector>();
        }

        #[test]
        #[should_panic]
        fn test_out_of_bounds_set() {
            out_of_bounds_set::<SparseVector>();
        }

        #[test]
        fn test_inner_product() {
            let v = test_vector::<SparseVector>();
            let u = test_vector::<SparseVector>();

            assert_approx_eq!(v.inner_product(&u), 25f64 + 36f64);
        }
    }
}
