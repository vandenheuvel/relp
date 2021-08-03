use std::cmp::Ordering;
use std::ops::{Neg, SubAssign};

use num_traits::Zero;

use crate::algorithm::two_phase::tableau::inverse_maintenance::ops;

/// Update column or "file".
///
/// R is given by `R = I + e_p r'` where `r'` is of the form
/// `r' = (0, 0, ..., 0, r_(p + 1), ..., r_m)` and `r' = u' U^-1` with
/// `u' = (0, 0, ..., 0, U_(p, p + 1), ..., U_(p, m)`.
#[derive(Eq, PartialEq, Clone, Debug)]
pub struct EtaFile<F> {
    values: Vec<(usize, F)>,
    pivot: usize,
    len: usize,
}

impl<F> EtaFile<F>
where
    F: ops::Field + ops::FieldHR,
{
    /// Create a new instance.
    ///
    /// # Arguments
    ///
    /// * `values`: Sorted tuples. Note that the lowest index should be larger than the pivot.
    /// * `pivot`: Row at which the values are located.
    /// * `len`: Dimension of the matrix.
    pub fn new(values: Vec<(usize, F)>, pivot: usize, len: usize) -> Self {
        debug_assert!(values.windows(2).all(|w| w[0].0 < w[1].0));
        debug_assert!(values.first().map_or(true, |&(i, _)| i > pivot));
        debug_assert!(values.last().map_or(true, |&(i, _)| i < len));
        debug_assert!(pivot < len);

        Self {
            values,
            pivot,
            len,
        }
    }

    /// Row-multiply with this matrix (from the left, i.e. x M)
    ///
    /// # Arguments
    ///
    /// * `vector`: Sparse vector of size `self.len`.
    pub fn apply_left(&self, vector: &mut Vec<(usize, F)>) {
        debug_assert!(vector.windows(2).all(|w| w[0].0 < w[1].0));
        debug_assert!(vector.last().map_or(true, |&(i, _)| i < self.len));

        let pivot_index = vector.binary_search_by_key(&self.pivot, |&(i, _)| i);
        if let Ok(pivot_index) = pivot_index {
            for &(j, ref value) in &self.values {
                let has_index = vector.binary_search_by_key(&j, |&(jj, _)| jj);

                // TODO(PERFORMANCE): Sort once at the end instead of inserting all the time.
                let difference = value * &vector[pivot_index].1;
                update_value(difference, has_index, j, vector);
            }

            debug_assert!(vector.windows(2).all(|w| w[0].0 < w[1].0));
        }
    }

    /// Column-multiply with this matrix (from the right, i.e. M x)
    ///
    /// # Arguments
    ///
    /// * `vector`: Sparse vector of size `self.len`.
    pub fn apply_right(&self, vector: &mut Vec<(usize, F)>) {
        debug_assert!(vector.windows(2).all(|w| w[0].0 < w[1].0));

        // We don't modify `vector` below this index
        let pivot_index = vector.binary_search_by_key(&self.pivot, |&(i, _)| i);

        let mut total = F::zero();
        let mut eta_index = 0;
        let mut vector_index = match pivot_index {
            Ok(index) | Err(index) => index,
        };

        while eta_index < self.values.len() && vector_index < vector.len() {
            match self.values[eta_index].0.cmp(&vector[vector_index].0) {
                Ordering::Less => {
                    eta_index += 1;
                }
                Ordering::Equal => {
                    total += &self.values[eta_index].1 * &vector[vector_index].1;
                    eta_index += 1;
                    vector_index += 1;
                }
                Ordering::Greater => {
                    vector_index += 1;
                }
            }
        }

        // TODO(PERFORMANCE): Sort once at the end instead of inserting all the time.
        update_value(total, pivot_index, self.pivot, vector);

        debug_assert!(vector.windows(2).all(|w| w[0].0 < w[1].0));
    }

    /// Modify the pivot index of the spike.
    ///
    /// During the update, multiples of rows are added to the pivot row in order to make all values
    /// except for the right-most one equal to zero. This method computes how that right-most value
    /// is affected by that.
    pub fn update_spike_pivot_value(&self, spike: &mut Vec<(usize, F)>) {
        debug_assert!(spike.windows(2).all(|w| w[0].0 < w[1].0));

        let has_pivot_index = spike.binary_search_by_key(&self.pivot, |&(i, _)| i);
        let search_index = match has_pivot_index {
            Ok(index) => index + 1,
            Err(index) => index,
        };

        let difference = self.values.iter()
            .filter_map(|&(j, ref value)| {
                let row = j;
                spike[search_index..]
                    .binary_search_by_key(&row, |&(i, _)| i)
                    .ok()
                    .map(|shift| {
                        value * &spike[search_index + shift].1
                    })
            })
            .sum::<F>();

        update_value(difference, has_pivot_index, self.pivot, spike);
    }
}

fn update_value<F>(
    difference: F,
    pivot_index: Result<usize, usize>,
    new_index: usize,
    vector: &mut Vec<(usize, F)>,
)
where
    F: SubAssign<F> + Neg<Output=F> + Zero,
{
    if !difference.is_zero() {
        match pivot_index {
            Ok(data_index) => {
                vector[data_index].1 -= difference;
                if vector[data_index].1.is_zero() {
                    vector.remove(data_index);
                }
            },
            Err(data_index) => vector.insert(data_index, (new_index, -difference)),
        }
    }
}

#[cfg(test)]
mod test {
    use relp_num::{R16, R8};

    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::eta_file::EtaFile;

    #[test]
    fn empty1() {
        let eta = EtaFile::new(vec![], 0, 1);
        let mut vector = vec![(0, R8!(1))];
        eta.apply_left(&mut vector);
        debug_assert_eq!(vector, vec![(0, R8!(1))]);
        eta.apply_right(&mut vector);
        debug_assert_eq!(vector, vec![(0, R8!(1))]);
    }

    #[test]
    fn empty_2() {
        let eta = EtaFile::new(vec![], 0, 2);
        let mut vector = vec![(0, R8!(1))];
        eta.apply_left(&mut vector);
        debug_assert_eq!(vector, vec![(0, R8!(1))]);
        eta.apply_right(&mut vector);
        debug_assert_eq!(vector, vec![(0, R8!(1))]);

        let eta = EtaFile::new(vec![], 1, 2);
        let mut vector = vec![(0, R8!(1))];
        eta.apply_left(&mut vector);
        debug_assert_eq!(vector, vec![(0, R8!(1))]);
        eta.apply_right(&mut vector);
        debug_assert_eq!(vector, vec![(0, R8!(1))]);
    }

    #[test]
    fn single_value_2() {
        let eta = EtaFile::new(vec![(1, R8!(1))], 0, 2);
        let mut vector = vec![(0, R8!(13)), (1, R8!(17))];
        eta.apply_right(&mut vector);
        debug_assert_eq!(vector, vec![(0, R8!(13 - 17)), (1, R8!(17))]);

        let mut vector = vec![(0, R8!(13)), (1, R8!(17))];
        eta.apply_left(&mut vector);
        debug_assert_eq!(vector, vec![(0, R8!(13)), (1, R8!(17 - 13))]);
    }

    #[test]
    fn single_value_2_empty() {
        let eta = EtaFile::new(vec![(1, R8!(1))], 0, 2);
        let mut vector = vec![];
        eta.apply_right(&mut vector);
        debug_assert_eq!(vector, vec![]);

        let mut vector = vec![];
        eta.apply_left(&mut vector);
        debug_assert_eq!(vector, vec![]);
    }

    #[test]
    fn two_values_3() {
        let eta = EtaFile::new(vec![(1, R8!(5)), (2, R8!(7))], 0, 3);
        let mut vector = vec![(0, R8!(13)), (1, R8!(17)), (2, R8!(19))];
        eta.apply_right(&mut vector);
        debug_assert_eq!(vector, vec![(0, R8!(13 - 5 * 17 - 7 * 19)), (1, R8!(17)), (2, R8!(19))]);

        let mut vector = vec![(0, R8!(13)), (1, R8!(17)), (2, R8!(19))];
        eta.apply_left(&mut vector);
        debug_assert_eq!(vector, vec![(0, R8!(13)), (1, R8!(-5 * 13 + 17)), (2, R8!(-7 * 13 + 19))]);
    }

    #[test]
    fn one_value_3() {
        let eta = EtaFile::new(vec![(1, R8!(5))], 0, 3);
        let mut vector = vec![(0, R8!(13)), (1, R8!(17)), (2, R8!(19))];
        eta.apply_right(&mut vector);
        debug_assert_eq!(vector, vec![(0, R8!(13 - 5 * 17)), (1, R8!(17)), (2, R8!(19))]);

        let mut vector = vec![(0, R8!(13)), (1, R8!(17)), (2, R8!(19))];
        eta.apply_left(&mut vector);
        debug_assert_eq!(vector, vec![(0, R8!(13)), (1, R8!(-5 * 13 + 17)), (2, R8!(19))]);
    }

    #[test]
    fn one_value_last_index_3() {
        let eta = EtaFile::new(vec![(2, R8!(5))], 0, 3);
        let mut vector = vec![(0, R8!(13)), (1, R8!(17)), (2, R8!(19))];
        eta.apply_right(&mut vector);
        debug_assert_eq!(vector, vec![(0, R8!(13 - 5 * 19)), (1, R8!(17)), (2, R8!(19))]);

        let mut vector = vec![(0, R8!(13)), (1, R8!(17)), (2, R8!(19))];
        eta.apply_left(&mut vector);
        debug_assert_eq!(vector, vec![(0, R8!(13)), (1, R8!(17)), (2, R8!(-5 * 13 + 19))]);
    }

    #[test]
    fn many() {
        let eta = EtaFile::new(vec![(1, R16!(2)), (2, R16!(3)), (5, R16!(5)), (7, R16!(7)), (11, R16!(11)), (12, R16!(13))], 0, 14);
        let mut vector = vec![(0, R16!(17)), (1, R16!(19)), (3, R16!(23)), (5, R16!(29)), (6, R16!(31)), (9, R16!(37)), (11, R16!(41))];
        eta.apply_right(&mut vector);
        debug_assert_eq!(vector, vec![(0, R16!(17 - 2 * 19 - 5 * 29 - 11 * 41)), (1, R16!(19)), (3, R16!(23)), (5, R16!(29)), (6, R16!(31)), (9, R16!(37)), (11, R16!(41))]);

        let mut vector = vec![(0, R16!(13)), (1, R16!(19)), (3, R16!(23)), (5, R16!(29)), (6, R16!(31)), (9, R16!(37)), (11, R16!(41))];
        eta.apply_left(&mut vector);
        debug_assert_eq!(vector, vec![(0, R16!(13)), (1, R16!(19 - 2 * 13)), (2, R16!(-3 * 13)), (3, R16!(23)), (5, R16!(29 -5 * 13)), (6, R16!(31)), (7, R16!(-7 * 13)), (9, R16!(37)), (11, R16!(41 - 11 * 13)), (12, R16!(-13 * 13))]);
    }
}
