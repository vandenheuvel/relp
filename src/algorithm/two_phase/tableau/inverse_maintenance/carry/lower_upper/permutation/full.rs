/// # Full permutation
///
/// Explicitly store both the permutation and its inverse. Used for the column and row permutation
/// produced by the pivoting during the decomposition.
use std::{fmt, mem};
use std::collections::HashSet;
use std::ops::Index;

use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::Permutation;

/// Full permutation.
///
/// Both the entire forward and backward permutation is stored explicitly.
#[derive(Eq, PartialEq, Clone, Debug)]
pub struct Full {
    /// Index of the value is mapped to the value at the index.
    forward: Vec<usize>,
    /// Index of the value is mapped to the value at the index.
    backward: Vec<usize>,
}

impl Full {
    /// Create a new instance.
    ///
    /// Computes the inverse by sorting.
    pub fn new(forward: Vec<usize>) -> Self {
        debug_assert_eq!(forward.iter().collect::<HashSet<_>>().len(), forward.len());
        debug_assert!(*forward.iter().max().unwrap() < forward.len());

        let mut backward = forward.iter()
            .enumerate()
            .map(|(i, j)| (*j, i))
            .collect::<Vec<_>>();
        backward.sort_unstable_by_key(|&(j, _)| j);
        let backward = backward.into_iter().map(|(_, i)| i).collect();

        Self {
            forward,
            backward,
        }
    }

    /// Create a new instance that maps an index onto itself.
    ///
    /// # Arguments
    ///
    /// * `n`: Number of elements in the permutation. Sometimes, this variable is saved as a field
    /// for debug assertions only, while sometimes, it is also necessary to determine the
    /// permutation.
    pub fn identity(n: usize) -> Self{
        Self {
            forward: (0..n).collect(),
            backward: (0..n).collect(),
        }
    }

    /// Invert the permutation, making the forward the backward direction and vice-versa.
    ///
    /// Done cheaply by swapping the backing arrays.
    pub fn invert(&mut self) {
        mem::swap(&mut self.forward, &mut self.backward);
    }

    pub fn swap(&mut self, i: usize, j: usize) {
        debug_assert!(i < self.len());
        debug_assert!(j < self.len());

        let i_target = self.forward[i];
        let j_target = self.forward[j];

        self.forward.swap(i, j);
        self.backward.swap(i_target, j_target);
    }

    pub fn swap_inverse(&mut self, i: usize, j: usize) {
        debug_assert!(i < self.len());
        debug_assert!(j < self.len());

        let i_target = self.backward[i];
        let j_target = self.backward[j];

        self.forward.swap(i_target, j_target);
        self.backward.swap(i, j);
    }

    pub fn rotate_right_from(&mut self, i: usize) {
        debug_assert!(i < self.len());

        self.forward[i..].rotate_right(1);
        self.backward[i..].rotate_left(1);
    }
}

impl Permutation for Full {
    fn forward(&self, i: usize) -> usize {
        debug_assert!(i < self.len());

        self.forward[i]
    }
    fn backward(&self, i: usize) -> usize {
        debug_assert!(i < self.len());

        self.backward[i]
    }

    fn len(&self) -> usize {
        self.forward.len()
        // == self.backward.len()
    }
}

impl Index<usize> for Full {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < self.len());

        &self.forward[index] // == &self.forward(index)
    }
}

impl fmt::Display for Full {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("(")?;
        if self.len() > 0 {
            self.forward[0].fmt(f)?;
            for i in &self.forward[1..] {
                write!(f, ", {}", i)?;
            }
        }
        f.write_str(")")
    }
}

#[cfg(test)]
mod test {
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::{FullPermutation, Permutation};

    #[test]
    fn identity() {
        let n = 4;
        let permutation = FullPermutation::identity(n);
        let mut i = 1;
        permutation.forward_ref(&mut i);

        assert_eq!(i, 1);
    }

    #[test]
    fn forward_backward() {
        let permutation = FullPermutation::new(vec![3, 1, 2, 0]);
        let mut i = 1;
        permutation.forward_ref(&mut i);
        assert_eq!(i, 1);

        let mut i = 0;
        permutation.forward_ref(&mut i);
        assert_eq!(i, 3);

        let mut i = 3;
        permutation.forward_ref(&mut i);
        assert_eq!(i, 0);

        let mut i = 1;
        permutation.backward_ref(&mut i);
        assert_eq!(i, 1);

        let mut i = 0;
        permutation.backward_ref(&mut i);
        assert_eq!(i, 3);

        let mut i = 3;
        permutation.backward_ref(&mut i);
        assert_eq!(i, 0);
    }

    #[test]
    fn back_and_forth() {
        let n = 4;
        let permutation = FullPermutation::new(vec![0, 1, 3, 2]);
        let original = (0..n).map(|i| (i, i)).collect::<Vec<_>>();
        let mut test = original.clone();
        permutation.forward_unsorted(&mut test);
        permutation.backward_unsorted(&mut test);
        assert_eq!(test, original);

        let n = 4;
        let mut permutation = FullPermutation::new(vec![0, 1, 3, 2]);
        let original = (0..n).map(|i| (i, i)).collect::<Vec<_>>();
        let mut test = original.clone();
        permutation.forward_unsorted(&mut test);
        permutation.invert();
        permutation.forward_unsorted(&mut test);
        assert_eq!(test, original);
    }

    #[test]
    fn swap_identity() {
        let n = 3;
        let mut p = FullPermutation::identity(n);
        p.swap(1, 2);
        for i in 0..n {
            let mut j = i;
            p.forward_ref(&mut j);
            p.backward_ref(&mut j);
            assert_eq!(j, i);
        }
    }

    #[test]
    fn swap_non_identity_matches() {
        let mut p = FullPermutation::new(vec![0, 2, 3, 1, 4, 9, 8, 7, 6, 5]);

        for i in 0..p.len() {
            for j in 0..p.len() {
                p.swap(i, j);

                for k in 0..p.len() {
                    let mut copy = k;

                    p.forward_ref(&mut copy);
                    p.backward_ref(&mut copy);

                    assert_eq!(copy, k, "i: {}, j: {}, k: {}", i, j, k);
                }
            }
        }
    }

    #[test]
    fn swap_non_identity() {
        let mut p = FullPermutation::new(vec![0, 2, 3, 1, 4, 9, 8, 7, 6, 5]);

        assert_eq!(p[6], 8);
        assert_eq!(p.backward(8), 6);
        assert_eq!(p[9], 5);
        assert_eq!(p.backward(5), 9);

        p.swap_inverse(8, 5);

        assert_eq!(p[6], 5);
        assert_eq!(p.backward(8), 9);
        assert_eq!(p[9], 8);
        assert_eq!(p.backward(5), 6);
    }

    #[test]
    fn rotate() {
        let n = 5;
        let mut p = FullPermutation::identity(n);
        p.rotate_right_from(3);
        for i in 0..n {
            let mut j = i;
            p.forward_ref(&mut j);
            p.backward_ref(&mut j);
            assert_eq!(j, i);
        }
    }
}
