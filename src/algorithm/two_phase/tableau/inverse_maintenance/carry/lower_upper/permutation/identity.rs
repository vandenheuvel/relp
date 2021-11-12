//! # Identity
//!
//! A permutation that doesn't change anything. Not in use.
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::Permutation;

/// Permutation that maps every element onto itself.
///
/// It's only field is the size, which is used only for debug purposes.
pub struct Identity(usize);
impl Identity {
    pub fn new(n: usize) -> Self {
        Self(n)
    }
}

impl Permutation for Identity {
    fn forward(&self, i: usize) -> usize {
        debug_assert!(i < self.0);

        i
    }

    fn backward(&self, i: usize) -> usize {
        debug_assert!(i < self.0);

        i
    }

    fn forward_sorted<T>(&self, items: &mut [(usize, T)]) {
        debug_assert!(items.iter().all(|&(i, _)| i < self.len()));
        debug_assert!(items.windows(2).all(|w| w[0].0 < w[1].0));
    }

    fn backward_sorted<T>(&self, items: &mut [(usize, T)]) {
        debug_assert!(items.iter().all(|&(i, _)| i < self.len()));
        debug_assert!(items.windows(2).all(|w| w[0].0 < w[1].0));
    }

    fn forward_unsorted<T>(&self, items: &mut [(usize, T)]) {
        debug_assert!(items.iter().all(|&(i, _)| i < self.len()));
    }

    fn backward_unsorted<T>(&self, items: &mut [(usize, T)]) {
        debug_assert!(items.iter().all(|&(i, _)| i < self.len()));
    }

    fn len(&self) -> usize {
        self.0
    }
}
