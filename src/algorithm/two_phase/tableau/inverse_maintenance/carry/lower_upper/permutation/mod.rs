/// # Permutations
///
/// Logic for permuting and more specifically rotating column and row indices.
mod full;
pub use full::Full as FullPermutation;

mod identity;
pub use identity::Identity as IdentityPermutation;

mod rotate_to_back;
pub use rotate_to_back::RotateToBack as RotateToBackPermutation;

mod swap;
pub use swap::Swap as SwapPermutation;

mod rotate;
pub use rotate::RotateLeft as RotateLeftPermutation;
pub use rotate::RotateRight as RotateRightPermutation;

/// Basic permutation behavior.
///
/// Contains a few trivial implementations to avoid duplication.
pub trait Permutation {
    /// Apply the permutation to an index in the forward direction.
    ///
    /// What this direction actually is (w.r.t. the backward direction) depends on the implementor.
    ///
    /// The choice for in-place mutation was made for idiomatic application of different
    /// transformations after each other.
    ///
    /// # Arguments
    ///
    /// * `i`: Value in range `0..self.len()` that will be mutated in place.
    fn forward(&self, i: &mut usize);

    /// Apply the permutation to an index in the forward direction.
    ///
    /// What this direction actually is (w.r.t. the backward direction) depends on the implementor.
    ///
    /// The choice for in-place mutation was made for idiomatic application of different
    /// transformations after each other.
    ///
    /// # Arguments
    ///
    /// * `i`: Value in range `0..self.len()` that will be mutated in place.
    fn backward(&self, i: &mut usize);

    /// Apply the forward permutation to a collection of items, keeping them sorted.
    ///
    /// # Arguments
    ///
    /// * `items`: Sorted slice of `(index, value)` tuples where the indices are unique.
    fn forward_sorted<T>(&self, items: &mut [(usize, T)]) {
        debug_assert!(items.iter().all(|&(i, _)| i < self.len()));
        debug_assert!(items.is_sorted_by_key(|&(i, _)| i));
        debug_assert!(items.windows(2).all(|w| w[0].0 < w[1].0));

        self.forward_unsorted(items);
        items.sort_unstable_by_key(|&(i, _)| i);
    }

    /// Apply the backward permutation to a collection of items, keeping them sorted.
    ///
    /// # Arguments
    ///
    /// * `items`: Sorted slice of `(index, value)` tuples where the indices are unique.
    fn backward_sorted<T>(&self, items: &mut [(usize, T)]) {
        debug_assert!(items.iter().all(|&(i, _)| i < self.len()));
        debug_assert!(items.is_sorted_by_key(|&(i, _)| i));
        debug_assert!(items.windows(2).all(|w| w[0].0 < w[1].0));

        self.backward_unsorted(items);
        items.sort_unstable_by_key(|&(i, _)| i);
    }

    /// Apply the forward permutation to each of the elements.
    ///
    /// Used when permutations get applied repeatedly, then sorting can be done manually after only
    /// once.
    ///
    /// # Arguments
    ///
    /// * `items`: Slice of `(index, value)` tuples that will be mutated in place. Does not need to
    /// be sorted.
    fn forward_unsorted<T>(&self, items: &mut [(usize, T)]) {
        debug_assert!(items.iter().all(|&(i, _)| i < self.len()));

        for (i, _) in items {
            Permutation::forward(self, i)
        }
    }

    /// Apply the backward permutation to each of the elements.
    ///
    /// Used when permutations get applied repeatedly, then sorting can be done manually after only
    /// once.
    ///
    /// # Arguments
    ///
    /// * `items`: Slice of `(index, value)` tuples that will be mutated in place. Does not need to
    /// be sorted.
    fn backward_unsorted<T>(&self, items: &mut [(usize, T)]) {
        debug_assert!(items.iter().all(|&(i, _)| i < self.len()));

        for (i, _) in items {
            Permutation::backward(self, i)
        }
    }

    /// Size of the permutation.
    ///
    /// Some implementors use this value internally to compute the permutation, others only for
    /// debug assertions.
    fn len(&self) -> usize;
}
