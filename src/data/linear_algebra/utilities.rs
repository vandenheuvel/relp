//! Helper algorithms for the `linear_algebra` module.
use std::clone::Clone;
use std::collections::HashSet;
use std::iter::IntoIterator;

/// Reduce the size of the vector by removing values.
///
/// There is another version of this algorithm implemented on `DenseVector` for `Vec<T>`, where `T`
/// is not necessarily `Copy`.
///
/// The method operates in place.
///
/// # Arguments
///
/// * `vector`: `Vec` to remove indices from.
/// * `indices`: A set of indices to remove from the vector, assumed sorted.
pub(super) fn remove_sparse_indices<T>(vector: &mut Vec<(usize, T)>, indices: &Vec<usize>) {
    debug_assert!(indices.is_sorted());
    // All values are unique
    debug_assert!(indices.clone().into_iter().collect::<HashSet<_>>().len() == indices.len());

    if indices.is_empty() || vector.is_empty() {
        return;
    }

    let mut nr_skipped_before = 0;
    vector.drain_filter(|(i, _)| {
        while nr_skipped_before < indices.len() && indices[nr_skipped_before] < *i {
            nr_skipped_before += 1;
        }

        if nr_skipped_before < indices.len() && indices[nr_skipped_before] == *i {
            true
        } else {
            *i -= nr_skipped_before;
            false
        }
    });
}

#[cfg(test)]
mod test {
    use crate::data::linear_algebra::utilities::remove_sparse_indices;

    #[test]
    fn test_remove_sparse_indices() {
        // Removing value not present
        let mut tuples = vec![(0, 3f64)];
        let indices = vec![1];
        remove_sparse_indices(&mut tuples, &indices);
        assert_eq!(tuples, vec![(0, 3f64)]);

        // Removing present value, index should be adjusted
        let mut tuples = vec![(0, 0f64), (2, 2f64)];
        let indices = vec![0];
        remove_sparse_indices(&mut tuples, &indices);
        assert_eq!(tuples, vec![(1, 2f64)]);

        // Empty vec
        let mut tuples: Vec<(usize, i32)> = vec![];
        let indices = vec![0, 1, 1_000];
        remove_sparse_indices(&mut tuples, &indices);
        assert_eq!(tuples, vec![]);

        // Empty vec, removing nothing
        let mut tuples: Vec<(usize, i32)> = vec![];
        let indices = vec![];
        remove_sparse_indices(&mut tuples, &indices);
        assert_eq!(tuples, vec![]);

        // Non-empty vec, removing nothing
        let mut tuples = vec![(1_000, 1f64)];
        let indices = vec![];
        remove_sparse_indices(&mut tuples, &indices);
        assert_eq!(tuples, vec![(1_000, 1f64)]);
    }
}
