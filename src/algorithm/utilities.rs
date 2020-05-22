//! # Utilities
//!
//! Helper functions for algorithms.
use std::clone::Clone;
use std::collections::HashSet;
use std::iter::IntoIterator;

/// Reduce the size of the vector by removing values.
///
/// # Arguments
///
/// * `vector` - `Vec` to remove indices from.
/// * `indices` - A set of indices to remove from the vector, assumed sorted.
pub(crate) fn remove_indices<T>(vector: &mut Vec<T>, indices: &Vec<usize>) {
    debug_assert!(indices.len() <= vector.len());
    debug_assert!(indices.is_sorted());
    // All values are unique
    debug_assert!(indices.clone().into_iter().collect::<HashSet<_>>().len() == indices.len());
    debug_assert!(indices.iter().all(|&i| i < vector.len()));

    let mut i = 0;
    let mut j = 0;
    vector.retain(|_| {
        if i < indices.len() && j < indices[i] {
            j += 1;
            true
        } else if i < indices.len() { // must have j == to_remove[i]
            j += 1;
            i += 1;
            false
        } else { // i == to_remove.len()
            j += 1;
            true
        }
    });
}

#[cfg(test)]
mod test {
    use crate::algorithm::utilities::remove_indices;

    #[test]
    fn test_remove_indices() {
        let mut v = vec![0f64, 1f64, 2f64];
        remove_indices(&mut v, &vec![1]);
        assert_eq!(v, vec![0f64, 2f64]);

        let mut v = vec![3f64, 0f64, 0f64];
        remove_indices(&mut v, &vec![0]);
        assert_eq!(v, vec![0f64, 0f64]);

        let mut v = vec![0f64, 0f64, 2f64, 3f64, 0f64, 5f64, 0f64, 0f64, 0f64, 9f64];
        remove_indices(&mut v,&vec![3, 4, 6]);
        assert_eq!(v, vec![0f64, 0f64, 2f64, 5f64, 0f64, 0f64, 9f64]);

        let mut v = vec![0f64];
        remove_indices(&mut v, &vec![0]);
        assert_eq!(v, vec![]);

        let mut v: Vec<i32> = vec![];
        remove_indices(&mut v, &vec![]);
        assert_eq!(v, vec![]);
    }
}
