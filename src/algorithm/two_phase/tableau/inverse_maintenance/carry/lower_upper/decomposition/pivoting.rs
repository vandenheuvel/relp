//! # Pivoting during LU
//!
//! Choosing the right column during LU decomposition reduces fill-in (and perhaps supports
//! numerical stability). Choosing the "best" pivot is often an intractable problem, so heuristics
//! are used.

use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::FullPermutation;

/// Choosing the next pivot.
///
/// A trait with associated methods allows implementors to speed up the search for a good pivot
/// using storage.
pub(super) trait PivotRule {
    fn new() -> Self;

    /// Choose the next pivot.
    ///
    /// # Arguments
    ///
    /// * `nnz_row`: For each row, the number of nonzeros that are left in the part of the matrix
    /// that still is to be decomposed.
    /// * `nnz_column`: For each column, the number of nonzeros that are left in the part of the
    /// matrix that still is to be decomposed.
    /// * `columns`: Rows from the start of the decomposition, but partially modified: some values in
    /// the lower rows (i < k) have been eliminated to move toward an upper triangular matrix.
    /// considered.
    fn choose_pivot<T>(
        &self,
        columns: &[Vec<(usize, T)>],
        row_permutation: &FullPermutation,
        k: usize,
    ) -> (usize, usize);
}

/// Markowitz's pivot rule minimizes `(nnz(row) - 1) * (nnz(row) - 1)` at each step.
///
/// If possible, the row and column are chosen to minimize the amount of swapping that needs to
/// happen in the decomposition.
pub(super) struct Markowitz;
impl PivotRule for Markowitz {
    fn new() -> Self {
        Self
    }

    fn choose_pivot<T>(
        &self,
        columns: &[Vec<(usize, T)>],
        row_permutation: &FullPermutation,
        k: usize,
    ) -> (usize, usize) {
        let m_minus_k = columns.len();
        debug_assert!(m_minus_k > 0);
        debug_assert!(columns.iter().all(|column| column.is_sorted_by_key(|&(i, _)| i)));

        let mut row_nnz = vec![0; m_minus_k];
        let mut column_nnz = vec![0; m_minus_k];

        // TODO(ENHANCEMENT): This is a very slow O(log(nnz(B)) * nnz(B)) search, see #15.
        for (j, column) in columns.iter().enumerate() {
            for &(old_space, _) in column {
                let new_space = row_permutation[old_space];

                if new_space >= k {
                    row_nnz[new_space - k] += 1;
                    column_nnz[j] += 1;
                }
            }
        }

        columns.iter().enumerate()
            .flat_map(|(j, column)| {
                column.iter().filter_map(move |&(old_space, _)| {
                    let new_space = row_permutation[old_space];

                    if new_space < k {
                        None
                    } else {
                        Some((new_space - k, j))
                    }
                })
            })
            .min_by_key(|&(i, j)| (row_nnz[i] - 1) * (column_nnz[j] - 1))
            .unwrap()
    }
}
