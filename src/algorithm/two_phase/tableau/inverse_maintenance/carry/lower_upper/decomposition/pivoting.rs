//! # Pivoting during LU
//!
//! Choosing the right column during LU decomposition reduces fill-in (and perhaps supports
//! numerical stability). Choosing the "best" pivot is often an intractable problem, so heuristics
//! are used.

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
    /// * `rows`: Rows from the start of the decomposition, but partially modified: some values in
    /// the lower rows (i < k) have been eliminated to move toward an upper triangular matrix.
    /// * `k`: Current row being "processed", meaning that the next pivot will be permuted to index
    /// `(k, k)`. Items at indices lower than `k` from the other arguments should typically not be
    /// considered.
    fn choose_pivot<T>(
        &self,
        nnz_row: &[usize], nnz_column: &[usize],
        rows: &[Vec<(usize, T)>],
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
        row_counts: &[usize], column_counts: &[usize],
        rows: &[Vec<(usize, T)>],
        k: usize,
    ) -> (usize, usize) {
        let m = rows.len();
        debug_assert!(m > 0);
        debug_assert!(row_counts.len() == m);
        debug_assert!(column_counts.len() == m);
        debug_assert!(row_counts[k..].iter().all(|&count| count > 0));
        debug_assert!(column_counts[k..].iter().all(|&count| count > 0));
        debug_assert!(rows.iter().all(|row| row.is_sorted_by_key(|&(j, _)| j)));

        // TODO(ENHANCEMENT): This is a very slow O(log(nnz(B)) * nnz(B)) search, see #15.
        let mut pairs = rows.iter()
            .enumerate()
            .skip(k)
            .flat_map(|(i, row)| {
                let first_relevant_index = match row.binary_search_by_key(&k, |&(j, _)| j) {
                    Ok(index) | Err(index) => index,
                };

                row[first_relevant_index..].iter().map(move |&(j, _)| (i, j))
            })
            .collect::<Vec<_>>();

        // We prefer to do row swaps, so try to find a low (= close to k) column first.
        // This could be either stable sorting (to keep row ordering from above) or unstable sorting
        // by the entire reversed tuple.
        pairs.sort_by_key(|&(_, j)| j);

        pairs
            .into_iter()
            .min_by_key(|&(i, j)| (row_counts[i] - 1) * (column_counts[j] - 1))
            .unwrap()
    }
}
