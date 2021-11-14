//! # Pivoting during LU
//!
//! Choosing the right column during LU decomposition reduces fill-in (and perhaps supports
//! numerical stability). Choosing the "best" pivot is often an intractable problem, so heuristics
//! are used.

use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::{FullPermutation, Permutation};
use relp_num::NonZero;

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
    /// * `columns`: Columns from the start of the decomposition, but partially modified: some
    /// values in the lower rows (i < k) have been eliminated to move toward an upper triangular
    /// matrix.
    /// * `k`: Current row being "processed", meaning that the next pivot will be permuted to index
    /// `(k, k)`. Items at indices lower than `k` from the other arguments should typically not be
    /// considered.
    fn choose_pivot<T: NonZero>(
        &self,
        k_from_below: usize, k_from_above: usize,
        nnz_row: &[usize], nnz_column: &[usize],
        row_permutation: &FullPermutation, column_permutation: &FullPermutation,
        columns: &[Vec<(usize, T)>],
    ) -> (usize, usize);
}

/// Markowitz's pivot rule minimizes `(nnz(row) - 1) * (nnz(column) - 1)` at each step.
///
/// If possible, the row and column are chosen to minimize the amount of swapping that needs to
/// happen in the decomposition.
pub(super) struct Markowitz;
impl PivotRule for Markowitz {
    fn new() -> Self {
        Self
    }

    fn choose_pivot<T: NonZero>(
        &self,
        k_from_below: usize, k_from_above: usize,
        row_counts: &[usize], column_counts: &[usize],
        row_permutation: &FullPermutation, column_permutation: &FullPermutation,
        columns: &[Vec<(usize, T)>],
    ) -> (usize, usize) {
        let m = columns.len();
        debug_assert!(m > 0);
        debug_assert!(row_counts.len() == m);
        debug_assert!(column_counts.len() == m);
        debug_assert!((k_from_below..=k_from_above).all(|i| row_counts[row_permutation.backward(i)] > 0));
        debug_assert!((k_from_below..=k_from_above).all(|j| column_counts[column_permutation.backward(j)] > 0));

        // TODO(ENHANCEMENT): This is a very slow O(nnz(B))  search, see #15.

        // Old row and column indices
        (k_from_below..=k_from_above)
            .map(|j| column_permutation.backward(j))
            .flat_map(|j| {
                columns[j].iter()
                    .filter(|(_, v)| v.is_not_zero())
                    .filter(|&&(i, _)| row_permutation[i] >= k_from_below && row_permutation[i] <= k_from_above)
                    .map(move |&(i, _)| (i, j))
            })
            .min_by_key(|&(i, j)| (row_counts[i] - 1) * (column_counts[j] - 1))
            .unwrap()
    }
}

#[derive(Eq, PartialEq, Clone, Debug)]
pub(super) struct NonZeroCounter {
    /// Number of non zero values in each row
    pub row: Vec<usize>,
    /// Number of non zero values in each column
    pub column: Vec<usize>,

    /// Rows containing exactly one non zero
    pub row_candidates: Vec<usize>,
    /// Column containing exactly one non zero
    pub column_candidates: Vec<usize>,
}

impl NonZeroCounter {
    /// Conduct an initial count the number of nonzero values in each row and column.
    ///
    /// Requires iterating over all values once.
    pub (super) fn new<T, S>(columns: &[Vec<(usize, T)>], rows: &[Vec<(usize, S)>]) -> Self {
        let n = columns.len();
        debug_assert!(n > 0);
        debug_assert!(columns.iter().all(|row| row.iter().all(|&(i, _)| i < n)));

        let (nnz_row, initial_row_candidates) = Self::make(rows);
        let (nnz_column, initial_column_candidates) = Self::make(columns);

        debug_assert_eq!(nnz_column.len(), n);
        debug_assert_eq!(nnz_row.len(), n);
        // We should be working with an invertible matrix
        debug_assert!(nnz_column.iter().all(|&count| count > 0));
        debug_assert!(nnz_row.iter().all(|&count| count > 0));

        Self {
            row: nnz_row,
            column: nnz_column,
            row_candidates: initial_row_candidates,
            column_candidates: initial_column_candidates,
        }
    }

    fn make<T>(columns: &[Vec<(usize, T)>]) -> (Vec<usize>, Vec<usize>) {
        let n = columns.len();

        let mut nnz_column = vec![0; n];
        let mut initial_column_candidates = Vec::new();

        // We fill the items back to front such that the candidates are increasing when popping off
        // the back. This makes debugging a bit easier, as the permutation will be closer to the
        // identity.
        for (j, column) in columns.iter().enumerate().rev() {
            nnz_column[j] = column.len();
            if column.len() == 1 {
                initial_column_candidates.push(j);
            }
        }

        (nnz_column, initial_column_candidates)
    }
}

#[cfg(test)]
mod test {
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::decomposition::build_index;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::decomposition::pivoting::NonZeroCounter;

    #[test]
    fn test_identity() {
        let columns = vec![vec![(0, 1)], vec![(1, 1)]];
        let rows = build_index(&columns);
        let counter = NonZeroCounter::new(&columns, &rows);
        assert_eq!(counter, NonZeroCounter {
            row: vec![1, 1],
            column: vec![1, 1],
            row_candidates: vec![1, 0],
            column_candidates: vec![1, 0],
        })
    }

    #[test]
    fn test_offdiagonal() {
        let columns = vec![vec![(0, 1)], vec![(0, 1), (1, 1)]];
        let rows = build_index(&columns);
        let counter = NonZeroCounter::new(&columns, &rows);
        assert_eq!(counter, NonZeroCounter {
            row: vec![2, 1],
            column: vec![1, 2],
            row_candidates: vec![1],
            column_candidates: vec![0],
        })
    }
}
