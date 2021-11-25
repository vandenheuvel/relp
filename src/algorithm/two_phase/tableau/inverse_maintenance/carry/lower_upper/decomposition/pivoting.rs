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
    fn new<T>(columns: &[Vec<(usize, T)>], rows: &[Vec<(usize, usize)>]) -> Self;

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

    fn pivot_on_column_singleton<T>(
        &mut self,
        columns: &[Vec<(usize, T)>],
        row_major_index: &[Vec<(usize, usize)>],
    ) -> Option<Pivot>;

    fn finalize_pivot(&mut self);

    fn times_pivoted(&self) -> usize;

    fn into_permutations(self) -> (FullPermutation, FullPermutation);
}

pub(super) struct Pivot {
    row_index: usize,
    column_index: usize,
    data_index: usize,
    k: usize,
}

/// Markowitz's pivot rule minimizes `(nnz(row) - 1) * (nnz(column) - 1)` at each step.
///
/// If possible, the row and column are chosen to minimize the amount of swapping that needs to
/// happen in the decomposition.
pub(super) struct Markowitz {
    non_zero_counter: NonZeroCounter,
    /// Permute from old to new
    row_permutation: FullPermutation,
    column_permutation: FullPermutation,
    k: usize,
}
impl PivotRule for Markowitz {
    fn new<T>(columns: &[Vec<(usize, T)>], rows: &[Vec<(usize, usize)>]) -> Self {
        let m = columns.len();

        Self {
            non_zero_counter: NonZeroCounter::new(columns, rows),
            row_permutation: FullPermutation::identity(m),
            column_permutation: FullPermutation::identity(m),
            k: m,
        }
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

    fn pivot_on_column_singleton<T>(
        &mut self,
        columns: &[Vec<(usize, T)>],
        row_major_index: &[Vec<(usize, usize)>],
    ) -> Option<Pivot> {
        self.non_zero_counter.column_candidates.pop()
            .map(|column| {
                // We search for the index of the pivot row in the column.
                // TODO(PERFORMANCE): Can this scan be avoided?
                let (data_index, row) = columns[column].iter()
                    .enumerate()
                    .map(|(data_index, &(i, _))| (data_index, i))
                    .find(|&(_, i)| self.row_permutation[i] >= self.k)
                    .expect("The pivot should exist.");

                // Update the permutations
                self.column_permutation.swap_inverse(column_permutation[pivot_column], k_from_below);
                self.row_permutation.swap_inverse(row_permutation[pivot_row], k_from_below);

                // Update the non zero counts in the active part of the matrix.
                //
                // It is not necessary to update the row counts, because all rows where this column has
                // non zero values are in rows which were already pivoted on (that is, these rows `i`
                // have `row_permutation[i] <= k_from_below`.
                for &(j, _) in &row_major_index[pivot_row] {
                    // TODO(PERFORMANCE): It is possible to remove this if statement and always subtract
                    //  if the value is not set to zero at the end?
                    if column_permutation[j] > k_from_below {
                        non_zero.column[j] -= 1;
                        debug_assert!(
                            non_zero.column[j] > 0,
                            "Each column in the active matrix should contain at least one non zero",
                        );
                        if non_zero.column[j] == 1 {
                            // Only one item is left in the column in the active part, so this column
                            // contains the / a next pivot.
                            non_zero.column_candidates.push(j);
                        }
                    }
                }

                // Note that no further row count updating needs to happen, as any non-pivot rows have
                // already been chosen. After sorting, there should be no non zeros below the pivot.
                // TODO(PERFORMANCE): These values are not used, updating the counts could be removed?
                self.non_zero_counter.set_zero(row, column);

                Pivot { row_index: row, column_index: column, data_index }
            })
    }

    fn finalize_pivot(&mut self) {
        self.k += 1;
    }

    fn into_permutations(self) -> (FullPermutation, FullPermutation) {
        (self.row_permutation, self.column_permutation)
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

    fn set_zero(&mut self, row_index: usize, column_index: usize) {
        debug_assert_ne!(self.row[row_index], 0);
        debug_assert_ne!(self.column[column_index], 0);
        debug_assert!(!self.row_candidates.contains(&row_index));
        debug_assert!(!self.column_candidates.contains(&column_index));

        self.row[row_index] = 0;
        self.column_candidates[column_index] = 0;
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
