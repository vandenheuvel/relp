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
    fn new<T: NonZero>(
        columns: &[Vec<(usize, T)>],
        row_major_index: &[Vec<(usize, usize)>],
        row_permutation: &FullPermutation, column_permutation: &FullPermutation,
        top_left_triangle_size: usize, highest_nucleus_index: usize,
        non_zero: &NonZeroCounter,
    ) -> Self;

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
        &mut self,
        k_from_below: usize, k_from_above: usize,
        nnz_row: &[usize], nnz_column: &[usize],
        row_permutation: &FullPermutation, column_permutation: &FullPermutation,
        columns: &[Vec<(usize, T)>],
        row_major_index: &[Vec<(usize, usize)>],
        non_zero: &NonZeroCounter,
    ) -> (usize, usize);
}

/// Markowitz's pivot rule minimizes `(nnz(row) - 1) * (nnz(column) - 1)` at each step.
///
/// If possible, the row and column are chosen to minimize the amount of swapping that needs to
/// happen in the decomposition.
pub(super) struct Markowitz;
impl PivotRule for Markowitz {
    fn new<T: NonZero>(
        columns: &[Vec<(usize, T)>],
        row_major_index: &[Vec<(usize, usize)>],
        row_permutation: &FullPermutation, column_permutation: &FullPermutation,
        top_left_triangle_size: usize, highest_nucleus_index: usize,
        non_zero: &NonZeroCounter,
    ) -> Self {
        Self
    }

    fn choose_pivot<T: NonZero>(
        &mut self,
        k_from_below: usize, k_from_above: usize,
        row_counts: &[usize], column_counts: &[usize],
        row_permutation: &FullPermutation, column_permutation: &FullPermutation,
        columns: &[Vec<(usize, T)>],
        row_major_index: &[Vec<(usize, usize)>],
        non_zero: &NonZeroCounter,
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

pub(super) struct MinimumDeficiency {
    costs: Vec<Vec<(usize, usize)>>,
}
impl PivotRule for MinimumDeficiency {
    fn new<T: NonZero>(
        columns: &[Vec<(usize, T)>],
        row_major_index: &[Vec<(usize, usize)>],
        row_permutation: &FullPermutation, column_permutation: &FullPermutation,
        top_left_triangle_size: usize, highest_nucleus_index: usize,
        non_zero: &NonZeroCounter,
    ) -> Self {
        let mut costs = vec![Vec::with_capacity(0); columns.len()];

        let mut work = vec![false; columns.len()];

        let nucleus_range = top_left_triangle_size..=highest_nucleus_index;
        let column_fill_in_range = top_left_triangle_size..;

        for new_column_index in nucleus_range.clone() {
            let pivot_column = column_permutation.backward(new_column_index);
            let column = &columns[pivot_column];

            let mut c = Vec::with_capacity(non_zero.column[pivot_column]);

            for &(pivot_row, ref value) in column {
                if !value.is_not_zero() {
                    continue;
                }

                if !nucleus_range.contains(&row_permutation[pivot_row]) {
                    continue;
                }

                // Compute the cost of pivoting on (pivot_row, pivot_column)

                let mut cost = (non_zero.column[pivot_column] - 1) * (non_zero.row[pivot_row] - 1);
                debug_assert!(cost != 0 || nucleus_range.start() == nucleus_range.end(),
                    "Free pivots should have been done during triangularization",
                );

                for &(j, data_index_in_column) in &row_major_index[pivot_row] {
                    if column_fill_in_range.contains(&column_permutation[j]) {
                        // Column is still active
                        if j != pivot_column {
                            if columns[j][data_index_in_column].1.is_not_zero() {
                                work[j] = true;
                            }
                        }
                    }
                }

                for &(i, ref value) in column {
                    if value.is_not_zero() {
                        if i != pivot_row {
                            if nucleus_range.contains(&row_permutation[i]) {
                                // Row is still active, imagine it is being erased

                                for &(j, data_index_in_column) in &row_major_index[i] {
                                    if column_fill_in_range.contains(&column_permutation[j]) {
                                        // Column is still active
                                        if j != pivot_column {
                                            if columns[j][data_index_in_column].1.is_not_zero() {
                                                if work[j] {
                                                    // No new fill-in will be introduced
                                                    cost -= 1;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                for &(j, data_index_in_column) in &row_major_index[pivot_row] {
                    if column_fill_in_range.contains(&column_permutation[j]) {
                        // Column is still active
                        if j != pivot_column {
                            if columns[j][data_index_in_column].1.is_not_zero() {
                                work[j] = false;
                            }
                        }
                    }
                }
                debug_assert!(work.iter().all(|&v| !v));

                c.push((pivot_row, cost));
            }

            costs[pivot_column] = c;
        }

        Self {
            costs,
        }
    }

    fn choose_pivot<T: NonZero>(
        &mut self,
        k_from_below: usize, k_from_above: usize,
        nnz_row: &[usize], nnz_column: &[usize],
        row_permutation: &FullPermutation, column_permutation: &FullPermutation,
        columns: &[Vec<(usize, T)>],
        row_major_index: &[Vec<(usize, usize)>],
        non_zero: &NonZeroCounter,
    ) -> (usize, usize) {
        *self = Self::new(
            columns,
            row_major_index,
            row_permutation,
            column_permutation,
            k_from_below,
            k_from_above,
            non_zero,
        );
        let nucleus_range = k_from_below..=k_from_above;

        let (pivot_row, pivot_column) = (k_from_below..=k_from_above)
            .map(|j| column_permutation.backward(j))
            .flat_map(|j| {
                self.costs[j].iter()
                    .filter(|&&(i, _)| nucleus_range.contains(&row_permutation[i]))
                    .map(move |&(i, cost)| (i, j, cost))
            })
            .min_by_key(|&(_, _, cost)| cost)
            .map(|(i, j, _)| (i, j))
            .unwrap();

        (pivot_row, pivot_column)
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
