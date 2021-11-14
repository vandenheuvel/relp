use core::mem;
/// # LU Decomposition
///

use std::cmp::Ordering;

use itertools::repeat_n;

use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::decomposition::pivoting::{Markowitz, NonZeroCounter, PivotRule};
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::LUDecomposition;
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::{FullPermutation, Permutation};
use crate::algorithm::two_phase::tableau::inverse_maintenance::ops;
use crate::data::linear_algebra::SparseTuple;
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::decomposition::row_subtraction::{subtract_rows, get_column};
use relp_num::NonZero;

mod pivoting;
mod row_subtraction;

impl<F> LUDecomposition<F>
where
    F: ops::Field + ops::FieldHR,
{
    /// Compute the factorization `PBQ = LU`.
    ///
    /// # Arguments
    ///
    /// * `columns`: A column major representation of the basis columns.
    #[must_use]
    pub fn decompose(mut columns: Vec<Vec<(usize, F)>>) -> Self {
        debug_assert!(columns.iter().all(|column| column.iter().is_sorted_by_key(|&(i, _)| i)));
        let m = columns.len();
        debug_assert!(m > 1, "Problems should have at least two unsolved variables, was {}", m);

        let mut structures = create_initial_structures(&columns, m);

        let top_left_triangle_size = top_left_upper_triangle(&mut structures, &mut columns);

        let size_left = m - top_left_triangle_size;
        structures.upper_diagonal.extend(repeat_n(Default::default(), size_left));
        let highest_nucleus_index = bottom_right_upper_triangle(&mut structures, &mut columns, top_left_triangle_size, m);

        nucleus(&mut structures, &mut columns, top_left_triangle_size, highest_nucleus_index);

        index_updates(&mut structures, &mut columns, top_left_triangle_size, highest_nucleus_index, m);

        let mut upper_triangular = columns;
        let Structures {
            mut lower_triangular,
            upper_diagonal,
            row_permutation,
            column_permutation,
            ..
        } = structures;

        // TODO(PERFORMANCE): Make sorting unnecessary by adapting other triangular solve algorithms
        for column in &mut upper_triangular {
            column.sort_unstable_by_key(|&(i, _)| i);
        }
        for column in &mut lower_triangular {
            column.sort_unstable_by_key(|&(i, _)| i);
        }

        Self::new(
            row_permutation,
            column_permutation,
            lower_triangular,
            upper_triangular,
            upper_diagonal,
        )
    }
}

struct Structures<F> {
    lower_triangular: Vec<Vec<(usize, F)>>,
    upper_diagonal: Vec<F>,

    row_major_index: Vec<Vec<(usize, usize)>>,

    index_scatter: Vec<Option<usize>>,
    non_zero: NonZeroCounter,

    row_permutation: FullPermutation,
    column_permutation: FullPermutation,
}

fn create_initial_structures<F>(columns: &Vec<Vec<(usize, F)>>, m: usize) -> Structures<F>
where
    F: Clone,
{
    // Column-major and indexed by new indices from the start. We keep columns indexed by the
    // old indices until the columns are not touched again.
    // TODO(OPTIMIZATION): Avoid creating all these values that will be erased
    let lower_triangular = vec![Vec::with_capacity(0); m];
    let upper_diagonal = Vec::with_capacity(m);
    // The `columns` structure will be the upper triangular part.

    // Supporting data structures
    // Map from row to (column, data_index) list
    let row_major_index = build_index(&columns);
    // TODO(PERFORMANCE): These Option<usize> values use too much space, how about using isize?
    // TODO(PERFORMANCE): Reuse this workspace.
    let index_scatter = vec![None; m];
    // Indexed by the old indices
    let non_zero = NonZeroCounter::new(&columns, &row_major_index);

    // Permute from old to new
    let row_permutation = FullPermutation::identity(m);
    let column_permutation = FullPermutation::identity(m);

    Structures {
        lower_triangular,
        upper_diagonal,

        row_major_index,

        index_scatter,
        non_zero,

        row_permutation,
        column_permutation,
    }
}

fn build_index<T>(columns: &[Vec<SparseTuple<T>>]) -> Vec<Vec<(usize, usize)>> {
    let n = columns.len();
    let mut rows = vec![Vec::new(); n];

    for (j, column) in columns.iter().enumerate() {
        for (data_index, &(i, _)) in column.iter().enumerate() {
            rows[i].push((j, data_index));
        }
    }

    return rows
}

fn top_left_upper_triangle<F>(
    Structures {
        upper_diagonal,
        row_major_index,
        non_zero,
        row_permutation,
        column_permutation,
        ..
    }: &mut Structures<F>,
    columns: &mut Vec<Vec<(usize, F)>>,
) -> usize
where
    F: ops::Field + ops::FieldHR,
{
    // Tracks the number of rows/columns already selected in this triangle
    let mut k_from_below = 0;
    while let Some(pivot_column) = non_zero.column_candidates.pop() {
        // There is still a pivot column with exactly one value (the pivot) in the active part
        // of the matrix

        // We search for the index of the pivot row in the column.
        // TODO(PERFORMANCE): Can this scan be avoided? Maybe at least for all the initial column candidates? How about with the rows?
        let (pivot_data_index, pivot_row) = columns[pivot_column].iter()
            .enumerate()
            .map(|(data_index, &(i, _))| (data_index, i))
            .find(|&(_, i)| row_permutation[i] >= k_from_below)
            .expect("The pivot should exist.");

        // Update the permutations
        column_permutation.swap_inverse(column_permutation[pivot_column], k_from_below);
        row_permutation.swap_inverse(row_permutation[pivot_row], k_from_below);

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

        // The column major structure is modified, which results in inconsistency with the row
        // major index. As this column is never modified or read again, so that's okay.
        let (_i, pivot) = columns[pivot_column].remove(pivot_data_index);
        debug_assert_eq!(_i, pivot_row);
        upper_diagonal.push(pivot);

        for (i, _) in &mut columns[pivot_column] {
            // Updating the index is the last edit made to the column, it should not be
            // interacted with again.
            row_permutation.forward_ref(i);
            debug_assert!(*i < k_from_below, "The element should be above the pivot");
        }

        // Update the nonzero counts
        // Note that no further row count updating needs to happen, as any non-pivot rows have
        // already been chosen. After sorting, there should be no non zeros below the pivot.
        // TODO(PERFORMANCE): These values are not used, updating the counts could be removed?
        non_zero.column[pivot_column] = 0;
        non_zero.row[pivot_row] = 0;

        k_from_below += 1;
    }

    k_from_below
}

fn bottom_right_upper_triangle<F>(
    Structures {
        upper_diagonal,
        row_major_index,
        non_zero,
        row_permutation,
        column_permutation,
        ..
    }: &mut Structures<F>,
    columns: &mut Vec<Vec<(usize, F)>>,
    top_left_triangle_size: usize,
    m: usize,
) -> usize
where
    F: ops::Field + ops::FieldHR,
{
    // Tracks the number of rows/columns already selected in this triangle
    let mut k_from_above = m - 1;
    // How many element were removed from each row.
    //
    // The values in `non_zero.row` can't be modified, as they are needed for the pivoting
    // decisions in the third stage.
    let mut removed_per_row = vec![0; m];
    while let Some(pivot_row) = non_zero.row_candidates.pop() {
        if row_permutation[pivot_row] < top_left_triangle_size {
            // The row already was used while building U
            continue;
        }

        // Find the column of this pivot
        let (pivot_column, pivot_data_index) = row_major_index[pivot_row].iter()
            .map(|&t| t)
            .find(|&(j, _)| (top_left_triangle_size..=k_from_above).contains(&column_permutation[j]))
            .expect("The pivot should exist.");

        // Move the pivot to the data structure for the diagonal
        let (_i, pivot_ref) = &mut columns[pivot_column][pivot_data_index];
        debug_assert_eq!(*_i, pivot_row);
        let pivot = mem::take(pivot_ref);
        upper_diagonal[k_from_above] = pivot;

        // Update the permutations
        row_permutation.swap_inverse(row_permutation[pivot_row], k_from_above);
        column_permutation.swap_inverse(column_permutation[pivot_column], k_from_above);

        // Update the row counts for the active part of the matrix `k_from_above` is excluded
        // because it has been filled in this iteration
        let active_range = top_left_triangle_size..k_from_above;
        for &(i, _) in &columns[pivot_column] {
            // TODO(PERFORMANCE): It is possible to remove this if statement and always subtract
            //  if the value is not set to zero at the end?
            if active_range.contains(&row_permutation[i]) {
                removed_per_row[i] += 1;
                debug_assert!(
                    non_zero.row[i] - removed_per_row[i] > 0,
                    "Each row in the active matrix should contain at least one non zero",
                );
                if non_zero.row[i] - removed_per_row[i] == 1 {
                    // Only one row element left in this row in the active part of the matrix
                    non_zero.row_candidates.push(i);
                }
            }
        }

        // Update the nonzero counts
        // TODO(PERFORMANCE): The column count is not used, just remove?
        non_zero.column[pivot_column] = 0;

        k_from_above -= 1;
    }

    k_from_above
}

fn nucleus<F>(
    Structures {
        lower_triangular,
        upper_diagonal,

        row_major_index,

        index_scatter,
        non_zero,

        row_permutation,
        column_permutation,
    }: &mut Structures<F>,
    columns: &mut Vec<Vec<(usize, F)>>,
    top_left_triangle_size: usize,
    highest_nucleus_index: usize,
)
where
    F: ops::Field + ops::FieldHR,
{
    let pivot_rule = Markowitz::new();
    let nucleus = top_left_triangle_size..=highest_nucleus_index;
    for k in nucleus.clone() {
        // Choose the next pivot
        let (pivot_row, pivot_column) = pivot_rule.choose_pivot(
            k, highest_nucleus_index,
            &non_zero.row, &non_zero.column,
            &row_permutation, &column_permutation,
            &columns,
        );

        // Update the permutations
        row_permutation.swap_inverse(row_permutation[pivot_row], k);
        column_permutation.swap_inverse(column_permutation[pivot_column], k);

        let (columns_before_pivot, pivot_and_after) = columns.split_at_mut(pivot_column);
        let (pivot_column_data, columns_after_pivot) = pivot_and_after.split_first_mut().unwrap();
        // TODO(PERFORMANCE): Get the data_index or pivot_ref directly from the pivot selection
        let (_, pivot_ref) = pivot_column_data.iter_mut()
            .find(|(i, _)| *i == pivot_row)
            .unwrap();
        let pivot = mem::take(pivot_ref);

        let nr_of_items_below_pivot = non_zero.column[pivot_column] - 1;
        let mut lower_column = Vec::with_capacity(nr_of_items_below_pivot);
        let active_range_after_this_iteration = (k + 1)..=highest_nucleus_index;

        // Zero the values below the pivot
        for &(i, ref value) in &*pivot_column_data {
            match row_permutation[i].cmp(&k) { // Column contains zeros below k_from_above
                Ordering::Less => {
                    // Row was already chosen and is no longer part of the active matrix
                }
                Ordering::Equal => {
                    // The pivot row
                    debug_assert_eq!(i, pivot_row);
                    debug_assert_eq!(value, &Default::default());

                    // Update the column counts
                    for &(j, data_index) in &row_major_index[pivot_row] {
                        // TODO(PERFORMANCE): Maybe this if statement can be removed, the inactive column
                        //  counts are not used anyway.
                        if active_range_after_this_iteration.contains(&column_permutation[j]) {
                            // Column is still active

                            let j_column = get_column(j, pivot_column, columns_before_pivot, columns_after_pivot);
                            if j_column[data_index].1.is_not_zero() {
                                non_zero.column[j] -= 1;
                            }
                        }
                    }
                }
                Ordering::Greater => {
                    // Row is still active

                    if !value.is_not_zero() {
                        // A value might be zero because previous operations created fill-in that
                        // was subsequently "erased" again, we just kept the zero value to not have
                        // to modify the row_major_index.
                        continue;
                    }

                    // Remove a count for the element being eliminated
                    non_zero.row[i] -= 1;

                    // Eliminate this value that is "below the pivot" in the new indexing
                    // TODO(PERFORMANCE): `value` is no longer used after this, can its storage
                    //  be reused?
                    let ratio = value / &pivot;

                    let (rows_before_pivot, pivot_and_after) = row_major_index.split_at_mut(pivot_row);
                    let (pivot_row_data, rows_after_pivot) = pivot_and_after.split_first_mut().unwrap();

                    let edit_row = if i < pivot_row {
                        &mut rows_before_pivot[i]
                    } else {
                        &mut rows_after_pivot[i - pivot_row - 1]
                    };

                    // Subtract ratio * pivot row from this row below the pivot
                    subtract_rows(
                        &ratio,
                        i,
                        edit_row,
                        pivot_row,
                        pivot_row_data,
                        columns_before_pivot,
                        columns_after_pivot,
                        pivot_column,
                        &column_permutation,
                        // We update also columns in the range `(highest_nucleus_index + 1)..m` range.
                        (k + 1)..,
                        highest_nucleus_index,
                        index_scatter,
                        non_zero,
                    );

                    lower_column.push((i, ratio));
                }
            }
        }
        debug_assert_eq!(lower_column.len(), nr_of_items_below_pivot);

        upper_diagonal[k] = pivot;
        lower_triangular[k] = lower_column;

        // Last edits to the column of U. The column should no longer be accessed.
        pivot_column_data.drain_filter(|(i, v)| {
            match row_permutation[*i].cmp(&k) {
                Ordering::Less => {
                    // Above the pivot, stays but index is updated
                    if v.is_not_zero() {
                        row_permutation.forward_ref(i);
                        false
                    } else {
                        true
                    }
                }
                Ordering::Equal => {
                    // The pivot, is already extracted
                    debug_assert_eq!(*i, pivot_row);
                    debug_assert_eq!(v, &F::default());
                    true
                }
                Ordering::Greater => {
                    // Below the pivot, was eliminated
                    true
                }
            }
        });

        // TODO(PERFORMANCE): Count is not used, remove these statements?
        non_zero.row[pivot_row] = 0;
        non_zero.column[pivot_column] = 0;

        for (i, row) in row_major_index.iter().enumerate() {
            for &(j, data_index) in row {
                if ((k + 1)..=highest_nucleus_index).contains(&column_permutation[j]) {
                    debug_assert_eq!(columns[j][data_index].0, i);
                }
            }
        }
    }
}

fn index_updates<F>(
    Structures {
        lower_triangular,
        row_permutation,
        column_permutation,
        ..
    }: &mut Structures<F>,
    columns: &mut Vec<Vec<(usize, F)>>,
    top_left_triangle_size: usize,
    highest_nucleus_index: usize,
    m: usize,
)
where
    F: NonZero + Clone,
{
    let nucleus_range = top_left_triangle_size..=highest_nucleus_index;

    // Update the lower triangle row indices to their final position (this can only happen once the
    // nucleus is completely factorized)
    for column in &mut lower_triangular[nucleus_range] {
        row_permutation.forward_unsorted(column);
    }

    // Sort the columns to the right spot
    let mut upper_triangular = vec![Vec::with_capacity(0); m];
    for (j, column) in columns.drain(..).enumerate() {
        upper_triangular[column_permutation[j]] = column;
    }
    *columns = upper_triangular;

    // Update the indices in the columns of the bottom-right triangle (this can only happen once the
    // nucleus is completely factorized)
    let start_of_bottom_right = highest_nucleus_index + 1;
    for (j, column) in columns.iter_mut().enumerate().skip(start_of_bottom_right) {
        column.drain_filter(|(i, v)| {
            debug_assert!(row_permutation[*i] <= j);
            let keep = row_permutation[*i] < j && v.is_not_zero();
            !keep
        });
        row_permutation.forward_unsorted(column);
    }
}

#[cfg(test)]
mod test {
    use relp_num::{NonZero, R64, R8, Rational64, Rational8, RB};

    use crate::algorithm::two_phase::matrix_provider::column::{Column, DenseSliceIterator, SparseSliceIterator};
    use crate::algorithm::two_phase::matrix_provider::column::identity::IdentityColumn;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::BasisInverse;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::decomposition::build_index;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::LUDecomposition;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::FullPermutation;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::ColumnComputationInfo;
    use crate::data::linear_algebra::vector::{SparseVector, Vector};

    #[test]
    fn test_build_index() {
        let columns = vec![vec![(0, 1)], vec![(0, 1), (1, 1)]];
        let rows = build_index(&columns);
        assert_eq!(rows, vec![vec![(0, 0), (1, 0)], vec![(1, 1)]]);
    }

    fn transpose<T: Clone>(rows: Vec<Vec<(usize, T)>>) -> Vec<Vec<(usize, T)>> {
        let m = rows.len();
        let mut columns = vec![vec![]; m];

        for (i, row) in rows.into_iter().enumerate() {
            for (j, value) in row {
                columns[j].push((i, value));
            }
        }

        columns
    }

    fn test_identity(n: usize) {
        let columns = (0..n).map(|i| vec![(i, R8!(1))]).collect();
        let result = LUDecomposition::<Rational8>::decompose(columns);
        for i in 0..n {
            let c = result.left_multiply_by_basis_inverse(IdentityColumn::new(i).iter());
            assert_eq!(c.into_column(), SparseVector::standard_basis_vector(i, n), "{}", i);
        }
    }

    #[test]
    fn identity_2() {
        let m = 2;
        test_identity(m);

        let columns = Vec::from([vec![(0, RB!(1))], vec![(1, RB!(1))]]);
        let result = LUDecomposition::decompose(columns);
        let expected = LUDecomposition {
            row_permutation: FullPermutation::identity(m),
            column_permutation: FullPermutation::identity(m),
            lower_triangular: vec![vec![]; m],
            upper_triangular: vec![vec![]; m],
            upper_diagonal: vec![RB!(1), RB!(1)],
            updates: vec![],
        };

        assert_eq!(result, expected);
    }

    #[test]
    fn identity_3() {
        test_identity(3);
    }

    #[test]
    fn identity_4() {
        test_identity(4);
    }

    #[test]
    fn offdiagonal_2_upper() {
        let rows = vec![
            vec![(0, R8!(1)), (1, R8!(1))],
            vec![(1, R8!(1))],
        ];
        let columns = transpose(rows);
        let result = LUDecomposition::decompose(columns);
        let expected = LUDecomposition {
            row_permutation: FullPermutation::identity(2),
            column_permutation: FullPermutation::identity(2),
            lower_triangular: vec![vec![], vec![]],
            upper_triangular: vec![vec![], vec![(0, R8!(1))]],
            upper_diagonal: vec![R8!(1), R8!(1)],
            updates: vec![],
        };

        assert_eq!(result, expected);
    }

    #[test]
    fn offdiagonal_2_lower() {
        let rows = vec![
            vec![(0, R8!(1))],
            vec![(0, R8!(1)), (1, R8!(1))],
        ];
        let columns = transpose(rows);
        let result = LUDecomposition::decompose(columns);
        let expected = LUDecomposition {
            row_permutation: FullPermutation::new(vec![1, 0]),
            column_permutation: FullPermutation::new(vec![1, 0]),
            lower_triangular: vec![vec![], vec![]],
            upper_triangular: vec![vec![], vec![(0, R8!(1))]],
            upper_diagonal: vec![R8!(1), R8!(1)],
            updates: vec![],
        };

        assert_eq!(result, expected);
    }

    #[test]
    fn offdiagonal_2_both() {
        let rows = vec![
            vec![(0, R8!(1)), (1, R8!(1))],
            vec![(0, R8!(1))],
        ];
        let columns = transpose(rows);
        let result = LUDecomposition::decompose(columns);
        let expected = LUDecomposition {
            row_permutation: FullPermutation::identity(2),
            column_permutation: FullPermutation::new(vec![1, 0]),
            lower_triangular: vec![vec![], vec![]],
            upper_triangular: vec![vec![], vec![(0, R8!(1))]],
            upper_diagonal: vec![R8!(1), R8!(1)],
            updates: vec![],
        };

        assert_eq!(result, expected);
    }

    #[test]
    fn wikipedia_example() {
        let rows = vec![
            vec![(0, R8!(4)), (1, R8!(3))],
            vec![(0, R8!(6)), (1, R8!(3))],
        ];
        let columns = transpose(rows);
        let result = LUDecomposition::decompose(columns);
        let expected = LUDecomposition {
            row_permutation: FullPermutation::identity(2),
            column_permutation: FullPermutation::identity(2),
            lower_triangular: vec![vec![(1, R8!(3, 2))], vec![]],
            upper_triangular: vec![vec![], vec![(0, R8!(3))]],
            upper_diagonal: vec![R8!(4), R8!(-3, 2)],
            updates: vec![],
        };

        assert_eq!(result, expected);
    }

    #[test]
    fn wikipedia_example2() {
        let rows = vec![
            vec![(0, R8!(-1)), (1, R8!(3, 2))],
            vec![(0, R8!(1)), (1, R8!(-1))],
        ];
        let columns = transpose(rows);
        let result = LUDecomposition::decompose(columns);
        let expected = LUDecomposition {
            row_permutation: FullPermutation::identity(2),
            column_permutation: FullPermutation::identity(2),
            lower_triangular: vec![vec![(1, R8!(-1))], vec![]],
            upper_triangular: vec![vec![], vec![(0, R8!(3, 2))]],
            upper_diagonal: vec![R8!(-1), R8!(1, 2)],
            updates: vec![],
        };

        assert_eq!(result, expected);

        assert_eq!(
            expected.left_multiply_by_basis_inverse(IdentityColumn::new(0).iter()).into_column(),
            SparseVector::new(vec![(0, R8!(2)), (1, R8!(2))], 2),
        );
        assert_eq!(
            expected.left_multiply_by_basis_inverse(IdentityColumn::new(1).iter()).into_column(),
            SparseVector::new(vec![(0, R8!(3)), (1, R8!(2))], 2),
        );
    }

    pub fn to_columns<const M: usize>(rows: &[[i32; M]; M]) -> Vec<Vec<(usize, Rational64)>> {
        let mut columns = vec![vec![]; M].into_iter().collect::<Vec<_>>();

        for (i, row) in rows.into_iter().enumerate() {
            for (j, v) in row.into_iter().enumerate() {
                if v.is_not_zero() {
                    columns[j].push((i, R64!(*v)));
                }
            }
        }

        columns
    }

    fn test_matrix<const M: usize>(rows: [[i32; M]; M]) {
        let columns = to_columns(&rows);
        let result = LUDecomposition::decompose(columns.clone());
        for (j, column) in columns.iter().enumerate() {
            assert_eq!(
                result.left_multiply_by_basis_inverse(SparseSliceIterator::new(column)).into_column(),
                SparseVector::standard_basis_vector(j, M),
                "Column {}", j,
            );
        }
        for (i, row) in rows.into_iter().enumerate() {
            assert_eq!(
                result.right_multiply_by_basis_inverse(DenseSliceIterator::new(&row)),
                SparseVector::standard_basis_vector(i, M),
                "Row {}", i,
            );
        }
    }

    #[test]
    fn test_triangle() {
        test_matrix([
            [2,  3,  0,  5],
            [0,  7, 11, 13],
            [0,  0, 17, 19],
            [0,  0,  0, 23],
        ]);
    }

    #[test]
    fn test_has_first_two_stages() {
        test_matrix([
            [ 2,  3,  0,  5],
            [ 0, 31, 11, 13],
            [ 0, 29, 17, 19],
            [ 0,  0, 37, 23],
        ]);
    }

    #[test]
    fn test_has_all_three_stages() {
        test_matrix([
            [2,  3,  0,  5],
            [0,  7, 11, 13],
            [0, 29, 17, 19],
            [0,  0,  0, 23],
        ]);
    }

    #[test]
    fn test_has_last_two_stages() {
        test_matrix([
            [ 2,  3,  0,  5],
            [31,  0, 11, 13],
            [ 0, 29, 17, 19],
            [ 0,  0,  0, 23],
        ]);
    }

    #[test]
    fn test_3x3_1() {
        test_matrix([
            [ 2,  3,  0],
            [ 5,  0, 11],
            [23, 29,  0],
        ]);
    }

    #[test]
    fn test_3x3_2() {
        test_matrix([
            [ 0,  0,  2],
            [ 3,  5,  7],
            [11, 13, 17],
        ]);
    }

    #[test]
    fn test_4x4_1() {
        test_matrix([
            [ 2,  3,  0,  5],
            [ 5,  0, 11, 13],
            [23, 29,  0, 57],
            [31, 37, 41,  0],
        ]);
    }

    #[test]
    fn test_4x4_2() {
        test_matrix([
            [-101,    0,    0,   -5],
            [-110,  -81,    0,    0],
            [   0,    0,    1, -111],
            [   0,   93,   69,    0],
        ]);
    }

    #[test]
    fn test_4x4_3() {
        test_matrix([
            [  0,   0, -84, 122],
            [  0,   9,   0,   0],
            [-39, 115,   0,  57],
            [  0, -12, 121,   0],
        ]);
    }

    #[test]
    fn test_4x4_4() {
        test_matrix([
            [19, 17,  0, 13],
            [ 0,  0, 11,  0],
            [ 0,  7,  5,  0],
            [ 3,  0,  0,  2],
        ]);
    }

    #[test]
    fn test_4x4_5() {
        test_matrix([
            [19, 17,  0, 13],
            [11,  0,  7,  0],
            [ 0,  5,  3,  0],
            [ 0,  0,  0,  2],
        ]);
    }

    #[test]
    fn test_4x4_6() {
        test_matrix([
            [2,  0,  0,  0],
            [0,  3,  5,  7],
            [0, 11, 13,  0],
            [0,  0,  0, 17],
        ]);
    }

    #[test]
    fn test_5x5_banded() {
        test_matrix([
            [2,  3,  0,  0,  0],
            [5,  7, 11,  0,  0],
            [0, 29, 13, 57,  0],
            [0,  0, 41, 17,  0],
            [0,  0,  0, 53, 51],
        ]);
    }

    #[test]
    fn test_5x5_1() {
        test_matrix([
            [29, 23,  0, 19, 0],
            [ 0,  0, 17, 13, 0],
            [ 0,  0,  7,  0, 0],
            [ 5,  0,  0,  3, 0],
            [ 0,  0,  0,  0, 2],
        ]);
    }

    #[test]
    fn test_5x5_2() {
        test_matrix([
            [ 2,  0,  0,  0,  0],
            [ 0,  3,  0,  5,  7],
            [ 0,  0, 11, 13,  0],
            [ 0, 17, 19,  0,  0],
            [ 0,  0,  0, 23, 29],
        ]);
    }

    #[test]
    fn test_5x5_3() {
        test_matrix([
            [ 2,  3,  0,  5,  7],
            [ 5,  0, 11, 13, 17],
            [23, 29,  0, 57, 59],
            [31, 37, 41,  0,  0],
            [43,  0, 47, 53, 51],
        ]);
    }

    #[test]
    fn test_5x5_4() {
        test_matrix([
            [ 2,  3,  0,  5,  7],
            [ 5,  0, 11, 13, 17],
            [23, 29,  0, 57, 59],
            [31, 37, 41,  0,  0],
            [43,  0, 47, 53, 51],
        ]);
    }

    #[test]
    fn test_5x5_5() {
        test_matrix([
            [   0,   54,   43,    0,   84],
            [   4,    0,    0,    0,    0],
            [   0, -111,  -27,    0,  -86],
            [  -6,    0,    0,   17,  -62],
            [-109,    0,    0,    0, -104],
        ]);
    }

    #[test]
    fn test_5x5_6() {
        test_matrix([
            [ -71, -124,    0,    0, -108],
            [   0,   66, -121,  -74,  -53],
            [   0,  104,    0,    0,    0],
            [   0,   55,    0,    1,   -3],
            [  93,    0,    0,    0,  104],
        ]);
    }

    #[test]
    fn test_5x5_7() {
        test_matrix([
            [ 2,  0,  0,  0,  0],
            [ 0,  3,  5,  0,  0],
            [ 0,  7,  0, 11, 13],
            [ 0,  0, 17, 19,  0],
            [ 0,  0,  0, 23, 29],
        ]);
    }

    #[test]
    fn test_6x6_1() {
        test_matrix([
            [   0,    0,    0,  -25,    0,    0],
            [ -15,   79,    0,    0,    0,    0],
            [   0,    0,    0,    0,    0,   14],
            [   0,    0,    0, -114,  -61,    0],
            [   0,    0,  109,    0,    0, -126],
            [  46,    0,    0,   50,   21,    0],
        ]);
    }

    #[test]
    fn test_6x6_2() {
        test_matrix([
            [   0,    0,  -26,  -68,   84,    0],
            [-125,   43,    0,    0,    0,  -63],
            [   0,    0,    1,   90,    0,    0],
            [   0,  -81,    0,    0,    0,    0],
            [ -15,    0,    0,  -81,    0,    0],
            [   0,  -12,    0,    0,    0,    1],
        ]);
    }

    #[test]
    fn test_10x10_1() {
        test_matrix([
            [   0,    0,    0,    0,    0,    0,  -60,    0,  -10,    0],
            [   0,    0,    0,    0,    0,    0,    0,  -84,    0,    0],
            [   0, -105,    0,    0,    0,    0,    0,    0,    0,    0],
            [   0,    0,    0,  -25,    0,    0,    0,    0,  116,    0],
            [   0,    0,    0,    0,  -18,    0,    0,    0,    0,    0],
            [   0,    0,    0,  -72,    0,    0,    0,    0,    0,    0],
            [   0,    0,   16,   48,    0,    0,    0,    0,    0,    0],
            [ -57,    0,    0,  -88,  107,    0,    0,    0,    0,    0],
            [-122, -108,    0,    0,    0,   91,    0,    0, -127,    0],
            [   0,   85,    0,    0,  106,    0,    0,    0,    0, -121],
        ]);
    }

    #[test]
    fn test_11x11_1() {
        test_matrix([
            [   0,    0,    0,    0,    0,    0,    0,    0,    0,  -13,    0],
            [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0, -122],
            [   0,    0,    0,    0,    0,  102,   82,    0,    0,    0,   13],
            [   0,    0,    0,    0,    0, -107,  -39,    0,    0,    0,    0],
            [   0,    0,    0,    0,    0,    0,  -39,   48, -113,    0,    0],
            [  24,    0,    0,    0,    0,    0,    0,    0,    0,  -93, -120],
            [-111,    0,    0,  -81,    0,    0,    0,    0,    0,    0,    0],
            [   0,    0,    0,    0,    0,   82,    0,    0,   76,    0,    0],
            [   0,    0,  -51,    0,    0,    0,  126,    0,    0,    0, -105],
            [   0,  118,    0,    0,    0,    0,    0,    0,    0,    0,   27],
            [   0,    0,  120,    0,  -31,    0,    0,    0,    0,    0,    0],
        ]);
    }
}
