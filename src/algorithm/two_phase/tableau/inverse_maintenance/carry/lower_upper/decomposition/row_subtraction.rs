use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::decomposition::pivoting::NonZeroCounter;
use crate::algorithm::two_phase::tableau::inverse_maintenance::ops;
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::FullPermutation;
use std::ops::RangeBounds;
use relp_num::NonZero;

pub(in super) fn subtract_rows<F>(
    ratio: &F,
    edit_row_index: usize,
    edit_row: &mut Vec<(usize, usize)>,
    pivot_row_index: usize,
    pivot_row: &Vec<(usize, usize)>,
    columns_before_pivot: &mut [Vec<(usize, F)>],
    columns_after_pivot: &mut [Vec<(usize, F)>],
    pivot_column_index: usize,
    column_permutation: &FullPermutation,
    update_range: impl RangeBounds<usize>,
    highest_nucleus_index: usize,
    index_scatter: &mut Vec<Option<usize>>,
    non_zero: &mut NonZeroCounter,
)
    where
        F: ops::Field + ops::FieldHR,
{
    scatter(
        pivot_row,
        pivot_column_index, column_permutation, &update_range,
        columns_before_pivot, columns_after_pivot,
        index_scatter,
    );
    update_matches(
        ratio,
        edit_row_index, edit_row,
        pivot_column_index, column_permutation, &update_range, highest_nucleus_index,
        columns_before_pivot, columns_after_pivot,
        index_scatter,
        non_zero,
    );
    add_non_zeros(
        ratio,
        edit_row_index, edit_row, pivot_row_index, pivot_row,
        pivot_column_index, column_permutation, update_range, highest_nucleus_index,
        columns_before_pivot, columns_after_pivot,
        index_scatter,
        non_zero,
    );

    debug_assert!(
        index_scatter.iter().all(|v| v.is_none()),
        "The scatter array should be empty at the end",
    );
}

fn scatter<F: NonZero>(
    pivot_row: &Vec<(usize, usize)>,
    pivot_column_index: usize,
    column_permutation: &FullPermutation,
    update_range: &impl RangeBounds<usize>,
    columns_before_pivot: &mut [Vec<(usize, F)>],
    columns_after_pivot: &mut [Vec<(usize, F)>],
    index_scatter: &mut Vec<Option<usize>>,
) {
    for &(j, pivot_row_data_index) in pivot_row {
        if update_range.contains(&column_permutation[j]) {
            let edit_column_data_ref = if j < pivot_column_index {
                &columns_before_pivot[j]
            } else {
                debug_assert_ne!(
                    j, pivot_column_index,
                    "We should be updating the active part of the matrix, not the pivot column",
                );
                &columns_after_pivot[j - pivot_column_index - 1]
            };
            if edit_column_data_ref[pivot_row_data_index].1.is_not_zero() {
                index_scatter[j] = Some(pivot_row_data_index);
            }
        }
    }
}

fn update_matches<F>(
    ratio: &F,
    edit_row_index: usize,
    edit_row: &mut Vec<(usize, usize)>,
    pivot_column_index: usize,
    column_permutation: &FullPermutation,
    update_range: &impl RangeBounds<usize>,
    highest_nucleus_index: usize,
    columns_before_pivot: &mut [Vec<(usize, F)>],
    columns_after_pivot: &mut [Vec<(usize, F)>],
    index_scatter: &mut Vec<Option<usize>>,
    non_zero: &mut NonZeroCounter,
)
where
    F: ops::Field + ops::FieldHR,
{
    for &(j, data_index) in edit_row.iter() {
        if update_range.contains(&column_permutation[j]) {
            if let Some(pivot_data_index) = index_scatter[j] {
                // There is a value at the same index, do addition

                let j_column = get_column(j, pivot_column_index, columns_before_pivot, columns_after_pivot);

                let pivot_coefficient = &j_column[pivot_data_index].1;
                debug_assert!(ratio.is_not_zero());
                debug_assert!(pivot_coefficient.is_not_zero());
                let update = ratio * pivot_coefficient;
                debug_assert!(update.is_not_zero());
                debug_assert!(pivot_coefficient.is_not_zero());

                let edit_coefficient = &mut j_column[data_index].1;
                let was_non_zero = edit_coefficient.is_not_zero();
                *edit_coefficient -= update;
                let is_non_zero = edit_coefficient.is_not_zero();
                match (was_non_zero, is_non_zero) {
                    (true, false) => {
                        non_zero.row[edit_row_index] -= 1;
                        if column_permutation[j] <= highest_nucleus_index {
                            non_zero.column[j] -= 1;
                        }
                    },
                    (false, true) => {
                        non_zero.row[edit_row_index] += 1;
                        if column_permutation[j] <= highest_nucleus_index {
                            non_zero.column[j] += 1;
                        }
                    },
                    _ => debug_assert_eq!(
                        (was_non_zero, is_non_zero), (true, true),
                        "Update is non zero, so either before or after, it should be non zero",
                    ),
                }

                index_scatter[j] = None;
            }
        }
    }
}

fn add_non_zeros<F>(
    ratio: &F,
    edit_row_index: usize,
    edit_row: &mut Vec<(usize, usize)>,
    pivot_row_index: usize,
    pivot_row: &Vec<(usize, usize)>,
    pivot_column_index: usize,
    column_permutation: &FullPermutation,
    update_range: impl RangeBounds<usize>,
    highest_nucleus_index: usize,
    columns_before_pivot: &mut [Vec<(usize, F)>],
    columns_after_pivot: &mut [Vec<(usize, F)>],
    index_scatter: &mut Vec<Option<usize>>,
    non_zero: &mut NonZeroCounter,
)
where
    F: ops::Field + ops::FieldHR,
{
    for pivot_row_data_index in 0..pivot_row.len() {
        let (j, pivot_row_element_column_data_index) = pivot_row[pivot_row_data_index];

        if index_scatter[j].is_some() {
            // There was no non zero present at this index `j` in the non-pivot
            // row that is being edited. We add a new non zero.
            debug_assert!(
                update_range.contains(&column_permutation[j]),
                "An index should only be `Some` in an index corresponding to the active part of the matrix",
            );

            let j_column = get_column(j, pivot_column_index, columns_before_pivot, columns_after_pivot);
            debug_assert!(
                j_column.iter().find(|&&(ii, _)| ii == edit_row_index).is_none(),
                "Introducing a new non zero is not necessary",
            );

            let new_index_in_column = j_column.len();
            let (_i, pivot_row_coefficient) = &j_column[pivot_row_element_column_data_index];
            debug_assert_eq!(*_i, pivot_row_index);
            let update = -ratio * pivot_row_coefficient;
            j_column.push((edit_row_index, update));
            edit_row.push((j, new_index_in_column));

            non_zero.row[edit_row_index] += 1;
            if column_permutation[j] <= highest_nucleus_index {
                non_zero.column[j] += 1;
            }

            index_scatter[j] = None;
        }
    }
}

pub fn get_column<'a, F>(
    j: usize, pivot_column_index: usize,
    columns_before_pivot: &'a mut [Vec<(usize, F)>], columns_after_pivot: &'a mut [Vec<(usize, F)>],
) -> &'a mut Vec<(usize, F)> {
    if j < pivot_column_index {
        &mut columns_before_pivot[j]
    } else {
        debug_assert_ne!(
            j, pivot_column_index,
            "We should be updating the active part of the matrix, not the pivot column",
        );
        &mut columns_after_pivot[j - pivot_column_index - 1]
    }
}

#[cfg(test)]
mod test {
    use relp_num::R8;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::decomposition::row_subtraction::subtract_rows;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::FullPermutation;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::decomposition::pivoting::NonZeroCounter;

    /// Matrix is
    /// [
    ///     [2, 3], <- pivot row
    ///     [5, 7], <- edit row
    ///      ^
    ///      L pivot column
    /// ]
    /// So the ratio is 5 / 2.
    #[test]
    fn test_subtract_rows_dense() {
        // Dense matrix
        let pivot_row = vec![(0, 0), (1, 0)];
        let mut edit_row = vec![(0, 1), (1, 1)];
        let mut columns_before_pivot = [];
        let mut columns_after_pivot = [vec![(0, R8!(3)), (1, R8!(7))]];

        let mut work = vec![None; 2];
        let mut non_zero = NonZeroCounter {
            row: vec![2, 2],
            column: vec![2, 2],
            row_candidates: vec![],
            column_candidates: vec![],
        };
        subtract_rows(
            &R8!(5, 2),
            1,
            &mut edit_row,
            0,
            &pivot_row,
            &mut columns_before_pivot,
            &mut columns_after_pivot,
            0,
            &FullPermutation::identity(2),
            1..,
            1, // column modified might have its count updated
            &mut work,
            &mut non_zero,
        );

        assert_eq!(edit_row, vec![(0, 1), (1, 1)]);
        assert!(columns_before_pivot.is_empty()); // did not change
        assert_eq!(columns_after_pivot, [vec![(0, R8!(3)), (1, R8!(-1, 2))]]);

        assert_eq!(work, vec![None; 2]);
        assert_eq!(non_zero, NonZeroCounter {
            row: vec![2, 2],
            column: vec![2, 2],
            row_candidates: vec![],
            column_candidates: vec![],
        });
    }

    /// Matrix is
    /// [
    ///     [2, 3], <- pivot row
    ///     [5, 0], <- edit row
    ///      ^
    ///      L pivot column
    /// ]
    /// So the ratio is 5 / 2.
    #[test]
    fn test_subtract_rows_new_non_zero() {
        let pivot_row = vec![(0, 0), (1, 0)];
        let mut edit_row = vec![(0, 1)];
        let mut columns_before_pivot = [];
        let mut columns_after_pivot = [vec![(0, R8!(3))]];

        let mut work = vec![None; 2];
        let mut non_zero = NonZeroCounter {
            row: vec![2, 1],
            column: vec![2, 1],
            row_candidates: vec![],
            column_candidates: vec![],
        };
        subtract_rows(
            &R8!(5, 2),
            1,
            &mut edit_row,
            0,
            &pivot_row,
            &mut columns_before_pivot,
            &mut columns_after_pivot,
            0,
            &FullPermutation::identity(2),
            1..,
            1, // column modified might have its count updated
            &mut work,
            &mut non_zero,
        );

        assert_eq!(edit_row, vec![(0, 1), (1, 1)]);
        assert!(columns_before_pivot.is_empty()); // did not change
        assert_eq!(columns_after_pivot, [vec![(0, R8!(3)), (1, R8!(-15, 2))]]);

        assert_eq!(work, vec![None; 2]);
        assert_eq!(non_zero, NonZeroCounter {
            row: vec![2, 2],
            column: vec![2, 2],
            row_candidates: vec![],
            column_candidates: vec![],
        });
    }

    /// Matrix is
    /// [
    ///     [2,   3], <- pivot row
    ///     [5, 7.5], <- edit row
    ///      ^
    ///      L pivot column
    /// ]
    /// So the ratio is 5 / 2, and 3 * 5 / 2 removes the bottom-right value.
    #[test]
    fn test_subtract_rows_remove_non_zero() {
        // Dense matrix
        let pivot_row = vec![(0, 0), (1, 0)];
        let mut edit_row = vec![(0, 1), (1, 1)];
        let mut columns_before_pivot = [];
        let mut columns_after_pivot = [vec![(0, R8!(3)), (1, R8!(15, 2))]];

        let mut work = vec![None; 2];
        let mut non_zero = NonZeroCounter {
            row: vec![2, 2],
            column: vec![2, 0],
            row_candidates: vec![],
            column_candidates: vec![],
        };
        subtract_rows(
            &R8!(5, 2),
            1,
            &mut edit_row,
            0,
            &pivot_row,
            &mut columns_before_pivot,
            &mut columns_after_pivot,
            0,
            &FullPermutation::identity(2),
            1..,
            0, // column modified will not have its count updated
            &mut work,
            &mut non_zero,
        );

        assert_eq!(edit_row, vec![(0, 1), (1, 1)]);
        assert!(columns_before_pivot.is_empty()); // did not change
        assert_eq!(columns_after_pivot, [vec![(0, R8!(3)), (1, R8!(0))]]);

        assert_eq!(work, vec![None; 2]);
        assert_eq!(non_zero, NonZeroCounter {
            row: vec![2, 1],
            column: vec![2, 0],
            row_candidates: vec![],
            column_candidates: vec![],
        });
    }

    /// Matrix is
    /// [
    ///     [2, 3], <- pivot row
    ///     [5, 0], <- edit row
    ///      ^
    ///      L pivot column
    /// ]
    /// and the bottom-right value is present as a previously filled-in zero.
    #[test]
    fn test_subtract_rows_new_non_zero_already_filled_in() {
        let pivot_row = vec![(0, 0), (1, 0)];
        let mut edit_row = vec![(0, 1), (1, 1)];
        let mut columns_before_pivot = [];
        let mut columns_after_pivot = [vec![(0, R8!(3)), (1, R8!(0))]];

        let mut work = vec![None; 2];
        let mut non_zero = NonZeroCounter {
            row: vec![2, 1],
            column: vec![2, 0],
            row_candidates: vec![],
            column_candidates: vec![],
        };
        subtract_rows(
            &R8!(5, 2),
            1,
            &mut edit_row,
            0,
            &pivot_row,
            &mut columns_before_pivot,
            &mut columns_after_pivot,
            0,
            &FullPermutation::identity(2),
            1..,
            0, // Column count will not be updated
            &mut work,
            &mut non_zero,
        );

        assert_eq!(edit_row, vec![(0, 1), (1, 1)]);
        assert!(columns_before_pivot.is_empty()); // did not change
        assert_eq!(columns_after_pivot, [vec![(0, R8!(3)), (1, R8!(-15, 2))]]);

        assert_eq!(work, vec![None; 2]);
        assert_eq!(non_zero, NonZeroCounter {
            row: vec![2, 2],
            column: vec![2, 0],
            row_candidates: vec![],
            column_candidates: vec![],
        });
    }
}