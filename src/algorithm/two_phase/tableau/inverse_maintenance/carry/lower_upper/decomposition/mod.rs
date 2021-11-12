/// # LU Decomposition
///

use std::cmp::Ordering;
use std::mem;
use std::ops::{Mul, Neg, Sub};

use num_traits::Zero;

use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::decomposition::pivoting::{Markowitz, PivotRule};
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::LUDecomposition;
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::{FullPermutation, Permutation, SwapPermutation};
use crate::algorithm::two_phase::tableau::inverse_maintenance::ops;

mod pivoting;

impl<F> LUDecomposition<F>
where
    F: ops::Field + ops::FieldHR,
{
    /// Compute the factorization `PBQ = LU`.
    ///
    /// # Arguments
    ///
    /// * `rows`: A row major representation of the basis columns.
    #[must_use]
    pub fn rows(mut rows: Vec<Vec<(usize, F)>>) -> Self {
        debug_assert!(rows.iter().all(|row| row.iter().is_sorted_by_key(|&(i, _)| i)));

        // The number of columns gets smaller over time.
        let m = rows.len();
        debug_assert!(m > 1);

        // These values will be returned.
        // They stay the same size, although nothing will be changed in the last indices once they
        // have been processed.
        let mut row_permutation = (0..m).collect::<Vec<_>>();
        let mut column_permutation = (0..m).collect::<Vec<_>>();
        let mut lower_triangular_row_major = vec![Vec::new(); m - 1];

        // These values get smaller over time.
        let (mut nnz_row, mut nnz_column) = count_non_zeros(&rows);
        let pivot_rule = Markowitz::new();
        for k in 0..m {
            let (pivot_row, pivot_column) = pivot_rule.choose_pivot(&nnz_row, &nnz_column, &rows, k);
            // Administration for swapping (pivot_row, pivot_column) to (k, k)
            swap(
                pivot_row, pivot_column,
                k,
                &mut row_permutation, &mut column_permutation,
                &mut nnz_row, &mut nnz_column,
                &mut rows,
                &mut lower_triangular_row_major,
            );

            // Update the nonzero counters for this row removal
            for &(j, _) in &rows[k] {
                nnz_row[k] -= 1;
                nnz_column[j] -= 1;
            }

            // Rest of the loop: row reduction operations to zero the values below the pivot
            let (current_row, remaining_rows) = {
                let (current_row, remaining_rows) = rows[k..].split_first_mut().unwrap();
                (&*current_row, remaining_rows)
            };
            let pivot_value = &current_row[0].1;

            // Compute the factor with which row k should be multiplied and subtracted from the
            // other rows
            let mut ratios_to_subtract = Vec::with_capacity(nnz_column[k]);
            for (i, row) in remaining_rows.iter_mut().enumerate() {
                debug_assert!(!row.is_empty(), "The first item exists (invertibility).");
                if row[0].0 == k {
                    ratios_to_subtract.push((i, row.remove(0).1 / pivot_value));
                    nnz_row[k + 1 + i] -= 1;
                    nnz_column[k] -= 1;
                }
            }

            // Eliminate the zeros below the pivot
            for (i, ratio) in ratios_to_subtract {
                let (nnz_row_net_difference, nnz_column_removed, nnz_column_added) = subtract_multiple_of_row_from_other_row(
                    &mut remaining_rows[i],
                    &ratio,
                    &current_row[1..],
                );

                nnz_row[k + 1 + i] -= nnz_row_net_difference.0;
                nnz_row[k + 1 + i] += nnz_row_net_difference.1;
                for column in nnz_column_removed {
                    nnz_column[column] -= 1;
                }
                for column in nnz_column_added {
                    nnz_column[column] += 1;
                }

                lower_triangular_row_major[k + 1 + i - 1].push((k, ratio));
            }

            debug_assert_eq!(nnz_row[k], 0);
            debug_assert_eq!(nnz_column[k], 0);
            debug_assert!(nnz_row[(k + 1)..].iter()
                .enumerate()
                .all(|(i, count)| rows[k + 1 + i].len() == *count));
        }

        // We collected row-major but need column major, so transpose
        let mut upper_triangular = vec![Vec::new(); m - 1];
        let mut upper_diagonal = Vec::with_capacity(m);
        for (i, row) in rows.into_iter().enumerate() {
            let mut iter = row.into_iter();
            let (index, diagonal) = iter.next().unwrap();
            debug_assert_eq!(index, i);
            upper_diagonal.push(diagonal);
            for (j, value) in iter {
                upper_triangular[j - 1].push((i, value));
            }
        }
        let mut lower_triangular = vec![Vec::new(); m - 1];
        for (i, row) in lower_triangular_row_major.into_iter().enumerate()
            // We collected starting at row 1, but the lowest row index was 1.
            .map(|(i, v)| (i + 1, v)) {
            for (j, v) in row {
                lower_triangular[j].push((i, v));
            }
        }

        // Compute the inverse permutations and invert
        let mut row_permutation = FullPermutation::new(row_permutation);
        row_permutation.invert();
        let mut column_permutation = FullPermutation::new(column_permutation);
        column_permutation.invert();

        Self {
            row_permutation,
            column_permutation,
            lower_triangular,
            upper_triangular,
            upper_diagonal,
            updates: Vec::new(),
        }
    }
}

fn subtract_multiple_of_row_from_other_row<T>(
    to_edit: &mut Vec<(usize, T)>,
    ratio: &T,
    being_removed: &[(usize, T)],
) -> ((usize, usize), Vec<usize>, Vec<usize>)
where
    for<'r> T: Mul<&'r T, Output=T> + Sub<T, Output=T> + Zero + Eq,
    for<'r> &'r T: Neg<Output=T> + Mul<&'r T, Output=T>,
{
    debug_assert!(!ratio.is_zero());

    let mut column_nnz_added = Vec::new();
    if to_edit.is_empty() {
        // TODO(PERFORMANCE): Is this case even possible?
        let row_nnz_added = being_removed.len();
        for (j, v) in being_removed {
            to_edit.push((*j, -ratio * v));
            column_nnz_added.push(*j);
        }
        ((0, row_nnz_added), Vec::with_capacity(0), column_nnz_added)
    } else {
        let old_row = mem::replace(to_edit, Vec::with_capacity(0));
        let old_row_nnz = old_row.len();

        let mut column_nnz_removed = Vec::new();
        let mut new = Vec::new();

        let mut index = 0;
        for (j, old_value) in old_row {
            while index < being_removed.len() && being_removed[index].0 < j {
                new.push((being_removed[index].0, -ratio * &being_removed[index].1));
                column_nnz_added.push(being_removed[index].0);
                index += 1;
            }

            if index < being_removed.len() && being_removed[index].0 == j {
                let product = ratio * &being_removed[index].1;
                if product != old_value {
                    new.push((j, old_value - product));
                } else {
                    column_nnz_removed.push(j);
                }
                index += 1;
            } else {
                new.push((j, old_value));
            }
        }
        while index < being_removed.len() {
            new.push((being_removed[index].0, -ratio * &being_removed[index].1));
            column_nnz_added.push(being_removed[index].0);
            index += 1;
        }

        let new_row_nnz = new.len();
        *to_edit = new;

        let (row_nnz_net_removed, row_nnz_net_added) = match new_row_nnz.cmp(&old_row_nnz) {
            Ordering::Less => (old_row_nnz - new_row_nnz, 0),
            Ordering::Equal => (0, 0),
            Ordering::Greater => (0, new_row_nnz - old_row_nnz),
        };

        ((row_nnz_net_removed, row_nnz_net_added), column_nnz_removed, column_nnz_added)
    }
}

/// Swap `pivot_row` and `pivot_column` to k.
///
/// # Arguments
///
/// * `pivot_row`: Index of row swapped with index k.
/// * `pivot_column`: Index of column swapped with index k.
/// * `k`: Step in the iteration and row and column index that will be swapped with.
/// * `row_permutation`: Maps from current row index to the index the row had in the original
/// matrix.
/// * `row_permutation`: Maps from current column index to the index the column had in the original
/// matrix.
/// * `nnz_row`: Number of nonzero elements per row.
fn swap<T>(
    pivot_row: usize, pivot_column: usize,
    k: usize,
    row_permutation: &mut [usize], column_permutation: &mut [usize],
    nnz_row: &mut [usize], nnz_column: &mut [usize],
    rows: &mut [Vec<(usize, T)>],
    lower_triangular_row_major: &mut [Vec<(usize, T)>],
) {
    let n = rows.len();
    debug_assert!(pivot_row < n);
    debug_assert!(pivot_column < n);
    debug_assert!(k < n);
    debug_assert_eq!(row_permutation.len(), n);
    debug_assert_eq!(column_permutation.len(), n);
    debug_assert_eq!(nnz_row.len(), n);
    debug_assert_eq!(nnz_column.len(), n);
    debug_assert_eq!(rows.len(), n);
    debug_assert_eq!(lower_triangular_row_major.len(), n - 1);

    // We work from low indices to high, and don't change the lower (< k) indices.
    debug_assert!(pivot_row >= k && pivot_column >= k);

    if pivot_row != k {
        row_permutation.swap(pivot_row, k);
        nnz_row.swap(pivot_row, k);
        rows.swap(pivot_row, k);

        if pivot_row == 0 && k > 0 {
            debug_assert!(lower_triangular_row_major[k - 1].is_empty());
        }
        if pivot_row > 0 && k == 0 {
            debug_assert!(lower_triangular_row_major[pivot_row - 1].is_empty());
        }
        if pivot_row > 0 && k > 0 {
            // If the pivot_row > 0 && k == 0, there is nothing to swap anyway
            lower_triangular_row_major.swap(pivot_row - 1, k - 1);
        }
    }

    if pivot_column != k {
        column_permutation.swap(pivot_column, k);
        nnz_column.swap(pivot_column, k);

        // TODO(ENHANCEMENT): Can some swapping by avoided by doing it all at the end?
        let swap = SwapPermutation::new((pivot_column, k), n);
        for row in rows {
            swap.forward_sorted(row);
        }
    }
}

/// Count the number of nonzero values in each row and column.
///
/// Requires iterating over all values once.
fn count_non_zeros<T>(rows: &[Vec<(usize, T)>]) -> (Vec<usize>, Vec<usize>) {
    let n = rows.len();
    debug_assert!(n > 0);
    debug_assert!(rows.iter().all(|row| row.iter().all(|&(i, _)| i < n)));

    let nnz_row = rows.iter()
        .map(Vec::len)
        .collect::<Vec<_>>();

    let nnz_column = {
        let mut counts = vec![0; n];
        for column in rows {
            for &(i, _) in column {
                counts[i] += 1;
            }
        }
        counts
    };

    debug_assert_eq!(nnz_column.len(), n);
    debug_assert_eq!(nnz_row.len(), n);
    // We should be working with an invertible matrix
    debug_assert!(nnz_column.iter().all(|&count| count > 0));
    debug_assert!(nnz_row.iter().all(|&count| count > 0));

    (nnz_row, nnz_column)
}

#[cfg(test)]
mod test {
    use relp_num::{RB, R8, RationalBig, NonZero, Rational8};

    use crate::algorithm::two_phase::matrix_provider::column::{Column, SparseSliceIterator, DenseSliceIterator};
    use crate::algorithm::two_phase::matrix_provider::column::identity::IdentityColumn;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::BasisInverse;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::LUDecomposition;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::FullPermutation;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::ColumnComputationInfo;
    use crate::data::linear_algebra::vector::{SparseVector, Vector};
    use std::collections::VecDeque;

    #[test]
    fn identity_2() {
        let rows = vec![vec![(0, RB!(1))], vec![(1, RB!(1))]];
        let result = LUDecomposition::rows(rows);
        let expected = LUDecomposition {
            row_permutation: FullPermutation::identity(2),
            column_permutation: FullPermutation::identity(2),
            lower_triangular: vec![vec![]],
            upper_triangular: vec![vec![]],
            upper_diagonal: vec![RB!(1), RB!(1)],
            updates: vec![],
        };

        assert_eq!(result, expected);
    }

    #[test]
    fn identity_3() {
        let rows = vec![vec![(0, RB!(1))], vec![(1, RB!(1))], vec![(2, RB!(1))]];
        let result = LUDecomposition::rows(rows);
        let expected = LUDecomposition {
            row_permutation: FullPermutation::identity(3),
            column_permutation: FullPermutation::identity(3),
            lower_triangular: vec![vec![], vec![]],
            upper_triangular: vec![vec![], vec![]],
            upper_diagonal: vec![RB!(1), RB!(1), RB!(1)],
            updates: vec![],
        };

        assert_eq!(result, expected);
    }

    #[test]
    fn offdiagonal_2_upper() {
        let rows = vec![vec![(0, RB!(1)), (1, RB!(1))], vec![(1, RB!(1))]];
        let result = LUDecomposition::rows(rows);
        let expected = LUDecomposition {
            row_permutation: FullPermutation::identity(2),
            column_permutation: FullPermutation::identity(2),
            lower_triangular: vec![vec![]],
            upper_triangular: vec![vec![(0, RB!(1))]],
            upper_diagonal: vec![RB!(1), RB!(1)],
            updates: vec![],
        };

        assert_eq!(result, expected);
    }

    #[test]
    fn offdiagonal_2_lower() {
        let rows = vec![vec![(0, RB!(1))], vec![(0, RB!(1)), (1, RB!(1))]];
        let result = LUDecomposition::rows(rows);
        let expected = LUDecomposition {
            row_permutation: FullPermutation::identity(2),
            column_permutation: FullPermutation::identity(2),
            lower_triangular: vec![vec![(1, RB!(1))]],
            upper_triangular: vec![vec![]],
            upper_diagonal: vec![RB!(1), RB!(1)],
            updates: vec![],
        };

        assert_eq!(result, expected);
    }

    #[test]
    fn offdiagonal_2_both() {
        let rows = vec![vec![(0, RB!(1)), (1, RB!(1))], vec![(0, RB!(1))]];
        let result = LUDecomposition::rows(rows);
        let expected = LUDecomposition {
            row_permutation: FullPermutation::new(vec![1, 0]),
            column_permutation: FullPermutation::identity(2),
            lower_triangular: vec![vec![(1, RB!(1))]],
            upper_triangular: vec![vec![]],
            upper_diagonal: vec![RB!(1), RB!(1)],
            updates: vec![],
        };

        assert_eq!(result, expected);
    }

    #[test]
    fn wikipedia_example() {
        let columns = vec![vec![(0, RB!(4)), (1, RB!(3))], vec![(0, RB!(6)), (1, RB!(3))]];
        let result = LUDecomposition::rows(columns);
        let expected = LUDecomposition {
            row_permutation: FullPermutation::identity(2),
            column_permutation: FullPermutation::identity(2),
            lower_triangular: vec![vec![(1, RB!(3, 2))]],
            upper_triangular: vec![vec![(0, RB!(3))]],
            upper_diagonal: vec![RB!(4), RB!(-3, 2)],
            updates: vec![],
        };

        assert_eq!(result, expected);
    }

    #[test]
    fn wikipedia_example2() {
        let rows = vec![vec![(0, RB!(-1)), (1, RB!(3, 2))], vec![(0, RB!(1)), (1, RB!(-1))]];
        let result = LUDecomposition::rows(rows);
        let expected = LUDecomposition {
            row_permutation: FullPermutation::identity(2),
            column_permutation: FullPermutation::identity(2),
            lower_triangular: vec![vec![(1, RB!(-1))]],
            upper_triangular: vec![vec![(0, RB!(3, 2))]],
            upper_diagonal: vec![RB!(-1), RB!(1, 2)],
            updates: vec![],
        };

        assert_eq!(result, expected);

        assert_eq!(
            expected.left_multiply_by_basis_inverse(IdentityColumn::new(0).iter()).into_column(),
            SparseVector::new(vec![(0, RB!(2)), (1, RB!(2))], 2),
        );
        assert_eq!(
            expected.left_multiply_by_basis_inverse(IdentityColumn::new(1).iter()).into_column(),
            SparseVector::new(vec![(0, RB!(3)), (1, RB!(2))], 2),
        );
    }

    pub fn to_columns<const M: usize>(rows: &[[i32; M]; M]) -> VecDeque<Vec<(usize, Rational8)>> {
        let mut columns = vec![vec![]; M].into_iter().collect::<VecDeque<_>>();

        for (i, row) in rows.into_iter().enumerate() {
            for (j, v) in row.into_iter().enumerate() {
                if v.is_not_zero() {
                    columns[j].push((i, R8!(*v)));
                }
            }
        }

        columns
    }

    fn test_matrix<const M: usize>(rows: [[i32; M]; M]) {
        let columns = to_columns(&rows);
        let result = LUDecomposition::<RationalBig>::rows(
            rows.iter().map(|row| {
                row.iter().enumerate()
                    .filter(|(_, v)| v.is_not_zero())
                    .map(|(i, v)| (i, v.into()))
                    .collect()
            }).collect()
        );
        for (j, column) in columns.iter().enumerate() {
            assert_eq!(
                result.left_multiply_by_basis_inverse(SparseSliceIterator::new(column)).into_column(),
                SparseVector::standard_basis_vector(j, M),
                "{}", j,
            );
        }
        for (j, row) in rows.into_iter().enumerate() {
            assert_eq!(
                result.right_multiply_by_basis_inverse(DenseSliceIterator::new(&row)),
                SparseVector::standard_basis_vector(j, M),
                "{}", j,
            );
        }
    }

    #[test]
    fn test_3x3() {
        test_matrix([
            [ 2,  3,  0],
            [ 5,  0, 11],
            [23, 29,  0],
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
            [29, 23,  0, 19, 0],
            [ 0,  0, 17, 13, 0],
            [ 0, 11,  7,  0, 0],
            [ 5,  0,  0,  3, 0],
            [ 0,  0,  0,  0, 2],
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

    mod subtract_multiple_of_column_from_other_column {
        use relp_num::R32;

        use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::decomposition::subtract_multiple_of_row_from_other_row;

        #[test]
        fn empty() {
            let mut column1 = vec![];
            let column2 = vec![];
            subtract_multiple_of_row_from_other_row(&mut column1, &R32!(1), &column2);
            assert_eq!(column1, vec![]);
        }

        #[test]
        fn edited_empty() {
            let mut column1 = vec![];
            let column2 = vec![(1, R32!(1))];
            subtract_multiple_of_row_from_other_row(&mut column1, &R32!(1), &column2);
            assert_eq!(column1, vec![(1, R32!(-1))]);
        }

        #[test]
        fn other_empty() {
            let mut column1 = vec![(1, R32!(1))];
            let column2 = vec![];
            subtract_multiple_of_row_from_other_row(&mut column1, &R32!(1), &column2);
            assert_eq!(column1, vec![(1, R32!(1))]);
        }

        #[test]
        fn single_before() {
            let mut column1 = vec![(1, R32!(1))];
            let column2 = vec![(2, R32!(3))];
            subtract_multiple_of_row_from_other_row(&mut column1, &R32!(1), &column2);
            assert_eq!(column1, vec![(1, R32!(1)), (2, R32!(-3))]);
        }

        #[test]
        fn single_at() {
            let mut column1 = vec![(1, R32!(1))];
            let column2 = vec![(1, R32!(3))];
            subtract_multiple_of_row_from_other_row(&mut column1, &R32!(1), &column2);
            assert_eq!(column1, vec![(1, R32!(-2))]);
        }

        #[test]
        fn single_at_make_zero() {
            let mut column1 = vec![(1, R32!(1))];
            let column2 = vec![(1, R32!(3))];
            subtract_multiple_of_row_from_other_row(&mut column1, &R32!(1, 3), &column2);
            assert_eq!(column1, vec![]);
        }

        #[test]
        fn single_after() {
            let mut column1 = vec![(1, R32!(1))];
            let column2 = vec![(0, R32!(3))];
            subtract_multiple_of_row_from_other_row(&mut column1, &R32!(1), &column2);
            assert_eq!(column1, vec![(0, R32!(-3)), (1, R32!(1))]);
        }
    }
}
