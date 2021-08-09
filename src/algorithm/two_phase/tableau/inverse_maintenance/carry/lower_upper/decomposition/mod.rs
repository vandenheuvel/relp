use std::collections::VecDeque;
/// # LU Decomposition
///

use std::mem;

use crate::algorithm::two_phase::matrix_provider::column::ColumnNumber;
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::decomposition::pivoting::{Markowitz, PivotRule};
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::LUDecomposition;
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::{FullPermutation, Permutation};
use crate::algorithm::two_phase::tableau::inverse_maintenance::ops;
use crate::data::linear_algebra::SparseTuple;

mod pivoting;

impl<F> LUDecomposition<F>
where
    F: ops::Field + ops::FieldHR,
{
    /// Compute the factorization `PBQ^{-1} = LU`.
    ///
    /// # Arguments
    ///
    /// * `rows`: A row major representation of the basis columns.
    #[must_use]
    pub fn decompose<G: ColumnNumber>(mut columns: VecDeque<Vec<(usize, G)>>) -> Self
    where
        F: ops::Column<G>,
    {
        let m = columns.len();
        debug_assert!(m >= 2);
        debug_assert!(columns.iter().all(|column| {
            column.windows(2).all(|w| w[0].0 < w[1].0) &&
                column.iter().last().map_or(false, |&(i, _)| i < m)
        }));

        // TODO(PERFORMANCE): Avoid recreating these work arrays.
        // These values will be returned.
        // They stay the same size, although nothing will be changed in the earlier indices once
        // they have been processed.

        // Permutation indicate where the row/column of the original index is now (either in L&U or
        // in the remaining data items.
        let mut row_permutation = FullPermutation::identity(m);
        let mut column_permutation = FullPermutation::identity(m);

        // Column major
        let mut lower: Vec<Vec<(usize, F)>> = Vec::with_capacity(m - 1);
        // Row major
        let mut upper = Vec::with_capacity(m - 1);
        let mut upper_diagonal = Vec::with_capacity(m);

        // TODO(PERFORMANCE): Merge the work and marker arrays
        // Triangular solve
        let mut discovered_triangle = Vec::new();
        let mut work_triangle_rhs = vec![F::zero(); m];
        let mut marker_non_zero_triangle = vec![false; m];
        let mut stack = Vec::new();
        // Matrix multiplication
        let mut discovered_matmul = Vec::new();
        let mut work_matmul_result = vec![F::zero(); m];
        let mut marker_non_zero_matmul = vec![false; m];

        // These values get smaller over time and keep corresponding to the data layout of the
        // remaining values.
        let pivot_rule = Markowitz::new();

        // Cases k = 0 is handled explicitly, because it always occurs
        {
            // In a separate scope to ensure variables defined for this iteration don't remain
            let k = 0;

            let mut column = {
                let (relative_pivot_row, relative_pivot_column)
                    = pivot_rule.choose_pivot(columns.make_contiguous(), &row_permutation, k);
                administration(
                    relative_pivot_row, relative_pivot_column,
                    k,
                    &mut row_permutation, &mut column_permutation,
                );
                let mut column = columns.pop_front().unwrap();
                let data_index = column.binary_search_by_key(&relative_pivot_row, |&(i, _)| i).ok().unwrap();
                column.swap(k, data_index);
                debug_assert!(column.is_sorted_by_key(|&(i, _)| row_permutation[i].cmp(&k)));
                column.into_iter()
            };

            let (row_index, diagonal) = column.next().unwrap();
            debug_assert_eq!(row_permutation[row_index], k);

            let lower_column = column
                .map(|(i, value)| {
                    debug_assert!(row_permutation[i] > k);

                    // This might be a conversion from a small to a big type
                    let large: F = value.into();
                    (i, large / &diagonal)
                })
                .collect();

            lower.push(lower_column);

            upper_diagonal.push(diagonal.into());
        }

        // Inductive step 0 < k < m - 1
        for k in 1..(m - 1) {
            debug_assert_eq!(lower.len(), k);
            debug_assert_eq!(upper.len(), k - 1);
            debug_assert_eq!(upper_diagonal.len(), k);
            debug_assert_eq!(row_permutation.len(), m);
            debug_assert_eq!(column_permutation.len(), m);
            debug_assert!(discovered_triangle.is_empty());
            debug_assert!(!marker_non_zero_triangle.iter().any(|&v| v));
            debug_assert!(stack.is_empty());
            debug_assert!(discovered_matmul.is_empty());
            debug_assert!(!marker_non_zero_matmul.iter().any(|&v| v));

            let mut column = {
                let (relative_pivot_row, relative_pivot_column) =
                    pivot_rule.choose_pivot(columns.make_contiguous(), &row_permutation, k);
                administration(
                    relative_pivot_row, relative_pivot_column,
                    k,
                    &mut row_permutation, &mut column_permutation,
                );
                let mut column = columns.pop_front().unwrap();
                sort_in_three_groups(&mut column, k, &row_permutation);
                column.into_iter()
            };

            let pivot_value = triangle_discover(
                &mut column,
                k,
                &mut stack,
                &mut discovered_triangle,
                &mut work_triangle_rhs,
                &mut marker_non_zero_triangle,
                &lower,
                &row_permutation,
            );

            let mut upper_column = Vec::with_capacity(discovered_triangle.len());
            let mut inner_product = F::zero();

            for row in discovered_triangle.drain(..).rev() {
                update_rhs(row, k, &mut work_triangle_rhs, &lower);
                update_inner_product(row, k, &mut inner_product, &work_triangle_rhs, &lower);
                update_matmul(row, k, &mut work_matmul_result, &mut discovered_matmul, &mut marker_non_zero_matmul, &work_triangle_rhs, &lower);

                upper_column.push((row, mem::take(&mut work_triangle_rhs[row])));
                marker_non_zero_triangle[row] = false;
            }

            let diagonal = {
                // We prefer the potentially smaller value to be the right-hand side
                let mut v: F = inner_product - pivot_value;
                v.negate();
                v
            };

            subtraction(column, &mut marker_non_zero_matmul, &mut discovered_matmul, &mut work_matmul_result);
            let lower_column = gather(&mut marker_non_zero_matmul, &mut discovered_matmul, &mut work_matmul_result, &diagonal);

            upper.push(upper_column);
            upper_diagonal.push(diagonal);
            lower.push(lower_column);
        }

        // Final k = m - 1 is handled explicitly, because it always occurs
        {
            let k = m - 1;
            debug_assert_eq!(lower.len(), k);
            debug_assert_eq!(upper.len(), k - 1);
            debug_assert_eq!(upper_diagonal.len(), k);
            debug_assert_eq!(row_permutation.len(), m);
            debug_assert_eq!(column_permutation.len(), m);
            debug_assert!(discovered_triangle.is_empty());
            debug_assert!(!marker_non_zero_triangle.iter().any(|&v| v));
            debug_assert!(stack.is_empty());
            debug_assert!(discovered_matmul.is_empty());
            debug_assert!(!marker_non_zero_matmul.iter().any(|&v| v));

            let mut column = {
                let (_relative_pivot_row, _relative_pivot_column) = (0, 0);
                let mut column = columns.pop_front().unwrap();
                debug_assert!(columns.is_empty());

                let data_index = column.binary_search_by_key(&row_permutation[k], |&(i, _)| i).ok().unwrap();
                let len = column.len();
                column.swap(len - 1, data_index);
                debug_assert!(column.is_sorted_by_key(|&(i, _)| row_permutation[i].cmp(&k)));
                column.into_iter()
            };

            let pivot_value = triangle_discover(
                &mut column,
                k,
                &mut stack,
                &mut discovered_triangle,
                &mut work_triangle_rhs,
                &mut marker_non_zero_triangle,
                &lower,
                &row_permutation,
            );
            debug_assert!(column.is_empty());

            let mut upper_column = Vec::with_capacity(discovered_triangle.len());
            let mut inner_product = F::zero();

            for row in discovered_triangle.drain(..).rev() {
                update_rhs(row, k, &mut work_triangle_rhs, &lower);
                update_inner_product(row, k, &mut inner_product, &work_triangle_rhs, &lower);

                upper_column.push((row, mem::take(&mut work_triangle_rhs[row])));
                marker_non_zero_triangle[row] = false;
            }

            let diagonal = {
                // We prefer the potentially smaller value to be the right-hand side
                let mut v: F = inner_product - pivot_value;
                v.negate();
                v
            };

            upper.push(upper_column);
            upper_diagonal.push(diagonal);
        }

        // TODO(PERFORMANCE): Can we keep these unsorted?
        // Convert to the new indices
        for column in &mut lower {
            for (i, _) in column.iter_mut() {
                row_permutation.forward(i);
            }
            column.sort_unstable_by_key(|&(i, _)| i);
        }
        for column in &mut upper {
            column.sort_unstable_by_key(|&(i, _)| i);
        }

        // TODO(CORRECTNESS): Permutation direction
        row_permutation.invert();
        column_permutation.invert();

        Self {
            row_permutation,
            column_permutation,
            lower_triangular: lower,
            upper_triangular: upper,
            upper_diagonal,
            updates: Vec::new(),
        }
    }
}

fn administration(
    relative_pivot_row: usize, relative_pivot_column: usize,
    k: usize,
    row_permutation: &mut FullPermutation, column_permutation: &mut FullPermutation,
) {
    row_permutation.swap(k, k + relative_pivot_row);
    column_permutation.swap(k, k + relative_pivot_column);
}

fn sort_in_three_groups<G>(
    unsorted_column: &mut Vec<SparseTuple<G>>,
    k: usize,
    permutation: &FullPermutation,
) {
    // Column is sorted in three groups; before the pivot, the pivot and after the pivot
    unsorted_column.sort_unstable_by_key(|&(mut i, _)| {
        permutation.backward(&mut i);
        i.cmp(&k)
    });
}

fn triangle_discover<F, G>(
    column: &mut std::vec::IntoIter<SparseTuple<G>>,
    k: usize,
    stack: &mut Vec<(usize, usize)>,
    discovered: &mut Vec<usize>, work: &mut Vec<F>, non_zero_marker: &mut Vec<bool>,
    lower: &Vec<Vec<SparseTuple<F>>>,
    row_permutation: &FullPermutation,
) -> G
where
    F: ops::Column<G>,
{
    loop {
        let (row_with_respect_to_old_index, value) = column.next().unwrap();
        let mut row = row_permutation[row_with_respect_to_old_index];

        if row < k {
            work[row] = value.into();

            if !non_zero_marker[row] {
                non_zero_marker[row] = true;

                let mut data_index = 0;
                'outer: loop {
                    if row < k - 1 {
                        let column = row;
                        for index in data_index..lower[column].len() {
                            let (i_with_respect_to_old_index, _) = lower[column][index];
                            let i = row_permutation[i_with_respect_to_old_index];

                            if i < k {
                                if !non_zero_marker[i] {
                                    non_zero_marker[i] = true;

                                    let next_data_index = index + 1;
                                    stack.push((column, next_data_index));

                                    row = i;
                                    data_index = 0;

                                    continue 'outer;
                                }
                            } else {
                                break;
                            }
                        }
                    }

                    discovered.push(row);

                    match stack.pop() {
                        Some((next_row, next_data_index)) => {
                            row = next_row;
                            data_index = next_data_index;
                        }
                        None => break,
                    }
                }
            }
        } else {
            debug_assert_eq!(row, k);
            break value;
        }
    }
}

fn update_rhs<F>(
    row: usize, k: usize,
    work_triangle_rhs: &mut Vec<F>,
    lower: &Vec<Vec<SparseTuple<F>>>,
)
where
    F: ops::Field + ops::FieldHR,
{
    if row < k - 1 {
        if work_triangle_rhs[row].is_not_zero() {
            let column = row;
            for (i, coefficient) in &lower[column] {
                debug_assert!(*i > row);

                if *i < k {
                    let product = &work_triangle_rhs[row] * coefficient;
                    work_triangle_rhs[*i] -= product;
                } else {
                    break;
                }
            }
        }
    }
}

fn update_inner_product<F>(
    row: usize, k: usize,
    inner_product: &mut F,
    work_triangle_rhs: &Vec<F>,
    lower: &Vec<Vec<SparseTuple<F>>>,
)
where
    F: ops::Field + ops::FieldHR,
{
    let column = row;
    if let Ok(index) = lower[column].binary_search_by_key(&k, |&(i, _)| i) {
        let (_, coefficient) = &lower[column][index];
        *inner_product += &work_triangle_rhs[row] * coefficient;
    }
}

fn update_matmul<F>(
    row: usize, k: usize,
    work_matmul_result: &mut Vec<F>,
    discovered_matmul: &mut Vec<usize>,
    marker_non_zero_matmul: &mut Vec<bool>,
    work_triangle_rhs: &Vec<F>,
    lower: &Vec<Vec<SparseTuple<F>>>,
)
where
    F: ops::Field + ops::FieldHR,
{
    let column = row;
    let start_index = lower[column].binary_search_by_key(&k, |&(i, _)| i).into_ok_or_err();
    for (i, coefficient) in &lower[column][start_index..] {
        if !marker_non_zero_matmul[*i] {
            marker_non_zero_matmul[*i] = true;
            discovered_matmul.push(*i);
            work_matmul_result[*i] = &work_triangle_rhs[row] * coefficient;
        } else {
            work_matmul_result[*i] += &work_triangle_rhs[row] * coefficient;
        }
    }
}

fn subtraction<F, G>(
    column: impl Iterator<Item=SparseTuple<G>>,
    marker_non_zero_matmul: &mut Vec<bool>,
    discovered_matmul: &mut Vec<usize>,
    work_matmul_result: &mut Vec<F>,
)
where
    F: ops::Field + ops::FieldHR + ops::Column<G>,
{
    for (i, coefficient) in column {
        if !marker_non_zero_matmul[i] {
            marker_non_zero_matmul[i] = true;
            discovered_matmul.push(i);
            work_matmul_result[i] = coefficient.into();
        } else {
            work_matmul_result[i] -= coefficient;
            work_matmul_result[i].negate();
        }
    }
}

fn gather<F, G>(
    marker_non_zero_matmul: &mut Vec<bool>,
    discovered_matmul: &mut Vec<usize>,
    work_matmul_result: &mut Vec<F>,
    diagonal: &G,
) -> Vec<SparseTuple<F>>
where
    F: ops::Field + ops::FieldHR + ops::Column<G>,
{
    // TODO(PERFORMANCE): Can this also be unsorted? This adds a log factor
    discovered_matmul.sort_unstable();
    let mut lower_column = Vec::with_capacity(discovered_matmul.len());
    for row in discovered_matmul.drain(..) {
        let numerator = mem::take(&mut work_matmul_result[row]);
        lower_column.push((row, numerator / diagonal));
        marker_non_zero_matmul[row] = false;
    }
    lower_column
}

#[cfg(test)]
mod test {
    use std::collections::VecDeque;

    use relp_num::{R8, RationalBig, RB};

    use crate::algorithm::two_phase::matrix_provider::column::Column;
    use crate::algorithm::two_phase::matrix_provider::column::identity::IdentityColumn;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::BasisInverse;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::decomposition::sort_in_three_groups;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::LUDecomposition;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::FullPermutation;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::ColumnComputationInfo;
    use crate::data::linear_algebra::vector::{SparseVector, Vector};

    #[test]
    fn identity_2() {
        let columns = VecDeque::from([vec![(0, RB!(1))], vec![(1, RB!(1))]]);
        let result = LUDecomposition::decompose(columns);
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

    fn test_identity(n: usize) {
        let columns = (0..n).map(|i| vec![(i, R8!(1))]).collect();
        let result = LUDecomposition::<RationalBig>::decompose(columns);
        for i in 0..n {
            let c = result.left_multiply_by_basis_inverse(IdentityColumn::new(i).iter());
            assert_eq!(c.into_column(), SparseVector::standard_basis_vector(i, n), "{}", i);
        }
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
        let columns = VecDeque::from([vec![(0, RB!(1))], vec![(0, RB!(1)), (1, RB!(1))]]);
        let result = LUDecomposition::decompose(columns);
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
        let columns = VecDeque::from([vec![(0, RB!(1)), (1, RB!(1))], vec![(1, RB!(1))]]);
        let result = LUDecomposition::decompose(columns);
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
        let columns = VecDeque::from([vec![(0, RB!(1)), (1, RB!(1))], vec![(0, RB!(1))]]);
        let result = LUDecomposition::decompose(columns);
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
    fn upper_triangular() {
        let columns = VecDeque::from([vec![(0, R8!(2))], vec![(0, R8!(3)), (1, R8!(5))], vec![(0, R8!(7)), (1, R8!(11)), (2, R8!(13))]]);
        let result = LUDecomposition::decompose(columns);
        let expected = LUDecomposition {
            row_permutation: FullPermutation::identity(3),
            column_permutation: FullPermutation::identity(3),
            lower_triangular: vec![vec![], vec![]],
            upper_triangular: vec![vec![(0, R8!(3))], vec![(0, R8!(7)), (1, R8!(11))]],
            upper_diagonal: vec![R8!(2), R8!(5), R8!(13)],
            updates: vec![],
        };

        assert_eq!(result, expected);
    }

    #[test]
    fn lower_triangular() {
        let columns = VecDeque::from([vec![(0, R8!(2)), (1, R8!(7)), (2, R8!(13))], vec![(1, R8!(3)), (2, R8!(11))], vec![(2, R8!(5))]]);
        let result = LUDecomposition::decompose(columns);
        let expected = LUDecomposition {
            row_permutation: FullPermutation::identity(3),
            column_permutation: FullPermutation::identity(3),
            lower_triangular: vec![vec![(1, R8!(7, 2)), (2, R8!(13, 2))], vec![(2, R8!(11, 3))]],
            upper_triangular: vec![vec![], vec![]],
            upper_diagonal: vec![R8!(2), R8!(3), R8!(5)],
            updates: vec![],
        };

        assert_eq!(result, expected);
    }

    #[test]
    fn wikipedia_example() {
        let columns = VecDeque::from([vec![(0, RB!(4)), (1, RB!(6))], vec![(0, RB!(3)), (1, RB!(3))]]);
        let result = LUDecomposition::decompose(columns);
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
        let columns = VecDeque::from([vec![(0, RB!(-1)), (1, RB!(1))], vec![(0, RB!(3, 2)), (1, RB!(-1))]]);
        let result = LUDecomposition::decompose(columns);
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

    #[test]
    fn test_sort() {
        let n = 4;
        let initial = (0..n).map(|i| (i, i)).collect::<Vec<_>>();
        let mut v = initial.clone();
        let permutation = FullPermutation::identity(n);
        sort_in_three_groups(&mut v, n / 2, &permutation);
        assert_eq!(v, initial);

        let n = 3;
        let mut v = (0..n).map(|i| (i, i)).collect::<Vec<_>>();
        let mut permutation = FullPermutation::identity(n);
        permutation.swap(0, 2);
        sort_in_three_groups(&mut v, 1, &permutation);
        assert_eq!(v, vec![(2, 2), (1, 1), (0, 0)]);


    }
}
