use std::cmp::Ordering;
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
    #[must_use]
    pub fn decompose<G: ColumnNumber>(mut columns: VecDeque<Vec<(usize, G)>>) -> Self
    where
        F: ops::Column<G>,
    {
        let m = columns.len();
        debug_assert!(m >= 2);
        debug_assert!(columns.iter().all(|column| {
            column.windows(2).all(|w| w[0].0 < w[1].0) &&
                column.iter().last().map_or(false, |&(i, _)| i < m) &&
                !column.is_empty()
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
        // All these arrays are indexed by the "old" indices.
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

        // Case k = 0 is handled explicitly, because it always occurs
        {
            // In a separate scope to ensure variables defined for this iteration don't remain
            let k = 0;

            let mut column = {
                let mut column = {
                    let (relative_pivot_row, relative_pivot_column)
                        = pivot_rule.choose_pivot(columns.make_contiguous(), &row_permutation, k);
                    println!("{} {}", relative_pivot_row, relative_pivot_column);
                    row_permutation.swap_inverse(k, k + relative_pivot_row);
                    column_permutation.swap_inverse(k, k + relative_pivot_column);
                    columns.swap_remove_front(relative_pivot_column).unwrap()
                };
                let k_inverse = row_permutation.backward(k);
                let data_index = column.binary_search_by_key(&k_inverse, |&(i, _)| i).ok()
                    .expect("pivot element not found in column");
                column.swap(0, data_index); // Move the pivot element to the front (sorting)
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
                .collect::<Vec<_>>();

            debug_assert!(lower_column.len() < m - k);
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
            debug_assert!(marker_non_zero_triangle.iter().all(|&v| !v));
            debug_assert!(stack.is_empty());
            debug_assert!(discovered_matmul.is_empty());
            debug_assert!(marker_non_zero_matmul.iter().all(|&v| !v));

            debug_assert!(
                (k..m).all(|i| {
                    let inverse = row_permutation.backward(i);
                    columns.iter().any(|column| column.iter().any(|&(ii, _)| ii == inverse))
                }),
                "k = {}, m = {}", k, m,
            );

            if k == 140 {
                println!("here");
            }

            let mut column = {
                let mut column = {
                    let (relative_pivot_row, relative_pivot_column) =
                        pivot_rule.choose_pivot(columns.make_contiguous(), &row_permutation, k);
                    println!("{} {}", relative_pivot_row, relative_pivot_column);
                    debug_assert!(relative_pivot_row < m - k && relative_pivot_column < m - k);

                    row_permutation.swap_inverse(k, k + relative_pivot_row);
                    column_permutation.swap_inverse(k, k + relative_pivot_column);
                    columns.swap_remove_front(relative_pivot_column).unwrap()
                };
                println!("column = {:?}\nrow_permutation = {}", column, row_permutation);
                debug_assert!(column.iter().find(|&&(i, _)| row_permutation[i] == k).is_some());
                sort_in_three_groups(&mut column, k, &row_permutation);
                column.into_iter()
            };

            debug_assert!(
                ((k + 1)..m).all(|i| {
                    let inverse = row_permutation.backward(i);
                    columns.iter().any(|column| column.iter().any(|&(ii, _)| ii == inverse))
                }),
                "k = {}, m = {}", k, m,
            );

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
            // TODO(PERFORMANCE): Put this in the work array instead
            let mut inner_product = F::zero();

            for row in discovered_triangle.drain(..).rev() {
                if work_triangle_rhs[row].is_not_zero() {
                    for &(i, ref coefficient) in &lower[row_permutation[row]] {
                        let product = &work_triangle_rhs[row] * coefficient;

                        match row_permutation[i].cmp(&k) {
                            Ordering::Less => update_triangle_rhs(i, product, &mut work_triangle_rhs),
                            Ordering::Equal => update_inner_product(product, &mut inner_product),
                            Ordering::Greater => update_matmul(
                                i, product,
                                &mut work_matmul_result, &mut discovered_matmul, &mut marker_non_zero_matmul,
                            ),
                        }
                    }

                    if work_triangle_rhs[row].is_not_zero() {
                        upper_column.push((row_permutation[row], mem::take(&mut work_triangle_rhs[row])));
                    }
                }

                marker_non_zero_triangle[row] = false;
            }
            debug_assert!(upper_column.len() <= k);

            let diagonal = {
                // We prefer the potentially smaller value to be the right-hand side
                let mut v: F = inner_product - pivot_value;
                v.negate();
                v
            };
            debug_assert!(diagonal.is_not_zero(), "the diagonal element should be non zero");

            subtraction(column, &mut marker_non_zero_matmul, &mut discovered_matmul, &mut work_matmul_result);
            let lower_column = gather(&mut marker_non_zero_matmul, &mut discovered_matmul, &mut work_matmul_result, &diagonal);
            debug_assert!(lower_column.len() < m - k);

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
                let (relative_pivot_row, relative_pivot_column) = (0, 0); // the last value
                debug_assert_eq!(
                    pivot_rule.choose_pivot(columns.make_contiguous(), &row_permutation, k),
                    (relative_pivot_row, relative_pivot_column),
                );
                let mut column = columns.pop_front().unwrap();
                debug_assert!(columns.is_empty());

                println!("column = {:?}\nrow_permutation = {}", column, row_permutation);
                let k_inverse = row_permutation.backward(k);
                let data_index = column.binary_search_by_key(&k_inverse, |&(i, _)| i).ok()
                    .expect("pivot element not found in column");
                let len = column.len();
                column.swap(len - 1, data_index); // Move the pivot element to the back (sorting)
                debug_assert!(column.is_sorted_by_key(|&(i, _)| row_permutation[i].cmp(&k)));
                debug_assert!(column.iter().find(|&&(i, _)| row_permutation[i] == k).is_some());
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
                if work_triangle_rhs[row].is_not_zero() {
                    for &(i, ref coefficient) in &lower[row_permutation[row]] {
                        let product = &work_triangle_rhs[row] * coefficient;

                        if row_permutation[i] < k {
                            update_triangle_rhs(i, product, &mut work_triangle_rhs);
                        } else {
                            debug_assert_eq!(row_permutation[i], k);
                            update_inner_product(product, &mut inner_product);
                        }
                    }

                    upper_column.push((row_permutation[row], mem::take(&mut work_triangle_rhs[row])));
                }

                marker_non_zero_triangle[row] = false;
            }

            let diagonal = {
                // We prefer the potentially smaller value to be the right-hand side
                let mut v: F = inner_product - pivot_value;
                v.negate();
                v
            };
            debug_assert!(diagonal.is_not_zero());

            upper.push(upper_column);
            upper_diagonal.push(diagonal);
        }

        // TODO(PERFORMANCE): Can we keep these unsorted?
        // Convert to the new indices
        for (k, column) in lower.iter_mut().enumerate() {
            // `column` is not necessarily sorted here
            row_permutation.forward_unsorted(column);
            column.sort_unstable_by_key(|&(i, _)| i);
            debug_assert!(column.windows(2).all(|w| w[0].0 < w[1].0));
            debug_assert!(column.iter().all(|&(i, _)| i > k));
        }
        for column in &mut upper {
            column.sort_unstable_by_key(|&(i, _)| i);
        }

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

fn sort_in_three_groups<G>(
    unsorted_column: &mut Vec<SparseTuple<G>>,
    k: usize,
    permutation: &FullPermutation,
) {
    // Column is sorted in three groups; before the pivot, the pivot and after the pivot
    unsorted_column.sort_unstable_by_key(|&(i, _)| permutation[i].cmp(&k));
}

fn triangle_discover<F, G>(
    column: &mut impl Iterator<Item=SparseTuple<G>>,
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
        let (mut row, value) = column.next().unwrap();

        if row_permutation[row] < k {
            work[row] = value.into();

            if !non_zero_marker[row] {
                non_zero_marker[row] = true;

                let mut data_index = 0;
                'outer: loop {
                    if row_permutation[row] < k - 1 {
                        let column = &lower[row_permutation[row]];
                        for index in data_index..column.len() {
                            let (i, _) = column[index];

                            if row_permutation[i] < k {
                                if !non_zero_marker[i] {
                                    non_zero_marker[i] = true;

                                    let next_data_index = index + 1;
                                    stack.push((row, next_data_index));

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
            debug_assert_eq!(row_permutation[row], k);
            debug_assert!(stack.is_empty());

            break value;
        }
    }
}

fn update_triangle_rhs<F>(
    row: usize, product: F,
    work_triangle_rhs: &mut Vec<F>,
)
where
    F: ops::Field + ops::FieldHR,
{
    work_triangle_rhs[row] -= product;
}

fn update_inner_product<F>(
    product: F,
    inner_product: &mut F,
)
where
    F: ops::Field + ops::FieldHR,
{
    *inner_product += product;
}

fn update_matmul<F>(
    row: usize,
    product: F,
    work_matmul_result: &mut Vec<F>,
    discovered_matmul: &mut Vec<usize>,
    marker_non_zero_matmul: &mut Vec<bool>,
)
where
    F: ops::Field + ops::FieldHR,
{
    if !marker_non_zero_matmul[row] {
        marker_non_zero_matmul[row] = true;
        discovered_matmul.push(row);
        work_matmul_result[row] = product;
    } else {
        work_matmul_result[row] += product;
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
            work_matmul_result[i].negate();
        } else {
            work_matmul_result[i] -= coefficient;
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
        if work_matmul_result[row].is_not_zero() {
            let numerator = mem::take(&mut work_matmul_result[row]);
            lower_column.push((row, -numerator / diagonal));
        }
        marker_non_zero_matmul[row] = false;
    }

    lower_column
}

#[cfg(test)]
pub mod test {
    use std::collections::VecDeque;

    use relp_num::{NonZero, R8, Rational8, RationalBig, RB, R32};

    use crate::algorithm::two_phase::matrix_provider::column::{Column, DenseSliceIterator, SparseSliceIterator};
    use crate::algorithm::two_phase::matrix_provider::column::identity::IdentityColumn;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::BasisInverse;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::decomposition::{sort_in_three_groups, triangle_discover};
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::LUDecomposition;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::FullPermutation;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::ColumnComputationInfo;
    use crate::data::linear_algebra::vector::{SparseVector, Vector};

    #[test]
    fn test_triangle_identity() {
        const M: usize = 2;
        let k = 1;
        let mut column = [(0, 1), (1, 1)].into_iter();

        let mut stack = vec![];
        let mut discovered = vec![];
        let mut work = vec![R32!(-123456789); M];
        let mut non_zero_marker = vec![false; M];
        let lower = vec![vec![(1, R32!(2)), (2, R32!(3))]];
        let diagonal = triangle_discover(
            &mut column,
            k,
            &mut stack,
            &mut discovered,
            &mut work,
            &mut non_zero_marker,
            &lower,
            &FullPermutation::identity(M),
        );

        debug_assert_eq!(discovered, vec![0]);
        debug_assert_eq!(work, vec![R32!(1), R32!(-123456789)]);
        debug_assert_eq!(non_zero_marker, vec![true, false]);
        debug_assert_eq!(diagonal, R32!(1));
    }

    #[test]
    fn test_triangle_subdiagonal() {
        const M: usize = 3;
        let k = 2;
        let mut column = [(0, 2), (1, 3), (2, 5)].into_iter();

        let mut stack = vec![];
        let mut discovered = vec![];
        let mut work = vec![R32!(-123456789); M];
        let mut non_zero_marker = vec![false; M];
        let lower = vec![vec![(1, R32!(7)), (2, R32!(11))]];
        let diagonal = triangle_discover(
            &mut column,
            k,
            &mut stack,
            &mut discovered,
            &mut work,
            &mut non_zero_marker,
            &lower,
            &FullPermutation::identity(M),
        );

        debug_assert_eq!(discovered, vec![1, 0]);
        debug_assert_eq!(work, vec![R32!(2), R32!(3), R32!(-123456789)]);
        debug_assert_eq!(non_zero_marker, vec![true, true, false]);
        debug_assert_eq!(diagonal, R32!(5));
    }

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
        let result = LUDecomposition::<RationalBig>::decompose(columns.clone());

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
    fn test_2x2_1() {
        test_matrix([
            [ 2,  3],
            [ 5,  0],
        ]);
    }

    #[test]
    fn test_2x2_2() {
        test_matrix([
            [ 2,  3],
            [ 5,  7],
        ]);
    }

    #[test]
    fn test_2x2_3() {
        test_matrix([
            [ 2,  0],
            [ 5,  3],
        ]);
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
