//! # LU decomposition
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::fmt::Display;

use num_traits::Zero;

use crate::algorithm::two_phase::matrix_provider::column::Column;
use crate::algorithm::two_phase::tableau::inverse_maintenance::{ColumnComputationInfo, ops};
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::{FullPermutation, Permutation};
use crate::data::linear_algebra::traits::SparseElement;
use crate::data::linear_algebra::vector::SparseVector;

mod decomposition;
mod permutation;
mod eta_file;
pub mod forrest_tomlin_update;

/// Decompose a matrix `B` into `PBQ = LU` where
///
/// * `P` is a row permutation
/// * `Q` is a column permutation
/// * `L` is lower triangular with `1`'s on the diagonal
/// * `U` is upper triangular
///
/// Note that permutations `P` and `Q` have the transpose equal to their inverse as they are
/// orthogonal matrices.
///
/// `P` and `Q` are "full" permutations, not to be confused with the simpler "rotating" permutations
/// in the `updates` field.
#[derive(Eq, PartialEq, Clone, Debug)]
pub struct LUDecomposition<F, Update> {
    /// Row permutation `P`.
    ///
    /// The `forward` application of the permutation to rows of `M` corresponds to `PM`.
    row_permutation: FullPermutation,
    /// Column permutation `Q`.
    ///
    /// The
    column_permutation: FullPermutation,
    /// Lower triangular matrix `L`.
    ///
    /// Column major, one's on diagonal implied. So the length of the vector is `m - 1`.
    lower_triangular: Vec<Vec<(usize, F)>>,
    /// Upper triangular matrix `U`.
    ///
    /// Column major.
    // TODO(PERFORMANCE): Consider storing the diagonal separately.
    upper_triangular: Vec<Vec<(usize, F)>>,

    // TODO(PERFORMANCE): Consider an Option<Permutation> instead to avoid doing identities
    // TODO(ARCHITECTURE): Consider grouping pivot index and eta file in a single update struct.
    updates: Vec<Update>,
}

impl<F, Update> LUDecomposition<F, Update>
where
    F: ops::Field + ops::FieldHR,
{
    fn identity(m: usize) -> Self {
        Self {
            row_permutation: FullPermutation::identity(m),
            column_permutation: FullPermutation::identity(m),
            lower_triangular: vec![vec![]; m - 1],
            upper_triangular: (0..m).map(|i| vec![(i, F::one())]).collect(),
            updates: Vec::new(),
        }
    }

    fn invert<C: Column>(columns: impl ExactSizeIterator<Item=C>) -> Self
    where
        F: ops::Column<C::F>,
    {
        let m = columns.len();
        let mut rows = vec![Vec::new(); m];
        for (j, column) in columns.into_iter().enumerate() {
            for (i, value) in column.iter() {
                rows[i].push((j, value.into()));
            }
        }
        debug_assert!(rows.iter().all(|row| row.is_sorted_by_key(|&(j, _)| j)));

        Self::rows(rows)
    }

    fn left_multiply_by_lower_inverse(&self, mut rhs: BTreeMap<usize, F>) -> Vec<(usize, F)> {
        let mut result = Vec::new();

        while let Some((row, rhs_value)) = rhs.pop_first() {
            let column = row;

            let result_value = rhs_value; // Implicit 1 on the diagonal.

            if column != self.len() - 1 {
                // Introduce new nonzeros
                for &(row, ref value) in &self.lower_triangular[column] {
                    insert_or_shift_maybe_remove(row, &result_value * value, &mut rhs);
                }
            }

            result.push((row, result_value));
        }

        result
    }

    fn left_multiply_by_upper_inverse(&self, mut rhs: BTreeMap<usize, F>) -> Vec<(usize, F)> {
        let mut result = Vec::new();

        while let Some((row, rhs_value)) = rhs.pop_last() {
            let column = row;
            let result_value = self.compute_result(column, rhs_value);
            self.update_rhs(column, &result_value, &mut rhs);
            result.push((row, result_value));
        }

        result.reverse();
        debug_assert!(result.is_sorted_by_key(|&(i, _)| i));

        result
    }

    fn left_multiply_by_upper_inverse_row(&self, mut rhs: BTreeMap<usize, F>, target_row: usize) -> Option<F> {
        while let Some((row, rhs_value)) = rhs.pop_last() {
            let column = row;
            match row.cmp(&target_row) {
                Ordering::Less => return None,
                Ordering::Equal => return Some(self.compute_result(column, rhs_value)),
                Ordering::Greater => {
                    let result_value = self.compute_result(column, rhs_value);
                    self.update_rhs(column, &result_value, &mut rhs);
                },
            }
        }

        None
    }

    fn compute_result(&self, column: usize, rhs_value: F) -> F {
        debug_assert_eq!(
            self.upper_triangular[column].last().unwrap().0,
            column, "Needs to have a diagonal element",
        );

        let diagonal_item = &self.upper_triangular[column].last().unwrap().1;
        rhs_value / diagonal_item
    }

    fn update_rhs(&self, column: usize, result_value: &F, rhs: &mut BTreeMap<usize, F>) {
        let nr_column_items = self.upper_triangular[column].len();
        for &(row, ref value) in &self.upper_triangular[column][..(nr_column_items - 1)] {
            insert_or_shift_maybe_remove(row, result_value * value, rhs);
        }
    }

    fn right_multiply_by_lower_inverse(&self, mut rhs: BTreeMap<usize, F>) -> Vec<(usize, F)> {
        let mut result = Vec::new();

        while let Some((column, rhs_value)) = rhs.pop_last() {
            let row = column;
            let result_value = rhs_value;

            // Introduce new nonzeros, by rows
            for j in 0..column {
                // TODO(PERFORMANCE): Avoid scanning all columns for row values
                let has_row = self.lower_triangular[j].binary_search_by_key(&row, |&(i, _)| i);
                if let Ok(data_index) = has_row {
                    let value = &self.lower_triangular[j][data_index].1;
                    insert_or_shift_maybe_remove(j, &result_value * value, &mut rhs);
                }
            }

            result.push((row, result_value));
        }

        result.reverse();
        debug_assert!(result.is_sorted_by_key(|&(i, _)| i));

        result
    }

    fn right_multiply_by_upper_inverse(&self, mut rhs: BTreeMap<usize, F>) -> Vec<(usize, F)> {
        let mut result = Vec::new();

        while let Some((column, rhs_value)) = rhs.pop_first() {
            let row = column;
            let diagonal_item = &self.upper_triangular[column].last().unwrap().1;
            let result_value = rhs_value / diagonal_item;

            // Introduce new nonzeros, by rows
            for j in (column + 1)..self.len() {
                // TODO(PERFORMANCE): Avoid scanning all columns for row values
                let has_row = self.upper_triangular[j].binary_search_by_key(&row, |&(i, _)| i);
                if let Ok(data_index) = has_row {
                    let value = &self.upper_triangular[j][data_index].1;
                    insert_or_shift_maybe_remove(j, &result_value * value, &mut rhs);
                }
            }

            result.push((column, result_value));
        }

        debug_assert!(result.windows(2).all(|w| w[0].0 < w[1].0));

        result
    }

    fn len(&self) -> usize {
        self.row_permutation.len()
        // == self.column_permutation.len()
        // == self.upper_triangular.len()
        // == self.lower_triangular.len() + 1
    }
}

fn insert_or_shift_maybe_remove<F>(index: usize, change: F, rhs: &mut BTreeMap<usize, F>)
where
    F: ops::Field + ops::FieldHR + Zero,
{
    match rhs.get_mut(&index) {
        None => {
            rhs.insert(index, -change);
        }
        Some(existing) => {
            *existing -= change;
            if existing.is_zero() {
                rhs.remove(&index);
            }
        }
    }
}

/// The generated column and the spike that was generated in the process wrapped together.
#[derive(Debug)]
pub struct ColumnAndSpike<F> {
    column: SparseVector<F, F>,
    spike: Vec<(usize, F)>,
}

impl<F: SparseElement<F>> ColumnComputationInfo<F> for ColumnAndSpike<F> {
    fn column(&self) -> &SparseVector<F, F> {
        &self.column
    }

    fn into_column(self) -> SparseVector<F, F> {
        self.column
    }
}

impl<F, Update> Display for LUDecomposition<F, Update>
where
    F: ops::Field + ops::FieldHR,
    Update: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let width = 10;
        let column_width = 3;

        writeln!(f, "Lower:")?;
        write!(f, "{:>width$} |", "", width = column_width)?;
        for j in 0..self.len() {
            write!(f, "{0:^width$}", j, width = width)?;
        }
        writeln!(f)?;
        let total_width = column_width + 1 + 1 + self.len() * width;
        writeln!(f, "{}", "-".repeat(total_width))?;

        for i in 0..self.len() {
            write!(f, "{0:>width$} |", i, width = column_width)?;
            for j in 0..self.len() {
                let value = match j.cmp(&i) {
                    Ordering::Greater => "".to_string(),
                    Ordering::Equal => "1".to_string(),
                    Ordering::Less => {
                        match self.lower_triangular[j].binary_search_by_key(&i, |&(i, _)| i) {
                            Ok(index) => self.lower_triangular[j][index].1.to_string(),
                            Err(_) => "0".to_string(),
                        }
                    }
                };
                write!(f, "{0:^width$}", value, width = width)?;
            }
            writeln!(f)?;
        }
        writeln!(f)?;

        writeln!(f, "Upper:")?;
        write!(f, "{:>width$} |", "", width = column_width)?;
        for j in 0..self.len() {
            write!(f, "{0:^width$}", j, width = width)?;
        }
        writeln!(f)?;
        writeln!(f, "{}", "-".repeat(total_width))?;

        for i in 0..self.len() {
            write!(f, "{0:>width$} |", i, width = column_width)?;
            for j in 0..self.len() {
                let value = match j.cmp(&i) {
                    Ordering::Equal | Ordering::Greater => {
                        match self.upper_triangular[j].binary_search_by_key(&i, |&(i, _)| i) {
                            Ok(index) => self.upper_triangular[j][index].1.to_string(),
                            Err(_) => "0".to_string(),
                        }
                    },
                    Ordering::Less => "".to_string(),
                };
                write!(f, "{0:^width$}", value, width = width)?;
            }
            writeln!(f)?;
        }
        writeln!(f)?;

        writeln!(f, "Row permutation:")?;
        writeln!(f, "{:?}", self.row_permutation)?;
        writeln!(f, "Column permutation:")?;
        writeln!(f, "{:?}", self.column_permutation)?;

        writeln!(f, "Updates:")?;
        for (i, update) in self.updates.iter().enumerate() {
            writeln!(f, "Update number {}: ", i)?;
            writeln!(f, "{}", update)?;
        }
        writeln!(f)
    }
}

#[cfg(test)]
mod test {
    use std::collections::BTreeMap;

    use relp_num::{R64, RB};
    use relp_num::{Rational64, RationalBig};

    use crate::algorithm::two_phase::matrix_provider::column::Column as ColumnTrait;
    use crate::algorithm::two_phase::matrix_provider::column::identity::IdentityColumn;
    use crate::algorithm::two_phase::matrix_provider::matrix_data::Column;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::BasisInverse;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::LUDecomposition;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::FullPermutation;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::ColumnComputationInfo;
    use crate::data::linear_algebra::vector::{SparseVector, Vector};

    mod matmul {
        use crate::algorithm::im_ops::Field;
        use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::forrest_tomlin_update::ForrestTomlinUpdate;

        use super::*;

        #[test]
        fn identity_empty() {
            let identity = LUDecomposition::<RationalBig, ()>::identity(2);

            let column = BTreeMap::new();
            let result = identity.left_multiply_by_upper_inverse(column);
            assert!(result.is_empty());
            let column = BTreeMap::new();
            let result = identity.right_multiply_by_upper_inverse(column);
            assert!(result.is_empty());
            let column = BTreeMap::new();
            let result = identity.left_multiply_by_lower_inverse(column);
            assert!(result.is_empty());
            let column = BTreeMap::new();
            let result = identity.right_multiply_by_lower_inverse(column);
            assert!(result.is_empty());
        }

        #[test]
        fn identity_single() {
            let identity = LUDecomposition::<RationalBig, ()>::identity(2);

            let column = vec![(0, RB!(1))];
            let result = identity.left_multiply_by_upper_inverse(column.clone().into_iter().collect());
            assert_eq!(result, column);
            let column = vec![(0, RB!(1))];
            let result = identity.right_multiply_by_upper_inverse(column.clone().into_iter().collect());
            assert_eq!(result, column);
            let column = vec![(0, RB!(1))];
            let result = identity.left_multiply_by_lower_inverse(column.clone().into_iter().collect());
            assert_eq!(result, column);
            let column = vec![(0, RB!(1))];
            let result = identity.right_multiply_by_lower_inverse(column.clone().into_iter().collect());
            assert_eq!(result, column);

            let column = vec![(1, RB!(1))];
            let result = identity.left_multiply_by_upper_inverse(column.clone().into_iter().collect());
            assert_eq!(result, column);
            let column = vec![(1, RB!(1))];
            let result = identity.right_multiply_by_upper_inverse(column.clone().into_iter().collect());
            assert_eq!(result, column);
            let column = vec![(1, RB!(1))];
            let result = identity.left_multiply_by_lower_inverse(column.clone().into_iter().collect());
            assert_eq!(result, column);
            let column = vec![(1, RB!(1))];
            let result = identity.right_multiply_by_lower_inverse(column.clone().into_iter().collect());
            assert_eq!(result, column);
        }

        #[test]
        fn identity_double() {
            let identity = LUDecomposition::<RationalBig, ()>::identity(2);

            let column = vec![(0, RB!(1)), (1, RB!(1))];
            let result = identity.left_multiply_by_upper_inverse(column.clone().into_iter().collect());
            assert_eq!(result, column);
            let column = vec![(0, RB!(1)), (1, RB!(1))];
            let result = identity.right_multiply_by_upper_inverse(column.clone().into_iter().collect());
            assert_eq!(result, column);
            let column = vec![(0, RB!(1)), (1, RB!(1))];
            let result = identity.left_multiply_by_lower_inverse(column.clone().into_iter().collect());
            assert_eq!(result, column);
            let column = vec![(0, RB!(1)), (1, RB!(1))];
            let result = identity.right_multiply_by_lower_inverse(column.clone().into_iter().collect());
            assert_eq!(result, column);
        }

        #[test]
        fn offdiagonal_empty() {
            let offdiag = LUDecomposition::<_, ForrestTomlinUpdate<_>> {
                row_permutation: FullPermutation::identity(2),
                column_permutation: FullPermutation::identity(2),
                lower_triangular: vec![vec![(1, RB!(1))]],
                upper_triangular: vec![vec![(0, RB!(1))], vec![(1, RB!(1))]],
                updates: vec![],
            };

            let column: Column<Rational64> = Column::Sparse {
                constraint_values: vec![],
                slack: None,
                mock_array: [],
            };
            let result = offdiag.left_multiply_by_basis_inverse(column.iter());
            assert_eq!(result.column, SparseVector::new(vec![], 2));
        }

        #[test]
        fn offdiagonal_single() {
            let offdiag = LUDecomposition::<_, ForrestTomlinUpdate<_>> {
                row_permutation: FullPermutation::identity(2),
                column_permutation: FullPermutation::identity(2),
                lower_triangular: vec![vec![(1, RB!(1))]],
                upper_triangular: vec![vec![(0, RB!(1))], vec![(1, RB!(1))]],
                updates: vec![],
            };

            let column = Column::Sparse {
                constraint_values: vec![(0, R64!(1))],
                slack: None,
                mock_array: [],
            };
            let result = offdiag.left_multiply_by_basis_inverse(column.iter());
            assert_eq!(result.column, SparseVector::new(vec![(0, RB!(1)), (1, RB!(-1))], 2));

            let column = Column::Sparse {
                constraint_values: vec![(1, R64!(1))],
                slack: None,
                mock_array: [],
            };
            let result = offdiag.left_multiply_by_basis_inverse(column.iter());
            assert_eq!(result.column, SparseVector::new(vec![(1, RB!(1))], 2));
        }

        #[test]
        fn dense() {
            // [
            //  [1, 2],
            //  [3, 4],
            // ]
            //
            // Inverse:
            // [
            //  [-2, 1],
            //  [1.5, -0.5],
            // ]
            let dense = LUDecomposition::<_, ForrestTomlinUpdate<_>> {
                row_permutation: FullPermutation::new(vec![1, 0]),
                column_permutation: FullPermutation::new(vec![0, 1]),
                lower_triangular: vec![vec![(1, RB!(1, 3))]],
                upper_triangular: vec![vec![(0, RB!(3))], vec![(0, RB!(4)), (1, RB!(2, 3))]],
                updates: vec![],
            };

            assert_eq!(
                dense.left_multiply_by_basis_inverse(IdentityColumn::new(0).iter()).into_column(),
                SparseVector::new(vec![(0, RB!(-2)), (1, RB!(3, 2))], 2),
            );
            assert_eq!(
                dense.left_multiply_by_basis_inverse(IdentityColumn::new(1).iter()).into_column(),
                SparseVector::new(vec![(0, RB!(1)), (1, RB!(-1, 2))], 2),
            );
            assert_eq!(
                dense.right_multiply_by_basis_inverse(IdentityColumn::new(0).iter()).into_column(),
                SparseVector::new(vec![(0, RB!(-2)), (1, RB!(1))], 2),
            );
            assert_eq!(
                dense.right_multiply_by_basis_inverse(IdentityColumn::new(1).iter()).into_column(),
                SparseVector::new(vec![(0, RB!(3, 2)), (1, RB!(-1, 2))], 2),
            );
        }
    }
}
