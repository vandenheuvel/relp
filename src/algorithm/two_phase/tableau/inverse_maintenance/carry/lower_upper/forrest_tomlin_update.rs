//! # Forrest & Tomlin LU Update
//!
//! Updating the LU factorization without completely refactoring it.
//!
//! See the 1972 paper by Forrest and Tomlin.
use std::collections::BTreeMap;
use std::fmt;

use crate::algorithm::two_phase::matrix_provider::column::{Column, ColumnIterator, ColumnNumber};
use crate::algorithm::two_phase::matrix_provider::column::identity::IdentityColumn;
use crate::algorithm::two_phase::tableau::inverse_maintenance::{ColumnComputationInfo, ops};
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::BasisInverse;
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::{ColumnAndSpike, LUDecomposition};
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::eta_file::EtaFile;
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::{Permutation, RotateToBackPermutation};
use crate::data::linear_algebra::SparseTuple;
use crate::data::linear_algebra::vector::{SparseVector, Vector};

/// Information stored for each step of the Forrest & Tomlin update.
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct ForrestTomlinUpdate<F> {
    /// The eta file is called "r" or "R" in the original paper.
    eta_file: EtaFile<F>,
    /// Represents which index the spike was entered into the upper triangular matrix.
    ///
    /// This rotation is both the conversion into upper Hessenberg form of U and the rotation moving
    /// the almost-eliminated row to the bottom.
    rotation: RotateToBackPermutation,
}

const ITERATIONS_BEFORE_REFACTOR: usize = 30;

impl<F> ForrestTomlinUpdate<F>
where
    F: ops::Field + ops::FieldHR,
{
    pub fn new(eta_file: Vec<SparseTuple<F>>, pivot_index: usize, len: usize) -> Self {
        debug_assert!(eta_file.iter().all(|&(i, _)| i < len));
        debug_assert!(pivot_index < len);

        Self {
            eta_file: EtaFile::new(eta_file, pivot_index, len),
            rotation: RotateToBackPermutation::new(pivot_index, len),
        }
    }
}

impl<F> From<(EtaFile<F>, RotateToBackPermutation)> for ForrestTomlinUpdate<F> {
    fn from((eta_file, rotation): (EtaFile<F>, RotateToBackPermutation)) -> Self {
        Self {
            eta_file,
            rotation,
        }
    }
}

impl<F> BasisInverse for LUDecomposition<F, ForrestTomlinUpdate<F>>
where
    F: ops::Field + ops::FieldHR,
{
    type F = F;
    type ColumnComputationInfo = ColumnAndSpike<Self::F>;

    fn identity(m: usize) -> Self {
        Self::identity(m)
    }

    fn invert<C: Column>(columns: impl ExactSizeIterator<Item=C>) -> Self
    where
        Self::F: ops::Column<C::F>,
    {
        Self::invert(columns)
    }

    fn change_basis(
        &mut self,
        pivot_row_index: usize,
        column: Self::ColumnComputationInfo,
    ) -> SparseVector<Self::F, Self::F> {
        let m = self.m();

        let pivot_column_index = {
            // Column with a pivot in `pivot_row_index` is leaving
            let mut pivot_column_index = pivot_row_index;
            self.column_permutation.forward(&mut pivot_column_index);
            for ForrestTomlinUpdate { rotation, .. } in &self.updates {
                rotation.forward(&mut pivot_column_index);
            }
            pivot_column_index
        };

        // Compute r
        let (u_bar, indices_to_zero): (Vec<_>, Vec<_>) = ((pivot_column_index + 1)..m)
            .filter_map(|j| {
                self.upper_triangular[j]
                    .binary_search_by_key(&pivot_column_index, |&(i, _)| i)
                    .ok()
                    .map(|data_index| (
                        (j, self.upper_triangular[j][data_index].1.clone()),
                        (j, data_index),
                    ))
            })
            .unzip();
        let u_bar = u_bar.into_iter().collect();
        let r = self.right_multiply_by_upper_inverse(u_bar);
        let eta_file = EtaFile::new(r, pivot_column_index, self.m());

        // We now have all information needed to recover the triangle, start modifying it
        // Zero out part of a row that will be rotated to the bottom
        for (j, data_index) in indices_to_zero {
            self.upper_triangular[j].remove(data_index);
        }
        let Self::ColumnComputationInfo { column, mut spike } = column;
        debug_assert!(spike.iter().is_sorted_by_key(|&(i, _)| i));

        eta_file.update_spike_pivot_value(&mut spike);
        debug_assert!(
            spike.binary_search_by_key(&pivot_column_index, |&(i, _)| i).is_ok(),
            "This value should be present to avoid singularity because it will be the bottom corner value.",
        );
        // Insert the spike for upper
        self.upper_triangular[pivot_column_index] = spike;

        // Move the spike to the end
        self.upper_triangular[pivot_column_index..].rotate_left(1);

        // Rotate the "empty except for one value" pivot row to the bottom
        let q = RotateToBackPermutation::new(pivot_column_index, self.m());
        for j in pivot_column_index..m {
            q.forward_sorted(&mut self.upper_triangular[j])
        }

        // `upper_triangular` is diagonal again
        debug_assert!(self.upper_triangular.iter().enumerate().all(|(j, column)| {
            column.last().unwrap().0 == j && column.is_sorted_by_key(|&(i, _)| i)
        }));
        self.updates.push(ForrestTomlinUpdate::from((eta_file, q)));

        column
    }

    fn left_multiply_by_basis_inverse<'a, G: 'a + ColumnNumber, I: ColumnIterator<'a, G>>(&self, iter: I) -> Self::ColumnComputationInfo
    where
        Self::F: ops::Column<G>,
    {
        let rhs = iter
            .map(|(mut i, v)| {
                self.row_permutation.forward(&mut i);
                (i, v.into())
            })
            // Also sorts after the row permutation
            .collect::<BTreeMap<_, _>>();
        let mut w = self.left_multiply_by_lower_inverse(rhs);

        for ForrestTomlinUpdate { eta_file, rotation } in &self.updates {
            eta_file.apply_right(&mut w);
            // Q^-1 c = (c^T Q)^T because Q orthogonal matrix. Here we
            rotation.forward_sorted(&mut w);
        }

        let spike = w.clone();

        let mut column = self.left_multiply_by_upper_inverse(w.into_iter().collect());

        for ForrestTomlinUpdate { eta_file: _, rotation } in self.updates.iter().rev() {
            rotation.backward_unsorted(&mut column);
        }
        self.column_permutation.backward_unsorted(&mut column);
        column.sort_unstable_by_key(|&(i, _)| i);

        Self::ColumnComputationInfo {
            column: SparseVector::new(column, self.m()),
            spike,
        }
    }

    fn right_multiply_by_basis_inverse<'a, G: 'a + ColumnNumber, I: ColumnIterator<'a, G>>(
        &self, row: I,
    ) -> SparseVector<Self::F, Self::F>
    where
        Self::F: ops::Column<G>,
    {
        let mut lhs = row
            .map(|(mut i, v)| {
                self.column_permutation.forward(&mut i);
                (i, v.into())
            })
            .collect::<Vec<_>>();

        for ForrestTomlinUpdate { eta_file: _, rotation } in &self.updates {
            rotation.forward_unsorted(&mut lhs);
        }

        let mut lhs = self.right_multiply_by_upper_inverse(lhs.into_iter().collect());

        for ForrestTomlinUpdate { eta_file, rotation } in self.updates.iter().rev() {
            rotation.backward_sorted(&mut lhs);
            eta_file.apply_left(&mut lhs);
        }

        let mut lhs = self.right_multiply_by_lower_inverse(lhs.into_iter().collect());
        self.row_permutation.backward_sorted(&mut lhs);

        SparseVector::new(lhs, self.m())
    }

    fn generate_element<'a, G: 'a + ColumnNumber, I: ColumnIterator<'a, G>>(
        &self, i: usize, original_column: I,
    ) -> Option<Self::F>
    where
        Self::F: ops::Column<G>,
    {
        // TODO(PERFORMANCE): Compute a single value only
        self.left_multiply_by_basis_inverse(original_column).into_column().get(i).cloned()
    }

    fn should_refactor(&self) -> bool {
        // TODO(ENHANCEMENT): What would be a good decision rule?
        self.updates.len() >= ITERATIONS_BEFORE_REFACTOR
    }

    fn basis_inverse_row(&self, mut row: usize) -> SparseVector<Self::F, Self::F> {
        self.right_multiply_by_basis_inverse(IdentityColumn::new(row).iter())
    }

    fn m(&self) -> usize {
        self.len()
    }
}

impl<F: fmt::Debug> fmt::Display for ForrestTomlinUpdate<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Pivot index: {}", self.rotation.index)?;
        writeln!(f, "R: {:?}", self.eta_file)
    }
}

#[cfg(test)]
mod test {
    use relp_num::RationalBig;
    use relp_num::RB;

    use crate::algorithm::two_phase::matrix_provider::column::Column;
    use crate::algorithm::two_phase::matrix_provider::column::identity::IdentityColumn;
    use crate::algorithm::two_phase::matrix_provider::matrix_data;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::BasisInverse;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::{ColumnAndSpike, LUDecomposition};
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::forrest_tomlin_update::ForrestTomlinUpdate;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::FullPermutation;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::ColumnComputationInfo;
    use crate::data::linear_algebra::vector::{SparseVector, Vector};

    #[test]
    fn wikipedia_example2() {
        let rows = vec![vec![(0, RB!(-1)), (1, RB!(3, 2))], vec![(0, RB!(1)), (1, RB!(-1))]];
        let result = LUDecomposition::<_, ForrestTomlinUpdate<_>>::rows(rows);
        let expected = LUDecomposition {
            row_permutation: FullPermutation::identity(2),
            column_permutation: FullPermutation::identity(2),
            lower_triangular: vec![vec![(1, RB!(-1))]],
            upper_triangular: vec![vec![(0, RB!(-1))], vec![(0, RB!(3, 2)), (1, RB!(1, 2))]],
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

    /// Spike is the column which is already there.
    #[test]
    fn no_change() {
        let mut initial = LUDecomposition::<RationalBig, ForrestTomlinUpdate<_>>::identity(3);

        let spike = vec![(1, RB!(1))];
        let column_computation_info = ColumnAndSpike {
            column: SparseVector::new(spike.clone(), 3),
            spike,
        };
        initial.change_basis(1, column_computation_info);
        let modified = initial;

        let mut expected = LUDecomposition::<RationalBig, ForrestTomlinUpdate<_>>::identity(3);
        expected.updates.push(ForrestTomlinUpdate::new(vec![], 1, 3));
        assert_eq!(modified, expected);
    }

    #[test]
    fn from_identity_2() {
        let mut identity = LUDecomposition::<RationalBig, ForrestTomlinUpdate<_>>::identity(2);

        let spike = vec![(0, RB!(1)), (1, RB!(1))];
        let column_computation_info = ColumnAndSpike {
            column: SparseVector::new(spike.clone(), 2),
            spike,
        };
        identity.change_basis(0, column_computation_info);
        let modified = identity;

        let expected = LUDecomposition {
            row_permutation: FullPermutation::identity(2),
            column_permutation: FullPermutation::identity(2),
            lower_triangular: vec![vec![]],
            upper_triangular: vec![vec![(0, RB!(1))], vec![(0, RB!(1)), (1, RB!(1))]],
            updates: vec![ForrestTomlinUpdate::new(vec![], 0, 2)],
        };
        assert_eq!(modified, expected);
    }

    /// Doesn't require any `r`, permutations only are sufficient
    #[test]
    fn from_5x5_identity_no_r() {
        let m = 5;
        let mut initial = LUDecomposition::<RationalBig, ForrestTomlinUpdate<_>>::identity(5);

        let spike = vec![(0, RB!(2)), (1, RB!(3)), (2, RB!(5)), (3, RB!(7))];
        let column_computation_info = ColumnAndSpike {
            column: SparseVector::new(spike.clone(), m),
            spike,
        };
        initial.change_basis(1, column_computation_info);
        let modified = initial;

        let expected = LUDecomposition {
            row_permutation: FullPermutation::identity(m),
            column_permutation: FullPermutation::identity(m),
            lower_triangular: vec![vec![]; m - 1],
            upper_triangular: vec![
                vec![(0, RB!(1))],
                vec![(1, RB!(1))],
                vec![(2, RB!(1))],
                vec![(3, RB!(1))],
                vec![(0, RB!(2)), (1, RB!(5)), (2, RB!(7)), (4, RB!(3))],
            ],
            updates: vec![ForrestTomlinUpdate::new(vec![], 1, m)],
        };
        assert_eq!(modified, expected);
    }

    /// Does require an `r`, permutations are not sufficient.
    #[test]
    fn from_4x4_identity() {
        let m = 4;
        let mut initial = LUDecomposition {
            row_permutation: FullPermutation::identity(m),
            column_permutation: FullPermutation::identity(m),
            lower_triangular: vec![vec![]; m - 1],
            upper_triangular: vec![
                vec![(0, RB!(1))],
                vec![(1, RB!(1))],
                vec![(2, RB!(4))],
                vec![(1, RB!(5)), (3, RB!(6))],
            ],
            updates: vec![],
        };

        let spike = vec![(1, RB!(2)), (2, RB!(3)), (3, RB!(4))];
        let column_computation_info = ColumnAndSpike {
            // They are the same in this case, row permutation and lower are identity
            column: SparseVector::new(spike.clone(), m),
            spike,
        };
        initial.change_basis(1, column_computation_info);
        let modified = initial;

        let expected = LUDecomposition {
            row_permutation: FullPermutation::identity(m),
            column_permutation: FullPermutation::identity(m),
            lower_triangular: vec![vec![]; m - 1],
            upper_triangular: vec![
                vec![(0, RB!(1))],
                vec![(1, RB!(4))],
                vec![(2, RB!(6))],
                vec![(1, RB!(3)), (2, RB!(4)), (3, -RB!(8, 6))],
            ],
            updates: vec![ForrestTomlinUpdate::new(vec![(3, RB!(5, 6))], 1, m)],
        };
        assert_eq!(modified, expected);

        // Columns
        assert_eq!(
            modified.left_multiply_by_basis_inverse(IdentityColumn::new(0).iter()).into_column(),
            SparseVector::standard_basis_vector(0, m),
        );
        assert_eq!(
            modified.left_multiply_by_basis_inverse(IdentityColumn::new(1).iter()).into_column(),
            SparseVector::new(vec![(1, RB!(-3, 4)), (2, RB!(9, 16)), (3, RB!(1, 2))], m),
        );
        assert_eq!(
            modified.left_multiply_by_basis_inverse(IdentityColumn::new(2).iter()).into_column(),
            SparseVector::new(vec![(2, RB!(1, 4))], m),
        );
        assert_eq!(
            modified.left_multiply_by_basis_inverse(IdentityColumn::new(3).iter()).into_column(),
            SparseVector::new(vec![(1, RB!(5, 8)), (2, RB!(-15, 32)), (3, RB!(-1, 4))], m),
        );

        // Rows
        assert_eq!(
            modified.basis_inverse_row(0),
            SparseVector::standard_basis_vector(0, m),
        );
        assert_eq!(
            modified.basis_inverse_row(1),
            SparseVector::new(vec![(1, RB!(-3, 4)), (3, RB!(5, 8))], m),
        );
        assert_eq!(
            modified.basis_inverse_row(2),
            SparseVector::new(vec![(1, RB!(9, 16)), (2, RB!(1, 4)), (3, RB!(-15, 32))], m),
        );
        assert_eq!(
            modified.basis_inverse_row(3),
            SparseVector::new(vec![(1, RB!(1, 2)), (3, RB!(-1, 4))], m),
        );
    }

    /// From "A review of the LU update in the simplex algorithm" by Joseph M. Elble and
    /// Nikolaes V. Sahinidis, Int. J. Mathematics in Operational Research, Vol. 4, No. 4, 2012.
    #[test]
    fn from_5x5_identity() {
        let m = 5;
        let mut initial = LUDecomposition {
            row_permutation: FullPermutation::identity(m),
            column_permutation: FullPermutation::identity(m),
            lower_triangular: vec![vec![]; m - 1],
            upper_triangular: vec![
                vec![(0, RB!(11))],
                vec![(0, RB!(12)), (1, RB!(22))],
                vec![(0, RB!(13)), (1, RB!(23)), (2, RB!(33))],
                vec![(0, RB!(14)), (1, RB!(24)), (2, RB!(34)), (3, RB!(44))],
                vec![(0, RB!(15)), (1, RB!(25)), (2, RB!(35)), (3, RB!(45)), (4, RB!(55))],
            ],
            updates: vec![],
        };

        let spike = vec![(0, RB!(12)), (1, RB!(22)), (2, RB!(32)), (3, RB!(42))];
        let column_computation_info = ColumnAndSpike {
            column: SparseVector::new(spike.clone(), m),
            spike,
        };
        initial.change_basis(1, column_computation_info);
        let modified = initial;

        let expected = LUDecomposition {
            row_permutation: FullPermutation::identity(m),
            column_permutation: FullPermutation::identity(m),
            lower_triangular: vec![vec![]; m - 1],
            upper_triangular: vec![
                vec![(0, RB!(11))],
                vec![(0, RB!(13)), (1, RB!(33))],
                vec![(0, RB!(14)), (1, RB!(34)), (2, RB!(44))],
                vec![(0, RB!(15)), (1, RB!(35)), (2, RB!(45)), (3, RB!(55))],
                vec![(0, RB!(12)), (1, RB!(32)), (2, RB!(42)), (4, RB!(-215, 363))],
            ],
            updates: vec![ForrestTomlinUpdate::new(
                vec![
                    (2, RB!(23, 33)),
                    (3, RB!(24 * 33 - 34 * 23, 33 * 44)),
                    (4, RB!(43, 7986)),
                ],
                1,
                m,
            )],
        };
        assert_eq!(modified, expected);

        // Columns
        assert_eq!(
            modified.left_multiply_by_basis_inverse(IdentityColumn::new(0).iter()).into_column(),
            SparseVector::new(vec![(0, RB!(1, 11))], m),
        );
        assert_eq!(
            modified.left_multiply_by_basis_inverse(IdentityColumn::new(1).iter()).into_column(),
            SparseVector::new(vec![(0, RB!(-2, 11)), (1, RB!(-363, 215)), (2, RB!(-1, 43)), (3, RB!(693, 430))], m),
        );
        assert_eq!(
            modified.left_multiply_by_basis_inverse(IdentityColumn::new(2).iter()).into_column(),
            SparseVector::new(vec![(0, RB!(1, 11)), (1, RB!(253, 215)), (2, RB!(2, 43)), (3, RB!(-483, 430))], m),
        );
        assert_eq!(
            modified.left_multiply_by_basis_inverse(IdentityColumn::new(3).iter()).into_column(),
            SparseVector::new(vec![(1, RB!(1, 86)), (2, RB!(-1, 43)), (3, RB!(1, 86))], m),
        );
        assert_eq!(
            modified.left_multiply_by_basis_inverse(IdentityColumn::new(4).iter()).into_column(),
            SparseVector::new(vec![(1, RB!(1, 110)), (3, RB!(-3, 110)), (4, RB!(1, 55))], m),
        );
        // Sum of two
        let iter = matrix_data::Column::TwoSlack([(0, RB!(1)), (1, RB!(1))], []);
        assert_eq!(
            modified.left_multiply_by_basis_inverse(iter.iter()).into_column(),
            SparseVector::new(vec![(0, RB!(-1, 11)), (1, RB!(-363, 215)), (2, RB!(-1, 43)), (3, RB!(693, 430))], m),
        );

        // Rows
        assert_eq!(
            modified.basis_inverse_row(0),
            SparseVector::new(vec![(0, RB!(1, 11)), (1, RB!(-2, 11)), (2, RB!(1, 11))], m),
        );
        assert_eq!(
            modified.basis_inverse_row(1),
            SparseVector::new(vec![(1, RB!(-363, 215)), (2, RB!(253, 215)), (3, RB!(1, 86)), (4, RB!(1, 110))], m),
        );
        assert_eq!(
            modified.basis_inverse_row(2),
            SparseVector::new(vec![(1, RB!(-1, 43)), (2, RB!(2, 43)), (3, RB!(-1, 43))], m),
        );
        assert_eq!(
            modified.basis_inverse_row(3),
            SparseVector::new(vec![(1, RB!(693, 430)), (2, RB!(-483, 430)), (3, RB!(1, 86)), (4, RB!(-3, 110))], m),
        );
        assert_eq!(
            modified.basis_inverse_row(4),
            SparseVector::new(vec![(4, RB!(1, 55))], m),
        );
    }
}
