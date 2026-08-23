//! # The Simplex algorithm
//!
//! This module contains all data structures and logic specific to the simplex algorithm. The
//! algorithm is implemented as described in chapters 2 and 4 of Combinatorial Optimization, a book
//! by Christos H. Papadimitriou and Kenneth Steiglitz.
use relp_num::Absorb;

use crate::algorithm::two_phase::matrix_provider::MatrixProvider;
use crate::algorithm::two_phase::matrix_provider::column::identity::Identity;
use crate::algorithm::two_phase::matrix_provider::column::{Column, ColumnNumber};
use crate::algorithm::two_phase::matrix_provider::filter::generic_wrapper::{
    IntoFilteredColumn, RemoveRows,
};
use crate::algorithm::two_phase::phase_one::{
    FeasibilityComputeTrait, FullInitialBasis, Rank, RankedFeasibilityResult,
};
use crate::algorithm::two_phase::strategy::pivot_rule::SteepestDescentAlongObjective;
use crate::algorithm::two_phase::tableau::Tableau;
use crate::algorithm::two_phase::tableau::inverse_maintenance::{InverseMaintainer, ops as im_ops};
use crate::algorithm::two_phase::tableau::kind::artificial::Cost as ArtificialCost;
use crate::algorithm::two_phase::tableau::kind::non_artificial::NonArtificial;
use crate::algorithm::{OptimizationResult, SolveRelaxation};
use crate::data::linear_algebra::vector::{DenseVector, Vector};

pub mod phase_one;
pub mod phase_two;

pub mod matrix_provider;
pub mod strategy;
pub mod tableau;

/// Spread a vector indexed by the kept rows over all rows, putting a zero at each removed one.
fn restore_removed_rows<F: crate::data::linear_algebra::traits::Element + num_traits::Zero>(
    kept: &DenseVector<F>,
    removed: &[usize],
    nr_rows: usize,
) -> DenseVector<F> {
    debug_assert!(removed.is_sorted());
    debug_assert_eq!(kept.len() + removed.len(), nr_rows);

    let mut values = Vec::with_capacity(nr_rows);
    let mut next_removed = 0;
    let mut next_kept = 0;
    for row in 0..nr_rows {
        if next_removed < removed.len() && removed[next_removed] == row {
            next_removed += 1;
            values.push(F::zero());
        } else {
            values.push(kept[next_kept].clone());
            next_kept += 1;
        }
    }

    DenseVector::new(values, nr_rows)
}

impl<MP> SolveRelaxation for MP
where
    MP: MatrixProvider<Column: Identity + IntoFilteredColumn>,
{
    // TODO(ENHANCEMENT): Specialize for MatrixProviders that can be filtered directly.
    default fn solve_relaxation<'provider, IM>(&'provider self) -> OptimizationResult<IM::F>
    where
        IM: InverseMaintainer<F: im_ops::FieldHR>,
        IM::F: Absorb<<<Self as MatrixProvider>::Column as Column>::F>,
        IM::F: Absorb<ArtificialCost>,
        IM::F: Absorb<MP::Cost<'provider>>,
        IM::F: Absorb<MP::Rhs>,
    {
        match self.compute_bfs_giving_im::<IM>() {
            RankedFeasibilityResult::Feasible {
                rank,
                nr_artificial_variables,
                basis,
                inverse_maintainer,
            } => match rank {
                Rank::Deficient(rows_to_remove) if !rows_to_remove.is_empty() => {
                    let rows_removed = RemoveRows::new(self, rows_to_remove);
                    let mut non_artificial =
                        Tableau::<_, NonArtificial<_>>::from_artificial_removing_rows(
                            inverse_maintainer,
                            nr_artificial_variables,
                            basis,
                            &rows_removed,
                        );
                    // TODO(ENHANCEMENT): Consider implementing a heuristic to decide these
                    //  strategies dynamically
                    let result = phase_two::primal::<_, _, SteepestDescentAlongObjective<_>>(
                        &mut non_artificial,
                    );

                    // The tableau was looking at a problem with rows missing, so its dual vector
                    // is indexed by the rows that were kept. Putting a zero where each removed row
                    // was gives a dual for the problem as it stands: a removed row is a
                    // combination of the kept ones, and asks nothing of the solution that they do
                    // not already ask.
                    match result {
                        OptimizationResult::FiniteOptimum(mut solution) => {
                            solution.dual = restore_removed_rows(
                                &solution.dual,
                                &rows_removed.rows_to_skip,
                                self.nr_rows(),
                            );
                            solution.redundant_rows = rows_removed.rows_to_skip.to_vec();
                            OptimizationResult::FiniteOptimum(solution)
                        }
                        other => other,
                    }
                }
                _ => {
                    let mut non_artificial_tableau =
                        Tableau::<_, NonArtificial<_>>::from_artificial(
                            inverse_maintainer,
                            nr_artificial_variables,
                            basis,
                            self,
                        );
                    // TODO(ENHANCEMENT): Consider implementing a heuristic to decide these
                    //  strategies dynamically
                    phase_two::primal::<_, _, SteepestDescentAlongObjective<_>>(
                        &mut non_artificial_tableau,
                    )
                }
            },
            RankedFeasibilityResult::Infeasible => OptimizationResult::Infeasible,
        }
    }
}

/// Skipping the entire first phase because a basic feasible solution could be cheaply provided.
///
/// TODO(ARCHITECTURE): If this is true, it's ot really two-phase, now is it? Should this solution
///  be provided elsewhere, or the module be renamed?
impl<MP: FullInitialBasis> SolveRelaxation for MP
where
    // TODO(ARCHITECTURE): The <MP as MatrixProvider>::Column: IdentityColumn bound is needed
    //  because of limitations of the specialization feature; overlap is not (yet) allowed.
    MP: MatrixProvider<Column: Identity + IntoFilteredColumn>,
    MP::Rhs: 'static + ColumnNumber,
{
    fn solve_relaxation<'provider, IM>(&'provider self) -> OptimizationResult<IM::F>
    where
        IM: InverseMaintainer<F: im_ops::FieldHR>,
        IM::F: Absorb<<<Self as MatrixProvider>::Column as Column>::F>,
        IM::F: Absorb<MP::Cost<'provider>>,
        IM::F: Absorb<Self::Rhs>,
    {
        let basis_indices = self.pivot_element_indices();
        // Sorting of identity matrix columns
        let inverse_maintainer = IM::from_basis_pivots(&basis_indices, self);

        let basis_indices = basis_indices
            .into_iter()
            .map(|(_row, column)| column)
            .collect();
        let mut tableau = Tableau::<_, NonArtificial<_>>::new_with_inverse_maintainer(
            self,
            inverse_maintainer,
            basis_indices,
        );
        // TODO(ENHANCEMENT): Consider implementing a heuristic to decide these strategies
        //  dynamically
        phase_two::primal::<_, _, SteepestDescentAlongObjective<_>>(&mut tableau)
    }
}

#[cfg(test)]
mod test;
