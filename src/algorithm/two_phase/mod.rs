//! # The Simplex algorithm
//!
//! This module contains all data structures and logic specific to the simplex algorithm. The
//! algorithm is implemented as described in chapters 2 and 4 of Combinatorial Optimization, a book
//! by Christos H. Papadimitriou and Kenneth Steiglitz.
use crate::algorithm::{OptimizationResult, SolveRelaxation};
use crate::algorithm::two_phase::matrix_provider::column::{Column, ColumnNumber};
use crate::algorithm::two_phase::matrix_provider::column::identity::Identity;
use crate::algorithm::two_phase::matrix_provider::filter::generic_wrapper::{IntoFilteredColumn, RemoveRows};
use crate::algorithm::two_phase::matrix_provider::MatrixProvider;
use crate::algorithm::two_phase::phase_one::{FeasibilityComputeTrait, FullInitialBasis, Rank, RankedFeasibilityResult};
use crate::algorithm::two_phase::strategy::pivot_rule::SteepestDescentAlongObjective;
use crate::algorithm::two_phase::tableau::inverse_maintenance::{InverseMaintener, ops as im_ops};
use crate::algorithm::two_phase::tableau::kind::artificial::Cost as ArtificialCost;
use crate::algorithm::two_phase::tableau::kind::non_artificial::NonArtificial;
use crate::algorithm::two_phase::tableau::Tableau;

pub mod phase_one;
pub mod phase_two;

pub mod tableau;
pub mod matrix_provider;
pub mod strategy;

impl<MP> SolveRelaxation for MP
where
    MP: MatrixProvider<Column: Identity + IntoFilteredColumn>,
{
    // TODO(ENHANCEMENT): Specialize for MatrixProviders that can be filtered directly.
    default fn solve_relaxation<IM>(&self) -> OptimizationResult<IM::F>
    where
        IM: InverseMaintener<F:
            im_ops::FieldHR +
            im_ops::Column<<<Self as MatrixProvider>::Column as Column>::F> +
            im_ops::Cost<ArtificialCost> +
            im_ops::Rhs<MP::Rhs> +
        >,
        for<'r> IM::F: im_ops::Cost<MP::Cost<'r>>,
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
                    let mut non_artificial = Tableau::<_, NonArtificial<_>>::from_artificial_removing_rows(
                        inverse_maintainer,
                        nr_artificial_variables,
                        basis,
                        &rows_removed,
                    );
                    // TODO(ENHANCEMENT): Consider implementing a heuristic to decide these
                    //  strategies dynamically
                    phase_two::primal::<_, _, SteepestDescentAlongObjective<_>>(&mut non_artificial)
                },
                _ => {
                    let mut non_artificial_tableau = Tableau::<_, NonArtificial<_>>::from_artificial(
                        inverse_maintainer,
                        nr_artificial_variables,
                        basis,
                        self,
                    );
                    // TODO(ENHANCEMENT): Consider implementing a heuristic to decide these
                    //  strategies dynamically
                    phase_two::primal::<_, _, SteepestDescentAlongObjective<_>>(&mut non_artificial_tableau)
                },
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
    fn solve_relaxation<IM>(&self) -> OptimizationResult<IM::F>
    where
        IM: InverseMaintener<F:
            im_ops::FieldHR +
            im_ops::Column<<<Self as MatrixProvider>::Column as Column>::F> +
            im_ops::Rhs<Self::Rhs> +
            im_ops::Column<Self::Rhs> +
        >,
        for<'r> IM::F: im_ops::Cost<MP::Cost<'r>>,
    {
        let basis_indices = self.pivot_element_indices();
        // Sorting of identity matrix columns
        let inverse_maintainer = IM::from_basis_pivots(&basis_indices, self);

        let basis_indices = basis_indices.into_iter().map(|(_row, column)| column).collect();
        let mut tableau = Tableau::<_, NonArtificial<_>>::new_with_inverse_maintainer(
            self, inverse_maintainer, basis_indices,
        );
        // TODO(ENHANCEMENT): Consider implementing a heuristic to decide these strategies
        //  dynamically
        phase_two::primal::<_, _, SteepestDescentAlongObjective<_>>(&mut tableau)
    }
}

#[cfg(test)]
mod test;
