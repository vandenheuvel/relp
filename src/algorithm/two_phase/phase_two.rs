//! # Phase two: improving a basic feasible solution
use crate::algorithm::OptimizationResult;
use crate::algorithm::two_phase::matrix_provider::column::Column;
use crate::algorithm::two_phase::matrix_provider::MatrixProvider;
use crate::algorithm::two_phase::strategy::pivot_rule::PivotRule;
use crate::algorithm::two_phase::tableau::{debug_assert_in_basic_feasible_solution_state, Tableau};
use crate::algorithm::two_phase::tableau::inverse_maintenance::{ColumnComputationInfo, InverseMaintainer, ops as im_ops};
use crate::algorithm::two_phase::tableau::kind::non_artificial::NonArtificial;

/// Reduces the cost of the basic feasible solution to the minimum.
///
/// While calling this method, a number of requirements should be satisfied:
/// - There should be a valid basis (not necessarily optimal <=> dual feasible <=> c >= 0)
/// - All constraint values need to be positive (primary feasibility)
///
/// TODO(CORRECTNESS): Write debug tests for these requirements
///
/// # Return value
///
/// An `OptimizationResult` indicating whether or not the problem has a finite optimum. It cannot be
/// infeasible, as a feasible solution is needed to start using this method.
pub fn primal<'provider, IM, MP, PR>(
    tableau: &mut Tableau<IM, NonArtificial<'provider, MP>>,
) -> OptimizationResult<IM::F>
where
    IM: InverseMaintainer<F:
        im_ops::FieldHR +
        im_ops::Column<<MP::Column as Column>::F> +
        im_ops::Cost<MP::Cost<'provider>> +
    >,
    MP: MatrixProvider,
    PR: PivotRule<IM::F>,
{
    let mut rule = PR::new(tableau);

    loop {
        debug_assert!({
            debug_assert_in_basic_feasible_solution_state(&tableau);
            true
        });

        match rule.select_primal_pivot_column(tableau) {
            Some((pivot_column_index, cost)) => {
                let column_with_info = tableau.generate_column(pivot_column_index);
                match tableau.select_primal_pivot_row(column_with_info.column()) {
                    Some(pivot_row_index) => {
                        let basis_change_computation_info = tableau.bring_into_basis(
                            pivot_column_index, pivot_row_index,
                            column_with_info, cost,
                        );
                        rule.after_basis_update(basis_change_computation_info,&tableau);
                    }
                    None => break OptimizationResult::Unbounded,
                }
            }
            None => break OptimizationResult::FiniteOptimum(tableau.current_bfs()),
        }
    }
}
