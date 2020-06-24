//! # The Simplex algorithm
//!
//! This module contains all data structures and logic specific to the simplex algorithm. The
//! algorithm is implemented as described in chapters 2 and 4 of Combinatorial Optimization, a book
//! by Christos H. Papadimitriou and Kenneth Steiglitz.
use std::collections::HashSet;

use crate::algorithm::{OptimizationResult, SolveRelaxation};
use crate::algorithm::two_phase::matrix_provider::MatrixProvider;
use crate::algorithm::two_phase::matrix_provider::remove_rows::RemoveRows;
use crate::algorithm::two_phase::strategy::pivot_rule::{FirstProfitable, PivotRule};
use crate::algorithm::two_phase::tableau::{is_in_basic_feasible_solution_state, Tableau};
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::CarryMatrix;
use crate::algorithm::two_phase::tableau::inverse_maintenance::InverseMaintenance;
use crate::algorithm::two_phase::tableau::kind::{Artificial, NonArtificial};
use crate::data::linear_algebra::traits::SparseElementZero;
use crate::data::number_types::traits::{Field, FieldRef, OrderedField, OrderedFieldRef};

pub mod tableau;
pub mod matrix_provider;
pub mod strategy;

impl<OF, OFZ, MP> SolveRelaxation<OF, OFZ> for MP
where
    OF: OrderedField,
    for<'r> &'r OF: OrderedFieldRef<OF>,
    OFZ: SparseElementZero<OF>,
    Self: MatrixProvider<OF, OFZ>,
{
    default fn solve_relaxation(&self) -> OptimizationResult<OF, OFZ> {
        // Default choice
        // TODO(ENHANCEMENT): Consider implementing a heuristic to decide these strategies
        //  dynamically
        type NonArtificialPR = FirstProfitable;
        // TODO: Extract the matrix inverse maintenance algorithm

        match self.compute_bfs_giving_im() {
            RankedFeasibilityResult::Feasible {
                rank,
                nr_artificial_variables,
                basis,
                inverse_maintainer,
            } => match rank {
                Rank::Deficient(rows_to_remove) if !rows_to_remove.is_empty() => {
                    let rows_removed = RemoveRows::new(self, rows_to_remove);
                    let mut non_artificial = Tableau::<_, _, CarryMatrix<_, _>, NonArtificial<_, _, _>>::from_artificial_removing_rows(
                        inverse_maintainer,
                        nr_artificial_variables,
                        basis,
                        &rows_removed,
                    );
                    primal::<_, _, _, _, NonArtificialPR>(&mut non_artificial)
                },
                _ => {
                    let mut non_artificial_tableau = Tableau::<_, _, _, NonArtificial<_, _, _>>::from_artificial(
                        inverse_maintainer,
                        nr_artificial_variables,
                        basis,
                        self,
                    );
                    primal::<_, _, _, _, NonArtificialPR>(&mut non_artificial_tableau)
                },
            },
            RankedFeasibilityResult::Infeasible => OptimizationResult::Infeasible,
        }
    }
}

/// Computing a feasible solution: the first phase of the two phase method.
///
/// This can happen either using the simplex method, or some more specialized method.
///
/// TODO(ENHANCEMENT): Problems that can only have full rank
pub trait FeasibilityComputeTrait {
    /// Compute a basic feasible solution.
    ///
    /// # Returns
    ///
    /// A value representing the basic feasible solution, or an indicator that there is none.
    fn compute_bfs_giving_im<OF, OFZ, IM>(&self) -> RankedFeasibilityResult<IM>
    where
        OF: OrderedField,
        for<'r> &'r OF: OrderedFieldRef<OF>,
        OFZ: SparseElementZero<OF>,
        IM: InverseMaintenance<OF, OFZ>,
        Self: MatrixProvider<OF, OFZ>,
    ;
}

/// Most generic implementation: finding a basic feasible solution using the Simplex method.
impl<MP> FeasibilityComputeTrait for MP {
    default fn compute_bfs_giving_im<OF, OFZ, IM>(&self) -> RankedFeasibilityResult<IM>
    where
        OF: OrderedField,
        for<'r> &'r OF: OrderedFieldRef<OF>,
        OFZ: SparseElementZero<OF>,
        IM: InverseMaintenance<OF, OFZ>,
        MP: MatrixProvider<OF, OFZ>,
    {
        // TODO(ENHANCEMENT): Consider implementing a heuristic to decide these strategies
        //  dynamically
        type PivotRule = FirstProfitable;

        let artificial_tableau = Tableau::<_, _, _, Artificial<_, _, _>>::new(self);
        artificial_primal::<_, _, _, _, PivotRule>(artificial_tableau)
    }
}

/// A few basis columns are already present in the problem.
///
/// Sometimes, a few variables (like positive slack variables) are available that result in less
/// artificial variables being needed.
///
/// TODO(ARCHITECTURE): Supertrait MatrixProvider?
pub trait PartialInitialBasis {
    /// Return the indices of all positive slack variables.
    ///
    /// This is used to find a basic feasible solution faster using the two phase method.
    ///
    /// # Return value
    ///
    /// Collection of tuples with row and column index of (positive) slack coefficients.
    fn pivot_element_indices(&self) -> Vec<(usize, usize)>;

    /// How many positive slacks there are in the problem.
    ///
    /// Is equal to the length of the value returned by `positive_slack_indices`.
    fn nr_initial_elements(&self) -> usize;
}

impl<MP: PartialInitialBasis> FeasibilityComputeTrait for MP {
    fn compute_bfs_giving_im<OF, OFZ, IM>(&self) -> RankedFeasibilityResult<IM>
        where
            OF: OrderedField,
            for<'r> &'r OF: OrderedFieldRef<OF>,
            OFZ: SparseElementZero<OF>,
            IM: InverseMaintenance<OF, OFZ>,
            Self: MatrixProvider<OF, OFZ>,
    {
        // TODO(ENHANCEMENT): Consider implementing a heuristic to decide these strategies
        //  dynamically
        type PivotRule = FirstProfitable;

        let artificial_tableau = Tableau::<_, _, _, Artificial<_, _, _>>::new_with_partial_basis(self);
        artificial_primal::<_, _, _, _, PivotRule>(artificial_tableau)
    }
}

/// A full basis is already present in the problem.
///
/// If the problem is of type Ax <= b, all (positive) slacks can be used as a basic feasible
/// solution (that is, x = 0 is feasible).
///
/// TODO(ENHANCEMENT): Is a marker trait enough, or should there be a specialized method?
/// TODO(ENHANCEMENT): What about the case where also an easy InverseMaintainer is available?
pub trait FullInitialBasis: PartialInitialBasis {
}

/// Skipping the entire first phase because a basic feasible solution could be cheaply provided.
///
/// TODO(ARCHITECTURE): If this is true, it's ot really two-phase, now is it? Should this solution
///  be provided elsewhere, or the module be renamed?
impl<OF, OFZ, MP> SolveRelaxation<OF, OFZ> for MP
where
    OF: OrderedField,
    for<'r> &'r OF: OrderedFieldRef<OF>,
    OFZ: SparseElementZero<OF>,
    MP: MatrixProvider<OF, OFZ> + FullInitialBasis,
{
    fn solve_relaxation(&self) -> OptimizationResult<OF, OFZ> {
        // TODO(ENHANCEMENT): Consider implementing a heuristic to decide these strategies
        //  dynamically
        type NonArtificialPR = FirstProfitable;
        // TODO: Extract IM type

        let basis_indices = self.pivot_element_indices();
        // Sorting of identity matrix columns
        let inverse_maintainer = CarryMatrix::from_basis_pivots(&basis_indices, self);

        let basis_indices: Vec<usize> = basis_indices.into_iter().map(|(_row, column)| column).collect();
        let basis_columns = basis_indices.iter().copied().collect();
        let mut tableau = Tableau::<_, _, _, NonArtificial<_, _, _>>::new_with_inverse_maintainer(
            self, inverse_maintainer, basis_indices, basis_columns,
        );
        primal::<_, _, _, _, NonArtificialPR>(&mut tableau)
    }
}


/// Reduces the artificial cost of the basic feasible solution to zero, if possible. In doing so, a
/// basic feasible solution to the standard form linear program is found.
///
/// # Arguments
///
/// * `tableau`: Artificial tableau with a valid basis. This basis will typically consist of only
/// artificial variables.
///
/// # Return value
///
/// Whether the tableau might allow a basic feasible solution without artificial variables.
pub(crate) fn artificial_primal<'provider, OF, OFZ, IM, MP, PR>(
    mut tableau: Tableau<OF, OFZ, IM, Artificial<'provider, OF, OFZ, MP>>,
) -> RankedFeasibilityResult<IM>
where
    OF: OrderedField,
    for<'r> &'r OF: OrderedFieldRef<OF>,
    OFZ: SparseElementZero<OF>,
    IM: InverseMaintenance<OF, OFZ>,
    MP: MatrixProvider<OF, OFZ> + 'provider,
    PR: PivotRule,
{
    let mut rule = PR::new();
    loop {
        debug_assert!(is_in_basic_feasible_solution_state(&tableau));

        match rule.select_primal_pivot_column(&tableau) {
            Some((column_nr, cost)) => {
                let column = tableau.generate_column(column_nr);
                match tableau.select_primal_pivot_row(&column) {
                    Some(row_nr) => tableau.bring_into_basis(column_nr, row_nr, &column, cost),
                    None => panic!("Artificial cost can not be unbounded."),
                }
            },
            None => break if tableau.objective_function_value().is_zero() {
                let rank = if tableau.has_artificial_in_basis() {
                    let rows_to_remove = remove_artificial_basis_variables(&mut tableau);
                    if rows_to_remove.is_empty() {
                        Rank::Full
                    } else {
                        Rank::Deficient(rows_to_remove)
                    }
                } else {
                    Rank::Full
                };

                let (im, nr_a, basis) = tableau.export_basis_representation();
                RankedFeasibilityResult::Feasible {
                    rank,
                    nr_artificial_variables: nr_a,
                    inverse_maintainer: im,
                    basis,
                }
            } else {
                RankedFeasibilityResult::Infeasible
            },
        }
    }
}

/// LP's can be either feasible (allowing at least one solution) or infeasible (allowing no
/// solutions).
///
/// If the problem is feasible, it can either have full rank, or be rank deficient.
#[derive(Debug, Eq, PartialEq)]
pub enum RankedFeasibilityResult<IM> {
    /// The problem is feasible and all information necessary to construct a bfs cheaply is in this
    /// variant.
    Feasible {
        /// Whether the problem needs rows to be removed.
        rank: Rank,
        /// The amount of artificial variables that were present in the problem.
        nr_artificial_variables: usize,
        /// The inverse basis that was maintained.
        inverse_maintainer: IM,
        /// Basis indices.
        ///
        /// Note that the second element is just a set version of the elements in the first element.
        basis: (Vec<usize>, HashSet<usize>),
    },
    /// The problem is not feasible.
    Infeasible,
}

/// A matrix or linear program either has full rank, or be rank deficient.
///
/// In case it is rank deficient, a sorted, deduplicated list of (row)indices should be provided,
/// that when removed, makes the matrix or linear program full rank.
#[derive(Debug, Eq, PartialEq)]
pub enum Rank {
    /// The matrix is full rank, no rows need to be removed.
    Full,
    /// The `Vec<usize>` is sorted and contains no duplicate values.
    Deficient(Vec<usize>),
}

/// Removes all artificial variables from the tableau by making a basis change "at zero level", or
/// without change of cost of the current solution.
///
/// # Arguments
///
/// * `tableau`: Tableau to change the basis for.
///
/// # Return value
///
/// A `Vec` with indices of rows that are redundant. Is sorted as a side effect of the algorithm.
///
/// Constraints containing only one variable (that is, they are actually bounds) are read as bounds
/// instead of constraints. All bounds are linearly independent among each other, and with respect
/// to all constraints. As such, they should never be among the redundant rows returned by this
/// method.
fn remove_artificial_basis_variables<F, FZ, IM, MP>(
    tableau: &mut Tableau<F, FZ, IM, Artificial<F, FZ, MP>>,
) -> Vec<usize>
where
    F: Field,
    for<'r> &'r F: FieldRef<F>,
    FZ: SparseElementZero<F>,
    IM: InverseMaintenance<F, FZ>,
    MP: MatrixProvider<F, FZ>,
{
    let mut artificial_variable_indices = tableau.artificial_basis_columns().into_iter().collect::<Vec<_>>();
    artificial_variable_indices.sort();
    let mut rows_to_remove = Vec::new();

    for artificial in artificial_variable_indices {
        // The problem was initialized with the artificial variable as a basis column, and it is still in the basis
        let pivot_row = tableau.pivot_row_from_artificial(artificial);
        let column_cost = (tableau.nr_artificial_variables()..tableau.nr_columns())
            .filter(|j| !tableau.is_in_basis(j))
            .map(|j| (j, tableau.relative_cost(j)))
            .filter(|(_, cost)| cost.is_zero())
            .find(|&(j, _)| !tableau.generate_element(pivot_row, j).is_zero());

        if let Some((pivot_column, cost)) = column_cost {
            let column = tableau.generate_column(pivot_column);
            tableau.bring_into_basis(pivot_column, pivot_row, &column, cost);
        } else {
            rows_to_remove.push(artificial);
        }
    }

    debug_assert!(rows_to_remove.is_sorted());
    rows_to_remove
}

/// Reduces the cost of the basic feasible solution to the minimum.
///
/// While calling this method, a number of requirements should be satisfied:
/// - There should be a valid basis (not necessarily optimal <=> dual feasible <=> c >= 0)
/// - All constraint values need to be positive (primary feasibility)
///
/// TODO: Write debug tests for these requirements
///
/// # Return value
///
/// An `OptimizationResult` indicating whether or not the problem has a finite optimum. It cannot be
/// infeasible, as a feasible solution is needed to start using this method.
pub(crate) fn primal<'provider, OF, OFZ, IM, MP, PR>(
    tableau: &mut Tableau<OF, OFZ, IM, NonArtificial<'provider, OF, OFZ, MP>>,
) -> OptimizationResult<OF, OFZ>
where
    OF: OrderedField,
    for<'r> &'r OF: OrderedFieldRef<OF>,
    OFZ: SparseElementZero<OF>,
    IM: InverseMaintenance<OF, OFZ>,
    MP: MatrixProvider<OF, OFZ> + 'provider,
    PR: PivotRule,
{
    let mut rule = PR::new();
    loop {
        debug_assert!(is_in_basic_feasible_solution_state(&tableau));

        match rule.select_primal_pivot_column(tableau) {
            Some((column_index, cost)) => {
                let column = tableau.generate_column(column_index);
                match tableau.select_primal_pivot_row(&column) {
                    Some(row_index) => tableau.bring_into_basis(
                        column_index,
                        row_index,
                        &column,
                        cost,
                    ),
                    None => break OptimizationResult::Unbounded,
                }
            },
            None => break OptimizationResult::FiniteOptimum(tableau.current_bfs()),
        }
    }
}

#[cfg(test)]
mod test {
    use num::FromPrimitive;
    use num::rational::Ratio;

    use crate::algorithm::{OptimizationResult, SolveRelaxation};
    use crate::algorithm::two_phase::{artificial_primal, primal, Rank, RankedFeasibilityResult};
    use crate::algorithm::two_phase::matrix_provider::matrix_data::{MatrixData, Variable};
    use crate::algorithm::two_phase::strategy::pivot_rule::FirstProfitable;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::CarryMatrix;
    use crate::algorithm::two_phase::tableau::kind::Artificial;
    use crate::algorithm::two_phase::tableau::Tableau;
    use crate::data::linear_algebra::matrix::{ColumnMajor, Order};
    use crate::data::linear_algebra::vector::{Dense as DenseVector, Sparse as SparseVector};
    use crate::data::linear_algebra::vector::test::TestVector;
    use crate::data::linear_program::elements::VariableType;
    use crate::R32;
    use crate::tests::problem_2::{create_matrix_data_data, matrix_data_form, tableau_form};

    #[test]
    fn simplex() {
        let (constraints, b) = create_matrix_data_data();
        let matrix_data_form = matrix_data_form(&constraints, &b);
        let mut tableau = tableau_form(&matrix_data_form);
        let result = primal::<_, _, _, _, FirstProfitable>(&mut tableau);
        assert!(matches!(result, OptimizationResult::FiniteOptimum(_)));
        assert_eq!(tableau.objective_function_value(), R32!(9, 2));
    }

    #[test]
    fn finding_bfs() {
        let (constraints, b) = create_matrix_data_data();
        let matrix_data_form = matrix_data_form(&constraints, &b);
        let tableau = Tableau::<_, _, CarryMatrix<_, _>, Artificial<_, _, _>>::new(&matrix_data_form);
        assert!(matches!(
            artificial_primal::<_, _, _, _, FirstProfitable>(tableau),
            RankedFeasibilityResult::Feasible { rank: Rank::Full, .. }
        ));
    }

    #[test]
    fn solve_matrix() {
        let (constraints, b) = create_matrix_data_data();
        let matrix_data_form = matrix_data_form(&constraints, &b);
        let result = matrix_data_form.solve_relaxation();
        //  Optimal value: R64!(4.5)
        assert_eq!(result, OptimizationResult::FiniteOptimum(SparseVector::from_test_tuples(vec![
            (1, 0.5f64),
            (3, 2.5f64),
            (4, 1.5f64),
        ], 5)));
    }

    #[test]
    fn solve_relaxation_1() {
        type T = Ratio<i32>;

        let constraints = ColumnMajor::from_test_data::<T, T, _, _>(&vec![
            vec![1, 0],
            vec![1, 1],
        ], 2);
        let b = DenseVector::from_test_data(vec![
            3f64 / 2f64,
            5f64 / 2f64,
        ]);
        let variables = vec![
            Variable {
                cost: R32!(-2),
                upper_bound: None,
                variable_type: VariableType::Integer,
            },
            Variable {
                cost: R32!(-1),
                upper_bound: None,
                variable_type: VariableType::Integer,
            },
        ];

        let data = MatrixData::new(&constraints, &b, 0, 2, 0, variables);

        let result = data.solve_relaxation();
        assert_eq!(result, OptimizationResult::FiniteOptimum(SparseVector::from_test_tuples(vec![
            (0, 3f64 / 2f64),
            (1, 1f64),
        ], 4)));
    }
}
