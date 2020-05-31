//! # High-level Simplex logic
//!
//! High level methods implementing the simplex algorithm. The details of this logic are hidden away
//! mostly in the `Tableau` type.
use crate::algorithm::simplex::data::{Artificial, is_in_basic_feasible_solution_state};
use crate::algorithm::simplex::data::NonArtificial;
use crate::algorithm::simplex::data::Tableau;
use crate::algorithm::simplex::matrix_provider::MatrixProvider;
use crate::algorithm::simplex::strategy::pivot_rule::PivotRule;
use crate::data::linear_algebra::vector::SparseVector;
use crate::data::number_types::traits::{Field, OrderedField};
use itertools::repeat_n;

/// Reduces the artificial cost of the basic feasible solution to zero, if possible. In doing so, a
/// basic feasible solution to the `CanonicalForm` linear program is found.
///
/// # Arguments
///
/// * `tableau`: Artificial tableau with a valid basis. This basis will typically consist of only
/// artificial variables.
///
/// # Return value
///
/// Whether the tableau might allow a basic feasible solution without artificial variables.
pub(crate) fn artificial_primal<'a, OF: OrderedField, MP: 'a + MatrixProvider<OF>, PR: PivotRule<Artificial>>(
    tableau: &'a mut Tableau<OF, Artificial, MP>,
) -> FeasibilityResult {
    let mut rule = PR::new();
    let mut i = 0;
    loop {
        debug_assert!(is_in_basic_feasible_solution_state(tableau));

        if i % 1 == 0 {
            println!("{}\t{}", i, tableau.objective_function_value());
            // println!(
            //     "{}\n{}",
            //     repeat_n("X", 196).collect::<Vec<_>>().concat(), tableau,
            // );
        }
        i += 1;
        match rule.select_primal_pivot_column(tableau) {
            Some((column_nr, cost)) => {
                let column = tableau.generate_column(column_nr);
                match tableau.select_primal_pivot_row(&column) {
                    Some(row_nr) => tableau.bring_into_basis(column_nr, row_nr, &column, cost),
                    None => panic!("Artificial cost can not be unbounded."),
                }
            },
            // TODO: We accept numerical errors to swing the objective function even below the trimming range by requiring `<= 0` instead of `== 0`
            None => break if tableau.objective_function_value() <= OF::additive_identity() {
                if tableau.has_artificial_in_basis() {
                    let rows_to_remove = remove_artificial_basis_variables(tableau);
                    FeasibilityResult::Feasible(Rank::Deficient(rows_to_remove))
                } else {
                    FeasibilityResult::Feasible(Rank::Full)
                }
            } else {
                FeasibilityResult::Infeasible
            },
        }
    }
}

/// LP's can be either feasible (allowing at least one solution) or infeasible (allowing no
/// solutions).
///
/// If the problem is feasible, it can either have full rank, or be rank deficient.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum FeasibilityResult {
    Feasible(Rank),
    Infeasible,
}

/// A matrix or linear program either has full rank, or be rank deficient.
///
/// In case it is rank deficient, a sorted, deduplicated list of (row)indices should be provided,
/// that when removed, makes the matrix or linear program full rank.
#[derive(Debug, Eq, PartialEq)]
pub (crate) enum Rank {
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
fn remove_artificial_basis_variables<'a, F: Field, MP>(
    tableau: &mut Tableau<F, Artificial, MP>,
) -> Vec<usize> where MP: 'a + MatrixProvider<F> {
    let artificial_variable_indices = tableau.artificial_basis_columns();
    let mut rows_to_remove = Vec::new();

    for artificial in artificial_variable_indices.into_iter() {
        let pivot_row = artificial; // The problem was initialized with the artificial variable as a basis column, and it is still in the basis
        let column_cost = (tableau.nr_artificial_variables()..tableau.nr_columns())
            .filter(|j| !tableau.is_in_basis(j))
            .map(|j| (j, tableau.relative_cost(j)))
            // TODO: Check this zero comparison, is no tolerance needed?
            .filter(|&(_, cost)| cost == F::additive_identity())
            // TODO: Check this zero comparison, is no tolerance needed?
            .filter(|&(j, _)| tableau.generate_element(pivot_row, j) != F::additive_identity())
            .nth(0); // Pick the first one

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
pub(crate) fn primal<'a, OF: OrderedField, MP: 'a + MatrixProvider<OF>, PR: PivotRule<NonArtificial>>(
    tableau: &mut Tableau<OF, NonArtificial, MP>,
) -> OptimizationResult<OF> {
    let mut rule = PR::new();
    loop {
        debug_assert!(is_in_basic_feasible_solution_state(&tableau));

        // println!("{}", tableau);
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

#[derive(Eq, PartialEq, Debug)]
pub enum OptimizationResult<F: Field> {
    Infeasible,
    FiniteOptimum(SparseVector<F>),
    Unbounded,
}

#[cfg(test)]
mod test {
    use num::rational::Ratio;

    use crate::algorithm::simplex::data::{Artificial, Tableau};
    use crate::algorithm::simplex::logic::{artificial_primal, FeasibilityResult, OptimizationResult, primal, Rank};
    use crate::algorithm::simplex::solve_relaxation;
    use crate::algorithm::simplex::strategy::pivot_rule::FirstProfitable;
    use crate::data::linear_algebra::vector::SparseVector;
    use crate::tests::problem_2::{create_matrix_data_data, matrix_data_form, tableau_form};

    #[test]
    fn test_simplex() {
        let (constraints, b) = create_matrix_data_data();
        let matrix_data_form = matrix_data_form(&constraints, &b);
        let mut tableau = tableau_form(&matrix_data_form);
        match primal::<_, _, FirstProfitable>(&mut tableau) {
            OptimizationResult::FiniteOptimum(_) => {
                assert_eq!(tableau.objective_function_value(), Ratio::<i32>::new(9, 2))
            },
            _ => assert!(false),
        }
    }

    #[test]
    fn test_finding_bfs() {
        let (constraints, b) = create_matrix_data_data();
        let matrix_data_form = matrix_data_form(&constraints, &b);
        let mut tableau = Tableau::<_, Artificial, _>::new(&matrix_data_form);
        assert_eq!(artificial_primal::<_, _, FirstProfitable>(&mut tableau), FeasibilityResult::Feasible(Rank::Full));
    }

    //    #[test]
    //    fn test_phase_two() {
    //        // TODO
    //    }

   #[test]
   fn test_solve_matrix() {
       let (constraints, b) = create_matrix_data_data();
       let matrix_data_form = matrix_data_form(&constraints, &b);
       let result = solve_relaxation::<_, _, FirstProfitable, FirstProfitable>(&matrix_data_form);
       // let expected = vec![
       //     (String::from("X1"), 0f64),
       //     (String::from("X2"), 0.5f64),
       //     (String::from("X3"), 0f64),
       //     (String::from("X4"), 2.5f64),
       //     (String::from("X5"), 1.5f64),
       // ]; Optimal value: R64!(4.5)
        assert_eq!(result, OptimizationResult::FiniteOptimum(SparseVector::from_test_tuples(vec![
            (1, 0.5f64),
            (3, 2.5f64),
            (4, 1.5f64),
        ], 5)));
   }

    // TODO
   // #[test]
   // fn test_solve_shortest_path() {
   //     let data = SparseMatrix::from_data(vec![vec![0f64, 1f64, 0f64, 2f64, 0f64],
   //                                             vec![0f64, 0f64, 1f64, 1f64, 0f64],
   //                                             vec![0f64, 0f64, 0f64, 1f64, 1f64],
   //                                             vec![0f64, 0f64, 0f64, 0f64, 2f64],
   //                                             vec![0f64, 0f64, 0f64, 0f64, 0f64]]);
   //     let graph = ShortestPathNetwork::new(data, 0, 4);
   //     let result = solve(&graph).unwrap().1;
   //     assert_eq!(result, 3f64);
   // }
}
