use std::collections::{HashMap, HashSet};

use algorithm::simplex::data::{CostRow, Tableau};
use algorithm::simplex::EPSILON;
use algorithm::simplex::tableau_provider::TableauProvider;
use data::linear_algebra::matrix::{DenseMatrix, Matrix, SparseMatrix};
use data::linear_algebra::vector::{DenseVector, SparseVector, Vector};
use data::linear_program::elements::{LinearProgramType, Variable, VariableType};

/// Solves a linear program using the Simplex method.
pub fn solve<T>(provider: &T) -> Result<(Vec<(String, f64)>, f64), String>
    where T: TableauProvider {
    let artificial_tableau = Tableau::new_artificial(provider);
    let (carry, basis) = match phase_one(artificial_tableau) {
        PhaseOneResult::FoundBFS(carry, basis) => (carry, basis),
        PhaseOneResult::Infeasible => return Err(format!("LP not feasible")),
        PhaseOneResult::ArtificialRemain(tableau, artificial_basis_columns) => {
            // TODO: MIPLIB problem 50v-10 doesn't work with this line. Neither does the graph problem
//            remove_artificial(tableau, artificial_basis_columns)
            (tableau.carry(), tableau.basis_columns())
        },
    };

    let tableau = Tableau::new(provider, carry, basis);
    match phase_two(tableau) {
        LinearProgramType::FiniteOptimum(optimum, cost) =>
            Ok((provider.human_readable_bfs(optimum), cost)),
        LinearProgramType::Infeasible => Err(format!("LP infeasible")),
        LinearProgramType::Unbounded => Err(format!("LP unbounded")),
    }
}

/// Reduces the artificial cost of the basic feasible solution to zero, if possible. In doing so, a
/// basic feasible solution to the `CanonicalForm` linear program is found.
fn phase_one<T>(mut tableau: Tableau<T>) -> PhaseOneResult<T> where T: TableauProvider {
    let mut artificial_basis_columns = (0..tableau.nr_rows()).collect::<HashSet<_>>();

    loop {
        match (0..tableau.nr_columns())
            .filter(|column| !tableau.is_in_basis(column) || artificial_basis_columns.contains(column))
            .find(|column| tableau.relative_cost(CostRow::Artificial, *column) < -EPSILON) {
            Some(column_nr) => {
                let column = tableau.generate_column(column_nr);
                let pivot_row = tableau.find_pivot_row(&column);
                artificial_basis_columns.remove(tableau.get_basis_column(&pivot_row).unwrap());
                tableau.bring_into_basis(pivot_row, column_nr, &column)
            },
            None => break if tableau.cost(CostRow::Artificial) < EPSILON {
                if artificial_basis_columns.is_empty() {
                    PhaseOneResult::FoundBFS(tableau.carry(), tableau.basis_columns())
                } else {
                    PhaseOneResult::ArtificialRemain(tableau, artificial_basis_columns)
                }
            } else {
                PhaseOneResult::Infeasible
            },
        }
    }
}

/// Removes all artificial variables from the tableau by making a basis change "at zero level", or
/// without change of cost of the current solution.
fn remove_artificial<T>(mut tableau: Tableau<T>, artificials: HashSet<usize>) -> (DenseMatrix, HashMap<usize, usize>)
    where T: TableauProvider {
    for artificial in artificials.into_iter() {
        debug_assert!(tableau.relative_cost(CostRow::Artificial, artificial).abs() < EPSILON);

        let artificial_column = tableau.generate_artificial_column(artificial);
        let pivot_row = artificial_column.values()
            .find(|(_, v)| (1f64 - v).abs() < EPSILON)
            .unwrap().0 - 1 - 1;

        for non_artificial in tableau.nr_rows()..tableau.nr_columns() {
            if !tableau.is_in_basis(&non_artificial) {
                if tableau.relative_cost(CostRow::Artificial, non_artificial).abs() < EPSILON {
                    let column = tableau.generate_column(non_artificial);
                    tableau.bring_into_basis(pivot_row, non_artificial, &column);
                    break;
                }
            }
        }

        debug_assert!(!tableau.is_in_basis(&artificial));
    }

    (tableau.carry(), tableau.basis_columns())
}

/// Reduces the cost of the basic feasible solution to the minimum.
fn phase_two<T>(mut tableau: Tableau<T>) -> LinearProgramType where T: TableauProvider {
    loop {
        // TODO: when is an LP unbounded?
        match tableau.profitable_column(CostRow::Actual) {
            Some(column_nr) => {
                let column = tableau.generate_column(column_nr);
                let pivot_row = tableau.find_pivot_row(&column);
                tableau.bring_into_basis(pivot_row, column_nr, &column);
            },
            None => break LinearProgramType::FiniteOptimum(tableau.current_bfs(),
                                                           tableau.cost(CostRow::Actual)),
        }
    }
}

/// After the first phase, a basic feasible solution is found, the problem is found to be
/// infeasible, or artificial variables remain in the basis.
#[derive(Debug)]
enum PhaseOneResult<'a, T: 'a> where T: TableauProvider {
    FoundBFS(DenseMatrix, HashMap<usize, usize>),
    Infeasible,
    ArtificialRemain(Tableau<'a, T>, HashSet<usize>),
}

#[cfg(test)]
mod test {

    use super::*;
    use algorithm::simplex::tableau_provider::network::ShortestPathNetwork;
    use algorithm::simplex::tableau_provider::matrix_data::MatrixData;
    use data::linear_program::canonical_form::CanonicalForm;

    fn matrix_data() -> MatrixData {
        let data = SparseMatrix::from_data(vec![vec![3f64, 2f64, 1f64, 0f64, 0f64],
                                                vec![5f64, 1f64, 1f64, 1f64, 0f64],
                                                vec![2f64, 5f64, 1f64, 0f64, 1f64]]);
        let cost = SparseVector::from_data(vec![1f64, 1f64, 1f64, 1f64, 1f64]);
        let b = DenseVector::from_data(vec![1f64, 2f64, 3f64]);
        let column_info = vec![String::from("X1"),
                               String::from("X2"),
                               String::from("X3"),
                               String::from("X4"),
                               String::from("X5")];
        MatrixData::new(data, cost, b, column_info)
    }

    fn tableau(data: &MatrixData) -> Tableau<MatrixData> {
        let carry = DenseMatrix::from_data(vec![vec![-6f64, 1f64, -1f64, -1f64],
                                                vec![0f64, 1f64, 1f64, 1f64],
                                                vec![1f64, 1f64, 0f64, 0f64],
                                                vec![2f64, -1f64, 1f64, 0f64],
                                                vec![3f64, -1f64, 0f64, 1f64]]);
        let mut basis_columns = HashMap::new();
        basis_columns.insert(0, 2);
        basis_columns.insert(1, 3);
        basis_columns.insert(2, 4);

        Tableau::new(data, carry, basis_columns)
    }

    #[test]
    fn test_simplex() {
        let data = matrix_data();
        let mut tableau = tableau(&data);
        if let LinearProgramType::FiniteOptimum(_, cost) = loop {
            match tableau.profitable_column(CostRow::Actual) {
                Some(column_nr) => {
                    let column = tableau.generate_column(column_nr);
                    let pivot_row = tableau.find_pivot_row(&column);
                    tableau.bring_into_basis(pivot_row, column_nr, &column)
                },
                None => break LinearProgramType::FiniteOptimum(tableau.current_bfs(),
                                                               tableau.cost(CostRow::Actual)),
            }
        } {
            assert_approx_eq!(cost, 9f64 / 2f64);
        } else {
            assert!(false);
        }

    }

    fn lp_canonical() -> CanonicalForm {
        let data = SparseMatrix::from_data(vec![vec![3f64, 2f64, 1f64, 0f64, 0f64],
                                                vec![5f64, 1f64, 1f64, 1f64, 0f64],
                                                vec![2f64, 5f64, 1f64, 0f64, 1f64]]);
        let b = DenseVector::from_data(vec![1f64,
                                            3f64,
                                            4f64]);
        let cost = SparseVector::from_data(vec![1f64, 1f64, 1f64, 1f64, 1f64]);
        let column_info = vec![Variable { name: "X1".to_string(), variable_type: VariableType::Continuous, offset: 0f64, },
                               Variable { name: "X2".to_string(), variable_type: VariableType::Continuous, offset: 0f64, },
                               Variable { name: "X3".to_string(), variable_type: VariableType::Continuous, offset: 0f64, },
                               Variable { name: "X4".to_string(), variable_type: VariableType::Continuous, offset: 0f64, },
                               Variable { name: "X5".to_string(), variable_type: VariableType::Continuous, offset: 0f64, }];

        CanonicalForm::new(data, b, cost, 0f64, column_info, Vec::new())
    }

    #[test]
    fn test_phase_one() {
        let canonical = lp_canonical();
        let data = MatrixData::from(canonical);
        let tableau = Tableau::new_artificial(&data);
        match phase_one(tableau) {
            PhaseOneResult::FoundBFS(_, map) => {
                let mut expected = HashMap::new();
                expected.insert(0, 1);
                expected.insert(1, 3);
                expected.insert(2, 4);
                assert_eq!(map, expected);
            },
            PhaseOneResult::Infeasible => assert!(false),
            PhaseOneResult::ArtificialRemain(_, _) => assert!(false),
        }
    }

    #[test]
    fn test_phase_two() {
        // TODO
    }

    #[test]
    fn test_solve_matrix() {
        let data = MatrixData::from(lp_canonical());
        let result = solve(&data).unwrap();
        let expected = (vec![(String::from("X1"), 0f64),
                             (String::from("X2"), 0.5f64),
                             (String::from("X3"), 0f64),
                             (String::from("X4"), 2.5f64),
                             (String::from("X5"), 1.5f64)], 4.5f64);

        for i in 0..expected.0.len() {
            assert_eq!(result.0[i].0, expected.0[i].0);
            assert_approx_eq!(result.0[i].1, expected.0[i].1);
        }
        assert_approx_eq!(result.1, expected.1);
    }

    #[test]
    fn test_solve_shortest_path() {
        let data = SparseMatrix::from_data(vec![vec![0f64, 1f64, 0f64, 2f64, 0f64],
                                                vec![0f64, 0f64, 1f64, 1f64, 0f64],
                                                vec![0f64, 0f64, 0f64, 1f64, 1f64],
                                                vec![0f64, 0f64, 0f64, 0f64, 2f64],
                                                vec![0f64, 0f64, 0f64, 0f64, 0f64]]);
        let graph = ShortestPathNetwork::new(data, 0, 4);
        let result = solve(&graph).unwrap().1;
        assert_eq!(result, 3f64);
    }
}
