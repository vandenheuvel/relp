use std::collections::{HashMap, HashSet};

use algorithm::simplex::data::{CostRow, create_artificial_tableau, Tableau};
use algorithm::simplex::EPSILON;
use data::linear_algebra::matrix::{DenseMatrix, Matrix, SparseMatrix};
use data::linear_algebra::vector::{DenseVector, SparseVector, Vector};
use data::linear_program::canonical_form::CanonicalForm;
use data::linear_program::elements::{LPCategory, Variable, VariableType};


/// Solves a linear program in `CanonicalForm` using the Simplex method.
pub fn solve(canonical: &CanonicalForm) -> Result<(Vec<(String, f64)>, f64), String> {
    let result = match phase_one(canonical) {
        PhaseOneResult::FoundBFS(carry, basis) => phase_two(&canonical, carry, basis),
        PhaseOneResult::Infeasible => return Err(format!("LP not feasible")),
        PhaseOneResult::ArtificialRemain(tableau) => {
            let mut tableau = tableau;
            // TODO: MIPLIB problem 50v-10 doesn't work with this line.
//            remove_artificial(&mut tableau);

            phase_two(&canonical, tableau.carry(), tableau.basis_columns_map())
        }
    };

    match result {
        PhaseTwoResult::FoundOptimum(optimum, cost) => Ok((combine_column_names(optimum, canonical), cost)),
        PhaseTwoResult::Unbounded => Err(format!("LP unbounded")),
    }
}

/// Combines the variable names to the values of a basic feasible solution.
fn combine_column_names(bfs: SparseVector, canonical: &CanonicalForm) -> Vec<(String, f64)> {
    debug_assert_eq!(bfs.len(), canonical.variable_info().len());

    let mut combined = Vec::new();
    let column_info = canonical.variable_info();
    for i in 0..column_info.len() {
        combined.push((column_info[i].clone(), bfs.get_value(i)));
    }
    combined
}

/// Reduces the artificial cost of the basic feasible solution to zero, if possible. In doing so, a
/// basic feasible solution to the `CanonicalForm` linear program is found.
fn phase_one(canonical: &CanonicalForm) -> PhaseOneResult {
    let mut tableau = create_artificial_tableau(canonical);
    let mut i = 0;

    let lp_category = loop {
        match tableau.profitable_column(CostRow::Artificial) {
            Some(column_nr) => {
                let column = tableau.generate_column(column_nr);
                let pivot_row = tableau.find_pivot_row(&column);
                tableau.bring_into_basis(pivot_row, column_nr, &column)
            },
            None => break LPCategory::FiniteOptimum(tableau.cost(CostRow::Artificial)),
        }
        i += 1;
    };
    match lp_category {
        LPCategory::Unbounded => PhaseOneResult::Infeasible,
        LPCategory::Infeasible => PhaseOneResult::Infeasible,
        LPCategory::FiniteOptimum(cost) if {
            cost < EPSILON && !tableau.has_artificial_in_basis()
        } => {
            PhaseOneResult::FoundBFS(tableau.carry(), tableau.basis_columns_map())
        },
        LPCategory::FiniteOptimum(cost) if cost >= EPSILON => PhaseOneResult::Infeasible,
        LPCategory::FiniteOptimum(_) => PhaseOneResult::ArtificialRemain(tableau),
    }
}

/// Removes all artificial variables from the tableau by making a basis change "at zero level", or
/// without change of cost of the current solution.
fn remove_artificial(tableau: &mut Tableau) {
    let mut artificials = tableau.basis_columns();
    artificials.retain(|&v| v < tableau.nr_rows());

    for artificial in artificials.into_iter() {
        debug_assert!(tableau.relative_cost(CostRow::Artificial, artificial).abs() < EPSILON);

        let artificial_column = tableau.generate_column(artificial);
        let pivot_row = artificial_column.values()
            .find(|(_, v)| (1f64 - v).abs() < EPSILON)
            .unwrap().0 - 1 - 1;

        for nonartificial in tableau.nr_rows()..tableau.nr_columns() {
            if !tableau.is_in_basis(nonartificial) {
                if tableau.relative_cost(CostRow::Artificial, nonartificial).abs() < EPSILON {
                    let column = tableau.generate_column(nonartificial);
                    tableau.bring_into_basis(pivot_row, nonartificial, &column);
                    break;
                }
            }
        }

        debug_assert!(!tableau.is_in_basis(artificial));
    }

    debug_assert!(!tableau.has_artificial_in_basis());
}

/// Reduces the cost of the basic feasible solution to the minimum.
fn phase_two(canonical: &CanonicalForm, carry: DenseMatrix, basis: HashMap<usize, usize>) -> PhaseTwoResult {
    let mut tableau = Tableau::create_from(canonical, carry, basis);

    let lp_category = loop {
        match tableau.profitable_column(CostRow::Actual) {
            Some(column_nr) => {
                let column = tableau.generate_column(column_nr);
                let pivot_row = tableau.find_pivot_row(&column);
                tableau.bring_into_basis(pivot_row, column_nr, &column);
            },
            None => break LPCategory::FiniteOptimum(tableau.cost(CostRow::Actual)),
        }
    };

    match lp_category {
        LPCategory::FiniteOptimum(cost) => {
            let values = tableau.current_bfs();
            PhaseTwoResult::FoundOptimum(values, cost)
        },
        LPCategory::Unbounded => PhaseTwoResult::Unbounded,
        LPCategory::Infeasible => panic!("We should have a bfs in phase two."),
    }
}

/// After the first phase, a basic feasible solution is found, the problem is found to be
/// infeasible, or artificial variables remain in the basis.
#[derive(Debug)]
enum PhaseOneResult {
    FoundBFS(DenseMatrix, HashMap<usize, usize>),
    Infeasible,
    ArtificialRemain(Tableau),
}

/// After the second phase, either an optimum is found or the problem is determined to be unbounded.
#[derive(Debug)]
enum PhaseTwoResult {
    FoundOptimum(SparseVector, f64),
    Unbounded,
}

#[cfg(test)]
mod test {

    use super::*;

    fn tableau() -> Tableau {
        let data = SparseMatrix::from_data(vec![vec![3f64, 2f64, 1f64, 0f64, 0f64],
                                                vec![5f64, 1f64, 1f64, 1f64, 0f64],
                                                vec![2f64, 5f64, 1f64, 0f64, 1f64]]);
        let original_c = SparseVector::from_data(vec![1f64, 1f64, 1f64, 1f64, 1f64]);
        let carry = DenseMatrix::from_data(vec![vec![-6f64, 1f64, -1f64, -1f64],
                                                vec![0f64, 1f64, 1f64, 1f64],
                                                vec![1f64, 1f64, 0f64, 0f64],
                                                vec![2f64, -1f64, 1f64, 0f64],
                                                vec![3f64, -1f64, 0f64, 1f64]]);
        let mut basis_columns = HashMap::new();
        basis_columns.insert(0, 2);
        basis_columns.insert(1, 3);
        basis_columns.insert(2, 4);
        let column_info = vec![String::from("X1"),
                               String::from("X2"),
                               String::from("X3"),
                               String::from("X4"),
                               String::from("X5")];

        Tableau::new(data, original_c, carry, basis_columns, column_info)
    }

    fn lp_canonical() -> CanonicalForm {
        let data = SparseMatrix::from_data(vec![vec![3f64, 2f64, 1f64, 0f64, 0f64],
                                                vec![5f64, 1f64, 1f64, 1f64, 0f64],
                                                vec![2f64, 5f64, 1f64, 0f64, 1f64]]);
        let b = DenseVector::from_data(vec![1f64,
                                            3f64,
                                            4f64]);
        let cost = SparseVector::from_data(vec![1f64, 1f64, 1f64, 1f64, 1f64]);
        let column_info = vec![Variable::new(String::from("X1"), VariableType::Continuous, 0f64),
                               Variable::new(String::from("X2"), VariableType::Continuous, 0f64),
                               Variable::new(String::from("X3"), VariableType::Continuous, 0f64),
                               Variable::new(String::from("X4"), VariableType::Continuous, 0f64),
                               Variable::new(String::from("X5"), VariableType::Continuous, 0f64)];

        CanonicalForm::new(data, b, cost, 0f64, column_info, Vec::new())
    }

    #[test]
    fn test_simplex() {
        let mut tableau = tableau();
        if let LPCategory::FiniteOptimum(cost) = loop {
            match tableau.profitable_column(CostRow::Actual) {
                Some(column_nr) => {
                    let column = tableau.generate_column(column_nr);
                    let pivot_row = tableau.find_pivot_row(&column);
                    tableau.bring_into_basis(pivot_row, column_nr, &column)
                },
                None => break LPCategory::FiniteOptimum(tableau.cost(CostRow::Actual)),
            }
        } {
            assert_approx_eq!(cost, 9f64 / 2f64);
        } else {
            assert!(false);
        }

    }

    #[test]
    fn test_phase_one() {
        let canonical = lp_canonical();
        match phase_one(&canonical) {
            PhaseOneResult::FoundBFS(_, map) => {
                let mut expected = HashMap::new();
                expected.insert(0, 4);
                expected.insert(1, 6);
                expected.insert(2, 7);
                assert_eq!(map, expected);
            },
            PhaseOneResult::Infeasible => assert!(false),
            PhaseOneResult::ArtificialRemain(_) => assert!(false),
        }
    }

    #[test]
    fn test_phase_two() {
        // TODO
    }

    #[test]
    fn test_solve() {
        let result = solve(&lp_canonical()).unwrap();
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
}
