use data::linear_program::elements::Variable;
use data::linear_algebra::matrix::{Matrix, SparseMatrix};
use data::linear_algebra::vector::{DenseVector, SparseVector, Vector};

/// A linear program in `CanonicalForm` is like a linear program in `GeneralForm`, but has only
/// equality as the constraint type. Moreover, the constraint vector `b` is non-negative.
#[derive(Debug, PartialEq)]
pub struct CanonicalForm {
    data: SparseMatrix,
    b: DenseVector,
    cost: SparseVector,
    fixed_cost: f64,

    variables: Vec<Variable>,
}

impl CanonicalForm {
    /// Create a new linear program in canonical form.
    pub fn new(data: SparseMatrix,
               b: DenseVector,
               cost: SparseVector,
               fixed_cost: f64,
               variable_info: Vec<Variable>,
               solution_values: Vec<(String, f64)>) -> CanonicalForm {
        let m = b.len();
        debug_assert_eq!(b.len(), m);
        debug_assert_eq!(data.nr_rows(), m);

        let n = cost.len();
        debug_assert_eq!(cost.len(), n);
        debug_assert_eq!(data.nr_columns(), n);
        debug_assert_eq!(variable_info.len(), n);

        CanonicalForm { data, b, cost, fixed_cost, variables: variable_info, }
    }
    /// Get the constraint vector `b`.
    pub fn b(&self) -> DenseVector {
        self.b.clone()
    }
    /// Get the cost vector `cost`.
    pub fn cost(&self) -> SparseVector {
        self.cost.clone()
    }
    /// Get the fixed cost value.
    pub fn fixed_cost(&self) -> f64 {
        self.fixed_cost
    }
    /// Get all variable names.
    pub fn variable_info(&self) -> Vec<String> {
        self.variables.clone()
            .into_iter()
            .map(|variable| variable.name)
            .collect()
    }
    /// Get a copy of the underlying data.
    pub fn data(&self) -> SparseMatrix {
        self.data.clone()
    }
    /// Get the number of variables.
    pub fn nr_variables(&self) -> usize {
        self.data.nr_columns()
    }
    /// Get the number of constraints.
    pub fn nr_constraints(&self) -> usize {
        self.data.nr_rows()
    }
}