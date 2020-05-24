//! # Representation of feasible solutions
//!
//! Once a linear program is fully solved, a solution is derived. This solution should contain also
//! any variables that were eliminated as part of a presolve process (constant variables, slack
//! variables, variables that don't interact with the rest of the problem, etc.).

/// Represents a full (including presolved variables, constants, etc.) solution to a linear program.
///
/// Should represent a solution that is feasible. This struct would probably be used to print the
/// optimal solution for the user.
#[derive(Eq, PartialEq, Debug)]
pub struct Solution<F> {
    /// Value of the objective function for this solution, including any constant that was included
    /// in the original problem.
    objective_value: F,
    /// (variable name, solution value) tuples for all variables, named as in the original problem.
    solution_values: Vec<(String, F)>,
}

impl<F> Solution<F> {
    /// Create a new `Solution` instance.
    ///
    /// A plain constructor.
    ///
    /// # Arguments
    ///
    /// * `objective`:
    pub fn new(objective: F, solution_values: Vec<(String, F)>) -> Self {
        Self {
            objective_value: objective,
            solution_values,
        }
    }
}
