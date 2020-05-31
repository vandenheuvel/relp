//! # Representation of feasible solutions
//!
//! Once a linear program is fully solved, a solution is derived. This solution should contain also
//! any variables that were eliminated as part of a presolve process (constant variables, slack
//! variables, variables that don't interact with the rest of the problem, etc.).

use crate::data::number_types::traits::Field;
use std::collections::HashMap;
use std::hash::Hash;

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

impl<F: Field> Solution<F> {
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

    pub fn is_probably_equal_to(&self, other: &Self, min_equal: f64) -> bool {
        if !(self.objective_value == other.objective_value) {
            return false;
        }

        if !(self.solution_values.len() == other.solution_values.len()) {
            return false;
        }

        let this_map = self.solution_values.iter().cloned().collect::<HashMap<_, _>>();
        let other_map = other.solution_values.iter().cloned().collect::<HashMap<_, _>>();

        if !{
            this_map.len() == other_map.len() && this_map.keys().all(|k| other_map.contains_key(k))
        } {
            return false;
        }

        let nr_total = self.solution_values.len();
        if nr_total < 10 {
            return true;
        }

        let mut nr_equal = 0;
        for name in this_map.keys() {
            if this_map.get(name) == other_map.get(name) {
                nr_equal += 1;
            }
        }

        nr_equal as f64 / nr_total as f64 > min_equal // At least a few solution values should be equal
    }
}
