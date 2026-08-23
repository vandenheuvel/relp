//! # Checking a solution against the problem it claims to solve
//!
//! An exact solver can do something an inexact one cannot: say for certain whether a solution is
//! feasible and optimal, rather than feasible and optimal to within a tolerance. Every comparison
//! here is exact, and there is no tolerance anywhere.
use std::fmt;
use std::ops::Neg;

use relp_num::{OrderedField, OrderedFieldRef};

use crate::data::linear_algebra::traits::{SparseComparator, SparseElement};
use crate::data::linear_program::elements::{Objective, RangedConstraintRelation};
use crate::data::linear_program::general_form::GeneralForm;
use crate::data::linear_program::solution::Solution;

/// A way in which a solution fails to be one.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Invalid {
    /// A different number of values than the problem has variables.
    WrongNumberOfValues {
        /// How many the problem has.
        expected: usize,
        /// How many the solution gave.
        found: usize,
    },
    /// A value is named differently from the variable in that position.
    WrongName {
        /// Where in the solution the name appeared.
        position: usize,
        /// The name the problem uses there.
        expected: String,
    },
    /// A variable lies outside its own bounds.
    VariableOutOfBounds {
        /// Index of the variable.
        variable: usize,
    },
    /// A row is not satisfied.
    RowViolated {
        /// Index of the row.
        row: usize,
    },
    /// The objective value reported is not the one the values give.
    ObjectiveValueWrong,
    /// A different number of multipliers than the problem has constraints.
    WrongNumberOfDuals {
        /// How many the problem has.
        expected: usize,
        /// How many the solution gave.
        found: usize,
    },
    /// A multiplier has a sign its row does not admit.
    DualSignWrong {
        /// Index of the row.
        row: usize,
    },
    /// A variable's reduced cost has a sign its position does not admit.
    ReducedCostSignWrong {
        /// Index of the variable.
        variable: usize,
    },
    /// A row is not tight although its multiplier is not zero.
    RowNotComplementary {
        /// Index of the row.
        row: usize,
    },
    /// A variable is not at a bound although its reduced cost is not zero.
    VariableNotComplementary {
        /// Index of the variable.
        variable: usize,
    },
}

impl fmt::Display for Invalid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Invalid::WrongNumberOfValues { expected, found } => {
                write!(f, "expected {} values, found {}", expected, found)
            }
            Invalid::WrongName { position, expected } => {
                write!(f, "the value at position {} is not {}", position, expected)
            }
            Invalid::VariableOutOfBounds { variable } => {
                write!(f, "variable {} is outside its bounds", variable)
            }
            Invalid::RowViolated { row } => write!(f, "row {} is not satisfied", row),
            Invalid::ObjectiveValueWrong => {
                f.write_str("the objective value does not match the solution values")
            }
            Invalid::WrongNumberOfDuals { expected, found } => {
                write!(f, "expected {} multipliers, found {}", expected, found)
            }
            Invalid::DualSignWrong { row } => {
                write!(f, "the multiplier of row {} has the wrong sign", row)
            }
            Invalid::ReducedCostSignWrong { variable } => {
                write!(
                    f,
                    "the reduced cost of variable {} has the wrong sign",
                    variable,
                )
            }
            Invalid::RowNotComplementary { row } => {
                write!(f, "row {} is slack but its multiplier is not zero", row)
            }
            Invalid::VariableNotComplementary { variable } => write!(
                f,
                "variable {} is between its bounds but its reduced cost is not zero",
                variable,
            ),
        }
    }
}

impl<F> GeneralForm<F>
where
    F: OrderedField + SparseElement<F> + SparseComparator,
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    /// Check a solution against this problem, exactly.
    ///
    /// `self` has to be the problem as it was read: before presolving, before standardizing and
    /// before scaling, all of which replace it with a different problem. Clone it first if it is
    /// also going to be solved.
    ///
    /// Checks that every variable respects its bounds, that every row is satisfied, and that the
    /// objective value reported is the one the values give.
    pub fn verify_solution<G>(&self, solution: &Solution<G>) -> Result<(), Invalid>
    where
        G: OrderedField + From<F>,
        for<'r> &'r G: OrderedFieldRef<G> + Neg<Output = G>,
    {
        let names = self.variable_names_for_verification();
        if solution.solution_values.len() != names.len() {
            return Err(Invalid::WrongNumberOfValues {
                expected: names.len(),
                found: solution.solution_values.len(),
            });
        }

        for (position, ((name, _), expected)) in
            solution.solution_values.iter().zip(names).enumerate()
        {
            if name != expected {
                return Err(Invalid::WrongName {
                    position,
                    expected: expected.clone(),
                });
            }
        }

        let values = solution
            .solution_values
            .iter()
            .map(|(_, value)| value)
            .collect::<Vec<_>>();

        self.check_bounds(&values)?;
        self.check_rows(&values)?;
        self.check_objective(&values, &solution.objective_value)?;

        match &solution.dual_values {
            None => Ok(()),
            Some(duals) => self.check_duals(&values, duals),
        }
    }

    /// The multipliers are a dual solution, and together with the values an optimal pair.
    ///
    /// Three things have to hold. Each multiplier has a sign its relation admits. Each reduced
    /// cost, `c - A^T y`, has a sign the position of its variable admits. And the two are
    /// complementary: a row that is not tight has a multiplier of zero, and a variable strictly
    /// between its bounds has a reduced cost of zero.
    ///
    /// Together these say that the solution is optimal, not merely feasible. Signs are stated
    /// relative to the direction of optimization: `s` below is `+1` for a minimum and `-1` for a
    /// maximum, so that a multiplier means the same thing either way and only what it is allowed
    /// to be changes.
    fn check_duals<G>(&self, values: &[&G], duals: &[(String, G)]) -> Result<(), Invalid>
    where
        G: OrderedField + From<F>,
        for<'r> &'r G: OrderedFieldRef<G> + Neg<Output = G>,
    {
        let nr_rows = self.constraint_names_for_verification().len();
        if duals.len() != nr_rows {
            return Err(Invalid::WrongNumberOfDuals {
                expected: nr_rows,
                found: duals.len(),
            });
        }
        let duals = duals.iter().map(|(_, value)| value).collect::<Vec<_>>();

        let (objective, _, _) = self.objective_for_verification();
        let minimizing = objective == Objective::Minimize;
        // `s * value`, so that one set of comparisons covers both directions.
        let oriented = |value: &G| if minimizing { value.clone() } else { -value };

        let zero = G::zero();

        // Sign of each multiplier, and complementary slackness for the rows.
        for (row, (relation, right_hand_side, coefficients)) in
            self.rows_for_verification().enumerate()
        {
            let activity = coefficients
                .map(|(column, coefficient)| &G::from(coefficient.clone()) * values[column])
                .sum::<G>();
            let right_hand_side = G::from(right_hand_side.clone());
            let dual = oriented(duals[row]);

            match relation {
                // An equality is tight by construction and admits a multiplier of either sign.
                RangedConstraintRelation::Equal => {}
                RangedConstraintRelation::Less => {
                    if dual > zero {
                        return Err(Invalid::DualSignWrong { row });
                    }
                    if dual != zero && activity != right_hand_side {
                        return Err(Invalid::RowNotComplementary { row });
                    }
                }
                RangedConstraintRelation::Greater => {
                    if dual < zero {
                        return Err(Invalid::DualSignWrong { row });
                    }
                    if dual != zero && activity != right_hand_side {
                        return Err(Invalid::RowNotComplementary { row });
                    }
                }
                // A range admits either sign, and the sign says which of its two ends the
                // multiplier is sitting on.
                RangedConstraintRelation::Range(range) => {
                    let lower = &right_hand_side - &G::from(range.clone());
                    if dual < zero && activity != right_hand_side {
                        return Err(Invalid::RowNotComplementary { row });
                    }
                    if dual > zero && activity != lower {
                        return Err(Invalid::RowNotComplementary { row });
                    }
                }
            }
        }

        // Reduced costs, from the multipliers and the original matrix.
        let mut reduced = self
            .objective_for_verification()
            .2
            .map(|cost| G::from(cost.clone()))
            .collect::<Vec<_>>();
        for (row, (_, _, coefficients)) in self.rows_for_verification().enumerate() {
            for (column, coefficient) in coefficients {
                reduced[column] -= &G::from(coefficient.clone()) * duals[row];
            }
        }

        for (variable, (lower, upper)) in self.variable_bounds_for_verification().enumerate() {
            let cost = oriented(&reduced[variable]);
            if cost == zero {
                continue;
            }

            let value = values[variable];
            let at_lower = lower.is_some_and(|bound| *value == G::from(bound.clone()));
            let at_upper = upper.is_some_and(|bound| *value == G::from(bound.clone()));

            // A variable with a positive reduced cost wants to be smaller, so it has to be held
            // there by its own lower bound; and the other way round.
            if cost > zero {
                if lower.is_none() {
                    return Err(Invalid::ReducedCostSignWrong { variable });
                }
                if !at_lower {
                    return Err(Invalid::VariableNotComplementary { variable });
                }
            } else {
                if upper.is_none() {
                    return Err(Invalid::ReducedCostSignWrong { variable });
                }
                if !at_upper {
                    return Err(Invalid::VariableNotComplementary { variable });
                }
            }
        }

        Ok(())
    }

    /// Every variable within its own bounds.
    fn check_bounds<G>(&self, values: &[&G]) -> Result<(), Invalid>
    where
        G: OrderedField + From<F>,
        for<'r> &'r G: OrderedFieldRef<G> + Neg<Output = G>,
    {
        for (variable, bounds) in self.variable_bounds_for_verification().enumerate() {
            let (lower, upper) = bounds;
            let value = values[variable];

            let too_low = lower.is_some_and(|bound| *value < G::from(bound.clone()));
            let too_high = upper.is_some_and(|bound| *value > G::from(bound.clone()));
            if too_low || too_high {
                return Err(Invalid::VariableOutOfBounds { variable });
            }
        }

        Ok(())
    }

    /// Every row satisfied by the activity the values give it.
    fn check_rows<G>(&self, values: &[&G]) -> Result<(), Invalid>
    where
        G: OrderedField + From<F>,
        for<'r> &'r G: OrderedFieldRef<G> + Neg<Output = G>,
    {
        for (row, (relation, right_hand_side, coefficients)) in
            self.rows_for_verification().enumerate()
        {
            let activity = coefficients
                .map(|(column, coefficient)| &G::from(coefficient.clone()) * values[column])
                .sum::<G>();
            let right_hand_side = G::from(right_hand_side.clone());

            let satisfied = match relation {
                RangedConstraintRelation::Equal => activity == right_hand_side,
                RangedConstraintRelation::Less => activity <= right_hand_side,
                RangedConstraintRelation::Greater => activity >= right_hand_side,
                RangedConstraintRelation::Range(range) => {
                    activity <= right_hand_side
                        && activity >= &right_hand_side - &G::from(range.clone())
                }
            };
            if !satisfied {
                return Err(Invalid::RowViolated { row });
            }
        }

        Ok(())
    }

    /// The objective value reported is the one the values give.
    fn check_objective<G>(&self, values: &[&G], reported: &G) -> Result<(), Invalid>
    where
        G: OrderedField + From<F>,
        for<'r> &'r G: OrderedFieldRef<G> + Neg<Output = G>,
    {
        let (objective, fixed_cost, costs) = self.objective_for_verification();
        let value = costs
            .enumerate()
            .map(|(column, cost)| &G::from(cost.clone()) * values[column])
            .sum::<G>()
            + G::from(fixed_cost.clone());

        // The direction does not enter into it: the objective value of a solution is the value of
        // the expression, whichever way the problem asks for it to be moved.
        let _ = objective;

        if &value == reported {
            Ok(())
        } else {
            Err(Invalid::ObjectiveValueWrong)
        }
    }
}

/// A guard so the objective direction stays part of this module's vocabulary.
const _: [Objective; 2] = [Objective::Minimize, Objective::Maximize];
