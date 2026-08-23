//! # Variables whose bounds leave one value
//!
//! Substituting the value into every row it appears in removes the column.
use std::ops::Neg;

use relp_num::OrderedFieldRef;

use crate::data::linear_program::general_form::presolve::postsolve::Reduction;
use crate::data::linear_program::general_form::presolve::problem::Problem;
use crate::data::linear_program::general_form::presolve::queues::{Queue, Queues};
use crate::data::linear_program::general_form::presolve::reducer::Reducer;
use crate::data::linear_program::general_form::presolve::rule::{
    Effect, LocalRule, RuleField, RuleId, Stop,
};

pub struct FixedColumn;

impl FixedColumn {
    const ID: RuleId = RuleId::FixedColumn;
}

impl<F: RuleField, const I: bool> LocalRule<F, I> for FixedColumn
where
    for<'r> &'r F: OrderedFieldRef<F> + Neg<Output = F>,
{
    type Trigger = usize;
    const ID: RuleId = RuleId::FixedColumn;

    fn queue(queues: &mut Queues) -> &mut Queue<usize> {
        &mut queues.fixed_column
    }

    fn is_applicable(column: usize, problem: &Problem<F>) -> bool {
        problem.columns[column].is_active && problem.columns[column].fixed_value().is_some()
    }

    fn apply(column: usize, reducer: &mut Reducer<'_, F, I>) -> Result<Effect, Stop> {
        let value = reducer.problem.columns[column]
            .fixed_value()
            .expect("checked by `is_applicable`")
            .clone();

        let cost = reducer.problem.columns[column].cost.clone();
        reducer.substitute_value(column, &value, Self::ID);
        let coefficients = reducer.remove_column(column, Self::ID);
        reducer.postsolve.push(Reduction::FixedColumn {
            column,
            value,
            coefficients,
            cost,
        });

        Ok(Effect::Reduced)
    }
}
