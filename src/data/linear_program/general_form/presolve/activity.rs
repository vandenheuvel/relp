//! # Row activities
//!
//! The activity of a row is the range of values `<a, x>` can take given the variable bounds. Its
//! two ends drive domain propagation: comparing them with the right hand side proves a row
//! redundant, forcing or unsatisfiable, and the slack between them bounds each variable in turn.
//!
//! An end can only be computed when every variable in the row is bounded in the direction that
//! contributes to it, so a count of how many are not is kept alongside. When that count reaches
//! one, the single variable still missing a bound can be given one; when it reaches zero, the end
//! itself can be computed.
use relp_num::{NonZeroSign, OrderedField, OrderedFieldRef};

use crate::data::linear_program::elements::BoundDirection;
use crate::data::linear_program::general_form::presolve::problem::Problem;

/// Per row, the two activity ends and the number of variable bounds each is still waiting for.
#[derive(Clone, Debug)]
pub struct Activity<F> {
    /// Cached value of each end, when it has been computed and is still valid.
    bounds: Vec<(Option<F>, Option<F>)>,
    /// Number of variables in the row lacking the bound that end needs.
    missing: Vec<(usize, usize)>,
}

/// Which end of a row's activity a variable bound contributes to.
///
/// A positive coefficient sends a variable's lower bound to the row's lower activity; a negative
/// one sends it to the upper activity.
pub fn contributes_to(direction: BoundDirection, sign: NonZeroSign) -> BoundDirection {
    direction * sign
}

impl<F> Activity<F>
where
    F: OrderedField,
    for<'r> &'r F: OrderedFieldRef<F>,
{
    /// Count, for every row, how many bounds each activity end is waiting for.
    pub fn new(problem: &Problem<F>) -> Self {
        let missing = (0..problem.matrix.nr_rows())
            .map(|row| {
                problem
                    .matrix
                    .row(row)
                    .fold((0, 0), |(lower, upper), (column, coefficient)| {
                        let sign = coefficient.non_zero_signum();
                        let has = |direction| {
                            problem.columns[column]
                                .bound(contributes_to(direction, sign))
                                .is_some()
                        };
                        (
                            lower + usize::from(!has(BoundDirection::Lower)),
                            upper + usize::from(!has(BoundDirection::Upper)),
                        )
                    })
            })
            .collect();

        Self {
            bounds: vec![(None, None); problem.matrix.nr_rows()],
            missing,
        }
    }

    /// How many bounds an activity end is still waiting for.
    pub fn missing(&self, row: usize, direction: BoundDirection) -> usize {
        match direction {
            BoundDirection::Lower => self.missing[row].0,
            BoundDirection::Upper => self.missing[row].1,
        }
    }

    /// Note that a variable gained a bound it did not have, so an end is waiting for one fewer.
    pub fn bound_added(&mut self, row: usize, direction: BoundDirection) {
        let counter = match direction {
            BoundDirection::Lower => &mut self.missing[row].0,
            BoundDirection::Upper => &mut self.missing[row].1,
        };
        debug_assert!(*counter > 0);
        *counter -= 1;
    }

    /// Note that a variable that was missing a bound left the row.
    pub fn missing_bound_left(&mut self, row: usize, direction: BoundDirection) {
        self.bound_added(row, direction);
    }

    /// Note that a variable that is missing a bound joined the row.
    pub fn missing_bound_arrived(&mut self, row: usize, direction: BoundDirection) {
        match direction {
            BoundDirection::Lower => self.missing[row].0 += 1,
            BoundDirection::Upper => self.missing[row].1 += 1,
        }
    }

    /// Forget the cached value of both ends of a row.
    ///
    /// Called whenever anything the value was computed from changes: a coefficient leaving the
    /// row, the right hand side moving, or a bound of a variable in it shifting. Recomputing is
    /// cheap relative to being wrong, and a cached activity that no longer matches the row is the
    /// kind of error that shows up as a confidently wrong answer.
    pub fn invalidate(&mut self, row: usize) {
        self.bounds[row] = (None, None);
    }

    /// The value of an activity end, computing it if it is not already known.
    ///
    /// Only call this when the end is waiting for no bounds.
    pub fn value(&mut self, problem: &Problem<F>, row: usize, direction: BoundDirection) -> &F {
        debug_assert_eq!(self.missing(row, direction), 0);

        let cached = match direction {
            BoundDirection::Lower => &mut self.bounds[row].0,
            BoundDirection::Upper => &mut self.bounds[row].1,
        };

        if cached.is_none() {
            *cached = Some(compute(problem, row, direction));
        }
        cached.as_ref().unwrap()
    }

    /// Recompute an end from scratch, for use in debug assertions.
    ///
    /// `debug_assert` still type checks its argument in a release build, so this cannot be
    /// compiled out; it is never called there.
    pub fn recompute(problem: &Problem<F>, row: usize, direction: BoundDirection) -> F {
        compute(problem, row, direction)
    }
}

/// Sum the contribution of every variable in a row to one end of its activity.
fn compute<F>(problem: &Problem<F>, row: usize, direction: BoundDirection) -> F
where
    F: OrderedField,
    for<'r> &'r F: OrderedFieldRef<F>,
{
    problem
        .matrix
        .row(row)
        .map(|(column, coefficient)| {
            let sign = coefficient.non_zero_signum();
            let bound = problem.columns[column]
                .bound(contributes_to(direction, sign))
                .expect("an activity end is only computed once every bound it needs is there");
            coefficient * bound
        })
        .sum()
}
