//! # Rational prescaling
//!
//! Prescaling rational linear programs.
use std::cmp::Ordering;
use std::collections::HashSet;
use std::hash::Hash;
use std::iter::once;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub};

use fifo_set::FIFOSet;
use num_traits::{One, Zero};
use relp_num::{Abs, NonZeroFactorizable, NonZeroFactorization, Sign, Signed};

use crate::data::linear_algebra::SparseTuple;
use crate::data::linear_algebra::traits::{SparseComparator, SparseElement};
use crate::data::linear_algebra::vector::Vector;
use crate::data::linear_program::general_form::GeneralForm;
use crate::data::linear_program::general_form::presolve::scale::{Scalable, scale, Scaling};

/// Multiplication traits necessary for scaling.
pub trait Multiplication<F> =
    One +
    MulAssign<F> +
    DivAssign<F> +
    MulAssign<Self> +
    DivAssign<Self> +
where
    for<'r> Self:
        MulAssign<&'r F> +
        DivAssign<&'r F> +
        MulAssign<&'r Self> +
        DivAssign<&'r Self> +
        Mul<&'r Self, Output=Self> +
    ,
;

/// Addition traits necessary for scaling.
pub trait Addition =
    Zero +
    Neg<Output=Self> +
    Sub<Output=Self> +
    Add<Output=Self> +
    AddAssign +
;

impl<R> Scalable<R> for GeneralForm<R>
where
    for<'r> R: NonZeroFactorizable<Power=i32> + Multiplication<R::Factor> + SparseElement<R> + SparseComparator,
    for<'r> &'r R: Mul<&'r R, Output=R> + Div<&'r R, Output=R>,
    R::Power: Addition + One + Ord,
{
    fn scale(&mut self) -> Scaling<R> {
        // Factorize all numbers at once
        let factorization = self.factorize();
        // Optimize per factor
        let scale_per_factor = factorization.solve();
        // Aggregate scaling per factor
        let scaling = combine_factors(scale_per_factor);
        // Apply the computed scaling
        scale(self, &scaling, |x, y| *x *= y, |x, y| *x /= y);

        scaling
    }

    fn scale_back(&mut self, scaling: Scaling<R>) {
        scale(self, &scaling, |x, y| *x /= y, |x, y| *x *= y);
    }
}

/// Notation shorthand for a factorization.
///
/// This type omits any sign information and only keeps the identified factors and their powers.
#[allow(type_alias_bounds)]
type Factorization<R: NonZeroFactorizable> = Vec<(R::Factor, R::Power)>;

/// Intermediate storage of the factorization of a `GeneralForm` linear program.
#[derive(Eq, PartialEq, Debug)]
struct GeneralFormFactorization<R: NonZeroFactorizable> {
    /// Set of factors identified as present in this program.
    ///
    /// Note that this set is not necessarily complete; depending on the factorization algorithm
    /// used, only some of the factors might be identified. Typically, it is not worth it to
    /// exhaustively search for all factors.
    factors: Vec<R::Factor>,

    /// Right-hand side.
    ///
    /// Zero values are None, as zero values are not affected by scaling. Non zero values have a
    /// factorization that might be empty, in case the value equals 1.
    ///
    /// Only right-hand side information of constraints is included here (and not the bound values).
    b: Vec<Option<Factorization<R>>>,

    /// Cost values.
    ///
    /// Contains values from original variables only. Zero values are None, non-zero values are Some
    /// but the factorization might be empty.
    c: Vec<Option<Factorization<R>>>,

    /// Variable bounds.
    ///
    /// For each variable, whether a lower bound or upper bound is present. The value is None when:
    ///
    /// * There is no bound value
    /// * There is a bound value, but it is zero
    ///
    /// In both these cases, the bound is not affected by the scaling.
    bounds: Vec<(Option<Factorization<R>>, Option<Factorization<R>>)>,

    /// Constraint coefficients, column major.
    constraints: Vec<Vec<SparseTuple<Factorization<R>>>>,
}

impl<R: NonZeroFactorizable> GeneralFormFactorization<R> {
    fn nr_constraints(&self) -> usize {
        self.b.len()
    }
    fn nr_bounds(&self) -> usize {
        self.bounds.iter()
            .map(|pair| match pair {
                (None, None) => 0,
                (Some(_), None) | (None, Some(_)) => 1,
                (Some(_), Some(_)) => 2,
            })
            .sum()
    }
    fn nr_variables(&self) -> usize {
        self.c.len()
    }
}

impl<R: NonZeroFactorizable<Power=i32>> GeneralFormFactorization<R> {
    /// Solve the scaling problem posed by this factorization per factor.
    ///
    /// For each factor that was identified, minimize how often it occurs.
    fn solve(mut self) -> Vec<(R::Factor, ((R::Power, Vec<R::Power>), Vec<R::Power>))> {
        let index = self.build_index();

        let mut results = Vec::with_capacity(self.factors.len());
        debug_assert!(self.factors.is_sorted());
        while !self.factors.is_empty() {
            let (factor, (row_scaling, column_scaling)) =
                self.solve_single(&index);
            results.push((factor, (row_scaling, column_scaling)));
        }

        results
    }

    /// Build a row major index of the constraint data.
    ///
    /// This index contains for each row a sequence of column indices and the data index describing
    /// the location of the value in the column major constraint data, such that it can be looked up
    /// directly.
    fn build_index(&self) -> Vec<Vec<(usize, usize)>> {
        let mut index = vec![Vec::new(); self.b.len()];

        for (j, column) in self.constraints.iter().enumerate() {
            for (data_index, &(i, _)) in column.iter().enumerate() {
                index[i].push((j, data_index));
            }
        }

        index
    }

    /// Solve the scaling for a single factor.
    ///
    /// This factor is per convention the last factor of the `self.factors` field.
    ///
    /// The algorithm alternates between optimizing rows and columns, greedily making progress when
    /// possible. Any rows directly affected by a column change (and vice versa) are then queued to
    /// be visited again. This guarantees that no greedy step is possible anymore after the this
    /// procedure.
    fn solve_single(
        &mut self,
        row_major_constraint_index: &Vec<Vec<(usize, usize)>>,
    ) -> (R::Factor, ((R::Power, Vec<R::Power>), Vec<R::Power>)) {
        // The changes in power, the return values
        let mut cost_change = 0;
        let mut constraint_changes = vec![0; self.b.len()];
        let mut variable_changes = vec![0; self.c.len()];

        // Queues tracking which rows / columns still should be tested
        let mut row_queue = (0..self.b.len()).map(RowToIncrement::ConstraintRow)
            .chain(once(RowToIncrement::CostRow))
            .collect::<FIFOSet<_>>();
        let mut column_queue = (0..self.c.len()).collect::<FIFOSet<_>>();

        // Main loop, repeatedly improving rows and columns
        while let Some(next_side) = self.constraint_or_variable(&mut row_queue, &mut column_queue) {
            match next_side {
                NextSide::Column(index) => self.update_column(
                    index,
                    &cost_change, &constraint_changes, &mut variable_changes[index],
                    &mut row_queue,
                ),
                NextSide::Row(row_to_increment) => match row_to_increment {
                    RowToIncrement::ConstraintRow(index) => self.do_constraint(
                        index,
                        &mut constraint_changes[index], &variable_changes,
                        &mut column_queue,
                        row_major_constraint_index,
                    ),
                    RowToIncrement::CostRow => self.do_cost(
                        &mut cost_change,
                        &variable_changes,
                        &mut column_queue,
                    ),
                },
            }
        }

        let factor = self.remove_factor_info();
        (factor, ((cost_change, constraint_changes), variable_changes))
    }

    /// Retrieve the next value to be tested.
    ///
    /// If both queues are empty, returns `None`. Otherwise, returns a value from the queue that is
    /// largest relative to its maximum size. This is a heuristic to avoid checking the same row or
    /// column again sooner than necessary.
    fn constraint_or_variable(
        &self,
        row_queue: &mut FIFOSet<RowToIncrement>, column_queue: &mut FIFOSet<usize>,
    ) -> Option<NextSide> {
        match (row_queue.len(), column_queue.len()) {
            (0, 0) => None,
            (constraints, variables) => Some({
                match (constraints * self.nr_variables()).cmp(&(variables * self.nr_constraints())) {
                    Ordering::Less => NextSide::Column(column_queue.pop().unwrap()),
                    Ordering::Equal => {
                        // Somewhat arbitrary choice. Because there are probably more columns, take
                        // from there.
                        NextSide::Column(column_queue.pop().unwrap())
                    }
                    Ordering::Greater => NextSide::Row(row_queue.pop().unwrap()),
                }
            })
        }
    }

    /// Try to improve the scaling of a single column.
    ///
    /// Repeatedly in- or decreases the change in power for the current factor for this column until
    /// that is no longer profitable.
    fn update_column(
        &self,
        variable: usize,
        cost_change: &R::Power, constraint_changes: &Vec<R::Power>, column_change: &mut R::Power,
        row_queue: &mut FIFOSet<RowToIncrement>,
    ) {
        let (bound_weight, constraint_weight) = self.relative_column_weight();

        let mut made_change = false;

        // TODO(OPTIMIZATION): This method computes the profitability of both the increase and
        //  decrease direction each time, but one moves in only one of those directions.

        while {
            // Changes in penalty
            let mut on_increase = 0;
            let mut on_decrease = 0;

            if let Some(factorization) = self.c[variable].as_ref() {
                self.count_positive(
                    factorization,
                    &mut on_increase, &mut on_decrease,
                    *cost_change, *column_change,
                    constraint_weight as isize,
                );
            }
            for (row, factorization) in &self.constraints[variable] {
                self.count_positive(
                    factorization,
                    &mut on_increase, &mut on_decrease,
                    constraint_changes[*row], *column_change,
                    constraint_weight as isize,
                );
            }

            self.count_negative(
                self.bounds[variable].0.as_ref(),
                &mut on_increase, &mut on_decrease,
                *column_change,
                bound_weight as isize,
            );
            self.count_negative(
                self.bounds[variable].1.as_ref(),
                &mut on_increase, &mut on_decrease,
                *column_change,
                bound_weight as isize,
            );

            let mut should_continue = false;

            debug_assert!(on_increase >= 0 || on_decrease >= 0);

            if on_increase < 0 {
                *column_change += 1;
                should_continue = true;
            }
            if on_decrease < 0 {
                *column_change -= 1;
                should_continue = true;
            }

            should_continue
        } {
            made_change = true;
        }

        if made_change {
            for (row, _) in &self.constraints[variable] {
                row_queue.push(RowToIncrement::ConstraintRow(*row));
            }
        }
    }

    /// Try to improve the scaling of a single row.
    ///
    /// Repeatedly in- or decreases the change in power for the current factor for this row until
    /// that is no longer profitable.
    fn do_constraint(
        &self,
        row: usize,
        row_change: &mut R::Power, column_changes: &Vec<R::Power>,
        column_queue: &mut FIFOSet<usize>,
        row_major_constraint_index: &Vec<Vec<(usize, usize)>>,
    ) {
        let (bound_weight, constraint_weight) = self.relative_column_weight();

        let mut made_change = false;

        while {
            // Changes in penalty
            let mut on_increase = 0;
            let mut on_decrease = 0;

            for &(column, data_index) in &row_major_constraint_index[row] {
                self.count_positive(
                    &self.constraints[column][data_index].1,
                    &mut on_increase, &mut on_decrease,
                    *row_change, column_changes[column],
                    constraint_weight as isize,
                )
            }

            if let Some(factorization) = self.b[row].as_ref() {
                self.count_positive(
                    factorization,
                    &mut on_increase, &mut on_decrease,
                    *row_change, 0,
                    bound_weight as isize,
                )
            }

            let mut should_continue = false;

            debug_assert!(on_increase >= 0 || on_decrease >= 0);

            if on_increase < 0 {
                *row_change += 1;
                should_continue = true;
            }
            if on_decrease < 0 {
                *row_change -= 1;
                should_continue = true;
            }

            should_continue
        } {
            made_change = true;
        }

        if made_change {
            for (column, _) in &row_major_constraint_index[row] {
                column_queue.push(*column);
            }
        }
    }

    /// Count in the reverse direction the effect of a change in the bounds.
    fn count_negative(
        &self,
        bound: Option<&Factorization<R>>,
        on_increase: &mut isize, on_decrease: &mut isize,
        column_change: R::Power,
        weight: isize,
    ) {
        if let Some(factorization) = bound {
            match (self.initial_exponent(factorization) - column_change).cmp(&0) {
                Ordering::Less => {
                    *on_increase += weight;
                    *on_decrease -= weight;
                }
                Ordering::Equal => {
                    *on_increase += weight;
                    *on_decrease += weight;
                }
                Ordering::Greater => {
                    *on_increase -= weight;
                    *on_decrease += weight;
                }
            }
        }
    }

    /// Try to improve the scaling of the cost row.
    ///
    /// Repeatedly in- or decreases the change in power for the current factor for until that is no
    /// longer profitable.
    fn do_cost(
        &self,
        change: &mut R::Power, column_changes: &Vec<R::Power>,
        column_queue: &mut FIFOSet<usize>,
    ) {
        let (_, constraint_weight) = self.relative_column_weight();

        let mut made_change = false;

        while {
            // Changes in penalty
            let mut on_increase = 0;
            let mut on_decrease = 0;

            for (column, coefficient) in self.c.iter().enumerate() {
                if let Some(factorization) = coefficient.as_ref() {
                    self.count_positive(
                        factorization,
                        &mut on_increase, &mut on_decrease,
                        *change, column_changes[column],
                        constraint_weight as isize,
                    )
                }
            }

            let mut should_continue = false;

            debug_assert!(on_increase >= 0 || on_decrease >= 0);

            if on_increase < 0 {
                *change += 1;
                should_continue = true;
            }
            if on_decrease < 0 {
                *change -= 1;
                should_continue = true;
            }

            should_continue
        } {
            made_change = true;
        }

        if made_change {
            for (column, coefficient) in self.c.iter().enumerate() {
                if coefficient.is_some() {
                    column_queue.push(column);
                }
            }
        }
    }

    /// Count in the the effect of a change in the right-hand side, constraint or cost coefficients.
    fn count_positive(
        &self,
        factorization: &Factorization<R>,
        on_increase: &mut isize, on_decrease: &mut isize,
        existing_change: R::Power, column_change: R::Power,
        weight: isize,
    ) {
        match (self.initial_exponent(factorization) + existing_change + column_change).cmp(&0) {
            Ordering::Less => {
                *on_increase -= weight;
                *on_decrease += weight;
            }
            Ordering::Equal => {
                *on_increase += weight;
                *on_decrease += weight;
            }
            Ordering::Greater => {
                *on_increase += weight;
                *on_decrease -= weight;
            }
        }
    }

    /// Compute the relative importance of the rhs column w.r.t. the constraint columns.
    ///
    /// The following weighting scheme was considered, but in practice not effective:
    ///
    /// > Call the number of constraints `m`, the number of variables `n`. A typical problem has
    /// (much) more columns than constraints, so `m << n`. Columns will on average be in the basis
    /// in `m` out of `n` times, so we weigh the cost of having prime factors in those columns
    /// accordingly: weight `n` for the rhs column and smaller weight `m` for the constraint
    /// columns.
    ///
    /// Instead, right-hand side and normal columns are weight equally.
    ///
    /// # Return value
    ///
    /// A tuple weights, first element for the rhs vector, second element for the constraint
    /// vectors.
    fn relative_column_weight(&self) -> (usize, usize) {
        // The ineffective weighting scheme as described in the comment
        // let normal_column_weight = self.nr_constraints() + self.nr_bounds();
        // let b_weight = self.nr_variables() + self.nr_bounds();

        let normal_column_weight = 1;
        let b_weight = 1;

        (b_weight, normal_column_weight)
    }

    /// The value the exponent originally had in the factorization.
    ///
    /// The factorization is sparse. The factor of interest should be the last value in the
    /// factorization.
    #[inline]
    fn initial_exponent(
        &self,
        factorization: &Vec<(R::Factor, R::Power)>,
    ) -> R::Power {
        let factor = self.factors.last().unwrap();

        factorization.last()
            .filter(|(f, _)| f == factor)
            .map(|&(_, power)| power)
            .unwrap_or_else(R::Power::zero)
    }

    fn remove_factor_info(&mut self) -> R::Factor {
        let factor = self.factors.pop().unwrap();

        for column in &mut self.constraints {
            for (_, factorization) in column {
                match factorization.last() {
                    Some((f, _)) if f == &factor => {factorization.pop();},
                    _ => {},
                }
            }
        }

        let remove_from = |data: &mut [Option<Vec<(R::Factor, R::Power)>>]| {
            for factorization in data.iter_mut().flatten() {
                match factorization.last() {
                    Some((f, _)) if f == &factor => {factorization.pop();},
                    _ => {},
                }
            }
        };

        remove_from(&mut self.b);
        remove_from(&mut self.c);

        for (maybe_lower, maybe_upper) in &mut self.bounds {
            if let Some(factorization) = maybe_lower {
                match factorization.last() {
                    Some((f, _)) if f == &factor => {factorization.pop();},
                    _ => {},
                }
            }
            if let Some(factorization) = maybe_upper {
                match factorization.last() {
                    Some((f, _)) if f == &factor => {factorization.pop();},
                    _ => {},
                }
            }
        }

        factor
    }
}

enum NextSide {
    Row(RowToIncrement),
    Column(usize),
}

#[derive(Eq, PartialEq, Hash, Copy, Clone)]
enum RowToIncrement {
    CostRow,
    /// Index of the constraint
    ConstraintRow(usize),
}

impl<R> GeneralForm<R>
where
    R: NonZeroFactorizable<Factor: Hash> + SparseElement<R> + SparseComparator,
{
    /// Factorize this `GeneralForm`.
    ///
    /// Compute an approximation to the prime factorization of most values in this program.
    fn factorize(&self) -> GeneralFormFactorization<R> {
        let mut all_factors = HashSet::new();
        let mut add = |factorization: &Factorization<R>|
            all_factors.extend(factorization.iter().map(|(f, _)| f).cloned());

        let b = self.b.iter()
            .map(|v| {
                if v.is_not_zero() {
                    let NonZeroFactorization { factors, .. } = v.factorize();
                    add(&factors);

                    Some(factors)
                } else {
                    None
                }
            })
            .collect();

        let (c, bounds): (Vec<_>, Vec<_>) = self.variables.iter()
            .map(|variable| {
                let factors = if variable.cost.is_not_zero() {
                    let NonZeroFactorization { factors, .. } = variable.cost.factorize();
                    add(&factors);
                    Some(factors)
                } else { None };

                let mut get_bound = |bound: Option<&R>| {
                    bound.map(|value| {
                        if value.is_not_zero() {
                            let NonZeroFactorization { factors, .. } = value.factorize();
                            add(&factors);
                            Some(factors)
                        } else { None }
                    }).flatten()
                };
                let lower_bound = get_bound(variable.lower_bound.as_ref());
                let upper_bound = get_bound(variable.upper_bound.as_ref());

                (factors, (lower_bound, upper_bound))
            })
            .unzip();

        let constraints = self.constraints.iter_columns().map(|column| {
            column.iter().map(|(i, v)| {
                let NonZeroFactorization { factors, .. } = v.factorize();
                all_factors.extend(factors.iter().map(|(f, _)| f).cloned());
                (*i, factors)
            }).collect()
        }).collect();

        let mut factors = all_factors.into_iter().collect::<Vec<_>>();
        factors.sort_unstable();
        GeneralFormFactorization { factors, b, c, bounds, constraints }
    }
}

/// The scaling is computed per factor but is applied at once. Combine the scaling of all factors.
fn combine_factors<R>(
    scale_per_factor: Vec<(R::Factor, ((R::Power, Vec<R::Power>), Vec<R::Power>))>,
) -> Scaling<R>
where
    R: NonZeroFactorizable<Power: Abs> + One,
    for<'r> R: MulAssign<&'r R::Factor> + DivAssign<&'r R::Factor>,
{
    debug_assert!(!scale_per_factor.is_empty());

    let nr_rows = scale_per_factor[0].1.0.1.len();
    let nr_columns = scale_per_factor[0].1.1.len();

    let mut cost_factor = R::one();
    let mut constraint_row_factors = vec![R::one(); nr_rows];
    let mut constraint_column_factors = vec![R::one(); nr_columns];

    for (factor, ((c_change, row_changes), column_changes)) in scale_per_factor {
        match c_change.signum() {
            Sign::Positive => {
                let mut iter = R::Power::zero();
                while iter < c_change.abs() {
                    cost_factor *= &factor;
                    iter += R::Power::one();
                }
            }
            Sign::Zero => {}
            Sign::Negative => {
                let mut iter = R::Power::zero();
                while iter < c_change.abs() {
                    cost_factor /= &factor;
                    iter += R::Power::one();
                }
            }
        }
        for (i, row_change) in row_changes.into_iter().enumerate() {
            match row_change.signum() {
                Sign::Positive => {
                    let mut iter = R::Power::zero();
                    while iter < row_change.abs() {
                        constraint_row_factors[i] *= &factor;
                        iter += R::Power::one();
                    }
                },
                Sign::Zero => {},
                Sign::Negative => {
                    let mut iter = R::Power::zero();
                    while iter < row_change.abs() {
                        constraint_row_factors[i] /= &factor;
                        iter += R::Power::one();
                    }
                },
            }
        }
        for (j, column_change) in column_changes.into_iter().enumerate() {
            match column_change.signum() {
                Sign::Positive => {
                    let mut iter = R::Power::zero();
                    while iter < column_change.abs() {
                        constraint_column_factors[j] /= &factor;
                        iter += R::Power::one();
                    }
                },
                Sign::Zero => (),
                Sign::Negative => {
                    let mut iter = R::Power::zero();
                    while iter < column_change.abs() {
                        constraint_column_factors[j] *= &factor;
                        iter += R::Power::one();
                    }
                },
            }
        }
    }

    Scaling {
        cost_factor,
        constraint_row_factors,
        constraint_column_factors,
    }
}

#[cfg(test)]
mod test;
