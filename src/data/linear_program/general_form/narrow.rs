//! # Handing the algorithm as little as the problem fits in
//!
//! The type a problem is read into has to be wide enough for presolving to compute in: bounds are
//! derived from bounds, columns are substituted into columns, and every one of those steps can
//! make a number bigger. The algorithm's needs are different. It only ever *reads* the problem
//! data, into an accumulator of its own, so what the problem is stored in and what it is computed
//! in are two separate questions. [`Absorb`] is the answer to the second one; this module is the
//! answer to the first.
//!
//! The saving is not small. A `RationalBig` holds two `SmallVec`s of eight words each, so a
//! coefficient costs around a hundred and fifty bytes whether or not it is the number three. A
//! `Rational32` costs twelve. For a constraint matrix that is the difference between fitting in
//! cache and not.
//!
//! # Nothing is rewritten here
//!
//! Only the type changes. Rewriting the rows to clear their fractions looks free — multiplying a
//! row by a positive number leaves `B^-1 A` and `B^-1 b` unchanged for every basis — and it is
//! not: the algorithm does not read this matrix alone, and a basis of cleared rows is more
//! expensive to factor than one of fractions. Measured over netlib, clearing takes the solving
//! time from 61 seconds to 277. [`RuleId::RowScaling`] says why in more detail; it is off by
//! default, and nothing of the kind happens here.
//!
//! [`Absorb`]: relp_num::Absorb
//! [`RuleId::RowScaling`]: crate::data::linear_program::general_form::RuleId::RowScaling

use relp_num::{
    Absorb, EncodingSize, Narrow as NarrowInto, OrderedField, OrderedFieldRef, Rational8,
    Rational16, Rational32, Rational64, Rational128,
};

use crate::algorithm::two_phase::matrix_provider::MatrixProvider;
use crate::algorithm::two_phase::matrix_provider::matrix_data::{BasisStatus, MatrixData};
use crate::algorithm::two_phase::tableau::inverse_maintenance::InverseMaintainer;
use crate::algorithm::two_phase::tableau::inverse_maintenance::ops as im_ops;
use crate::algorithm::two_phase::tableau::kind::artificial::Cost as ArtificialCost;
use crate::algorithm::{OptimizationResult, SolveRelaxation};
use crate::data::linear_algebra::matrix::{ColumnMajor, MatrixOrder, SparseMatrix};
use crate::data::linear_algebra::traits::{
    SparseComparator, SparseElement as LinearAlgebraElement,
};
use crate::data::linear_algebra::vector::{DenseVector, SparseVector, Vector};
use crate::data::linear_program::elements::RangedConstraintRelation;
use crate::data::linear_program::general_form::{GeneralForm, Variable};

/// What a number type has to offer for a problem to be stored in less than it.
pub trait Narrowable:
    EncodingSize
    + NarrowInto<Rational8>
    + NarrowInto<Rational16>
    + NarrowInto<Rational32>
    + NarrowInto<Rational64>
    + NarrowInto<Rational128>
{
}

impl<F> Narrowable for F where
    F: EncodingSize
        + NarrowInto<Rational8>
        + NarrowInto<Rational16>
        + NarrowInto<Rational32>
        + NarrowInto<Rational64>
        + NarrowInto<Rational128>
{
}

/// What a wide accumulator has to be able to read a problem stored in `N`.
///
/// A matrix provider hands out its coefficients directly and its costs as `Option`s, so both have
/// to be readable. The lift from one to the other is [`Absorb`]'s, not this trait's.
pub trait Reads<N>: Absorb<N> + for<'a> Absorb<Option<&'a N>> {}
impl<W, N> Reads<N> for W where W: Absorb<N> + for<'a> Absorb<Option<&'a N>> {}

/// Everything an accumulator has to read to solve a narrowed problem, whatever it narrowed to.
pub trait ReadsNarrowed<F>:
    Reads<Rational8>
    + Reads<Rational16>
    + Reads<Rational32>
    + Reads<Rational64>
    + Reads<Rational128>
    + Reads<F>
{
}

impl<W, F> ReadsNarrowed<F> for W where
    W: Reads<Rational8>
        + Reads<Rational16>
        + Reads<Rational32>
        + Reads<Rational64>
        + Reads<Rational128>
        + Reads<F>
{
}

/// The narrowest fixed width rational a value fits in.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Width {
    /// `Rational8`.
    Bits8,
    /// `Rational16`.
    Bits16,
    /// `Rational32`.
    Bits32,
    /// `Rational64`.
    Bits64,
    /// `Rational128`.
    Bits128,
    /// None of them, so the problem stays in the type it was read into.
    Wide,
}

impl Width {
    /// The narrowest width that holds a numerator and a denominator of this many bits.
    ///
    /// Both parts are stored unsigned and the sign beside them, so the width is decided by the
    /// larger of the two.
    fn holding(bits: u32) -> Self {
        match bits {
            0..=8 => Width::Bits8,
            9..=16 => Width::Bits16,
            17..=32 => Width::Bits32,
            33..=64 => Width::Bits64,
            65..=128 => Width::Bits128,
            _ => Width::Wide,
        }
    }

    /// A name to print in a report.
    pub fn name(self) -> &'static str {
        match self {
            Width::Bits8 => "8 bits",
            Width::Bits16 => "16 bits",
            Width::Bits32 => "32 bits",
            Width::Bits64 => "64 bits",
            Width::Bits128 => "128 bits",
            Width::Wide => "arbitrary precision",
        }
    }
}

/// The standardized problem, held in one number type.
///
/// This is exactly what a [`MatrixData`] reads and nothing else. In particular the variables carry
/// only their cost and their upper bound; the record of how each of them was standardized stays
/// with the problem that was standardized, which is where it is needed.
#[derive(Debug)]
struct Standardized<N> {
    constraints: SparseMatrix<N, N, ColumnMajor>,
    right_hand_side: DenseVector<N>,
    ranges: Vec<N>,
    variables: Vec<Variable<N>>,
    constraint_type_counts: [usize; 4],
}

impl<N> Standardized<N>
where
    N: 'static + OrderedField + LinearAlgebraElement<N>,
    for<'r> &'r N: OrderedFieldRef<N>,
{
    /// The problem as the algorithm reads it.
    fn matrix_data(&self) -> MatrixData<'_, N> {
        let [equality, range, upper, lower] = self.constraint_type_counts;
        MatrixData::new(
            &self.constraints,
            &self.right_hand_side,
            self.ranges.iter().collect(),
            equality,
            range,
            upper,
            lower,
            &self.variables,
        )
    }
}

/// The standardized problem, stored in the narrowest number type that holds it exactly.
#[derive(Debug)]
pub struct Narrowed<F> {
    stored: Stored<F>,
}

#[derive(Debug)]
enum Stored<F> {
    Bits8(Standardized<Rational8>),
    Bits16(Standardized<Rational16>),
    Bits32(Standardized<Rational32>),
    Bits64(Standardized<Rational64>),
    Bits128(Standardized<Rational128>),
    Wide(Standardized<F>),
}

/// What solving the standardized problem concluded.
///
/// Not a `Result`: an infeasible or unbounded problem is an answer, not a failure to produce one.
#[derive(Debug, Eq, PartialEq)]
pub enum Solved<G> {
    /// An optimal point, with the multipliers that certify it.
    Optimum(StandardizedSolution<G>),
    /// The problem has no feasible point.
    Infeasible,
    /// The objective can be improved without limit.
    Unbounded,
}

/// An optimal primal and dual pair, against the standardized problem's own rows and columns.
///
/// Which is what [`GeneralForm::compute_full_solution_with_duals`] takes.
#[derive(Debug, Eq, PartialEq)]
pub struct StandardizedSolution<G> {
    /// The value of each column of the standardized problem.
    pub primal: SparseVector<G, G>,
    /// The multiplier of each of its constraint rows.
    pub duals: Vec<G>,
    /// Where each of its variables sits relative to its own bounds.
    pub statuses: Vec<BasisStatus>,
}

impl<F> Narrowed<F> {
    /// What the problem ended up being stored in.
    pub fn width(&self) -> Width {
        match self.stored {
            Stored::Bits8(_) => Width::Bits8,
            Stored::Bits16(_) => Width::Bits16,
            Stored::Bits32(_) => Width::Bits32,
            Stored::Bits64(_) => Width::Bits64,
            Stored::Bits128(_) => Width::Bits128,
            Stored::Wide(_) => Width::Wide,
        }
    }
}

impl<F> Narrowed<F>
where
    F: 'static + OrderedField + LinearAlgebraElement<F>,
    for<'r> &'r F: OrderedFieldRef<F>,
{
    /// Solve, and give the answer back in terms of the problem that was standardized.
    ///
    /// The problem stored here is the standardized problem itself, only held in less, so what
    /// comes back needs nothing done to it.
    pub fn solve_relaxation<IM>(&self) -> Solved<IM::F>
    where
        IM: InverseMaintainer<F: im_ops::FieldHR>,
        IM::F: Absorb<ArtificialCost> + ReadsNarrowed<F>,
    {
        match &self.stored {
            Stored::Bits8(problem) => solve::<IM, Rational8>(problem),
            Stored::Bits16(problem) => solve::<IM, Rational16>(problem),
            Stored::Bits32(problem) => solve::<IM, Rational32>(problem),
            Stored::Bits64(problem) => solve::<IM, Rational64>(problem),
            Stored::Bits128(problem) => solve::<IM, Rational128>(problem),
            Stored::Wide(problem) => solve::<IM, F>(problem),
        }
    }
}

/// A bound that may not be there, in a narrower type.
///
/// The outer `Option` is whether it fitted and the inner one whether there is a bound at all,
/// which is the wrong way round for `?`.
fn narrow_bound<N: NarrowInto<M>, M>(bound: &Option<N>) -> Option<Option<M>> {
    match bound {
        None => Some(None),
        Some(value) => value.narrow().map(Some),
    }
}

/// Solve one narrowed problem, whatever it was narrowed to.
fn solve<IM, N>(problem: &Standardized<N>) -> Solved<IM::F>
where
    IM: InverseMaintainer<F: im_ops::FieldHR>,
    IM::F: Absorb<ArtificialCost> + Reads<N>,
    N: 'static + OrderedField + LinearAlgebraElement<N>,
    for<'r> &'r N: OrderedFieldRef<N>,
{
    let data = problem.matrix_data();
    match data.solve_relaxation::<IM>() {
        OptimizationResult::FiniteOptimum(optimum) => Solved::Optimum(StandardizedSolution {
            duals: data.constraint_duals(&optimum.dual),
            statuses: data.variable_statuses(&optimum.basic),
            primal: data.reconstruct_solution(optimum.primal),
        }),
        OptimizationResult::Infeasible => Solved::Infeasible,
        OptimizationResult::Unbounded => Solved::Unbounded,
    }
}

impl<N> Standardized<N>
where
    N: LinearAlgebraElement<N> + SparseComparator,
{
    /// The widest thing in here, which is what the whole problem has to be stored in.
    fn width(&self) -> Width
    where
        N: EncodingSize,
    {
        let mut widest = Width::Bits8;
        let mut consider = |value: &N| {
            let bits = value.numerator_bits().max(value.denominator_bits());
            widest = widest.max(Width::holding(bits));
        };

        for column in self.constraints.iter_columns() {
            for (_, coefficient) in column {
                consider(coefficient);
            }
        }
        for value in self.right_hand_side.iter() {
            consider(value);
        }
        for range in &self.ranges {
            consider(range);
        }
        for variable in &self.variables {
            consider(&variable.cost);
            if let Some(bound) = &variable.upper_bound {
                consider(bound);
            }
        }

        widest
    }

    /// The same problem in a narrower type, when every value in it fits.
    fn narrow<M>(&self) -> Option<Standardized<M>>
    where
        N: NarrowInto<M>,
        M: LinearAlgebraElement<M> + SparseComparator + num_traits::Zero,
    {
        let columns = self
            .constraints
            .iter_columns()
            .map(|column| {
                column
                    .iter()
                    .map(|(row, value)| Some((*row, value.narrow()?)))
                    .collect::<Option<Vec<_>>>()
            })
            .collect::<Option<Vec<_>>>()?;

        let variables = self
            .variables
            .iter()
            .map(|variable| {
                Some(Variable {
                    variable_type: variable.variable_type,
                    cost: variable.cost.narrow()?,
                    lower_bound: narrow_bound(&variable.lower_bound)?,
                    upper_bound: narrow_bound(&variable.upper_bound)?,
                    // Only the cost and the upper bound are read from here. What a variable was
                    // shifted and flipped by belongs to the problem that was standardized, which
                    // is where the solution is turned back into one of the problem that was read.
                    shift: M::zero(),
                    flipped: false,
                })
            })
            .collect::<Option<Vec<_>>>()?;

        Some(Standardized {
            constraints: ColumnMajor::new(
                columns,
                self.constraints.nr_rows(),
                self.constraints.nr_columns(),
            ),
            right_hand_side: DenseVector::new(
                self.right_hand_side
                    .iter()
                    .map(NarrowInto::narrow)
                    .collect::<Option<Vec<_>>>()?,
                self.right_hand_side.len(),
            ),
            ranges: self
                .ranges
                .iter()
                .map(NarrowInto::narrow)
                .collect::<Option<Vec<_>>>()?,
            variables,
            constraint_type_counts: self.constraint_type_counts,
        })
    }
}

impl<OF> GeneralForm<OF>
where
    OF: 'static + OrderedField + LinearAlgebraElement<OF> + Narrowable,
    for<'r> &'r OF: OrderedFieldRef<OF>,
{
    /// The standardized problem, stored in the smallest type that holds it.
    ///
    /// Call this in place of [`GeneralForm::derive_matrix_data`], with the same argument, and
    /// solve what comes back. What is left of the pipeline is unchanged: the primal and the
    /// multipliers the result carries are the ones
    /// [`GeneralForm::compute_full_solution_with_duals`] expects.
    pub fn narrow(&self, constraint_type_counts: [usize; 4]) -> Narrowed<OF> {
        let [equalities, ranges, ..] = constraint_type_counts;
        debug_assert_eq!(
            constraint_type_counts.iter().sum::<usize>(),
            self.nr_constraints(),
        );

        let copied = Standardized {
            constraints: ColumnMajor::new(
                self.constraints
                    .iter_columns()
                    .map(|column| column.to_vec())
                    .collect(),
                self.nr_constraints(),
                self.variables.len(),
            ),
            right_hand_side: DenseVector::new(
                self.b.iter().cloned().collect(),
                self.nr_constraints(),
            ),
            ranges: (equalities..equalities + ranges)
                .map(|row| match &self.constraint_types[row] {
                    RangedConstraintRelation::Range(range) => range.clone(),
                    _ => unreachable!("the rows were sorted by type, and these are the ranges"),
                })
                .collect(),
            variables: self
                .variables
                .iter()
                .map(|variable| Variable {
                    variable_type: variable.variable_type,
                    cost: variable.cost.clone(),
                    lower_bound: variable.lower_bound.clone(),
                    upper_bound: variable.upper_bound.clone(),
                    // Only the cost and the upper bound are read from here. What a variable was
                    // shifted and flipped by belongs to the problem that was standardized, which
                    // is where the solution is turned back into one of the problem that was read.
                    shift: OF::zero(),
                    flipped: false,
                })
                .collect(),
            constraint_type_counts,
        };

        let fits = "the width was measured over exactly these values";
        let stored = match copied.width() {
            Width::Bits8 => Stored::Bits8(copied.narrow().expect(fits)),
            Width::Bits16 => Stored::Bits16(copied.narrow().expect(fits)),
            Width::Bits32 => Stored::Bits32(copied.narrow().expect(fits)),
            Width::Bits64 => Stored::Bits64(copied.narrow().expect(fits)),
            Width::Bits128 => Stored::Bits128(copied.narrow().expect(fits)),
            Width::Wide => Stored::Wide(copied),
        };

        Narrowed { stored }
    }
}

#[cfg(test)]
mod test {
    use relp_num::{RB, RationalBig};

    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::basis_inverse_rows::BasisInverseRows;
    use crate::data::linear_algebra::matrix::{ColumnMajor, MatrixOrder};
    use crate::data::linear_algebra::vector::DenseVector;
    use crate::data::linear_algebra::vector::test::TestVector;
    use crate::data::linear_program::elements::{
        Objective, RangedConstraintRelation, VariableType,
    };
    use crate::data::linear_program::general_form::narrow::{Solved, Width};
    use crate::data::linear_program::general_form::{GeneralForm, Variable};

    type T = RationalBig;

    /// `min x0 + x1` subject to one row, with the given coefficients and bounds.
    fn problem(coefficients: Vec<T>, right_hand_side: T, upper_bound: T) -> GeneralForm<T> {
        let nr_variables = coefficients.len();
        GeneralForm::new(
            Objective::Minimize,
            ColumnMajor::from_test_data::<T, T, _>(&[coefficients], nr_variables),
            vec![RangedConstraintRelation::Greater],
            DenseVector::from_test_data(vec![right_hand_side]),
            (0..nr_variables)
                .map(|_| Variable {
                    variable_type: VariableType::Continuous,
                    cost: RB!(1),
                    lower_bound: Some(RB!(0)),
                    upper_bound: Some(upper_bound.clone()),
                    shift: RB!(0),
                    flipped: false,
                })
                .collect(),
            (0..nr_variables).map(|j| format!("x{j}")).collect(),
            RB!(0),
        )
    }

    #[test]
    fn the_width_is_decided_by_the_larger_of_the_two_parts() {
        assert_eq!(Width::holding(0), Width::Bits8);
        assert_eq!(Width::holding(8), Width::Bits8);
        assert_eq!(Width::holding(9), Width::Bits16);
        assert_eq!(Width::holding(32), Width::Bits32);
        assert_eq!(Width::holding(33), Width::Bits64);
        assert_eq!(Width::holding(128), Width::Bits128);
        assert_eq!(Width::holding(129), Width::Wide);
        assert!(Width::Bits8 < Width::Wide);
    }

    #[test]
    fn a_problem_of_small_numbers_is_stored_in_a_byte() {
        let mut general = problem(vec![RB!(3), RB!(-5)], RB!(7), RB!(200));
        let counts = general.standardize();
        assert_eq!(general.narrow(counts).width(), Width::Bits8);
    }

    /// One value decides for all of them: the whole problem is stored in one type.
    #[test]
    fn one_large_value_widens_everything() {
        let mut general = problem(vec![RB!(3), RB!(-5)], RB!(7), RB!(300));
        let counts = general.standardize();
        assert_eq!(general.narrow(counts).width(), Width::Bits16);

        let mut general = problem(vec![RB!(3), RB!(-5)], RB!(1_i64 << 40), RB!(200));
        let counts = general.standardize();
        assert_eq!(general.narrow(counts).width(), Width::Bits64);
    }

    /// A denominator counts as much as a numerator.
    #[test]
    fn a_fraction_is_measured_at_both_ends() {
        let mut general = problem(vec![RB!(1, 300), RB!(-5)], RB!(7), RB!(200));
        let counts = general.standardize();
        assert_eq!(general.narrow(counts).width(), Width::Bits16);
    }

    #[test]
    fn what_no_fixed_width_holds_stays_where_it_is() {
        let huge = &RB!(1_i64 << 62) * &RB!(1_i64 << 62);
        let mut general = problem(vec![RB!(3), RB!(-5)], &huge * &huge, RB!(200));
        let counts = general.standardize();
        assert_eq!(general.narrow(counts).width(), Width::Wide);
    }

    /// The narrowed problem is the same problem: it has to give the same answer.
    #[test]
    fn narrowing_does_not_change_the_answer() {
        for upper_bound in [RB!(200), RB!(300), RB!(1_i64 << 40)] {
            let mut general = problem(vec![RB!(3), RB!(-5)], RB!(7), upper_bound);
            let expected = {
                let mut wide = general.clone();
                let counts = wide.standardize();
                let data = wide.derive_matrix_data(counts);
                let solved = crate::algorithm::SolveRelaxation::solve_relaxation::<
                    Carry<T, BasisInverseRows<_>>,
                >(&data);
                let crate::algorithm::OptimizationResult::FiniteOptimum(optimum) = solved else {
                    panic!("the problem has a finite optimum");
                };
                use crate::algorithm::two_phase::matrix_provider::MatrixProvider;
                wide.compute_full_solution_with_reduced_solution::<T>(
                    data.reconstruct_solution(optimum.primal),
                )
            };

            let counts = general.standardize();
            let Solved::Optimum(optimum) =
                general.narrow(counts).solve_relaxation::<Carry<T, BasisInverseRows<_>>>()
            else {
                panic!("the problem has a finite optimum");
            };
            let narrowed = general.compute_full_solution_with_reduced_solution::<T>(optimum.primal);

            assert_eq!(narrowed, expected);
        }
    }
}
