//! # Pivot rules
//!
//! Strategies for moving from basis to basis, whether primal or dual.
use std::ops::Range;

use num_traits::One;
use relp_num::{Absorb, Signed};

use crate::algorithm::two_phase::matrix_provider::column::Column;
use crate::algorithm::two_phase::tableau::inverse_maintenance::{
    ColumnComputationInfo, InverseMaintainer, ops as im_ops,
};
use crate::algorithm::two_phase::tableau::kind::Kind;
use crate::algorithm::two_phase::tableau::kind::artificial::Artificial;
use crate::algorithm::two_phase::tableau::{BasisChangeComputationInfo, Tableau};
use crate::data::linear_algebra::SparseTuple;
use crate::data::linear_algebra::vector::Vector;

/// Deciding how to pivot.
///
/// During the Simplex method, one needs to decide how to move from basic solution to basic
/// solution. The pivot rule describes that behavior.
///
/// Once the column has been selected for a primal pivot (or the row for a dual pivot), a row
/// (column) needs to be found. This decision is currently made independent of the strategy.
pub trait PivotRule<F: im_ops::Field> {
    /// Create a new instance.
    fn new<IM, K>(tableau: &Tableau<IM, K>) -> Self
    where
        IM: InverseMaintainer<F = F>,
        K: Kind,
        F: Absorb<<K::Column as Column>::F>,
        F: Absorb<K::Cost>;

    /// Column selection rule for the primal Simplex method.
    fn select_primal_pivot_column<IM, K>(
        &mut self,
        tableau: &Tableau<IM, K>,
    ) -> Option<SparseTuple<IM::F>>
    where
        IM: InverseMaintainer<F = F>,
        K: Kind,
        F: Absorb<<K::Column as Column>::F>,
        F: Absorb<K::Cost>;

    fn after_basis_update<IM, K>(
        &mut self,
        _info: BasisChangeComputationInfo<IM::F>,
        _tableau: &Tableau<IM, K>,
    ) where
        IM: InverseMaintainer<F = F>,
        K: Kind,
        F: Absorb<<K::Column as Column>::F>,
        F: Absorb<K::Cost>,
    {
    }
}

/// Helper trait to facilitate different implementations per tableau kind.
trait StartIndex {
    /// Index of the non artificial variable with the lowest index.
    ///
    /// Artificial variables have the lowest indices.
    fn start_index(&self) -> usize;
}

impl<IM, K> StartIndex for Tableau<IM, K>
where
    K: Kind,
{
    default fn start_index(&self) -> usize {
        0
    }
}

impl<IM, A> StartIndex for Tableau<IM, A>
where
    A: Artificial,
{
    fn start_index(&self) -> usize {
        self.nr_artificial_variables()
    }
}

/// Entering half of Bland's rule: the profitable column with the lowest index.
///
/// The leaving half is implemented in
/// [`select_primal_pivot_row`](Tableau::select_primal_pivot_row), which breaks ties in the ratio
/// test by picking the leaving column with the lowest index. Bland's theorem states that when both
/// halves are used, no basis is ever visited twice, so the Simplex method terminates. Applying only
/// one of the two halves provides no such guarantee.
fn bland_pivot_column<IM, K>(tableau: &Tableau<IM, K>) -> Option<SparseTuple<IM::F>>
where
    IM: InverseMaintainer,
    K: Kind,
    IM::F: Absorb<<K::Column as Column>::F>,
    IM::F: Absorb<K::Cost>,
{
    (tableau.start_index()..tableau.nr_columns())
        .filter(|&column| !tableau.is_in_basis(column))
        .map(|column| (column, tableau.relative_cost(column)))
        .find(|(_, cost)| cost.is_negative())
}

/// Simply pivot on the first column, which has a negative relative cost.
///
/// This is the entering half of Bland's rule; combined with the leaving rule of
/// [`select_primal_pivot_row`](Tableau::select_primal_pivot_row), the method cannot cycle.
///
/// The behavior is the same for both the primal and dual simplex method.
pub struct FirstProfitable;
impl<F> PivotRule<F> for FirstProfitable
where
    F: im_ops::Field,
{
    fn new<IM, K>(_tableau: &Tableau<IM, K>) -> Self {
        Self
    }

    fn select_primal_pivot_column<IM, K>(
        &mut self,
        tableau: &Tableau<IM, K>,
    ) -> Option<SparseTuple<IM::F>>
    where
        IM: InverseMaintainer<F = F>,
        K: Kind,
        F: Absorb<<K::Column as Column>::F>,
        F: Absorb<K::Cost>,
    {
        bland_pivot_column(tableau)
    }
}

/// Number of consecutive degenerate pivots after which [`BlandFallback`] switches to Bland's rule.
///
/// Degeneracy is common and mostly harmless, so this value is large enough that merely degenerate
/// problems keep being solved with the wrapped rule, and small enough that a cycle is left behind
/// quickly.
pub const DEFAULT_MAX_DEGENERATE_PIVOTS: usize = 100;

/// Fall back to Bland's rule when the Simplex method might be cycling.
///
/// Rules like [`SteepestDescentAlongObjective`] pick the most attractive column rather than the one
/// with the lowest index, so they don't implement the entering half of Bland's rule and, by
/// themselves, offer no protection against cycling. Wrapping such a rule in this type adds that
/// protection at almost no cost: it counts the pivots that didn't change the objective value and,
/// after `MAX_DEGENERATE_PIVOTS` of them in a row, selects the entering column with Bland's rule
/// until the objective value improves again.
///
/// That is enough to guarantee termination. Suppose the method would not terminate. There are
/// finitely many bases and the objective value never increases, so from some pivot onwards it is
/// constant and every pivot is degenerate. The counter then never resets, so at most
/// `MAX_DEGENERATE_PIVOTS` pivots later Bland's rule is in charge and stays in charge. From that
/// basis onwards, both halves of Bland's rule are applied to every pivot, which by Bland's theorem
/// terminates: a contradiction. The value of `MAX_DEGENERATE_PIVOTS` is therefore irrelevant for
/// correctness; it only trades a longer degenerate run against pivoting by index instead of by
/// attractiveness.
pub struct BlandFallback<R, const MAX_DEGENERATE_PIVOTS: usize = DEFAULT_MAX_DEGENERATE_PIVOTS> {
    /// Rule to use while the objective value still improves regularly.
    inner: R,
    /// Number of pivots since the last one that changed the objective value.
    nr_degenerate_pivots: usize,
}

impl<R, const MAX_DEGENERATE_PIVOTS: usize> BlandFallback<R, MAX_DEGENERATE_PIVOTS> {
    /// Number of pivots since the last pivot that changed the objective value.
    ///
    /// Once this reaches `MAX_DEGENERATE_PIVOTS`, the entering column is selected with Bland's rule
    /// instead of with the wrapped rule.
    pub fn nr_degenerate_pivots(&self) -> usize {
        self.nr_degenerate_pivots
    }
}

impl<F, R, const MAX_DEGENERATE_PIVOTS: usize> PivotRule<F>
    for BlandFallback<R, MAX_DEGENERATE_PIVOTS>
where
    F: im_ops::Field,
    R: PivotRule<F>,
{
    fn new<IM, K>(tableau: &Tableau<IM, K>) -> Self
    where
        IM: InverseMaintainer<F = F>,
        K: Kind,
        F: Absorb<<K::Column as Column>::F>,
        F: Absorb<K::Cost>,
    {
        Self {
            inner: R::new(tableau),
            nr_degenerate_pivots: 0,
        }
    }

    fn select_primal_pivot_column<IM, K>(
        &mut self,
        tableau: &Tableau<IM, K>,
    ) -> Option<SparseTuple<IM::F>>
    where
        IM: InverseMaintainer<F = F>,
        K: Kind,
        F: Absorb<<K::Column as Column>::F>,
        F: Absorb<K::Cost>,
    {
        if self.nr_degenerate_pivots >= MAX_DEGENERATE_PIVOTS {
            bland_pivot_column(tableau)
        } else {
            self.inner.select_primal_pivot_column(tableau)
        }
    }

    fn after_basis_update<IM, K>(
        &mut self,
        info: BasisChangeComputationInfo<IM::F>,
        tableau: &Tableau<IM, K>,
    ) where
        IM: InverseMaintainer<F = F>,
        K: Kind,
        F: Absorb<<K::Column as Column>::F>,
        F: Absorb<K::Cost>,
    {
        let pivot_row_index = info.pivot_row_index;
        // The wrapped rule keeps maintaining its data, also while the fallback is active, such that
        // it can take over again immediately.
        self.inner.after_basis_update(info, tableau);

        // The entering variable is now the basic variable of the pivot row, and its value equals the
        // step length that was taken. The objective value changed if and only if that step was
        // nonzero, because the relative cost of the entering column was strictly negative.
        if tableau.constraint_value(pivot_row_index).is_not_zero() {
            self.nr_degenerate_pivots = 0;
        } else {
            self.nr_degenerate_pivots += 1;
        }
    }
}

/// Small modification w.r.t. the `FirstProfitable` rule; it starts the search from the last
/// column selected.
pub struct FirstProfitableWithMemory {
    last_selected: Option<usize>,
}
impl<F> PivotRule<F> for FirstProfitableWithMemory
where
    F: im_ops::Field,
{
    fn new<IM, K>(_tableau: &Tableau<IM, K>) -> Self {
        Self {
            last_selected: None,
        }
    }

    fn select_primal_pivot_column<IM, K>(
        &mut self,
        tableau: &Tableau<IM, K>,
    ) -> Option<SparseTuple<IM::F>>
    where
        IM: InverseMaintainer<F = F>,
        K: Kind,
        F: Absorb<<K::Column as Column>::F>,
        F: Absorb<K::Cost>,
    {
        let find = |to_consider: Range<usize>| {
            to_consider
                .filter(|&column| !tableau.is_in_basis(column))
                .map(|column| (column, tableau.relative_cost(column)))
                .find(|(_, cost)| cost.is_negative())
        };

        let potential = self.last_selected.map_or_else(
            || find(tableau.start_index()..tableau.nr_columns()),
            |last| {
                find((last + 1)..tableau.nr_columns()).or_else(|| find(tableau.start_index()..last))
            },
        );

        self.last_selected = potential.as_ref().map(|&(i, _)| i);
        potential
    }
}

/// Simply pivot on the column, which has the most negative relative cost.
pub struct SteepestDescentAlongVariable;
impl<F> PivotRule<F> for SteepestDescentAlongVariable
where
    F: im_ops::Field,
    for<'r> &'r F: Ord,
{
    fn new<IM, K>(_tableau: &Tableau<IM, K>) -> Self {
        Self
    }

    fn select_primal_pivot_column<IM, K>(
        &mut self,
        tableau: &Tableau<IM, K>,
    ) -> Option<SparseTuple<IM::F>>
    where
        IM: InverseMaintainer<F = F>,
        K: Kind,
        F: Absorb<<K::Column as Column>::F>,
        F: Absorb<K::Cost>,
    {
        let mut smallest = None;
        for (j, cost) in (tableau.start_index()..tableau.nr_columns())
            .filter(|&column| !tableau.is_in_basis(column))
            .map(|column| (column, tableau.relative_cost(column)))
            .filter(|(_, cost)| cost.is_negative())
        {
            if let Some((existing_j, existing_cost)) = smallest.as_mut() {
                if &cost < existing_cost {
                    *existing_j = j;
                    *existing_cost = cost;
                }
            } else {
                smallest = Some((j, cost))
            }
        }

        smallest
    }
}

/// Goldfarb et. al's 1977 "A practicable steepest-edge simplex algorithm".
pub struct SteepestDescentAlongObjective<F> {
    /// The gamma equal 1 + norm(B^-1 a_j)^2.
    ///
    /// There is a value for each column. Basis columns have a None, as do artificial variables who
    /// should never be considered for entering into the basis.
    gamma: Vec<Option<F>>,
}

impl<F> PivotRule<F> for SteepestDescentAlongObjective<F>
where
    F: im_ops::Field + im_ops::FieldHR,
{
    fn new<IM, K>(tableau: &Tableau<IM, K>) -> Self
    where
        IM: InverseMaintainer<F = F>,
        K: Kind,
        F: Absorb<<K::Column as Column>::F>,
        F: Absorb<K::Cost>,
    {
        Self {
            gamma: (0..tableau.nr_columns())
                .map(|j| {
                    if j >= tableau.start_index() && !tableau.is_in_basis(j) {
                        Some(initial_gamma(j, tableau))
                    } else {
                        None
                    }
                })
                .collect(),
        }
    }

    fn select_primal_pivot_column<IM, K>(
        &mut self,
        tableau: &Tableau<IM, K>,
    ) -> Option<SparseTuple<IM::F>>
    where
        IM: InverseMaintainer<F = F>,
        K: Kind,
        F: Absorb<<K::Column as Column>::F>,
        F: Absorb<K::Cost>,
    {
        (tableau.start_index()..tableau.nr_columns())
            .filter(|&j| !tableau.is_in_basis(j))
            .map(|j| (j, tableau.relative_cost(j)))
            .filter(|(_, cost)| cost.is_negative())
            // Take the maximum, because we square the negative cost
            .max_by_key(|(j, cost)| {
                debug_assert!(self.gamma[*j].is_some());

                // gamma is the squared norm, so to compare, we need also to square the cost
                cost * cost / self.gamma[*j].as_ref().unwrap()
            })
    }

    fn after_basis_update<IM, K>(
        &mut self,
        info: BasisChangeComputationInfo<IM::F>,
        tableau: &Tableau<IM, K>,
    ) where
        IM: InverseMaintainer<F = F>,
        K: Kind,
        F: Absorb<<K::Column as Column>::F>,
        F: Absorb<K::Cost>,
    {
        // Update entering column gamma
        self.gamma[info.pivot_column_index] = None;

        // Update leaving column gamma
        let gamma_q = IM::F::one() + info.column_before_change.squared_norm();

        // Update all other gammas, leaving column gamma is still None and will be skipped
        for (j, maybe_gamma) in self
            .gamma
            .iter_mut()
            .enumerate()
            .skip(tableau.start_index())
        {
            if let Some(gamma) = maybe_gamma {
                let original_column = tableau.original_column(j);

                let alpha_j_bar: IM::F = info
                    .basis_inverse_row
                    .sparse_inner_product(original_column.iter());

                let alternative = if alpha_j_bar.is_not_zero() {
                    let alpha_j_bar_squared = &alpha_j_bar * &alpha_j_bar;

                    let inner_product: F = info
                        .work_vector
                        .sparse_inner_product(original_column.iter());
                    if inner_product.is_not_zero() {
                        let first = alpha_j_bar * inner_product;
                        for _ in 0..2 {
                            *gamma -= &first;
                        }
                    }

                    let second = &alpha_j_bar_squared * &gamma_q;
                    *gamma += second;

                    IM::F::one() + alpha_j_bar_squared
                } else {
                    IM::F::one()
                };

                if *gamma < alternative {
                    *gamma = alternative;
                }

                debug_assert_eq!(gamma, &initial_gamma(j, tableau));
            }
        }

        let w_p = info.column_before_change.get(info.pivot_row_index).unwrap();
        self.gamma[info.leaving_column_index] = Some(gamma_q / (w_p * w_p));
    }
}

fn initial_gamma<IM, K>(j: usize, tableau: &Tableau<IM, K>) -> IM::F
where
    IM: InverseMaintainer<F: im_ops::FieldHR>,
    K: Kind,
    IM::F: Absorb<<K::Column as Column>::F>,
    IM::F: Absorb<K::Cost>,
{
    IM::F::one() + tableau.generate_column(j).into_column().squared_norm()
}

#[cfg(test)]
mod test {
    use crate::algorithm::two_phase::strategy::pivot_rule::{FirstProfitable, PivotRule};
    use crate::data::linear_algebra::vector::SparseVector;
    use crate::data::linear_algebra::vector::test::TestVector;
    use crate::tests::problem_2;

    #[test]
    fn find_profitable_column() {
        let (constraints, b, variables) = problem_2::create_matrix_data_data();
        let matrix_data = problem_2::matrix_data_form(&constraints, &b, &variables);
        let artificial_tableau = problem_2::artificial_tableau_form(&matrix_data);
        let mut rule = <FirstProfitable as PivotRule<_>>::new(&artificial_tableau);
        assert!(matches!(
            rule.select_primal_pivot_column(&artificial_tableau),
            Some((3, _))
        ));

        let tableau = problem_2::tableau_form(&matrix_data);
        let mut rule = <FirstProfitable as PivotRule<_>>::new(&tableau);
        assert_eq!(rule.select_primal_pivot_column(&tableau), None);
    }

    #[test]
    fn find_pivot_row() {
        let (constraints, b, variables) = problem_2::create_matrix_data_data();
        let matrix_data = problem_2::matrix_data_form(&constraints, &b, &variables);
        let artificial_tableau = problem_2::artificial_tableau_form(&matrix_data);
        let column = SparseVector::from_test_data(vec![3, 5, 2]);
        assert_eq!(artificial_tableau.select_primal_pivot_row(&column), Some(0));

        let column = SparseVector::from_test_data(vec![2, 1, 5]);
        assert_eq!(artificial_tableau.select_primal_pivot_row(&column), Some(0));

        let tableau = problem_2::tableau_form(&matrix_data);
        let column = SparseVector::from_test_data(vec![3, 2, -1]);
        assert_eq!(tableau.select_primal_pivot_row(&column), Some(0));

        let column = SparseVector::from_test_data(vec![2, -1, 3]);
        assert_eq!(tableau.select_primal_pivot_row(&column), Some(0));
    }
}
