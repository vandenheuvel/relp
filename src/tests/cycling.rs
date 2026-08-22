//! # Cycling
//!
//! Beale's 1955 example, the classic problem on which the Simplex method cycles when it selects
//! the entering column by attractiveness and breaks ties in the ratio test by row index.
//!
//! The problem is scaled by a factor `100` with respect to the original, such that all data is
//! integer. Scaling rows and the objective function doesn't change which pivots are made.
use num_traits::Zero;
use relp_num::{R64, RB, Rational64, RationalBig};

use crate::algorithm::OptimizationResult;
use crate::algorithm::two_phase::matrix_provider::matrix_data::MatrixData;
use crate::algorithm::two_phase::phase_one::PartialInitialBasis;
use crate::algorithm::two_phase::phase_two;
use crate::algorithm::two_phase::strategy::pivot_rule::{
    BlandFallback, FirstProfitable, PivotRule, SteepestDescentAlongObjective,
};
use crate::algorithm::two_phase::tableau::Tableau;
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::basis_inverse_rows::BasisInverseRows;
use crate::algorithm::two_phase::tableau::inverse_maintenance::{
    ColumnComputationInfo, InverseMaintainer,
};
use crate::algorithm::two_phase::tableau::kind::Kind;
use crate::algorithm::two_phase::tableau::kind::non_artificial::NonArtificial;
use crate::data::linear_algebra::SparseTuple;
use crate::data::linear_algebra::matrix::{ColumnMajor, MatrixOrder, SparseMatrix};
use crate::data::linear_algebra::vector::test::TestVector;
use crate::data::linear_algebra::vector::{DenseVector, SparseVector, Vector};
use crate::data::linear_program::elements::VariableType;
use crate::data::linear_program::general_form::Variable;

type T = Rational64;
type S = RationalBig;
type IM = Carry<S, BasisInverseRows<S>>;

fn create_matrix_data_data() -> (
    SparseMatrix<T, T, ColumnMajor>,
    DenseVector<T>,
    Vec<Variable<T>>,
) {
    let constraints = ColumnMajor::from_test_data(
        &[
            vec![25, -6000, -4, 900],
            vec![50, -9000, -2, 300],
            vec![0, 0, 1, 0],
        ],
        4,
    );

    let b = DenseVector::from_test_data(vec![0, 0, 1]);

    let variables = [-75, 15_000, -2, 600]
        .into_iter()
        .map(|cost| Variable {
            cost: R64!(cost),
            lower_bound: Some(R64!(0)),
            upper_bound: None,
            shift: R64!(0),
            variable_type: VariableType::Continuous,
            flipped: false,
        })
        .collect();

    (constraints, b, variables)
}

fn matrix_data_form<'a>(
    constraints: &'a SparseMatrix<T, T, ColumnMajor>,
    b: &'a DenseVector<T>,
    variables: &'a [Variable<T>],
) -> MatrixData<'a, T> {
    MatrixData::new(constraints, b, Vec::with_capacity(0), 0, 0, 3, 0, variables)
}

/// Tableau with the degenerate basis of all three slack variables, two of which have value zero.
fn tableau_form<'a>(
    data: &'a MatrixData<'a, T>,
) -> Tableau<IM, NonArtificial<'a, MatrixData<'a, T>>> {
    let pivots = data.pivot_element_indices();
    let inverse_maintainer = IM::from_basis_pivots(&pivots, data);
    let basis = pivots.into_iter().map(|(_row, column)| column).collect();

    Tableau::new_with_inverse_maintainer(data, inverse_maintainer, basis)
}

/// The unique optimum: `x_1 = 1 / 25`, `x_3 = 1` and a slack of `3` in the first row.
fn optimum() -> OptimizationResult<S> {
    OptimizationResult::FiniteOptimum(SparseVector::new(
        vec![(0, RB!(1, 25)), (2, RB!(1)), (4, RB!(3))],
        7,
    ))
}

/// Bland's rule: the entering half from `FirstProfitable`, the leaving half from
/// `select_primal_pivot_row`.
#[test]
fn bland() {
    let (constraints, b, variables) = create_matrix_data_data();
    let data = matrix_data_form(&constraints, &b, &variables);
    let mut tableau = tableau_form(&data);

    assert_eq!(
        phase_two::primal::<_, _, FirstProfitable>(&mut tableau),
        optimum(),
    );
}

/// Wrapping a rule shouldn't change the outcome, only the pivots made during a degenerate run.
#[test]
fn fallback_wrapping_steepest_descent() {
    let (constraints, b, variables) = create_matrix_data_data();
    let data = matrix_data_form(&constraints, &b, &variables);
    let mut tableau = tableau_form(&data);

    assert_eq!(
        phase_two::primal::<_, _, BlandFallback<SteepestDescentAlongObjective<_>>>(&mut tableau),
        optimum(),
    );
}

/// A rule that panics when it is asked to select a column.
///
/// It is used to observe which of the two rules the fallback selects with.
struct Unreachable;
impl PivotRule<S> for Unreachable {
    fn new<IM2, K>(_tableau: &Tableau<IM2, K>) -> Self {
        Self
    }

    fn select_primal_pivot_column<IM2, K>(
        &mut self,
        _tableau: &Tableau<IM2, K>,
    ) -> Option<SparseTuple<S>>
    where
        IM2: InverseMaintainer<F = S>,
        K: Kind,
    {
        panic!("the wrapped rule was consulted");
    }
}

/// Without a budget for degenerate pivots, all columns are selected with Bland's rule.
#[test]
fn fallback_takes_over() {
    let (constraints, b, variables) = create_matrix_data_data();
    let data = matrix_data_form(&constraints, &b, &variables);
    let mut tableau = tableau_form(&data);

    assert_eq!(
        phase_two::primal::<_, _, BlandFallback<Unreachable, 0>>(&mut tableau),
        optimum(),
    );
}

/// As long as there is budget left, columns are selected with the wrapped rule.
#[test]
#[should_panic(expected = "the wrapped rule was consulted")]
fn wrapped_rule_is_used_first() {
    let (constraints, b, variables) = create_matrix_data_data();
    let data = matrix_data_form(&constraints, &b, &variables);
    let mut tableau = tableau_form(&data);

    phase_two::primal::<_, _, BlandFallback<Unreachable, 1>>(&mut tableau);
}

/// A pivot is degenerate exactly when the leaving variable is at zero before the basis change.
#[test]
fn degenerate_pivots_are_counted() {
    let (constraints, b, variables) = create_matrix_data_data();
    let data = matrix_data_form(&constraints, &b, &variables);
    let mut tableau = tableau_form(&data);

    let mut rule = <BlandFallback<FirstProfitable> as PivotRule<S>>::new(&tableau);
    let mut nr_degenerate_pivots = 0;
    let mut nr_degenerate_pivots_total = 0;
    let mut nr_pivots = 0;

    while let Some((column, cost)) = rule.select_primal_pivot_column(&tableau) {
        let column_with_info = tableau.generate_column(column);
        let row = tableau
            .select_primal_pivot_row(column_with_info.column())
            .unwrap();

        // The step length is the value of the leaving variable divided by the pivot element.
        if tableau.constraint_value(row).is_zero() {
            nr_degenerate_pivots += 1;
            nr_degenerate_pivots_total += 1;
        } else {
            nr_degenerate_pivots = 0;
        }

        let info = tableau.bring_into_basis(column, row, column_with_info, cost);
        rule.after_basis_update(info, &tableau);
        nr_pivots += 1;

        assert_eq!(rule.nr_degenerate_pivots(), nr_degenerate_pivots);
    }

    // Both branches are exercised: this problem starts with a degenerate pivot and ends with a
    // pivot that improves the objective value.
    assert!(nr_degenerate_pivots_total > 0);
    assert!(nr_pivots > nr_degenerate_pivots_total);
    assert_eq!(nr_degenerate_pivots, 0);
}
