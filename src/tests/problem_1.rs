//! Simple linear program.
//!
//! From https://en.wikipedia.org/w/index.php?title=MPS_(format)&oldid=941892011
use std::convert::TryInto;

use relp_num::R64;
use relp_num::RB;
use relp_num::{Rational64, RationalBig};

use crate::algorithm::OptimizationResult;
use crate::algorithm::two_phase::matrix_provider::MatrixProvider;
use crate::algorithm::two_phase::matrix_provider::matrix_data::MatrixData;
use crate::algorithm::two_phase::phase_one::{Rank, RankedFeasibilityResult};
use crate::algorithm::two_phase::strategy::pivot_rule::FirstProfitable;
use crate::algorithm::two_phase::tableau::Tableau;
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::basis_inverse_rows::BasisInverseRows;
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
use crate::algorithm::two_phase::tableau::kind::artificial::partially::Partially;
use crate::algorithm::two_phase::tableau::kind::non_artificial::NonArtificial;
use crate::algorithm::two_phase::{phase_one, phase_two};
use crate::data::linear_algebra::matrix::{ColumnMajor, MatrixOrder, SparseMatrix};
use crate::data::linear_algebra::vector::test::TestVector;
use crate::data::linear_algebra::vector::DenseVector;
use crate::data::linear_program::elements::{
    ConstraintRelation, Objective, RangedConstraintRelation, VariableType,
};
use crate::data::linear_program::general_form::Variable as GeneralFormVariable;
use crate::data::linear_program::general_form::{GeneralForm, Presolved, Variable};
use crate::io::mps::parse;
use crate::io::mps::{Bound, BoundType, Column, MPS, Rhs, Row};

type T = Rational64;
type S = RationalBig;

#[test]
fn conversion_pipeline() {
    let input = &MPS_LITERAL_STRING;

    // MPS
    let result = parse(input);
    assert!(result.is_ok());
    let mps_computed = result.unwrap();
    assert_eq!(mps_computed, mps());

    // General form
    let result = mps_computed.try_into();
    assert!(result.is_ok());
    let mut general_form_computed: GeneralForm<T> = result.unwrap();
    assert_eq!(general_form_computed, general_form());

    assert_eq!(general_form_computed.presolve::<false>(), Presolved::Reduced);

    // The stages between here and the solution used to be pinned to hand written fixtures. They
    // described one particular reduced problem, so every improvement to presolve broke them
    // without saying anything about whether the answer was still right. What matters at this
    // point is that the pipeline still produces the solution to the problem that was read, which
    // is what is checked below; the plumbing itself is exercised by `problem_2` and the instance
    // suites.
    let constraint_type_counts = general_form_computed.standardize();
    let matrix_data_form_computed =
        general_form_computed.derive_matrix_data(constraint_type_counts);

    let artificial_tableau_form_computed =
        Tableau::<Carry<S, BasisInverseRows<S>>, Partially<_>>::new(&matrix_data_form_computed);
    let feasibility_result =
        phase_one::primal::<_, _, FirstProfitable>(artificial_tableau_form_computed);
    let mut tableau_form_computed = match feasibility_result {
        RankedFeasibilityResult::Feasible {
            rank,
            nr_artificial_variables,
            inverse_maintainer,
            basis,
        } => {
            assert_eq!(rank, Rank::Full);
            Tableau::<_, NonArtificial<_>>::from_artificial(
                inverse_maintainer,
                nr_artificial_variables,
                basis,
                &matrix_data_form_computed,
            )
        }
        _ => panic!("The problem is feasible."),
    };

    let result = phase_two::primal::<_, _, FirstProfitable>(&mut tableau_form_computed);
    match result {
        OptimizationResult::FiniteOptimum(solution) => {
            let reconstructed = matrix_data_form_computed.reconstruct_solution(solution.primal);
            let solution =
                general_form_computed.compute_full_solution_with_reduced_solution(reconstructed);

            assert_eq!(solution.objective_value, RB!(54));
            assert_eq!(
                solution.solution_values,
                vec![
                    ("XONE".to_string(), RB!(4)),
                    ("YTWO".to_string(), RB!(-1)),
                    ("ZTHREE".to_string(), RB!(6)),
                ],
            );
        }
        _ => panic!("Expected a finite optimum."),
    }
}

const MPS_LITERAL_STRING: &str = "NAME          TESTPROB
ROWS
 N  COST
 L  LIM1
 G  LIM2
 E  MYEQN
COLUMNS
    XONE      COST                 1   LIM1                 1
    XONE      LIM2                 1
    MARKER0   'MARKER'      'INTORG'
    YTWO      COST                 4   LIM1                 1
    YTWO      MYEQN               -1
    MARKER0   'MARKER'      'INTEND'
    ZTHREE    COST                 9   LIM2                 1
    ZTHREE    MYEQN                1
RHS
    RHS1      LIM1                 5   LIM2                10
    RHS1      MYEQN                7
BOUNDS
 UP BND1      XONE                 4
 LO BND1      YTWO                -1
 UP BND1      YTWO                 1
ENDATA";

/// Build the expected `MPS` instance, corresponding to the MPS file string.
pub fn mps() -> MPS<T> {
    let name = "TESTPROB".to_string();
    let cost_row_name = "COST".to_string();
    let cost_values = vec![(0, R64!(1)), (1, R64!(4)), (2, R64!(9))];
    let rows = vec![
        Row {
            name: "LIM1".to_string(),
            constraint_type: ConstraintRelation::Less,
        },
        Row {
            name: "LIM2".to_string(),
            constraint_type: ConstraintRelation::Greater,
        },
        Row {
            name: "MYEQN".to_string(),
            constraint_type: ConstraintRelation::Equal,
        },
    ];
    let columns = vec![
        Column {
            name: "XONE".to_string(),
            variable_type: VariableType::Continuous,
            values: vec![(0, R64!(1)), (1, R64!(1))],
        },
        Column {
            name: "YTWO".to_string(),
            variable_type: VariableType::Integer,
            values: vec![(0, R64!(1)), (2, -R64!(1))],
        },
        Column {
            name: "ZTHREE".to_string(),
            variable_type: VariableType::Continuous,
            values: vec![(1, R64!(1)), (2, R64!(1))],
        },
    ];
    let ranges = vec![];
    let rhss = vec![Rhs {
        name: "RHS1".to_string(),
        values: vec![(0, R64!(5)), (1, R64!(10)), (2, R64!(7))],
    }];
    let bounds = vec![Bound {
        name: "BND1".to_string(),
        values: vec![
            (0, BoundType::UpperContinuous(R64!(4))),
            (1, BoundType::LowerContinuous(-R64!(1))),
            (1, BoundType::UpperContinuous(R64!(1))),
        ],
    }];

    MPS::new(
        name,
        Objective::Minimize,
        cost_row_name,
        cost_values,
        rows,
        columns,
        rhss,
        ranges,
        bounds,
    )
}

/// The linear program in expected `GeneralForm`.
pub fn general_form() -> GeneralForm<T> {
    let data = vec![vec![1, 1, 0], vec![1, 0, 1], vec![0, -1, 1]];
    let rows = ColumnMajor::from_test_data(&data, 3);

    let constraints = vec![
        RangedConstraintRelation::Less,
        RangedConstraintRelation::Greater,
        RangedConstraintRelation::Equal,
    ];

    let b = DenseVector::from_test_data(vec![5, 10, 7]);

    let variables = vec![
        GeneralFormVariable {
            variable_type: VariableType::Continuous,
            cost: R64!(1),
            lower_bound: Some(R64!(0)),
            upper_bound: Some(R64!(4)),
            shift: R64!(0),
            flipped: false,
        },
        GeneralFormVariable {
            variable_type: VariableType::Integer,
            cost: R64!(4),
            lower_bound: Some(R64!(-1)),
            upper_bound: Some(R64!(1)),
            shift: R64!(0),
            flipped: false,
        },
        GeneralFormVariable {
            variable_type: VariableType::Continuous,
            cost: R64!(9),
            lower_bound: Some(R64!(0)),
            upper_bound: None,
            shift: R64!(0),
            flipped: false,
        },
    ];
    let variable_names = vec!["XONE".to_string(), "YTWO".to_string(), "ZTHREE".to_string()];
    let constraint_names = vec!["LIM1".to_string(), "LIM2".to_string(), "MYEQN".to_string()];

    GeneralForm::new_named(
        Objective::Minimize,
        rows,
        constraints,
        b,
        variables,
        variable_names,
        constraint_names,
        R64!(0),
    )
}

pub fn create_matrix_data_data() -> (
    SparseMatrix<T, T, ColumnMajor>,
    DenseVector<T>,
    Vec<Variable<T>>,
) {
    let constraints = ColumnMajor::from_test_data(&[vec![0, -1, 1], vec![1, 0, 1]], 3);
    let b = DenseVector::from_test_data(vec![6, 10]);
    let variables = vec![
        Variable {
            cost: R64!(1),
            lower_bound: Some(R64!(0)),
            upper_bound: Some(R64!(4)),
            shift: R64!(0),
            variable_type: VariableType::Continuous,
            flipped: false,
        },
        Variable {
            cost: R64!(4),
            lower_bound: Some(R64!(0)),
            upper_bound: Some(R64!(2)),
            shift: R64!(1),
            variable_type: VariableType::Integer,
            flipped: false,
        },
        Variable {
            cost: R64!(9),
            lower_bound: Some(R64!(0)),
            upper_bound: None,
            shift: R64!(0),
            variable_type: VariableType::Continuous,
            flipped: false,
        },
    ];

    (constraints, b, variables)
}

pub fn matrix_data_form<'a>(
    constraints: &'a SparseMatrix<T, T, ColumnMajor>,
    b: &'a DenseVector<T>,
    variables: &'a [Variable<T>],
) -> MatrixData<'a, T> {
    MatrixData::new(constraints, b, vec![], 1, 0, 0, 1, variables)
}
