//! Simple linear program.
//!
//! From https://en.wikipedia.org/w/index.php?title=MPS_(format)&oldid=941892011
use std::convert::TryInto;

use num::FromPrimitive;

use crate::algorithm::OptimizationResult;
use crate::algorithm::two_phase::{phase_one, phase_two};
use crate::algorithm::two_phase::matrix_provider::{Column as ColumnTrait, MatrixProvider};
use crate::algorithm::two_phase::matrix_provider::matrix_data::{MatrixData, Variable as MatrixDataVariable};
use crate::algorithm::two_phase::phase_one::{Rank, RankedFeasibilityResult};
use crate::algorithm::two_phase::strategy::pivot_rule::FirstProfitable;
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::Carry;
use crate::algorithm::two_phase::tableau::kind::artificial::partially::Partially;
use crate::algorithm::two_phase::tableau::kind::non_artificial::NonArtificial;
use crate::algorithm::two_phase::tableau::Tableau;
use crate::data::linear_algebra::matrix::{ColumnMajor, Order, Sparse};
use crate::data::linear_algebra::vector::{Dense, Sparse as SparseVector};
use crate::data::linear_algebra::vector::test::TestVector;
use crate::data::linear_program::elements::{ConstraintType, Objective, VariableType};
use crate::data::linear_program::general_form::GeneralForm;
use crate::data::linear_program::general_form::Variable as GeneralFormVariable;
use crate::data::number_types::rational::Rational64;
use crate::io::mps::{Bound, BoundType, Column, MPS, Rhs, Row};
use crate::io::mps::parse;
use crate::R64;

type T = Rational64;

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
    let mut general_form_computed: GeneralForm<_> = result.unwrap();
    assert_eq!(general_form_computed, general_form());

    assert!(general_form_computed.standardize().is_ok());
    // General form, standardized
    assert_eq!(general_form_computed, general_form_standardized());

    // Matrix data form
    let result = general_form_computed.derive_matrix_data();
    assert!(result.is_ok());

    let (constraints, b) = create_matrix_data_data();
    let matrix_data_form_computed = result.unwrap();
    assert_eq!(matrix_data_form_computed, matrix_data_form(&constraints, &b));

    // Artificial tableau form
    let artificial_tableau_form_computed = Tableau::<_, Partially<_>>::new(&matrix_data_form_computed);
    assert_eq!(artificial_tableau_form_computed, artificial_tableau_form(&matrix_data_form(&constraints, &b)));

    // Get to a basic feasible solution
    let feasibility_result = phase_one::primal::<_, _, MatrixData<T>, FirstProfitable>(artificial_tableau_form_computed);
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
        },
        _ => panic!(),
    };

    // Non-artificial tableau form
    assert_eq!(tableau_form_computed, tableau_form(&matrix_data_form(&constraints, &b)));

    // Get to a basic feasible solution
    let result = phase_two::primal::<_, _, FirstProfitable>(&mut tableau_form_computed);
    assert_eq!(result, OptimizationResult::FiniteOptimum(SparseVector::from_test_tuples(vec![
        (0, 2),
        (5, 2),
        (6, 2),
    ], 7)));
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
        Row { name: "LIM1".to_string(), constraint_type: ConstraintType::Less },
        Row { name: "LIM2".to_string(), constraint_type: ConstraintType::Greater },
        Row { name: "MYEQN".to_string(), constraint_type: ConstraintType::Equal },
    ];
    let columns = vec![
        Column {
            name: "XONE".to_string(),
            variable_type: VariableType::Continuous,
            values: vec![
                (0, R64!(1)),
                (1, R64!(1)),
            ],
        },
        Column {
            name: "YTWO".to_string(),
            variable_type: VariableType::Integer,
            values: vec![
                (0, R64!(1)),
                (2, -R64!(1)),
            ],
        },
        Column {
            name: "ZTHREE".to_string(),
            variable_type: VariableType::Continuous,
            values: vec![
                (1, R64!(1)),
                (2, R64!(1)),
            ],
        },
    ];
    let ranges = vec![];
    let rhss = vec![
        Rhs {
            name: "RHS1".to_string(),
            values: vec![(0, R64!(5)), (1, R64!(10)), (2, R64!(7))],
        }
    ];
    let bounds = vec![
        Bound {
            name: "BND1".to_string(),
            values: vec![
                (0, BoundType::UpperContinuous(R64!(4))),
                (1, BoundType::LowerContinuous(-R64!(1))),
                (1, BoundType::UpperContinuous(R64!(1))),
            ],
        }
    ];

    MPS::new(
        name,
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
    let data = vec![
        vec![1, 1, 0],
        vec![1, 0, 1],
        vec![0, -1, 1],
    ];
    let rows = ColumnMajor::from_test_data(&data, 3);

    let constraints = vec![
        ConstraintType::Less,
        ConstraintType::Greater,
        ConstraintType::Equal,
    ];

    let b = Dense::from_test_data(vec![
        5,
        10,
        7,
    ]);

    let variables = vec![
        GeneralFormVariable {
            variable_type: VariableType::Continuous,
            cost: R64!(1),
            lower_bound: Some(R64!(0)),
            upper_bound: Some(R64!(4)),
            shift: R64!(0),
            flipped: false
        },
        GeneralFormVariable {
            variable_type: VariableType::Integer,
            cost: R64!(4),
            lower_bound: Some(R64!(-1)),
            upper_bound: Some(R64!(1)),
            shift: R64!(0),
            flipped: false
        },
        GeneralFormVariable {
            variable_type: VariableType::Continuous,
            cost: R64!(9),
            lower_bound: Some(R64!(0)),
            upper_bound: None,
            shift: R64!(0),
            flipped: false
        },
    ];
    let variable_names = vec!["XONE".to_string(), "YTWO".to_string(), "ZTHREE".to_string()];

    GeneralForm::new(
        Objective::Minimize,
        rows,
        constraints,
        b,
        variables,
        variable_names,
        R64!(0),
    )
}

pub fn general_form_standardized() -> GeneralForm<T> {
    let data = vec![
        vec![1, 0, 1],
        vec![0, -1, 1],
    ];
    let constraints = ColumnMajor::from_test_data(&data, 3);

    let constraint_types = vec![
        ConstraintType::Greater,
        ConstraintType::Equal,
    ];

    let b = Dense::from_test_data(vec![
        2,
        0,
    ]);

    let variables = vec![
        GeneralFormVariable {
            variable_type: VariableType::Continuous,
            cost: R64!(1),
            lower_bound: Some(R64!(0)),
            upper_bound: Some(R64!(2)),
            shift: R64!(-2),
            flipped: false
        },
        GeneralFormVariable {
            variable_type: VariableType::Integer,
            cost: R64!(4),
            lower_bound: Some(R64!(0)),
            upper_bound: Some(R64!(2)),
            shift: R64!(1),
            flipped: false
        },
        GeneralFormVariable {
            variable_type: VariableType::Continuous,
            cost: R64!(9),
            lower_bound: Some(R64!(0)),
            upper_bound: Some(R64!(2)),
            shift: R64!(-6),
            flipped: false
        },
    ];
    let variable_names = vec!["XONE".to_string(), "YTWO".to_string(), "ZTHREE".to_string()];

    GeneralForm::new(
        Objective::Minimize,
        constraints,
        constraint_types,
        b,
        variables,
        variable_names,
        -R64!(-2 * 1) - R64!(1 * 4) - R64!(-6 * 9),
    )
}

pub fn create_matrix_data_data() -> (Sparse<T, T, ColumnMajor>, Dense<T>) {
    let constraints = ColumnMajor::from_test_data(
        &vec![
            vec![0, -1, 1],
            vec![1, 0, 1],
        ],
        3,
    );
    let b = Dense::from_test_data(vec![
        0,
        2,
    ]);

    (constraints, b)
}

pub fn matrix_data_form<'a>(
    constraints: &'a Sparse<T, T, ColumnMajor>,
    b: &'a Dense<T>,
) -> MatrixData<'a, T> {
    let variables = vec![
        MatrixDataVariable {
            cost: R64!(1),
            upper_bound: Some(R64!(2)),
            variable_type: VariableType::Continuous,
        },
        MatrixDataVariable {
            cost: R64!(4),
            upper_bound: Some(R64!(2)),
            variable_type: VariableType::Integer,
        },
        MatrixDataVariable {
            cost: R64!(9),
            upper_bound: Some(R64!(2)),
            variable_type: VariableType::Continuous,
        },
    ];
    MatrixData::new(
        constraints,
        b,
        1,
        0,
        1,
        variables,
    )
}

pub fn artificial_tableau_form<MP: MatrixProvider<Column: ColumnTrait<F=T>>>(
    provider: &MP,
) -> Tableau<Carry<T>, Partially<MP>> {
    let m = 5;
    let carry = {
        let minus_objective = R64!(-2);
        let minus_pi = Dense::from_test_data(vec![-1, -1, 0, 0, 0]);
        let b = Dense::from_test_data(vec![0, 2, 2, 2, 2]);
        let basis_inverse_rows = (0..m)
            .map(|i| SparseVector::standard_basis_vector(i, m))
            .collect();
        Carry::new(minus_objective, minus_pi, b, basis_inverse_rows)
    };
    let artificials = vec![0, 1];
    let mut basis_indices = artificials.clone();
    basis_indices.extend(vec![2 + 4, 2 + 5, 2 + 6]);
    let basis_columns = basis_indices.iter().copied().collect();

    Tableau::<_, Partially<_>>::new_with_basis(
        provider,
        carry,
        basis_indices,
        basis_columns,
        artificials,
    )
}

pub fn tableau_form<MP: MatrixProvider<Column: ColumnTrait<F=T>>>(
    provider: &MP,
) -> Tableau<Carry<T>, NonArtificial<MP>> {
    let carry = {
        let minus_objective = R64!(-2);
        let minus_pi = Dense::from_test_data(vec![-8, -1, 0, 0, 0]);
        let b = Dense::from_test_data(vec![0, 2, 0, 2, 2]);
        let basis_inverse_rows = vec![
            SparseVector::from_test_data(vec![1, 0, 0, 0, 0]),
            SparseVector::from_test_data(vec![-1, 1, 0, 0, 0]),
            SparseVector::from_test_data(vec![1, -1, 1, 0, 0]),
            SparseVector::from_test_data(vec![0, 0, 0, 1, 0]),
            SparseVector::from_test_data(vec![-1, 0, 0, 0, 1]),
        ];
        Carry::new(minus_objective, minus_pi, b, basis_inverse_rows)
    };
    let basis_indices = vec![2, 0, 4, 5, 6];
    let basis_columns = basis_indices.iter().copied().collect();

    Tableau::<_, NonArtificial<_>>::new_with_inverse_maintainer(
        provider,
        carry,
        basis_indices,
        basis_columns,
    )
}
