//! Simple linear program.
//!
//! From https://en.wikipedia.org/w/index.php?title=MPS_(format)&oldid=941892011
use std::collections::HashSet;
use std::convert::{TryFrom, TryInto};

use num::FromPrimitive;
use num::rational::Ratio;

use crate::{R32, RF};
use crate::algorithm::simplex::data::{Artificial, CarryMatrix, NonArtificial, Tableau};
use crate::algorithm::simplex::logic::{artificial_primal, FeasibilityResult, OptimizationResult, primal, Rank};
use crate::algorithm::simplex::matrix_provider::matrix_data::{MatrixData, Variable as MatrixDataVariable};
use crate::algorithm::simplex::matrix_provider::MatrixProvider;
use crate::algorithm::simplex::strategy::pivot_rule::FirstProfitable;
use crate::data::linear_algebra::matrix::{MatrixOrder, RowMajorOrdering};
use crate::data::linear_algebra::vector::{DenseVector, SparseVector};
use crate::data::linear_algebra::vector::test::TestVector;
use crate::data::linear_program::elements::{ConstraintType, Objective, VariableType};
use crate::data::linear_program::general_form::GeneralForm;
use crate::data::linear_program::general_form::Variable as GeneralFormVariable;
use crate::data::number_types::traits::RealField;
use crate::io::mps::{Bound, BoundType, Constraint, Rhs, Variable};
use crate::io::mps::parsing::{into_atom_lines, UnstructuredBound, UnstructuredColumn, UnstructuredMPS, UnstructuredRhs, UnstructuredRow};
use crate::io::mps::structuring::MPS;

#[test]
fn test_conversion_pipeline() {
    let input = &MPS_LITERAL_STRING;

    // Unstructured MPS
    let lines = into_atom_lines(input);
    let result = UnstructuredMPS::try_from(lines);
    assert!(result.is_ok());
    let unstructured_mps_computed: UnstructuredMPS = result.unwrap();
    assert_abs_diff_eq!(unstructured_mps_computed, unstructured_mps_form());

    // MPS
    let result = unstructured_mps_computed.try_into();
    assert!(result.is_ok());
    let structured_mps_computed: MPS = result.unwrap();
    assert_abs_diff_eq!(structured_mps_computed, mps_form());

    // General form
    let result = structured_mps_computed.try_into();
    assert!(result.is_ok());
    let mut general_form_computed: GeneralForm<_> = result.unwrap();
    assert_eq!(general_form_computed, general_form());

    // Matrix data form
    let result = general_form_computed.derive_matrix_data();
    // General form, canonicalized
    assert_eq!(general_form_computed, general_form_canonicalized());
    // The resulting matrix data form
    assert!(result.is_ok());
    let matrix_data_form_computed = result.unwrap();
    assert_eq!(matrix_data_form_computed, matrix_data_form());

    // Artificial tableau form
    let mut artificial_tableau_form_computed = Tableau::<_, Artificial, _>::new(&matrix_data_form_computed);
    assert_eq!(artificial_tableau_form_computed, artificial_tableau_form(&matrix_data_form()));

    // Get to a basic feasible solution
    let feasibility_result = artificial_primal::<_, _, FirstProfitable>(&mut artificial_tableau_form_computed);
    assert_eq!(feasibility_result, FeasibilityResult::Feasible(Rank::Full));

    // Non-artificial tableau form
    let mut tableau_form_computed = Tableau::<_, NonArtificial, _>::from_artificial(artificial_tableau_form_computed);
    assert_eq!(tableau_form_computed, tableau_form(&matrix_data_form()));

    // Get to a basic feasible solution
    let result = primal::<_, _, FirstProfitable>(&mut tableau_form_computed);
    assert_eq!(result, OptimizationResult::FiniteOptimum(R32!(58)));
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
pub(super) fn unstructured_mps_form() -> UnstructuredMPS<'static> {
    UnstructuredMPS {
        name: "TESTPROB",
        cost_row_name: "COST",
        rows: vec![
            UnstructuredRow {
                name: "LIM1",
                constraint_type: ConstraintType::Less,
            },
            UnstructuredRow {
                name: "LIM2",
                constraint_type: ConstraintType::Greater,
            },
            UnstructuredRow {
                name: "MYEQN",
                constraint_type: ConstraintType::Equal,
            },
        ],
        columns: vec![
            UnstructuredColumn {
                name: "XONE",
                variable_type: VariableType::Continuous,
                row_name: "COST",
                value: 1f64,
            },
            UnstructuredColumn {
                name: "XONE",
                variable_type: VariableType::Continuous,
                row_name: "LIM1",
                value: 1f64,
            },
            UnstructuredColumn {
                name: "XONE",
                variable_type: VariableType::Continuous,
                row_name: "LIM2",
                value: 1f64,
            },
            UnstructuredColumn {
                name: "YTWO",
                variable_type: VariableType::Integer,
                row_name: "COST",
                value: 4f64,
            },
            UnstructuredColumn {
                name: "YTWO",
                variable_type: VariableType::Integer,
                row_name: "LIM1",
                value: 1f64,
            },
            UnstructuredColumn {
                name: "YTWO",
                variable_type: VariableType::Integer,
                row_name: "MYEQN",
                value: -1f64,
            },
            UnstructuredColumn {
                name: "ZTHREE",
                variable_type: VariableType::Continuous,
                row_name: "COST",
                value: 9f64,
            },
            UnstructuredColumn {
                name: "ZTHREE",
                variable_type: VariableType::Continuous,
                row_name: "LIM2",
                value: 1f64,
            },
            UnstructuredColumn {
                name: "ZTHREE",
                variable_type: VariableType::Continuous,
                row_name: "MYEQN",
                value: 1f64,
            },
        ],
        rhss: vec![
            UnstructuredRhs {
                name: "RHS1",
                row_name: "LIM1",
                value: 5f64,
            },
            UnstructuredRhs {
                name: "RHS1",
                row_name: "LIM2",
                value: 10f64,
            },
            UnstructuredRhs {
                name: "RHS1",
                row_name: "MYEQN",
                value: 7f64,
            },
        ],
        bounds: vec![
            UnstructuredBound {
                name: "BND1",
                bound_type: BoundType::UpperContinuous,
                column_name: "XONE",
                value: 4f64,
            },
            UnstructuredBound {
                name: "BND1",
                bound_type: BoundType::LowerContinuous,
                column_name: "YTWO",
                value: -1f64,
            },
            UnstructuredBound {
                name: "BND1",
                bound_type: BoundType::UpperContinuous,
                column_name: "YTWO",
                value: 1f64,
            },
        ],
    }
}

/// Build the expected `MPS` instance, corresponding to the MPS file string.
pub fn mps_form() -> MPS {
    let name = "TESTPROB".to_string();
    let cost_row_name = "COST".to_string();
    let cost_values = vec![(0, 1f64), (1, 4f64), (2, 9f64)];
    let row_names = vec!["LIM1", "LIM2", "MYEQN"].into_iter().map(String::from).collect();
    let rows = vec![
        Constraint { name: 0, constraint_type: ConstraintType::Less },
        Constraint { name: 1, constraint_type: ConstraintType::Greater },
        Constraint { name: 2, constraint_type: ConstraintType::Equal },
    ];
    let column_names = vec!["XONE", "YTWO", "ZTHREE"].into_iter().map(String::from).collect();
    let columns = vec![
        Variable {
            name: 0,
            variable_type: VariableType::Continuous,
            values: vec![
                (0, 1f64),
                (1, 1f64),
            ],
        },
        Variable {
            name: 1,
            variable_type: VariableType::Integer,
            values: vec![
                (0, 1f64),
                (2, -1f64),
            ],
        },
        Variable {
            name: 2,
            variable_type: VariableType::Continuous,
            values: vec![
                (1, 1f64),
                (2, 1f64),
            ],
        },
    ];
    let rhss = vec![
        Rhs {
            name: "RHS1".to_string(),
            values: vec![(0, 5f64), (1, 10f64), (2, 7f64)],
        }
    ];
    let bounds = vec![
        Bound {
            name: "BND1".to_string(),
            values: vec![
                (BoundType::UpperContinuous, 0, 4f64),
                (BoundType::LowerContinuous, 1, -1f64),
                (BoundType::UpperContinuous, 1, 1f64),
            ],
        }
    ];

    MPS::new(
        name,
        cost_row_name,
        cost_values,
        row_names,
        rows,
        column_names,
        columns,
        rhss,
        bounds,
    )
}

/// The linear program in expected `GeneralForm`.
pub fn general_form<RF: RealField>() -> GeneralForm<RF> {
    let data = vec![
        vec![1f64, 1f64, 0f64],
        vec![1f64, 0f64, 1f64],
        vec![0f64, -1f64, 1f64],
    ];
    let rows = RowMajorOrdering::from_test_data(&data);

    let constraints = vec![
        ConstraintType::Less,
        ConstraintType::Greater,
        ConstraintType::Equal,
    ];

    let b = DenseVector::from_test_data(vec![
        5f64,
        10f64,
        7f64,
    ]);

    let variables = vec![
        GeneralFormVariable {
            name: "XONE".to_string(),
            variable_type: VariableType::Continuous,
            cost: RF!(1),
            lower_bound: Some(RF!(0)),
            upper_bound: Some(RF!(4)),
            shift: RF!(0),
            flipped: false
        },
        GeneralFormVariable {
            name: "YTWO".to_string(),
            variable_type: VariableType::Integer,
            cost: RF!(4),
            lower_bound: Some(RF!(-1)),
            upper_bound: Some(RF!(1)),
            shift: RF!(0),
            flipped: false
        },
        GeneralFormVariable {
            name: "ZTHREE".to_string(),
            variable_type: VariableType::Continuous,
            cost: RF!(9),
            lower_bound: Some(RF!(0)),
            upper_bound: None,
            shift: RF!(0),
            flipped: false
        },
    ];

    GeneralForm::new(
        Objective::Minimize,
        rows,
        constraints,
        b,
        variables,
        RF!(0),
    )
}

pub fn general_form_canonicalized<RF: RealField>() -> GeneralForm<RF> {
    let data = vec![
        vec![1f64, 1f64, 0f64],
        vec![1f64, 0f64, 1f64],
        vec![0f64, -1f64, 1f64],
    ];
    let constraints = RowMajorOrdering::from_test_data(&data);

    let constraint_types = vec![
        ConstraintType::Less,
        ConstraintType::Greater,
        ConstraintType::Equal,
    ];

    let b = DenseVector::from_test_data(vec![
        6f64,
        10f64,
        6f64,
    ]);

    let variables = vec![
        GeneralFormVariable {
            name: "XONE".to_string(),
            variable_type: VariableType::Continuous,
            cost: RF!(1),
            lower_bound: Some(RF!(0)),
            upper_bound: Some(RF!(4)),
            shift: RF!(0),
            flipped: false
        },
        GeneralFormVariable {
            name: "YTWO".to_string(),
            variable_type: VariableType::Integer,
            cost: RF!(4),
            lower_bound: Some(RF!(0)),
            upper_bound: Some(RF!(2)),
            shift: RF!(1),
            flipped: false
        },
        GeneralFormVariable {
            name: "ZTHREE".to_string(),
            variable_type: VariableType::Continuous,
            cost: RF!(9),
            lower_bound: Some(RF!(0)),
            upper_bound: None,
            shift: RF!(0),
            flipped: false
        },
    ];

    GeneralForm::new(
        Objective::Minimize,
        constraints,
        constraint_types,
        b,
        variables,
        RF!(4),
    )
}

pub fn matrix_data_form<RF: RealField>() -> MatrixData<RF> {
    let equal = vec![vec![(1, RF!(-1)), (2, RF!(1))]];
    let upper_bounded = vec![vec![(0, RF!(1)), (1, RF!(1))]];
    let lower_bounded = vec![vec![(0, RF!(1)), (2, RF!(1))]];
    let b = DenseVector::from_test_data(vec![
        6f64,
        6f64,
        10f64,
    ]);
    let variables = vec![
        MatrixDataVariable {
            cost: RF!(1),
            upper_bound: Some(RF!(4)),
            variable_type: VariableType::Continuous,
        },
        MatrixDataVariable {
            cost: RF!(4),
            upper_bound: Some(RF!(2)),
            variable_type: VariableType::Integer,
        },
        MatrixDataVariable {
            cost: RF!(9),
            upper_bound: None,
            variable_type: VariableType::Continuous,
        },
    ];
    MatrixData::new(
        equal,
        upper_bounded,
        lower_bounded,
        b,
        variables,
        Vec::with_capacity(0),
    )
}

pub fn artificial_tableau_form<RF: RealField, MP: MatrixProvider<RF>>(provider: &MP) -> Tableau<RF, Artificial, MP> {
    let m = 5;
    let carry = {
        let minus_objective = RF!(-28);
        let minus_pi = DenseVector::from_test_data(vec![-1f64; m]);
        let b = DenseVector::from_test_data(vec![6f64, 6f64, 10f64, 4f64, 2f64]);
        let basis_inverse_rows = (0..m)
            .map(|i| SparseVector::standard_basis_vector(i, m))
            .collect();
        CarryMatrix::create(minus_objective, minus_pi, b, basis_inverse_rows)
    };
    let basis_indices = (0..m).collect();
    let basis_columns = (0..m).collect();

    Tableau::new_with_basis(
        &provider,
        carry,
        basis_indices,
        basis_columns,
    )
}

pub fn tableau_form<RF: RealField, MP: MatrixProvider<RF>>(provider: &MP) -> Tableau<RF, NonArtificial, MP> {
    let carry = {
        let minus_objective = RF!(-84);
        let minus_pi = DenseVector::from_test_data(vec![-9f64, -1f64, 0f64, 0f64, -12f64]);
        let b = DenseVector::from_test_data(vec![2f64, 2f64, 8f64, 4f64, 0f64]);
        let basis_inverse_rows = vec![
            SparseVector::from_test_data(vec![1f64, 1f64, -1f64, 0f64, 0f64]),
            SparseVector::from_test_data(vec![0f64, 0f64, 0f64, 0f64, 1f64]),
            SparseVector::from_test_data(vec![1f64, 0f64, 0f64, 0f64, 1f64]),
            SparseVector::from_test_data(vec![0f64, 1f64, 0f64, 0f64, -1f64]),
            SparseVector::from_test_data(vec![0f64, -1f64, 0f64, 1f64, 1f64]),
        ];
        CarryMatrix::<RF>::create(minus_objective, minus_pi, b, basis_inverse_rows)
    };
    let basis_indices = vec![4, 1, 2, 0, 5];
    let mut basis_columns = HashSet::new();
    basis_columns.insert(4);
    basis_columns.insert(1);
    basis_columns.insert(2);
    basis_columns.insert(0);
    basis_columns.insert(5);

    Tableau::new_with_basis(
        provider,
        carry,
        basis_indices,
        basis_columns,
    )
}
