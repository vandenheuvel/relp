use relp_num::R32;
use relp_num::Rational32;

use crate::data::linear_algebra::matrix::{ColumnMajor, MatrixOrder};
use crate::data::linear_program::general_form::presolve::matrix::Matrix;

type T = Rational32;

/// A 2 by 3 matrix with one structural zero.
fn matrix() -> Matrix<T> {
    Matrix::from_column_major(ColumnMajor::from_test_data::<T, T, _>(
        &[vec![1, 0, 2], vec![3, 4, 5]],
        3,
    ))
}

#[test]
fn walks_both_directions() {
    let matrix = matrix();

    assert_eq!(matrix.nr_rows(), 2);
    assert_eq!(matrix.nr_columns(), 3);
    assert_eq!(matrix.nonzeros(), 5);

    assert_eq!(
        matrix.row(0).map(|(j, v)| (j, *v)).collect::<Vec<_>>(),
        vec![(0, R32!(1)), (2, R32!(2))],
    );
    assert_eq!(
        matrix.column(1).map(|(i, v)| (i, *v)).collect::<Vec<_>>(),
        vec![(1, R32!(4))],
    );
    assert_eq!(matrix.row_len(0), 2);
    assert_eq!(matrix.column_len(1), 1);
    assert_eq!(matrix.get(1, 2), Some(&R32!(5)));
    assert_eq!(matrix.get(0, 1), None);
}

#[test]
fn taking_a_row_empties_it_in_both_directions() {
    let mut matrix = matrix();

    assert_eq!(matrix.take_row(0), vec![(0, R32!(1)), (2, R32!(2))]);

    assert_eq!(matrix.row_len(0), 0);
    assert_eq!(matrix.column_len(0), 1);
    assert_eq!(matrix.column_len(2), 1);
    assert_eq!(matrix.nonzeros(), 3);
    assert_eq!(matrix.get(0, 0), None);
    assert_eq!(matrix.get(1, 0), Some(&R32!(3)));
}

#[test]
fn taking_a_column_empties_it_in_both_directions() {
    let mut matrix = matrix();

    assert_eq!(matrix.take_column(2), vec![(0, R32!(2)), (1, R32!(5))]);

    assert_eq!(matrix.column_len(2), 0);
    assert_eq!(matrix.row_len(0), 1);
    assert_eq!(matrix.row_len(1), 2);
    assert_eq!(matrix.nonzeros(), 3);
}

#[test]
fn setting_inserts_overwrites_and_removes() {
    let mut matrix = matrix();

    // Insert where there was a structural zero.
    assert_eq!(matrix.set(0, 1, R32!(7)), None);
    assert_eq!(matrix.get(0, 1), Some(&R32!(7)));
    assert_eq!(matrix.nonzeros(), 6);
    assert_eq!(
        matrix.row(0).map(|(j, _)| j).collect::<Vec<_>>(),
        vec![0, 1, 2],
        "the row must stay sorted",
    );
    assert_eq!(
        matrix.column(1).map(|(i, _)| i).collect::<Vec<_>>(),
        vec![0, 1],
        "the column must stay sorted",
    );

    // Overwrite.
    assert_eq!(matrix.set(0, 1, R32!(8)), Some(R32!(7)));
    assert_eq!(matrix.nonzeros(), 6);

    // A zero removes.
    assert_eq!(matrix.set(0, 1, R32!(0)), Some(R32!(8)));
    assert_eq!(matrix.get(0, 1), None);
    assert_eq!(matrix.nonzeros(), 5);
}

#[test]
fn freed_slots_are_reused() {
    let mut matrix = matrix();

    matrix.take(0, 0).unwrap();
    matrix.set(0, 1, R32!(9));

    assert_eq!(matrix.get(0, 1), Some(&R32!(9)));
    assert_eq!(matrix.get(0, 0), None);
    assert_eq!(matrix.get(0, 2), Some(&R32!(2)), "the arena must not have shifted");
    assert_eq!(matrix.nonzeros(), 5);
}
