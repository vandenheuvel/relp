use relp_num::R32;
use relp_num::Rational32;

use crate::data::linear_program::general_form::presolve::integrality::Roundable;

type T = Rational32;

#[test]
fn rounds_positive_values() {
    assert_eq!(Roundable::floor(&R32!(7, 2)), R32!(3));
    assert_eq!(Roundable::ceil(&R32!(7, 2)), R32!(4));
    assert_eq!(Roundable::floor(&R32!(5)), R32!(5));
    assert_eq!(Roundable::ceil(&R32!(5)), R32!(5));
    assert_eq!(Roundable::floor(&R32!(1, 3)), R32!(0));
    assert_eq!(Roundable::ceil(&R32!(1, 3)), R32!(1));
}

#[test]
fn rounds_negative_values() {
    assert_eq!(Roundable::floor(&R32!(-7, 2)), R32!(-4));
    assert_eq!(Roundable::ceil(&R32!(-7, 2)), R32!(-3));
    assert_eq!(Roundable::floor(&R32!(-5)), R32!(-5));
    assert_eq!(Roundable::ceil(&R32!(-5)), R32!(-5));
    assert_eq!(Roundable::floor(&R32!(-1, 3)), R32!(-1));
    assert_eq!(Roundable::ceil(&R32!(-1, 3)), R32!(0));
}

#[test]
fn rounds_zero() {
    assert_eq!(Roundable::floor(&R32!(0)), R32!(0));
    assert_eq!(Roundable::ceil(&R32!(0)), R32!(0));
}

#[test]
fn recognises_integers() {
    assert!(Roundable::is_integer(&R32!(0)));
    assert!(Roundable::is_integer(&R32!(-12)));
    assert!(!Roundable::is_integer(&R32!(1, 2)));
    assert!(!Roundable::is_integer(&R32!(-1, 2)));
}

/// The generic implementation walks in steps, so a value far from zero must still be cheap and
/// correct.
#[test]
fn rounds_values_far_from_zero() {
    let value: T = R32!(1_000_001, 2);
    assert_eq!(Roundable::floor(&value), R32!(500_000));
    assert_eq!(Roundable::ceil(&value), R32!(500_001));
    assert_eq!(Roundable::floor(&-value), R32!(-500_001));
}
