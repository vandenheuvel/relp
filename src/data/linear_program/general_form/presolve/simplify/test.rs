use relp_num::{EncodingSize, R64, RB, Rational64, RationalBig};

use crate::data::linear_program::general_form::presolve::simplify::Simplify;

/// The values a row would hand over, with the factor that clears them.
fn factor(values: &[RationalBig]) -> Option<RationalBig> {
    RationalBig::simplifying_factor(values.iter())
}

#[test]
fn a_row_of_coprime_integers_needs_nothing() {
    assert_eq!(factor(&[RB!(2), RB!(3)]), None);
    assert_eq!(factor(&[RB!(1)]), None);
    assert_eq!(factor(&[RB!(-1), RB!(5)]), None);
}

#[test]
fn nothing_but_zeros_asks_for_nothing() {
    assert_eq!(factor(&[]), None);
    assert_eq!(factor(&[RB!(0), RB!(0)]), None);
}

#[test]
fn denominators_are_cleared() {
    // 1/2 x + 1/3 y = 1  ->  3x + 2y = 6
    assert_eq!(factor(&[RB!(1, 2), RB!(1, 3), RB!(1)]), Some(RB!(6)));
}

#[test]
fn a_common_factor_is_divided_out() {
    // 4x + 6y = 10  ->  2x + 3y = 5
    assert_eq!(factor(&[RB!(4), RB!(6), RB!(10)]), Some(RB!(1, 2)));
}

#[test]
fn the_factor_is_positive_whatever_the_signs_are() {
    let factor = factor(&[RB!(-4), RB!(-6)]).unwrap();
    assert!(factor > RB!(0));
    assert_eq!(factor, RB!(1, 2));
}

#[test]
fn zeros_among_the_values_are_passed_over() {
    assert_eq!(factor(&[RB!(0), RB!(1, 2), RB!(0)]), Some(RB!(2)));
}

/// Multiplying by the factor has to leave integers with nothing in common.
#[test]
fn what_comes_out_is_coprime_integers() {
    for values in [
        vec![RB!(1, 2), RB!(1, 3), RB!(1, 5)],
        vec![RB!(7, 15), RB!(14, 25)],
        vec![RB!(-3, 4), RB!(9, 8), RB!(0), RB!(15, 16)],
        vec![RB!(1000), RB!(250), RB!(125)],
    ] {
        let factor = RationalBig::simplifying_factor(values.iter()).unwrap();
        let scaled = values.iter().map(|value| value * &factor).collect::<Vec<_>>();

        for value in &scaled {
            assert_eq!(value.denominator_bits(), 1, "{value} is not an integer");
        }
        assert_eq!(
            RationalBig::simplifying_factor(scaled.iter()),
            None,
            "{scaled:?} still has something in common",
        );
    }
}

/// A fixed width type answers too, right up to where it runs out of room.
#[test]
fn a_narrow_type_answers_until_it_cannot() {
    assert_eq!(
        Rational64::simplifying_factor([R64!(1, 2), R64!(1, 3)].iter()),
        Some(R64!(6)),
    );

    // The least common multiple of these denominators does not fit in the type: they are coprime,
    // being odd, not multiples of five, and ten apart, so it is their product.
    let huge = [
        R64!(1, 5_000_000_029),
        R64!(1, 5_000_000_039),
    ];
    assert_eq!(Rational64::simplifying_factor(huge.iter()), None);
}

/// The default implementation declines rather than being wrong.
#[test]
fn the_size_of_a_value_is_reported_where_it_can_be() {
    assert_eq!(Simplify::encoding_bits(&RB!(0)), Some(1));
    assert_eq!(Simplify::encoding_bits(&RB!(1)), Some(2));
    assert_eq!(Simplify::encoding_bits(&RB!(255, 256)), Some(17));
}
