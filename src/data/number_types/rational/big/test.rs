use num::FromPrimitive;

use crate::data::number_types::rational::big::Big;
use crate::data::number_types::rational::Rational64;
use crate::data::number_types::traits::Abs;

#[test]
fn eq() {
    // with Self
    let x = Big::new(1, 1);
    let y = Big::new(2, 2);
    assert_eq!(x, y);

    let x = Big::new(2, 1);
    let y = Big::new(2, 2);
    assert_ne!(x, y);

    let x = Big::new(-1, 1);
    let y = Big::new(2, -2);
    assert_eq!(x, y);

    let x = Big::new(0, 1);
    let y = Big::new(-0, 2);
    assert_eq!(x, y);

    // with Rational64
    let x = Big::new(1, 1);
    let y = Rational64::new(2, 2);
    assert_eq!(x, y);

    let x = Big::new(2, 1);
    let y = Rational64::new(2, 2);
    assert_ne!(x, y);

    let x = Big::new(-1, 1);
    let y = Rational64::new(2, -2);
    assert_eq!(x, y);

    let x = Big::new(0, 1);
    let y = Rational64::new(-0, 2);
    assert_eq!(x, y);
}

#[test]
fn from() {
    let x = Rational64::new(4, 3);
    let y = Big::from(x);
    let z = Big::new(4, 3);
    assert_eq!(y, z);

    let y = Big::from_f64(4f64 / 3f64).unwrap();
    let z = Big::new(4, 3);
    assert!((y - z).abs() < Big::new(1, 2 << 10));
}

#[test]
fn add() {
    // with Self
    let x = Big::new(1, 1);
    let y = Big::new(2, 2);
    assert_eq!(x + y, Big::new(2, 1));

    let x = Big::new(2, 1);
    let y = Big::new(2, 2);
    assert_eq!(x + &y, Big::new(3, 1));

    let x = Big::new(-1, 1);
    let y = Big::new(2, -2);
    assert_eq!(x + y, Big::new(-2, 1));

    let x = Big::new(0, 1);
    let y = &Big::new(-0, 2);
    assert_eq!(x + y, Big::new(0, 1));

    // with Rational64
    let mut x = Big::new(1, 1);
    let y = Big::new(2, 2);
    x += y;
    assert_eq!(x, Big::new(2, 1));

    let mut x = Big::new(2, 1);
    let y = Big::new(2, 2);
    x += &y;
    assert_eq!(x, Big::new(3, 1));

    let mut x = Big::new(1, 1);
    let y = Rational64::new(2, 2);
    x += y;
    assert_eq!(x, Big::new(2, 1));

    let mut x = Big::new(2, 1);
    let y = Rational64::new(2, 2);
    x += &y;
    assert_eq!(x, Big::new(3, 1));
}

#[test]
fn mul() {
    let x = Big::new(1, 2);
    let y = Big::new(3, 4);
    assert_eq!(x * y, Big::new(3, 8));

    let x = Big::new(5, 6);
    let y = Big::new(7, 8);
    assert_eq!(x * &y, Big::new(35, 48));

    let x = Big::new(-11, 12);
    let y = Rational64::new(13, -14);
    assert_eq!(x * &y, Big::new(11 * 13, 12 * 14));

    let x = Big::new(0, 1);
    let y = &Rational64::new(-0, 2);
    assert_eq!(x * y, Big::new(0, 1));

    let mut x = Big::new(1, 1);
    let y = Big::new(2, 2);
    x *= y;
    assert_eq!(x, Big::new(1, 1));

    let mut x = Big::new(2, 1);
    let y = Big::new(2, 2);
    x *= &y;
    assert_eq!(x, Big::new(8, 4));
}
