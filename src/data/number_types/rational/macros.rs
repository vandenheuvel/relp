/// Shorthand for creating a rational number in tests.
#[macro_export]
macro_rules! R32 {
    ($value:expr) => {
        Rational32::from_f64($value as f64).unwrap()
    };
    ($numer:expr, $denom:expr) => {
        Rational32::new($numer, $denom)
    };
}

/// Shorthand for creating a rational number in tests.
#[macro_export]
macro_rules! R64 {
    ($value:expr) => {
        Rational64::from_f64($value as f64).unwrap()
    };
    ($numer:expr, $denom:expr) => {
        Rational64::new($numer, $denom)
    };
}

/// Shorthand for creating a rational number in tests.
#[macro_export]
macro_rules! R128 {
    ($value:expr) => {
        Rational128::from_f64($value as f64).unwrap()
    };
    ($numer:expr, $denom:expr) => {
        Rational128::new($numer, $denom)
    };
}

/// Shorthand for creating a rational number in tests.
#[macro_export]
macro_rules! RB {
    ($value:expr) => {
        RationalBig::from_f64($value as f64).unwrap()
    };
    ($numer:expr, $denom:expr) => {
        RationalBig::new($numer, $denom)
    };
}
