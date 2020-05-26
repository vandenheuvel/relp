//! # Floating point numbers
//!
//! Assumed to have a better performance than rational numbers in the simplex algorithm. Correctness
//! guarantees are harder (impossible?) to give due to (accumulating) rounding errors.
use core::fmt;
use std::cmp::Ordering;
use std::fmt::{Debug, Display, Formatter};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use noisy_float::checkers::FiniteChecker;
use noisy_float::FloatChecker;
use num::{Float, FromPrimitive, ToPrimitive, Unsigned};
use num::traits::{One, Zero};

use crate::data::number_types::float::numerical_precision::close_heuristic_fraction;
use crate::data::number_types::traits::{DedekindComplete, Field, OrderedField, RealField};

pub mod numerical_precision;

// TODO: How can fused multiply add be utilized?

/// Probably choose OPS same size as T in bits
#[derive(Copy, Clone, Debug)]
pub struct FiniteErrorControlledFloat<F: Derived, OPS: Derived> {
    value: F,
    operations: OPS,
}
type FECF<F, OPS> = FiniteErrorControlledFloat<F, OPS>;

/// All traits that the inner float and inner operations value should satisfy, such that we can
/// derive them for the `FECF` struct.
pub trait Derived: Copy + Clone + Debug {}
/// Basic field operations necessary to calculate with the inner float.
pub trait InnerF: Derived + Float + AddAssign + SubAssign + MulAssign + DivAssign + Display {}
/// Basic numerical operations necessary from the operations counter.
pub trait InnerOPS: Derived + Unsigned + Ord + FromPrimitive + ToPrimitive + AddAssign + Eq + Copy + Display {}

macro_rules! impl_FECF {
    ($f_t:ident, $ops_t:ident) => {
        impl Derived for $f_t {}
        impl Derived for $ops_t {}

        impl InnerF for $f_t {}
        impl InnerOPS for $ops_t {}
    }
}
impl_FECF!(f64, u64);
impl_FECF!(f32, u32);

/// Implementations of `RealField` for `FECF` types.
///
/// This type will might the `RealField` contract. Finite precision might
/// lead to overflows.
impl<F: InnerF, OPS: InnerOPS> RealField for FECF<F, OPS> {
    fn floor(self) -> Self {
        Self { value: self.value.floor(), operations: OPS::one(), }
    }

    fn ceil(self) -> Self {
        Self { value: self.value.ceil(), operations: OPS::one(), }
    }

    fn round(self) -> Self {
        Self { value: self.value.round(), operations: OPS::one(), }
    }
}

/// Not actually Dedekind complete, and not actually an ordered field.
///
/// Because of limited precision and limited maximum number size.
impl<F: InnerF, OPS: InnerOPS> DedekindComplete for FECF<F, OPS> {}
impl<F: InnerF, OPS: InnerOPS> OrderedField for FECF<F, OPS> {}

/// Basic computational operations.
impl<F: InnerF, OPS: InnerOPS> Field for FECF<F, OPS> {
    /// Create a zero valued `FECF`.
    ///
    /// The number of operations is zero, because it can be represented exactly.
    fn additive_identity() -> Self {
        Self { value: F::zero(), operations: OPS::zero(), }
    }

    /// Create a one valued `FECF`.
    ///
    /// The number of operations is zero, because it can be represented exactly.
    fn multiplicative_identity() -> Self {
        Self { value: F::one(), operations: OPS::zero(), }
    }
}

impl<F: InnerF, OPS: InnerOPS> FECF<F, OPS> {
    /// If sufficiently close to a number which is likely to be closer to the true result of a
    /// computation rather than a computation result with (accumulated) rounding errors, round and reset
    /// the operations counter.
    fn trim_iid_rounding_errors(&mut self) {
        // if self.operations < OPS::from_u64(1e6 as u64).unwrap() {
        //     return
        // }

        let candidate = self.candidate_to_round_to();
        if self.is_close_to_candidate(candidate, self.operations) {
            self.value = candidate;
            self.operations = OPS::zero();
        }
    }

    /// Generate a (heuristic) fraction that is close to the original value.
    ///
    /// TODO: Currently, this is just the nearest integer. Consider expanding this to other fractions.
    /// See https://stackoverflow.com/questions/5124743/algorithm-for-simplifying-decimal-to-fractions/42085412#42085412
    /// and the `close_heuristic_fraction` function below.
    /// # Arguments
    ///
    /// * `value`: Float (potentially having rounding errors) that we try to find a candidate for.
    ///
    /// # Return value
    ///
    /// Float representation (with rounding error, but only a single one) of a fraction that is close to
    /// the input.
    fn candidate_to_round_to(&self) -> F {
        close_heuristic_fraction::<_, i64, u64>(self.value,30, Self::base_epsilon())
    }

    fn is_close_to_candidate(&self, candidate: F, operations: OPS) -> bool {
        // Equality, in case epsilon
        (candidate - self.value).abs() <= Self::accumulation_factor(operations) * Self::base_epsilon()
    }

    /// Heuristic multiplication factor for the expected magnitude of accumulated errors.
    fn accumulation_factor(operations: OPS) -> F {
        F::from(OPS::one() + operations).unwrap().sqrt()
    }

    /// Numerical error assumed to accumulate for each operation.
    ///
    /// TODO: A relative epsilon rather than an absolute one is (probably) needed
    /// TODO: Different epsilons per operation?
    fn base_epsilon() -> F {
        // On a problem of size 2000 x 4000, this resulted in good results from 2 ** 14 to 2 ** 30
        // Let's go with 1_024 instead of 1_024 ** 2, because we also have f32 now, f32::eps == 1e-6
        F::from(1e5).unwrap() * F::epsilon()
    }
}

/// Converting basic types to `FECF`.
impl<F: InnerF, OPS: InnerOPS> FromPrimitive for FECF<F, OPS> {
    fn from_i64(n: i64) -> Option<Self> {
        if let Some(value) = F::from(n) {
            if FiniteChecker::check(value) {
                Some(Self { value, operations: OPS::one(), })
            } else { None }
        } else {
            None
        }
    }

    fn from_u64(n: u64) -> Option<Self> {
        if let Some(value) = F::from(n) {
            if FiniteChecker::check(value) {
                Some(Self { value, operations: OPS::one(), })
            } else { None }
        } else {
            None
        }
    }

    fn from_f64(n: f64) -> Option<Self> {
        let potential = F::from(n);
        if let Some(value) = potential {
            if FiniteChecker::check(value) {
                Some(Self { value, operations: OPS::one(), })
            } else {
                None
            }
        } else {
            None
        }
    }
}

macro_rules! impl_trait_for_FECF_move {
    ($trait:ident, $trait_method: ident) => {
        impl<F: InnerF, OPS: InnerOPS> $trait for FECF<F, OPS> {
            type Output = Self;

            fn $trait_method(self, rhs: Self) -> Self::Output {
                let mut result = Self {
                    value: $trait::$trait_method(self.value, rhs.value),
                    operations: self.operations + rhs.operations,
                };
                result.trim_iid_rounding_errors();
                FiniteChecker::assert(result.value);
                result
            }
        }
    }
}

impl_trait_for_FECF_move!(Add, add);
impl_trait_for_FECF_move!(Sub, sub);
impl_trait_for_FECF_move!(Mul, mul);
impl_trait_for_FECF_move!(Div, div);

impl<F: InnerF, OPS: InnerOPS> Neg for FECF<F, OPS> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self { value: -self.value, operations: self.operations, }
    }
}

macro_rules! impl_trait_for_FECF {
    ($trait:ident, $trait_method: ident) => {
        impl<F: InnerF, OPS: InnerOPS> $trait for FECF<F, OPS> {
            fn $trait_method(&mut self, rhs: Self) {
                $trait::$trait_method(&mut self.value, rhs.value);
                self.operations += rhs.operations;
                self.trim_iid_rounding_errors();
                FiniteChecker::assert(self.value);
            }
        }
    }
}
impl_trait_for_FECF!(AddAssign, add_assign);
impl_trait_for_FECF!(SubAssign, sub_assign);
impl_trait_for_FECF!(MulAssign, mul_assign);
impl_trait_for_FECF!(DivAssign, div_assign);

macro_rules! impl_trait_for_FECF_fold {
    ($trait:ident, $trait_method: ident, $initial_value: expr, $base_trait: ident, $base_trait_method: ident) => {
        impl<F: InnerF, OPS: InnerOPS> $trait for FECF<F, OPS> {
            fn $trait_method<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold($initial_value, $base_trait::$base_trait_method)
            }
        }
    }
}
impl_trait_for_FECF_fold!(Sum, sum, Self::zero(), Add, add);
impl_trait_for_FECF_fold!(Product, product, Self::one(), Mul, mul);

impl<F: InnerF, OPS: InnerOPS> Zero for FECF<F, OPS> {
    fn zero() -> Self {
        Self { value: F::zero(), operations: OPS::zero(), }
    }
    fn is_zero(&self) -> bool {
        self.is_close_to_candidate(F::zero(), OPS::zero())
    }
}
impl<F: InnerF, OPS: InnerOPS> One for FECF<F, OPS> {
    fn one() -> Self {
        Self { value: F::one(), operations: OPS::zero(), }
    }
}

impl<F: InnerF, OPS: InnerOPS> Eq for FECF<F, OPS> {}
impl<F: InnerF, OPS: InnerOPS> PartialEq for FECF<F, OPS> {
    fn eq(&self, other: &Self) -> bool {
        self.is_close_to_candidate(other.value, self.operations + other.operations)
    }
}

impl<F: InnerF, OPS: InnerOPS> Ord for FECF<F, OPS> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl<F: InnerF, OPS: InnerOPS> PartialOrd for FECF<F, OPS> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.eq(other) {
            Some(Ordering::Equal)
        } else if self.value > other.value {
            Some(Ordering::Greater)
        } else if self.value < other.value {
            Some(Ordering::Less)
        } else {
            panic!()
        }
    }
}

impl<F: InnerF, OPS: InnerOPS> Display for FECF<F, OPS> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.value, self.operations)
    }
}
/// Shorthand for creating a `FECF` number in tests.
#[macro_export]
macro_rules! F32 {
    ($value:expr) => {
        FiniteErrorControlledFloat::<f32, u32>::from_f64($value as f64).unwrap()
    };
}
/// Shorthand for creating a `FECF` number in tests.
#[macro_export]
macro_rules! F64 {
    ($value:expr) => {
        FiniteErrorControlledFloat::<f64, u64>::from_f64($value as f64).unwrap()
    };
}

#[cfg(test)]
mod test {
    use crate::data::number_types::float::FiniteErrorControlledFloat;
    use crate::num::FromPrimitive;

    /// TODO: Below is a well-known problematic float example. Is it even important (or feasible)
    ///  that this can be dealth with?
    #[test]
    #[ignore]
    fn relative_error() {
        let x = F32!(64919121) * F32!(205117922);
        let y = F32!(159018721) * F32!(83739041) + F32!(1);
        assert_eq!(x, y);
        let x = F64!(64919121) * F64!(205117922);
        let y = F64!(159018721) * F64!(83739041);
        let z = y + F64!(1);
        assert_eq!(y, z);
        assert_eq!(x, y);
    }

    macro_rules! test_FECF {
        ($t:ident, $test_module_name:ident, $f_t:ident, $ops_t:ident) => {
            mod $test_module_name {
                use num::traits::{One, Zero};
                use crate::num::FromPrimitive;

                use crate::data::number_types::float::FECF;
                use crate::data::number_types::traits::Field;
                use crate::data::number_types::float::FiniteErrorControlledFloat;

                #[test]
                fn test_field_identities() {
                    assert_eq!($t!(0), FECF::<$f_t, $ops_t>::additive_identity());
                    assert_eq!($t!(1), FECF::<$f_t, $ops_t>::multiplicative_identity());
                }

                #[test]
                #[should_panic]
                fn test_panic_create_infinite() {
                    let _result = $t!(f64::INFINITY);
                }

                #[test]
                #[should_panic]
                fn test_panic_create_nan() {
                    let _result = $t!(f64::NAN);
                }

                #[test]
                fn test_eq() {
                    assert_eq!($t!(3), $t!(3));
                    assert_eq!($t!(0), $t!(0));
                    assert_eq!($t!(-1), $t!(-1));
                }

                #[test]
                fn test_add() {
                    assert_eq!($t!(3) + $t!(6), $t!(9));
                    assert_eq!($t!(0) + $t!(0), $t!(0));
                    assert_eq!($t!(0) + $t!(2), $t!(2));

                    let mut x = $t!(0);
                    for _ in 0..1000 {
                        x = x + $t!(1);
                    }
                    assert_eq!(x, $t!(1000));
                }

                #[test]
                fn test_sub() {
                    assert_eq!($t!(3) - $t!(6), $t!(-3));
                    assert_eq!($t!(0) - $t!(0), $t!(0));
                    assert_eq!($t!(0) - $t!(16), $t!(-16));
                }

                #[test]
                fn test_mul() {
                    assert_eq!($t!(3) * $t!(6), $t!(18));
                    assert_eq!($t!(0) * $t!(0), $t!(0));
                    assert_eq!($t!(0) * $t!(0), $t!(0));
                }

                #[test]
                fn test_div() {
                    assert_eq!($t!(3) / $t!(3), FECF::<$f_t, $ops_t>::one());
                    assert_eq!($t!(-2) / $t!(-2), FECF::<$f_t, $ops_t>::one());
                    assert_eq!($t!(0) / $t!(2), FECF::<$f_t, $ops_t>::zero());
                    assert_eq!($t!(0) / $t!(-6), FECF::<$f_t, $ops_t>::zero());
                    assert_eq!($t!(6) / $t!(9), $t!(6f64 / 9f64));
                }

                #[test]
                #[should_panic]
                fn test_panic_divide_zero_by_zero() {
                    let _result = $t!(0) / $t!(0);
                }

                #[test]
                #[should_panic]
                fn test_panic_divide_nonzero_by_zero() {
                    let _result = $t!(3) / $t!(0);
                }
            }
        }
    }

    test_FECF!(F32, test_f32_u32, f32, u32);
    test_FECF!(F64, test_f64_u64, f64, u64);
}