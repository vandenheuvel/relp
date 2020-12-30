//! # Wrapping existing methods
//!
//! Standard operations on types that are somewhat general, often simply wrapping the operations
//! already defined on the inner type.
use std::fmt;

use crate::data::number_types::rational::big::Big;

mod creation {
    use std::str::FromStr;

    use num::FromPrimitive;

    use crate::data::number_types::rational::big::Big;

    impl FromPrimitive for Big {
        fn from_i64(n: i64) -> Option<Self> {
            FromPrimitive::from_i64(n).map(Self)
        }

        fn from_u64(n: u64) -> Option<Self> {
            FromPrimitive::from_u64(n).map(Self)
        }

        fn from_f32(n: f32) -> Option<Self> {
            FromPrimitive::from_f32(n).map(Self)
        }

        fn from_f64(n: f64) -> Option<Self> {
            FromPrimitive::from_f64(n).map(Self)
        }
    }

    impl From<&str> for Big {
        fn from(input: &str) -> Self {
            Self(num::BigRational::from_str(input).unwrap())
        }
    }

    impl From<&Big> for Big {
        fn from(value: &Big) -> Self {
            value.clone()
        }
    }
}

mod field {
    mod add {
        use std::iter::Sum;
        use std::ops::{Add, AddAssign};

        use crate::data::number_types::rational::big::Big;

        impl Add for Big {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self(self.0.add(rhs.0))
            }
        }

        impl Add<&Big> for Big {
            type Output = Self;

            fn add(self, rhs: &Self) -> Self::Output {
                Self(self.0.add(&rhs.0))
            }
        }

        impl Add<Big> for &Big {
            type Output = Big;

            fn add(self, rhs: Big) -> Self::Output {
                Big(Add::add(&self.0, rhs.0))
            }
        }

        impl Add for &Big {
            type Output = Big;

            fn add(self, rhs: Self) -> Self::Output {
                Big(Add::add(&self.0,&rhs.0))
            }
        }

        impl AddAssign<Big> for Big {
            fn add_assign(&mut self, rhs: Self) {
                self.0.add_assign(rhs.0)
            }
        }

        impl AddAssign<&Big> for Big {
            fn add_assign(&mut self, rhs: &Self) {
                self.0.add_assign(&rhs.0)
            }
        }

        impl Sum for Big {
            fn sum<I: Iterator<Item=Self>>(iter: I) -> Self {
                Self(iter.map(|item| item.0).sum())
            }
        }
    }

    mod sub {
        use std::ops::{Sub, SubAssign};

        use crate::data::number_types::rational::big::Big;

        impl Sub<Big> for Big {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                Self(self.0.sub(rhs.0))
            }
        }

        impl Sub<&Big> for Big {
            type Output = Self;

            fn sub(self, rhs: &Self) -> Self::Output {
                Self(self.0.sub(&rhs.0))
            }
        }

        impl Sub<Big> for &Big {
            type Output = Big;

            fn sub(self, rhs: Big) -> Self::Output {
                Big(Sub::sub(&self.0, rhs.0))
            }
        }

        impl Sub for &Big {
            type Output = Big;

            fn sub(self, rhs: Self) -> Self::Output {
                Big(Sub::sub(&self.0,&rhs.0))
            }
        }

        impl SubAssign<Big> for Big {
            fn sub_assign(&mut self, rhs: Self) {
                self.0.sub_assign(rhs.0)
            }
        }

        impl SubAssign<&Big> for Big {
            fn sub_assign(&mut self, rhs: &Self) {
                self.0.sub_assign(&rhs.0)
            }
        }
    }

    mod mul {
        use std::ops::{Mul, MulAssign};

        use crate::data::number_types::rational::big::Big;

        impl Mul for Big {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self::Output {
                Self(self.0.mul(rhs.0))
            }
        }

        impl Mul<&Big> for Big {
            type Output = Self;

            fn mul(self, rhs: &Self) -> Self::Output {
                Self(self.0.mul(&rhs.0))
            }
        }

        impl Mul<Big> for &Big {
            type Output = Big;

            fn mul(self, rhs: Big) -> Self::Output {
                Big(Mul::mul(&self.0, rhs.0))
            }
        }

        impl Mul for &Big {
            type Output = Big;

            fn mul(self, rhs: Self) -> Self::Output {
                Big(Mul::mul(&self.0,&rhs.0))
            }
        }

        impl MulAssign<Big> for Big {
            fn mul_assign(&mut self, rhs: Self) {
                self.0.mul_assign(rhs.0)
            }
        }

        impl MulAssign<&Big> for Big {
            fn mul_assign(&mut self, rhs: &Self) {
                self.0.mul_assign(&rhs.0)
            }
        }
    }

    mod div {
        use std::ops::{Div, DivAssign};

        use crate::data::number_types::rational::big::Big;

        impl Div<Big> for Big {
            type Output = Self;

            fn div(self, rhs: Self) -> Self::Output {
                Self(self.0.div(rhs.0))
            }
        }

        impl Div<&Big> for Big {
            type Output = Self;

            fn div(self, rhs: &Self) -> Self::Output {
                Self(self.0.div(&rhs.0))
            }
        }

        impl Div<Big> for &Big {
            type Output = Big;

            fn div(self, rhs: Big) -> Self::Output {
                Big(Div::div(&self.0, rhs.0))
            }
        }

        impl Div for &Big {
            type Output = Big;

            fn div(self, rhs: Self) -> Self::Output {
                Big(Div::div(&self.0,&rhs.0))
            }
        }

        impl DivAssign<Big> for Big {
            fn div_assign(&mut self, rhs: Self) {
                self.0.div_assign(rhs.0)
            }
        }

        impl DivAssign<&Big> for Big {
            fn div_assign(&mut self, rhs: &Self) {
                self.0.div_assign(&rhs.0)
            }
        }
    }

    mod neg {
        use std::ops::Neg;

        use crate::data::number_types::rational::big::Big;

        impl Neg for Big {
            type Output = Self;

            fn neg(self) -> Self::Output {
                Self(self.0.neg())
            }
        }

        impl Neg for &Big {
            type Output = Big;

            fn neg(self) -> Self::Output {
                Big(Neg::neg(&self.0))
            }
        }
    }

    mod identities {
        use num::{One, Zero};

        use crate::data::number_types::rational::big::Big;

        impl Zero for Big {
            fn zero() -> Self {
                Self(num::BigRational::zero())
            }

            fn set_zero(&mut self) {
                self.0.set_zero()
            }

            fn is_zero(&self) -> bool {
                self.0.is_zero()
            }
        }

        impl One for Big {
            fn one() -> Self {
                Self(num::BigRational::one())
            }

            fn set_one(&mut self) {
                self.0.set_one()
            }

            fn is_one(&self) -> bool where
                Self: PartialEq, {
                self.0.is_one()
            }
        }
    }
}

impl fmt::Display for Big {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
