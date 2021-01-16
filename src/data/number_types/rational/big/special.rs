//! # Non-standard implementations
//!
//! Operations with specific types from this crate.
mod creation {
    use num::One;

    use crate::algorithm::two_phase::matrix_provider::column::identity;
    use crate::data::number_types::rational::big::Big;

    impl From<identity::One> for Big {
        fn from(_: identity::One) -> Self {
            Self::one()
        }
    }

    impl From<&identity::One> for Big {
        fn from(_: &identity::One) -> Self {
            Self::one()
        }
    }
}

mod field {
    use num::One;
    use num::Zero;

    use crate::algorithm::two_phase::matrix_provider::column::identity;
    use crate::algorithm::two_phase::tableau::kind::artificial::Cost;
    use crate::data::number_types::rational::big::Big;

    mod add {
        use std::ops::{Add, AddAssign};

        use super::*;

        impl Add<Cost> for Big {
            type Output = Big;

            fn add(self, rhs: Cost) -> Self::Output {
                match rhs {
                    Cost::Zero => self,
                    Cost::One => {
                        let (numer, denom): (num::BigInt, num::BigInt) = self.0.into();
                        Self(num::BigRational::new(numer + &denom, denom))
                    },
                }
            }
        }

        impl Add<Option<&Big>> for Big {
            type Output = Big;

            fn add(self, rhs: Option<&Big>) -> Self::Output {
                // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
                match rhs {
                    None => self,
                    Some(rhs) => Add::add(self, rhs),
                }
            }
        }

        impl Add<Option<&Big>> for &Big {
            type Output = Big;

            fn add(self, rhs: Option<&Big>) -> Self::Output {
                // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
                let copy = self.clone();
                match rhs {
                    None => copy,
                    Some(rhs) => Add::add(copy, rhs),
                }
            }
        }

        impl Add<&identity::One> for Big {
            type Output = Self;

            fn add(self, _: &identity::One) -> Self::Output {
                self + Self::one()
            }
        }

        impl AddAssign<&identity::One> for Big {
            fn add_assign(&mut self, _: &identity::One) {
                self.0 += num::BigRational::one()
            }
        }
    }

    mod mul {
        use std::ops::Mul;

        use super::*;

        impl Mul<Cost> for Big {
            type Output = Big;

            fn mul(self, rhs: Cost) -> Self::Output {
                match rhs {
                    Cost::Zero => Self::Output::zero(),
                    Cost::One => self,
                }
            }
        }

        impl Mul<Cost> for &Big {
            type Output = Big;

            fn mul(self, rhs: Cost) -> Self::Output {
                match rhs {
                    Cost::Zero => Self::Output::zero(),
                    Cost::One => self.clone(),
                }
            }
        }

        impl Mul<Option<&Big>> for &Big {
            type Output = Big;

            fn mul(self, rhs: Option<&Big>) -> Self::Output {
                match rhs {
                    None => Big::zero(),
                    Some(rhs) => Mul::mul(self, rhs)
                }
            }
        }

        impl Mul<&identity::One> for &Big {
            type Output = Big;

            fn mul(self, _: &identity::One) -> Self::Output {
                self.clone()
            }
        }
    }

    mod div {
        use std::ops::Div;

        use super::*;

        impl Div<&identity::One> for Big {
            type Output = Big;

            fn div(self, _: &identity::One) -> Self::Output {
                self
            }
        }
    }
}
