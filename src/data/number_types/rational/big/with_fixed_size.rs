//! # Interactions with fixed size ratios
use std::ops::{Add, AddAssign, Div, Mul};

use num::Zero;

use crate::data::number_types::rational::{Rational128, Rational32, Rational64};

use super::Big;

macro_rules! define_interations {
    ($small:ident, $small_backing:ident, $module_name:ident) => {
        mod $module_name {
            use super::*;

            mod creation {
                use super::*;

                impl From<$small> for Big {
                    fn from(value: $small) -> Self {
                        let (numer, denom) = value.into();
                        Self((numer.into(), denom.into()).into())
                    }
                }

                impl From<&$small> for Big {
                    fn from(value: &$small) -> Self {
                        Self::from(value.clone())
                    }
                }
            }

            mod compare {
                use super::*;

                impl PartialEq<$small> for Big {
                    fn eq(&self, other: &$small) -> bool {
                        // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
                        let (numer, denom) = (*other.numer(), *other.denom());
                        let uptyped = num::BigRational::new(numer.into(), denom.into());
                        self.0.eq(&uptyped)
                    }
                }
            }

            mod field {
                use super::*;

                mod add {
                    use super::*;

                    impl Add<&$small> for Big {
                        type Output = Self;

                        fn add(mut self, rhs: &$small) -> Self::Output {
                            // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
                            let (numer, denom) = (*rhs.numer(), *rhs.denom());
                            let uptyped = num::BigRational::new(numer.into(), denom.into());
                            self.0 += uptyped;
                            self
                        }
                    }

                    impl Add<Option<&$small>> for Big {
                        type Output = Self;

                        fn add(self, rhs: Option<&$small>) -> Self::Output {
                            // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
                            match rhs {
                                None => self,
                                Some(rhs) => Add::add(self, rhs),
                            }
                        }
                    }

                    impl Add<&$small> for &Big {
                        type Output = Big;

                        fn add(self, rhs: &$small) -> Self::Output {
                            // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
                            self.clone().add(rhs)
                        }
                    }

                    impl Add<Option<&$small>> for &Big {
                        type Output = Big;

                        fn add(self, rhs: Option<&$small>) -> Self::Output {
                            // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
                            let copy = self.clone();
                            match rhs {
                                None => copy,
                                Some(rhs) => Add::add(copy, rhs),
                            }
                        }
                    }

                    impl AddAssign<$small> for Big {
                        fn add_assign(&mut self, rhs: $small) {
                            // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
                            let (numer, denom): ($small_backing, $small_backing) = rhs.into();
                            let uptyped = num::BigRational::new(numer.into(), denom.into());
                            self.0 += uptyped;
                        }
                    }

                    impl AddAssign<&$small> for Big {
                        fn add_assign(&mut self, rhs: &$small) {
                            // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
                            let (numer, denom) = (*rhs.numer(), *rhs.denom());
                            let uptyped = num::BigRational::new(numer.into(), denom.into());
                            self.0 += uptyped;
                        }
                    }
                }

                mod mul {
                    use super::*;

                    impl Mul<&$small> for Big {
                        type Output = Big;

                        fn mul(self, rhs: &$small) -> Self::Output {
                            // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
                            let (numer, denom) = (rhs.numer(), rhs.denom());
                            let uptyped = num::BigRational::new((*numer).into(), (*denom).into());
                            Self(self.0 * uptyped)
                        }
                    }

                    impl Mul<&$small> for &Big {
                        type Output = Big;

                        fn mul(self, rhs: &$small) -> Self::Output {
                            // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
                            self.clone() * rhs
                        }
                    }

                    impl Mul<Option<&$small>> for &Big {
                        type Output = Big;

                        fn mul(self, rhs: Option<&$small>) -> Self::Output {
                            // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
                            match rhs {
                                None => Big::zero(),
                                Some(rhs) => Mul::mul(self, rhs),
                            }
                        }
                    }
                }

                mod div {
                    use super::*;

                    impl Div<&$small> for Big {
                        type Output = Big;

                        fn div(self, rhs: &$small) -> Self::Output {
                            // TODO(PERFORMANCE): Make sure that this is just as efficient as a native algorithm.
                            let (numer, denom) = (rhs.numer(), rhs.denom());
                            let uptyped = num::BigRational::new((*numer).into(), (*denom).into());
                            Self(self.0 / uptyped)
                        }
                    }
                }
            }
        }
    }
}

define_interations!(Rational32, i32, rational32);
define_interations!(Rational64, i64, rational64);
define_interations!(Rational128, i128, rational128);
