use std::ops::{Add, AddAssign, Mul};

use num::{One, Zero};

use crate::data::number_types::binary::Binary;
use crate::data::number_types::rational::RationalBig;

impl From<Binary> for RationalBig {
    fn from(from: Binary) -> Self {
        match from {
            Binary::Zero => Self::zero(),
            Binary::One => Self::one(),
        }
    }
}

impl From<&Binary> for RationalBig {
    fn from(from: &Binary) -> Self {
        match from {
            Binary::Zero => Self::zero(),
            Binary::One => Self::one(),
        }
    }
}

impl Add<&Binary> for RationalBig {
    type Output = RationalBig;

    fn add(self, rhs: &Binary) -> Self::Output {
        match rhs {
            Binary::Zero => self,
            Binary::One => self + Self::one(),
        }
    }
}

impl AddAssign<&Binary> for RationalBig {
    fn add_assign(&mut self, rhs: &Binary) {
        match rhs {
            Binary::Zero => {}
            Binary::One => *self += RationalBig::one(),
        }
    }
}

impl Mul<&Binary> for RationalBig {
    type Output = RationalBig;

    fn mul(self, rhs: &Binary) -> Self::Output {
        match rhs {
            Binary::Zero => Zero::zero(),
            Binary::One => self,
        }
    }
}

impl Mul<&Binary> for &RationalBig {
    type Output = RationalBig;

    fn mul(self, rhs: &Binary) -> Self::Output {
        match rhs {
            Binary::Zero => Zero::zero(),
            Binary::One => self.clone(),
        }
    }
}
