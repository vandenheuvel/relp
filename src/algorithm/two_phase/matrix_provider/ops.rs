//! # Properties of number types
use num::Zero;

use crate::data::linear_algebra::traits::Element;

pub trait Rhs =
    Zero +
    Eq +
    Element +
;
