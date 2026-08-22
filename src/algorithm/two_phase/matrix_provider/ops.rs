//! # Properties of number types
use relp_num::NonZero;

use crate::data::linear_algebra::traits::Element;

/// Operations that should be possible with the right-hand side values of a matrix provider.
pub trait Rhs = NonZero + Eq + Element;
