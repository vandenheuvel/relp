//! # Number types
//!
//! Aliases for the number types needed when reading and converting MPS structures.
use std::fmt::{Debug, Display};
use std::ops::{Neg, Sub};

use num::{One, Zero};

use crate::data::number_types::traits::Abs;

pub trait Field =
    Zero +
    One +

    Neg<Output = Self> +
    Sub<Output = Self> +
    Abs +

    Eq +
    Ord +

    Debug +
    Display +
    Clone +
;
