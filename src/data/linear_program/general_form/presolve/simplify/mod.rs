//! # Clearing the fractions out of a set of values
//!
//! A floating point solver scales a row for conditioning. Exact arithmetic has no conditioning to
//! worry about, which is usually where the subject is left — but it replaces that concern with
//! another one that does not go away: the encoding of the numbers. A row of fractions costs more
//! to hold than the same row written over a common denominator, and, more importantly, every
//! denominator in the data is inherited by everything computed from it.
//!
//! Dividing a row by the greatest common divisor of its values leaves integers with no common
//! factor, which is the canonical form of that constraint and the smallest way to write it as
//! integers. That divisor is the whole of what this module needs from the number type.
//!
//! As with [`Scalable`], the default implementation declines: not every ordered field is a field
//! of fractions, and one that is may still not be able to say what its parts are. Unlike
//! [`Scalable`], declining is silent, because a number type that cannot do this is not misusing
//! anything — the rule simply has nothing to do.
//!
//! [`Scalable`]: crate::data::linear_program::general_form::Scalable
use relp_num::{Abs, EncodingSize, Gcd, OrderedField, OrderedFieldRef};

/// Turning a set of values into integers with nothing left in common.
pub trait Simplify: Sized {
    /// The positive factor to multiply these values by to make them coprime integers.
    ///
    /// `None` when there is nothing to gain — the values are already in that form, or they are all
    /// zero — and also when the number type cannot answer, so that a caller can treat "no factor"
    /// as one thing.
    fn simplifying_factor<'a, I>(values: I) -> Option<Self>
    where
        I: Iterator<Item = &'a Self>,
        Self: 'a;

    /// What it costs to store this value, in bits, when the type can say.
    ///
    /// Used to check that clearing the fractions of a row makes it smaller rather than larger,
    /// which is not automatic: the least common multiple of a row's denominators multiplies into
    /// every numerator, and a row whose denominators are coprime pays for all of them at once.
    fn encoding_bits(&self) -> Option<u64>;
}

impl<F: OrderedField> Simplify for F {
    default fn simplifying_factor<'a, I>(_values: I) -> Option<Self>
    where
        I: Iterator<Item = &'a Self>,
        Self: 'a,
    {
        None
    }

    default fn encoding_bits(&self) -> Option<u64> {
        None
    }
}

/// A field of fractions that can be taken apart answers for real.
impl<F: OrderedField + Gcd + Abs + EncodingSize> Simplify for F {
    fn simplifying_factor<'a, I>(values: I) -> Option<Self>
    where
        I: Iterator<Item = &'a Self>,
        Self: 'a,
    {
        let mut divisor: Option<F> = None;
        for value in values {
            // Zero is an integer multiple of everything, so it neither restricts the divisor nor
            // needs one.
            if value.is_zero() {
                continue;
            }

            divisor = Some(match divisor {
                None => value.clone().abs(),
                Some(current) => current.gcd(value)?,
            });
        }

        // Already coprime integers, or nothing but zeros.
        let divisor = divisor?;
        if divisor.is_one() {
            return None;
        }

        Some(F::one() / divisor)
    }

    fn encoding_bits(&self) -> Option<u64> {
        Some(EncodingSize::encoding_bits(self))
    }
}

/// Whether multiplying these values by the factor leaves them no larger than they are.
///
/// Clearing denominators usually shrinks a row and always makes what is computed from it cheaper,
/// but it is not free in general: the least common multiple of the denominators ends up in every
/// numerator, so values over denominators that share nothing pay for all of them at once. What is
/// allowed is stated as a fraction of what the values cost now.
pub fn worth_clearing<F>(values: &[&F], factor: &F, growth: (u32, u32)) -> bool
where
    F: Simplify,
    for<'r> &'r F: OrderedFieldRef<F>,
{
    let mut before = 0_u64;
    let mut after = 0_u64;
    for value in values {
        // A type that cannot measure itself cannot be second guessed, so it is taken at its word.
        let (Some(one), Some(other)) = (value.encoding_bits(), (*value * factor).encoding_bits())
        else {
            return true;
        };
        before += one;
        after += other;
    }

    u128::from(after) * u128::from(growth.1) <= u128::from(before) * u128::from(growth.0)
}

#[cfg(test)]
mod test;
