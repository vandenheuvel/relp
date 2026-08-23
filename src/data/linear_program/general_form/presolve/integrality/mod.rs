//! # Rounding for integer variables
//!
//! Presolve serves two masters: it either treats every variable as continuous, giving reductions
//! that are valid for the relaxation, or it respects integrality. The difference is small and
//! local, and it is expressed by the `RESPECT_INTEGRALITY` constant that the rules are generic
//! over. What that constant needs from the number type is the ability to round, which is what this
//! module provides.
use relp_num::{OrderedField, OrderedFieldRef};

/// Rounding to the nearest integer, in either direction.
///
/// Unlike the `Scalable` trait, whose default implementation does nothing and warns, the default
/// here is *correct* for every ordered field: it needs nothing but one, addition, negation and
/// comparison. It is merely slow, at `O(log |v|)` field operations. A concrete number type that
/// can divide its numerator by its denominator should say so by overriding it.
pub trait Roundable {
    /// The largest integer that is at most this value.
    fn floor(&self) -> Self;
    /// The smallest integer that is at least this value.
    fn ceil(&self) -> Self;
    /// Whether this value is an integer.
    fn is_integer(&self) -> bool;
}

impl<F> Roundable for F
where
    F: OrderedField,
    for<'r> &'r F: OrderedFieldRef<F>,
{
    default fn floor(&self) -> Self {
        generic_floor(self)
    }

    default fn ceil(&self) -> Self {
        -generic_floor(&-self)
    }

    default fn is_integer(&self) -> bool {
        &generic_floor(self) == self
    }
}

/// The largest integer that is at most `value`, using only field operations.
///
/// Doubles a step size until the value is bracketed, then bisects. Both halves take `O(log |v|)`
/// iterations, so a value with a large numerator costs proportionally to its bit length rather
/// than to its magnitude.
fn generic_floor<F>(value: &F) -> F
where
    F: OrderedField,
    for<'r> &'r F: OrderedFieldRef<F>,
{
    let zero = F::zero();
    let one = F::one();

    if value >= &zero {
        // Find `high` with `value < high`, doubling from one.
        let mut low = zero;
        let mut high = one.clone();
        while &high <= value {
            low = high.clone();
            high = &high + &high;
        }
        // Invariant: `low <= value < high`, both integers.
        bisect(low, high, value, &one)
    } else {
        // Find `low <= value`, doubling downwards.
        let mut high = zero;
        let mut low = -one.clone();
        while &low > value {
            high = low.clone();
            low = &low + &low;
        }
        // Invariant: `low <= value < high`, both integers.
        bisect(low, high, value, &one)
    }
}

/// Narrow `low <= value < high` until `high - low == 1`, and return `low`.
fn bisect<F>(mut low: F, mut high: F, value: &F, one: &F) -> F
where
    F: OrderedField,
    for<'r> &'r F: OrderedFieldRef<F>,
{
    let two = one + one;
    while &(&high - &low) > one {
        // `low` and `high` are integers an even distance apart or one more than that, so the
        // midpoint of `low` and `high` rounded towards `low` is an integer.
        let middle = &low + &half_of_gap(&low, &high, &two, one);
        if &middle <= value {
            low = middle;
        } else {
            high = middle;
        }
    }

    low
}

/// Half the distance from `low` to `high`, rounded down to an integer.
fn half_of_gap<F>(low: &F, high: &F, two: &F, one: &F) -> F
where
    F: OrderedField,
    for<'r> &'r F: OrderedFieldRef<F>,
{
    let gap = high - low;
    let half = &gap / two;
    // `gap` is a positive integer; when it is odd, the halving leaves a fraction. Stepping back to
    // the integer below keeps both ends of the bracket integral.
    let mut candidate = half;
    while &candidate * two > gap {
        candidate = &candidate - one;
    }
    if candidate < *one { one.clone() } else { candidate }
}

#[cfg(test)]
mod test;
