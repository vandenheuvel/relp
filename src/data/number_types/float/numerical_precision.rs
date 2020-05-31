//! Helper methods for the accuracy of floating point calculations.
use num::{Float, Integer, Signed, ToPrimitive, Unsigned};

/// See the docstring of `candidate_to_round_to`: this function could be used instead of the simpler
/// integer rounding.
pub fn close_heuristic_fraction<F, N, D>(value: F, rounds: u32, epsilon: F) -> F
where
    F: Float,
    N: Signed + Integer + ToPrimitive + Copy,
    D: Unsigned + Integer + ToPrimitive + Copy,
{
    debug_assert!(epsilon > F::zero());

    let floor = value.floor();
    let fraction = value - floor;

    let (numerator, denominator) = if fraction < epsilon {
        // Close to zero
        (N::zero(), D::one())
    } else if fraction > F::one() - epsilon {
        // Close to one
        (N::one(), D::one())
    } else {
        // Somewhere in the middle
        let (mut numerator_lb, mut numerator_ub) = (N::zero(), N::one());
        let (mut denominator_lb, mut denominator_ub) = (D::one(), D::one());
        let (mut numerator_middle, mut denominator_middle) = (N::one(), D::one() + D::one());

        for _ in 0..rounds {
            if F::from(denominator_middle).unwrap() * (fraction + epsilon) < F::from(numerator_middle).unwrap() {
                numerator_ub = numerator_middle;
                denominator_ub = denominator_middle;
            } else if F::from(denominator_middle).unwrap() * (fraction - epsilon) > F::from(numerator_middle).unwrap() {
                numerator_lb = numerator_middle;
                denominator_lb = denominator_middle;
            } else {
                break
            }

            numerator_middle = numerator_lb + numerator_ub;
            denominator_middle = denominator_lb + denominator_ub;
        }

        (numerator_middle, denominator_middle)
    };

    floor + F::from(numerator).unwrap() / F::from(denominator).unwrap()
}

#[cfg(test)]
mod test {
    use crate::data::number_types::float::numerical_precision::close_heuristic_fraction;

    #[test]
    fn test_close_heuristic_fraction() {
        let epsilon = 1e-10f64;
        for x in vec![0.6584f64, 0.684684f64, 0.121484948f64] {
            assert_relative_eq!(close_heuristic_fraction::<_, i128, u128>(x, 150, epsilon), x, epsilon=epsilon);
        }
        for x in vec![984654.65484f64, 4984.6484684f64, -4681984.8484948f64] {
            assert_abs_diff_eq!(close_heuristic_fraction::<_, i64, u64>(x, 150, epsilon), x, epsilon=epsilon);
        }
        for x in vec![0f64, 1e-16f64, 654654f64 - 1e-16f64] {
            assert_abs_diff_eq!(close_heuristic_fraction::<_, i32, u32>(x, 50, epsilon), x, epsilon=epsilon);
        }
    }
}
