//! Helper methods for the accuracy of floating point calculations.
use num::Float;

/// See the docstring of `candidate_to_round_to`: this function could be used instead of the simpler
/// integer rounding.
pub fn close_heuristic_fraction<F: Float>(value: F, epsilon: F) -> F {
    debug_assert!(epsilon > F::zero());

    let floor = value.floor();
    let fraction = value - floor;

    let close_fraction = if fraction < epsilon {
        // Close to zero
        fraction
    } else if fraction > F::one() - epsilon {
        // Close to one
        fraction + F::one()
    } else {
        // Somewhere in the middle
        let (mut numerator_lb, mut numerator_ub) = (0u64, 1u64);
        let (mut denominator_lb, mut denominator_ub) = (1u64, 1u64);
        let (mut numerator_middle, mut denominator_middle) = (1u64, 2u64);

        loop {
            if F::from(denominator_middle).unwrap() * (fraction + epsilon)  < F::from(numerator_middle).unwrap() {
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

        F::from(numerator_middle).unwrap() / F::from(denominator_middle).unwrap()
    };

    floor + close_fraction
}

#[cfg(test)]
mod test {
    use crate::data::number_types::float::numerical_precision::close_heuristic_fraction;

    #[test]
    fn test_close_heuristic_fraction() {
        let epsilon = 1e-15f64;
        for x in vec![0.6584f64, 0.684684f64, 0.121484948f64] {
            assert_relative_eq!(x, close_heuristic_fraction(x, epsilon));
        }
        for x in vec![984654.65484f64, 4984.6484684f64, -4681984.8484948f64] {
            assert_abs_diff_eq!(x, close_heuristic_fraction(x, epsilon));
        }
        for x in vec![0f64, 1e-16f64, 654654f64 - 1e-16f64] {
            assert_abs_diff_eq!(x, close_heuristic_fraction(x, epsilon));
        }
    }
}
