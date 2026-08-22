//! # Number parsing
//!
//! Reading numbers from strings.
use std::convert::TryFrom;

use relp_num::Rational64;
use relp_num::Sign;

use crate::io::error::Parse as ParseError;
use crate::io::error::ParseResult;

/// Parsing a number read from an MPS file.
pub trait Parse: Sized {
    /// Read a string representation from a decimal (abc.xyz) number.
    ///
    /// # Errors
    ///
    /// When the number could not be parsed, an error.
    fn parse(text: &str) -> ParseResult<Self>;
}

impl Parse for f64 {
    fn parse(text: &str) -> ParseResult<Self> {
        let value: Self = text.parse().map_err(|error| {
            ParseError::wrap_other(
                error,
                format!("Failed to parse value text \"{}\" into f64", text),
            )
        })?;

        let minimum_magnitude = 1e-10;
        let maximum_magnitude = 1e10;
        if value.abs() < minimum_magnitude {
            Err(ParseError::new(format!(
                "Parsed value was {}, but can't be smaller than {}.",
                value, minimum_magnitude,
            )))
        } else if value.abs() > maximum_magnitude {
            Err(ParseError::new(format!(
                "Parsed value was {}, but can't be larger than {}.",
                value, maximum_magnitude,
            )))
        } else {
            Ok(value)
        }
    }
}

impl Parse for Rational64 {
    fn parse(text: &str) -> ParseResult<Self> {
        Self::try_from(Raw::try_from(text)?)
    }
}

impl TryFrom<Raw> for Rational64 {
    type Error = ParseError;

    fn try_from(value: Raw) -> Result<Self, Self::Error> {
        let Raw {
            sign,
            integer,
            decimal_steps_from_right,
        } = value;
        debug_assert!(sign != Sign::Zero || integer == 0);

        let numerator = integer;
        // Twenty digits after the point already leave `u64`, and `pow` wraps rather than saying so
        // in a release build: `10_u64.pow(21)` is `3875820019684212736`, a perfectly plausible
        // denominator that `new_signed` accepts without complaint.
        let denominator = 10_u64.checked_pow(decimal_steps_from_right).ok_or_else(|| {
            ParseError::new(format!(
                "A value with {} decimals needs a denominator that does not fit u64.",
                decimal_steps_from_right,
            ))
        })?;

        // A power of ten is never zero, and the sign agrees with the numerator by the assertion
        // above, so the two cases `new_signed` rejects are both ruled out here.
        Ok(Self::new_signed(sign, numerator, denominator)
            .expect("a power of ten is not zero and the sign matches the numerator"))
    }
}

/// Intermediate form of read number.
///
/// TODO(PERFORMANCE): Reduce the size of this struct to 64 bits.
#[derive(Eq, PartialEq, Clone, Debug)]
pub(crate) struct Raw {
    sign: Sign,                    // Need 2 bits
    integer: u64,                  // We need about 40 bits to represent
    decimal_steps_from_right: u32, // Need about 5 bits
}

impl TryFrom<&str> for Raw {
    type Error = ParseError;

    fn try_from(text: &str) -> Result<Self, Self::Error> {
        let (has_minus, text) = match &text[0..1] {
            "-" => (true, &text[1..]),
            _ => (false, text),
        };

        let parse = |text: &str, number_part| {
            if text.is_empty() {
                Ok(0)
            } else {
                text.parse().map_err(|error| {
                    ParseError::wrap_other(
                        error,
                        format!("Failed to parse {} \"{}\" as u64.", number_part, text),
                    )
                })
            }
        };

        let (integer, decimal_steps_from_right): (u64, _) = match text.find('.') {
            None => (parse(text, "entire value")?, 0),
            Some(index) => {
                let from_right = u32::try_from(text.len() - index - 1)
                    .expect("It wouldn't make sense for the number text to be so large.");

                let integer_part = parse(&text[0..index], "integer part")?;
                let mantissa_part = parse(&text[(index + 1)..], "mantissa part")?;

                // Shifting the integer part past the decimals leaves `u64` at roughly twenty
                // significant digits, and in a release build it wraps into a wrong value rather
                // than reporting anything.
                let shifted = 10_u64
                    .checked_pow(from_right)
                    .and_then(|shift| integer_part.checked_mul(shift))
                    .and_then(|shifted| shifted.checked_add(mantissa_part))
                    .ok_or_else(|| {
                        ParseError::new(format!(
                            "Value \"{}\" has too many significant digits to fit u64.",
                            text,
                        ))
                    })?;

                (shifted, from_right)
            }
        };

        let sign = match (has_minus, integer) {
            (_, 0) => Sign::Zero,
            (false, _) => Sign::Positive,
            (true, _) => Sign::Negative,
        };

        Ok(Self {
            sign,
            integer,
            decimal_steps_from_right,
        })
    }
}

#[cfg(test)]
mod test {
    use std::convert::TryFrom;

    use relp_num::Sign;
    use relp_num::{R64, Rational64};

    use crate::io::mps::number::parse::Raw;

    #[test]
    fn parse() {
        // dot location, single digit
        assert_eq!(
            Raw::try_from("1").unwrap(),
            Raw {
                sign: Sign::Positive,
                integer: 1,
                decimal_steps_from_right: 0,
            }
        );
        assert_eq!(
            Raw::try_from("2.").unwrap(),
            Raw {
                sign: Sign::Positive,
                integer: 2,
                decimal_steps_from_right: 0,
            }
        );
        assert_eq!(
            Raw::try_from(".3").unwrap(),
            Raw {
                sign: Sign::Positive,
                integer: 3,
                decimal_steps_from_right: 1,
            }
        );

        // sign
        assert_eq!(
            Raw::try_from("-1").unwrap(),
            Raw {
                sign: Sign::Negative,
                integer: 1,
                decimal_steps_from_right: 0,
            }
        );
        assert_eq!(
            Raw::try_from("-2.").unwrap(),
            Raw {
                sign: Sign::Negative,
                integer: 2,
                decimal_steps_from_right: 0,
            }
        );
        assert_eq!(
            Raw::try_from("-.3").unwrap(),
            Raw {
                sign: Sign::Negative,
                integer: 3,
                decimal_steps_from_right: 1,
            }
        );

        // dot location, multiple digits
        assert_eq!(
            Raw::try_from("16456").unwrap(),
            Raw {
                sign: Sign::Positive,
                integer: 16456,
                decimal_steps_from_right: 0,
            }
        );
        assert_eq!(
            Raw::try_from("64896848.").unwrap(),
            Raw {
                sign: Sign::Positive,
                integer: 64_896_848,
                decimal_steps_from_right: 0,
            }
        );
        assert_eq!(
            Raw::try_from(".984654684").unwrap(),
            Raw {
                sign: Sign::Positive,
                integer: 984_654_684,
                decimal_steps_from_right: 9,
            }
        );
        assert_eq!(
            Raw::try_from("-.95").unwrap(),
            Raw {
                sign: Sign::Negative,
                integer: 95,
                decimal_steps_from_right: 2,
            }
        );

        // multiple digits, both sides
        assert_eq!(
            Raw::try_from("15465.2").unwrap(),
            Raw {
                sign: Sign::Positive,
                integer: 154652,
                decimal_steps_from_right: 1,
            }
        );
        assert_eq!(
            Raw::try_from("1234.56789").unwrap(),
            Raw {
                sign: Sign::Positive,
                integer: 123_456_789,
                decimal_steps_from_right: 5,
            }
        );
        assert_eq!(
            Raw::try_from("1.24654").unwrap(),
            Raw {
                sign: Sign::Positive,
                integer: 124_654,
                decimal_steps_from_right: 5,
            }
        );

        // Zero
        assert_eq!(
            Raw::try_from("0").unwrap(),
            Raw {
                sign: Sign::Zero,
                integer: 0,
                decimal_steps_from_right: 0,
            }
        );
        assert_eq!(
            Raw::try_from("0.").unwrap(),
            Raw {
                sign: Sign::Zero,
                integer: 0,
                decimal_steps_from_right: 0,
            }
        );
        assert_eq!(
            Raw::try_from(".0").unwrap(),
            Raw {
                sign: Sign::Zero,
                integer: 0,
                decimal_steps_from_right: 1,
            }
        );
    }

    #[test]
    fn convert() {
        // dot location, single digit
        assert_eq!(
            Rational64::try_from(Raw {
                sign: Sign::Positive,
                integer: 1,
                decimal_steps_from_right: 0,
            }).unwrap(),
            R64!(1)
        );
        assert_eq!(
            Rational64::try_from(Raw {
                sign: Sign::Positive,
                integer: 2,
                decimal_steps_from_right: 0,
            }).unwrap(),
            R64!(2)
        );
        assert_eq!(
            Rational64::try_from(Raw {
                sign: Sign::Positive,
                integer: 3,
                decimal_steps_from_right: 1,
            }).unwrap(),
            R64!(3, 10)
        );

        // sign
        assert_eq!(
            Rational64::try_from(Raw {
                sign: Sign::Negative,
                integer: 1,
                decimal_steps_from_right: 0,
            }).unwrap(),
            R64!(-1, 1)
        );
        assert_eq!(
            Rational64::try_from(Raw {
                sign: Sign::Negative,
                integer: 2,
                decimal_steps_from_right: 0,
            }).unwrap(),
            R64!(-2, 1)
        );
        assert_eq!(
            Rational64::try_from(Raw {
                sign: Sign::Negative,
                integer: 3,
                decimal_steps_from_right: 1,
            }).unwrap(),
            R64!(-3, 10)
        );

        // dot location, multiple digits
        assert_eq!(
            Rational64::try_from(Raw {
                sign: Sign::Positive,
                integer: 16456,
                decimal_steps_from_right: 0,
            }).unwrap(),
            R64!(16456, 1)
        );
        assert_eq!(
            Rational64::try_from(Raw {
                sign: Sign::Positive,
                integer: 64_896_848,
                decimal_steps_from_right: 0,
            }).unwrap(),
            R64!(64_896_848, 1)
        );
        assert_eq!(
            Rational64::try_from(Raw {
                sign: Sign::Positive,
                integer: 984_654_684,
                decimal_steps_from_right: 9,
            }).unwrap(),
            R64!(246_163_671, 250_000_000)
        );

        // multiple digits, both sides
        assert_eq!(
            Rational64::try_from(Raw {
                sign: Sign::Positive,
                integer: 12,
                decimal_steps_from_right: 1,
            }).unwrap(),
            R64!(12, 10)
        );
        assert_eq!(
            Rational64::try_from(Raw {
                sign: Sign::Positive,
                integer: 123_456_789,
                decimal_steps_from_right: 5,
            }).unwrap(),
            R64!(123_456_789, 100_000)
        );
        assert_eq!(
            Rational64::try_from(Raw {
                sign: Sign::Positive,
                integer: 124_654,
                decimal_steps_from_right: 5,
            }).unwrap(),
            R64!(62_327, 50_000)
        );

        // Zero
        assert_eq!(
            Rational64::try_from(Raw {
                sign: Sign::Zero,
                integer: 0,
                decimal_steps_from_right: 0,
            }).unwrap(),
            R64!(0, 1)
        );
        assert_eq!(
            Rational64::try_from(Raw {
                sign: Sign::Zero,
                integer: 0,
                decimal_steps_from_right: 1,
            }).unwrap(),
            R64!(0, 1)
        );
    }

    /// Overflowing the denominator or the shifted integer part has to be reported, not wrapped.
    ///
    /// `10_u64.pow(21)` wraps to `3875820019684212736` in a release build, and `new_signed` then
    /// accepts that as an ordinary denominator, so a long enough decimal field silently parsed to
    /// a different number. Rust's `FromStr` accepts leading zeros, which is why the first case is
    /// a way of writing one.
    #[test]
    fn test_too_many_decimals_is_an_error() {
        use relp_num::Rational64;

        // Twenty one decimals, value one
        let text = "1.000000000000000000000";
        assert!(Raw::try_from(text).and_then(Rational64::try_from).is_err(), "{text}");

        // Leading zeros parse fine as u64, so the digit count is what matters
        let text = "0.000000000000000000001";
        assert!(Raw::try_from(text).and_then(Rational64::try_from).is_err(), "{text}");

        // Enough significant digits that shifting the integer part leaves u64
        let text = "123456789012.34567890123";
        assert!(Raw::try_from(text).and_then(Rational64::try_from).is_err(), "{text}");

        // Just inside the range, and exact
        let text = "1.0000000000000000000";
        let value = Raw::try_from(text).and_then(Rational64::try_from);
        assert_eq!(value.unwrap(), Rational64::new(1, 1).unwrap(), "{text}");

        let text = "1.5";
        let value = Raw::try_from(text).and_then(Rational64::try_from);
        assert_eq!(value.unwrap(), Rational64::new(3, 2).unwrap(), "{text}");

        let text = "-0.25";
        let value = Raw::try_from(text).and_then(Rational64::try_from);
        assert_eq!(value.unwrap(), Rational64::new(-1, 4).unwrap(), "{text}");
    }
}
