use num::FromPrimitive;

use rust_lp::data::number_types::rational::RationalBig;
use rust_lp::data::number_types::traits::Abs;
use rust_lp::RB;

use crate::netlib::solve;

#[test]
#[ignore = "Has a row name consisting of only numbers, the parser doesn't support that."]
fn test_25FV47() {
    let result = solve("25FV47");
    assert!((result.objective_value - RB!(5.5018459e+03)).abs() < RB!(1e-5)); // Gurobi
}

#[test]
#[ignore = "Too computationally intensive"]
fn test_80BAU3B() {
    let result = solve("80BAU3B");
    assert!((result.objective_value - RB!(9.872241924e+05)).abs() < RB!(1e-5)); // Gurobi
}

#[test]
fn test_ADLITTLE() {
    let result = solve("ADLITTLE");
    assert!((result.objective_value - RB!(2.254949632e+05)).abs() < RB!(1e-3)); // Gurobi
}
