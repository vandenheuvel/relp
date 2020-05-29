//! # A linear program solver
//!
//! Linear programs are solved using the Simplex Method as described in the book Combinatorial
//! Optimization by Christos H. Papadimitriou and Kenneth Steiglitz.
#![warn(missing_docs)]
#![feature(is_sorted)]
#![feature(cmp_min_max_by)]
#![feature(exclusive_range_pattern, half_open_range_patterns)]
#![feature(or_patterns)]
#![feature(vec_drain_as_slice)]
#![feature(fixed_size_array)]
#![feature(box_patterns)]

#[macro_use]
extern crate approx;
extern crate num;
extern crate daggy;

pub mod algorithm;
pub mod io;

pub mod data;

#[cfg(test)]
mod tests;
