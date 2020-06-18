//! # A linear program solver
//!
//! Linear programs are solved using the Simplex Method as described in the book Combinatorial
//! Optimization by Christos H. Papadimitriou and Kenneth Steiglitz.
#![warn(missing_docs)]
#![feature(is_sorted)]
#![feature(or_patterns)]
#![feature(drain_filter)]
#![feature(result_cloned)]

extern crate daggy;
extern crate num;

pub mod algorithm;
pub mod data;
pub mod io;

#[cfg(test)]
mod tests;
