//! # A linear program solver.
//!
//! Linear programs are solved using the Simplex Method as described in the book Combinatorial
//! Optimization by Christos H. Papadimitriou.
#![feature(try_from)]
#![feature(slice_patterns)]
#![feature(convert_id)]

#[macro_use] extern crate approx;
extern crate core;

pub mod algorithm;
pub mod io;

mod data;

#[cfg(test)]
mod test;
