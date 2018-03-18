//! # A linear program solver.
//!
//! Linear programs are solved using the Simplex Method as described in the book Combinatorial
//! Optimization by Christos H. Papadimitriou.
#![feature(match_default_bindings)]
#![feature(try_from)]
#![feature(underscore_lifetimes)]
#![feature(use_extern_macros)]
#![feature(conservative_impl_trait)]
#![feature(slice_patterns)]

#[macro_use] extern crate assert_approx_eq;

pub mod algorithm;
pub mod io;

mod data;

#[cfg(test)]
mod test;
