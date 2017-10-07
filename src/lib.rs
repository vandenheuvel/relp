//! # A linear program solver.
//!
//! Linear programs are solved using the Simplex Method as described in the book Combinatorial
//! Optimization by Christos H. Papadimitriou.
#![feature(underscore_lifetimes)]
#![feature(match_default_bindings)]
#![feature(use_extern_macros)]

#[macro_use] extern crate assert_approx_eq;

pub mod algorithm;
pub mod io;

mod data;
