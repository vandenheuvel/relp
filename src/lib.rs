//! # A linear program solver
//!
//! Linear programs are solved using the Simplex Method as described in the book Combinatorial
//! Optimization by Christos H. Papadimitriou and Kenneth Steiglitz.
#![warn(missing_docs)]

#![allow(incomplete_features)]
#![feature(specialization)]
#![feature(type_alias_impl_trait)]
#![feature(trait_alias)]
#![feature(impl_trait_in_assoc_type)]

pub mod algorithm;
pub mod data;
pub mod io;

#[cfg(test)]
mod tests;
