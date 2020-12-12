//! # Integration tests
//!
//! Integration tests completely external from the crate. All code written in this module could be
//! written by an external user of the crate.
//!
//! ## Note
//!
//! The tests in this module are only ran when the a specific feature is enabled as these tests
//! take a long time to run.

#![allow(non_snake_case)]

#[cfg(all(feature = "burkardt"))]
mod burkardt;
#[cfg(all(feature = "miplib"))]
mod miplib;
#[cfg(all(feature = "netlib"))]
mod netlib;
#[cfg(all(feature = "unicamp"))]
mod unicamp;
#[cfg(all(feature = "cook"))]
mod cook;
