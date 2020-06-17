//! Integration tests completely external from the crate.
//!
//! All code written in this module could be written by an external user of the crate.
//!
//! ## Note
//!
//! The tests in this module are only ran when the a specific feature is enabled as these tests
//! take a long time to run.
#[cfg(all(test, feature = "burkardt"))]
mod burkardt;
#[cfg(all(test, feature = "miplib"))]
mod miplib;
#[cfg(all(test, feature = "unicamp"))]
mod unicamp;
