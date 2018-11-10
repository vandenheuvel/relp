//! Integration tests completely external from the crate.
//!
//! All code written in this module could be written by an external user of the crate.

#[cfg(all(test, feature = "miplib"))]
mod miplib;
