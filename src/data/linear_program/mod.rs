//! # Representing linear programs
//!
//! This module contains different representations of linear programs. Linear programs in general
//! form may contain any type of constraint, while linear programs in standard form may contain
//! equality constraints only.
pub mod elements;
pub mod general_form;
pub mod network;
pub mod solution;
