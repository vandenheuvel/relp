//! # Presolving rules
//!
//! These rules, one per submodule, can be applied to simplify a linear program.
mod bound_constraint;
mod domain_propagation;
mod fixed_variable;
mod slack;
