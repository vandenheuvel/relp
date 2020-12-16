//! # Presolving rules
//!
//! These rules, one per submodule, can be applied to simplify a linear program.
mod fixed_variable;
mod bound_constraint;
mod slack;
mod domain_propagation;
