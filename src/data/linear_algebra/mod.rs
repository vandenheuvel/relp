//! # Linear algebra primitives
//!
//! Linear algebra primitives used to represent data in either a dense or a sparse format.
pub mod matrix;
pub mod vector;
mod utilities;

/// Inner value for the `SparseVector` and `SparseMatrix` type.
pub type SparseTuples<F> = Vec<(usize, F)>;
