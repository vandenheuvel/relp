//! # Linear algebra primitives
//!
//! Linear algebra primitives used to represent data in either a dense or a sparse format.
pub mod matrix;
pub mod vector;
mod utilities;
pub mod traits;

/// Inner value for the `SparseVector` and `SparseMatrix` type.
pub type SparseTuple<F> = (usize, F);
/// Shorthand for common type used in sparse data structures (vector, matrix, the carry matrix,
/// etc.)
pub type SparseTupleVec<F> = Vec<SparseTuple<F>>;
