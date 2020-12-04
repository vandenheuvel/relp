//! # Representing linear programs for rapid read access
//!
//! The Simplex method algorithms work on a tableau. Because this tableau is very sparse in
//! practice, we store in a matrix that describes the current basis together with the original
//! (also sparse) matrix data. This module contains structures that can provide a matrix.
use crate::data::linear_algebra::SparseTuple;
use crate::data::linear_algebra::vector::{Dense, Sparse as SparseVector};
use crate::data::linear_program::elements::BoundDirection;
use crate::data::number_types::traits::Field;

pub mod matrix_data;
pub mod filter;
pub mod variable;

/// Abstract interface for a matrix and constraint vector.
///
/// This is the data of the "problem relative to the initial basis"; that is, nothing in
/// data structures implementing this trait determines a basis. The implementors of this trait
/// should be primarily read-only, with basis changes, the `Carry` fields of the `Tableau`
/// should change instead.
///
/// Note that a this trait doesn't have to be implemented by a (sparse) matrix data structure per
/// se; it could also be implemented by a graph, which lets itself be represented by data in a
/// matrix.
/// The indexing for the variables and constraints is as follows:
///
/// /                 || Vars of which we want a solution | Constraint slack vars | Bound slack vars |
/// ==================||==================================|=======================|==================|-----
/// Constraints       ||            constants             |       constants       |         0        || b |
/// ------------------||----------------------------------|-----------------------|------------------||---|
///                   ||                                  |                       | +/- 1            |
/// Bound constraints ||    constants (one 1 per row)     |           0           |       +/- 1      |
///                   ||                                  |                       |            +/- 1 |
/// --------------------------------------------------------------------------------------------------
pub trait MatrixProvider {
    /// Type used to represent a column of the matrix.
    ///
    /// TODO(ARCHITECTURE): When GATs are working, cloning can be avoided in some implementations,
    ///  such as the ones that explicitly store the column data, by giving this associated type a
    ///  lifetime parameter. Keep an eye on https://github.com/rust-lang/rust/issues/44265.
    /// TODO(ARCHITECTURE): When specializing on the generic arguments of trait methods is possible,
    ///  the columns no longer need to be ordered necessarily and the bound can be removed here.
    type Column: Column + OrderedColumn;

    /// Column of the problem.
    ///
    /// # Arguments
    ///
    /// * `j`: Column index.
    ///
    /// # Return value
    ///
    /// A sparse vector.
    fn column(&self, j: usize) -> Self::Column;

    /// Cost of a variable.
    ///
    /// # Arguments
    ///
    /// * `j`: Column index.
    ///
    /// # Return value
    ///
    /// Cost value.
    fn cost_value(&self, j: usize) -> &<Self::Column as Column>::F;

    /// Constraint values.
    ///
    /// Note: constraint values of both the constraints and bounds. Lengths should be
    /// `self.nr_rows()`.
    ///
    /// TODO(OPTIMIZATION): Can this clone be avoided?
    ///
    /// # Return value
    ///
    /// A dense vector of constraint values, often called `b` in mathematical notation.
    fn constraint_values(&self) -> Dense<<Self::Column as Column>::F>;

    /// Index of the row of a virtual bound, if any.
    ///
    /// TODO(ARCHITECTURE): Currently, the return value is a row index. Make this relative to
    ///  `self.nr_constraints`?
    ///
    /// # Arguments
    ///
    /// * `j`: Variable index for the bound, if it exists.
    /// * `bound_type`: Whether it concerns a lower or upper bound.
    ///
    /// # Return value
    ///
    /// The index of the row in which the bound is virtually represented, if the bound exists.
    fn bound_row_index(&self, j: usize, bound_type: BoundDirection) -> Option<usize>;

    /// The number of constraints in the problem. This excludes simple variable bounds.
    fn nr_constraints(&self) -> usize;

    /// The number of simple variable bounds in the problem. This excludes more complicated
    /// constraints.
    fn nr_bounds(&self) -> usize;

    /// The total number of rows in the provided virtual matrix.
    fn nr_rows(&self) -> usize {
        self.nr_constraints() + self.nr_bounds()
    }

    /// The total number of columns in the provided virtual matrix. This does not include artificial
    /// variables; those are virtually represented by the `Artificial` `TableauType`.
    fn nr_columns(&self) -> usize;

    /// Reconstruct a solution.
    ///
    /// Not all variables that a provider presents to the solution algorithms might be relevant for
    /// the final solution. Free variables that are split, for example, could be recombined here.
    ///
    /// # Arguments
    ///
    /// * `column_values`: A solution for each of the variables that this provider presents.
    ///
    /// # Return value
    ///
    /// A solution that might be smaller than the number of variables in this problem.
    fn reconstruct_solution(
        &self,
        column_values: SparseVector<<Self::Column as Column>::F, <Self::Column as Column>::F>,
    ) -> SparseVector<<Self::Column as Column>::F, <Self::Column as Column>::F>;
}

/// Columns represent part of a (virtual) data matrix.
///
/// This column is sparse.
///
/// A column might be expensive to compute. It can store computed values and once GATs work
/// (better), also references to items stored in the matrix provider that yields instances of this
/// trait.
///
/// It can't necessarily be iterated over directly. That needs to happen many times, for example
/// when computing many inner products with a data matrix. The trait has an associated type to be
/// used for iteration, that should be cheaply cloneable and probably not store any values itself.
/// Rather, it should describe how this column should be iterated over.
// TODO(ARCHITECTURE): Once GATs work, consider giving this trait a lifetime parameter.
pub trait Column {
    /// Input data type.
    ///
    /// Items of this type get read and used in additions and multiplications often.
    // TODO(ENHANCEMENT): Don't work with a field type directly, but an `Into<F>` type to separate.
    type F: 'static + Field;

    /// Type of struct to iterate over this column.
    ///
    /// It should be somewhat cheaply cloneable and as such not be too large.
    ///
    /// Note that we use a Generic Associated Type (GAT) here, and that these are (currently) part
    /// of an unfinished feature. Keep an eye on https://github.com/rust-lang/rust/issues/44265 for
    /// stabilization and possible bugs.
    type Iter<'a>: Iterator<Item = &'a SparseTuple<Self::F>> + Clone;

    /// Derive the iterator object.
    ///
    /// Because this column might need to be iterated over many times, it doesn't consume the
    /// column but instead produces a struct that might keep references to this column.
    fn iter(&self) -> Self::Iter<'_>;

    /// Format an index of the column.
    ///
    /// Note that this index might not be explicitly stored due to the column being sparse.
    fn index_to_string(&self, i: usize) -> String;
}

/// Column that can be iterated over in-order.
///
/// This trait is simply a marker trait to be used in specialization.
///
/// TODO(ENHANCEMENT): At the time of writing, it is not possible to specialize the generic
///  arguments of trait methods. That is why this trait and the standard `Column` trait are
///  currently both needed.
///
// TODO(ARCHITECTURE): Once GATs work, consider giving this trait a lifetime parameter.
pub trait OrderedColumn: Column {}
