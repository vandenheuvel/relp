//! # The constraint matrix during presolving
//!
//! Presolving needs to walk both rows and columns, and it needs to remove and occasionally change
//! coefficients. This module provides a representation that supports all three.
use std::mem;
use std::ops::MulAssign;

use num_traits::Zero;

use crate::data::linear_algebra::SparseTupleVec;
use crate::data::linear_algebra::matrix::{ColumnMajor, SparseMatrix};
use crate::data::linear_algebra::traits::{SparseComparator, SparseElement};

/// Position of a coefficient in the arena.
type Element = u32;

/// One coefficient, as seen from a row or from a column.
///
/// In a row list, `minor` is a column index; in a column list, a row index.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Entry {
    minor: u32,
    element: Element,
}

/// The constraint matrix, in a form that can be walked and edited in both directions.
///
/// Each coefficient is stored once, in `values`. A row-major and a column-major list of positions
/// into that arena are maintained, both sorted by minor index, so a row or a column is walked in
/// index order and the twin entry of a coefficient is found by binary search.
///
/// Rows and columns are never renumbered: an index means the same thing for as long as the matrix
/// lives. Removal is eager, so every entry in every list refers to a coefficient that is still
/// part of the problem. That is what lets a walk borrow nothing but the matrix; a representation
/// that filtered lazily against a counter would have to borrow that counter too, and every caller
/// wanting to change something while walking would have to copy its way around the borrow.
#[derive(Clone, Debug)]
pub struct Matrix<F> {
    /// The coefficients. A slot that is not in use holds a zero and is listed in `free`.
    values: Vec<F>,
    /// Slots in `values` that are not in use.
    free: Vec<Element>,

    /// Per row, the coefficients in it, sorted by column index.
    rows: Vec<Vec<Entry>>,
    /// Per column, the coefficients in it, sorted by row index.
    columns: Vec<Vec<Entry>>,

    nonzeros: usize,
}

impl<F> Matrix<F> {
    /// Number of rows, including any that have no coefficients left.
    pub fn nr_rows(&self) -> usize {
        self.rows.len()
    }
    /// Number of columns, including any that have no coefficients left.
    pub fn nr_columns(&self) -> usize {
        self.columns.len()
    }
    /// Number of coefficients in the matrix.
    pub fn nonzeros(&self) -> usize {
        self.nonzeros
    }
    /// Number of coefficients left in a row.
    pub fn row_len(&self, row: usize) -> usize {
        self.rows[row].len()
    }
    /// Number of coefficients left in a column.
    pub fn column_len(&self, column: usize) -> usize {
        self.columns[column].len()
    }

    /// The coefficients of a row, by ascending column index.
    pub fn row(&self, row: usize) -> impl Iterator<Item = (usize, &F)> + '_ {
        self.rows[row]
            .iter()
            .map(move |entry| (entry.minor as usize, &self.values[entry.element as usize]))
    }
    /// The coefficients of a column, by ascending row index.
    pub fn column(&self, column: usize) -> impl Iterator<Item = (usize, &F)> + '_ {
        self.columns[column]
            .iter()
            .map(move |entry| (entry.minor as usize, &self.values[entry.element as usize]))
    }
    /// The columns a row has coefficients in, ascending.
    pub fn row_indices(&self, row: usize) -> impl Iterator<Item = usize> + '_ {
        self.rows[row].iter().map(|entry| entry.minor as usize)
    }
    /// The rows a column has coefficients in, ascending.
    pub fn column_indices(&self, column: usize) -> impl Iterator<Item = usize> + '_ {
        self.columns[column].iter().map(|entry| entry.minor as usize)
    }

    /// A single coefficient, if there is one.
    pub fn get(&self, row: usize, column: usize) -> Option<&F> {
        let (list, needle) = if self.rows[row].len() <= self.columns[column].len() {
            (&self.rows[row], column)
        } else {
            (&self.columns[column], row)
        };

        list.binary_search_by_key(&(needle as u32), |entry| entry.minor)
            .ok()
            .map(|position| &self.values[list[position].element as usize])
    }
}

impl<F> Matrix<F>
where
    for<'r> F: MulAssign<&'r F>,
{
    /// Multiply every coefficient of a row by a value that is not zero.
    ///
    /// Which coefficients a row has does not change, so neither list is touched: only the values
    /// the two of them point at.
    pub fn scale_row(&mut self, row: usize, factor: &F) {
        for position in 0..self.rows[row].len() {
            let element = self.rows[row][position].element as usize;
            self.values[element] *= factor;
        }
    }
}

impl<F: Zero> Matrix<F> {
    /// Build from the column major representation a `GeneralForm` holds, moving the coefficients.
    pub fn from_column_major(matrix: SparseMatrix<F, F, ColumnMajor>) -> Self
    where
        F: SparseElement<F>,
        F: SparseComparator,
    {
        let nr_rows = matrix.nr_rows();
        let nr_columns = matrix.nr_columns();

        let mut values = Vec::with_capacity(matrix.size());
        let mut rows = vec![Vec::new(); nr_rows];
        let mut columns = Vec::with_capacity(nr_columns);

        // Columns are visited in order and each is already sorted by row index, so the row lists
        // come out sorted by column index without a sort.
        for column in matrix.data.into_iter() {
            let mut entries = Vec::with_capacity(column.len());
            for (row, value) in column {
                let element = values.len() as Element;
                values.push(value);

                entries.push(Entry {
                    minor: row as u32,
                    element,
                });
                rows[row].push(Entry {
                    minor: columns.len() as u32,
                    element,
                });
            }
            columns.push(entries);
        }

        let nonzeros = values.len();
        Self {
            values,
            free: Vec::new(),
            rows,
            columns,
            nonzeros,
        }
    }

    /// Take a coefficient out of the matrix.
    ///
    /// Returns `None` when there was none.
    pub fn take(&mut self, row: usize, column: usize) -> Option<F> {
        let position = self.rows[row]
            .binary_search_by_key(&(column as u32), |entry| entry.minor)
            .ok()?;
        let element = self.rows[row].remove(position).element;

        let twin = self.columns[column]
            .binary_search_by_key(&(row as u32), |entry| entry.minor)
            .expect("the two directions describe the same matrix");
        self.columns[column].remove(twin);

        Some(self.release(element))
    }

    /// Take a whole row out of the matrix, leaving it empty.
    pub fn take_row(&mut self, row: usize) -> SparseTupleVec<F> {
        let entries = mem::take(&mut self.rows[row]);
        entries
            .into_iter()
            .map(|entry| {
                let column = entry.minor as usize;
                let twin = self.columns[column]
                    .binary_search_by_key(&(row as u32), |other| other.minor)
                    .expect("the two directions describe the same matrix");
                self.columns[column].remove(twin);

                (column, self.release(entry.element))
            })
            .collect()
    }

    /// Take a whole column out of the matrix, leaving it empty.
    pub fn take_column(&mut self, column: usize) -> SparseTupleVec<F> {
        let entries = mem::take(&mut self.columns[column]);
        entries
            .into_iter()
            .map(|entry| {
                let row = entry.minor as usize;
                let twin = self.rows[row]
                    .binary_search_by_key(&(column as u32), |other| other.minor)
                    .expect("the two directions describe the same matrix");
                self.rows[row].remove(twin);

                (row, self.release(entry.element))
            })
            .collect()
    }

    /// Set a coefficient, inserting or removing an entry as needed.
    ///
    /// A zero value removes the coefficient; the matrix never stores an explicit zero.
    ///
    /// Returns the value that was there before, if any.
    pub fn set(&mut self, row: usize, column: usize, value: F) -> Option<F> {
        if value.is_zero() {
            return self.take(row, column);
        }

        match self.rows[row].binary_search_by_key(&(column as u32), |entry| entry.minor) {
            Ok(position) => {
                let element = self.rows[row][position].element as usize;
                Some(mem::replace(&mut self.values[element], value))
            }
            Err(position) => {
                let element = self.acquire(value);
                self.rows[row].insert(
                    position,
                    Entry {
                        minor: column as u32,
                        element,
                    },
                );

                let twin = self.columns[column]
                    .binary_search_by_key(&(row as u32), |entry| entry.minor)
                    .expect_err("the two directions describe the same matrix");
                self.columns[column].insert(
                    twin,
                    Entry {
                        minor: row as u32,
                        element,
                    },
                );

                self.nonzeros += 1;
                None
            }
        }
    }

    /// Put a value in the arena, reusing a free slot when there is one.
    fn acquire(&mut self, value: F) -> Element {
        match self.free.pop() {
            Some(element) => {
                self.values[element as usize] = value;
                element
            }
            None => {
                self.values.push(value);
                (self.values.len() - 1) as Element
            }
        }
    }

    /// Take a value out of the arena and mark its slot reusable.
    fn release(&mut self, element: Element) -> F {
        self.free.push(element);
        self.nonzeros -= 1;
        mem::replace(&mut self.values[element as usize], F::zero())
    }
}

#[cfg(test)]
mod test;
