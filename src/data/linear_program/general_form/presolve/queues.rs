//! # Work lists
//!
//! A rule is not tried everywhere, it is tried where something changed. Each rule has a queue of
//! places worth trying, which other rules add to as they go.
use std::collections::VecDeque;
use std::collections::HashSet;
use std::hash::Hash;

use crate::data::linear_program::elements::BoundDirection;

/// A queue that holds each item at most once.
///
/// Rules enqueue eagerly and often redundantly, so keeping duplicates out here is what stops the
/// work lists from growing faster than they are drained.
#[derive(Clone, Debug)]
pub struct Queue<T> {
    order: VecDeque<T>,
    present: HashSet<T>,
}

impl<T> Default for Queue<T> {
    fn default() -> Self {
        Self {
            order: VecDeque::new(),
            present: HashSet::new(),
        }
    }
}

impl<T: Copy + Eq + Hash> Queue<T> {
    pub fn push(&mut self, item: T) {
        if self.present.insert(item) {
            self.order.push_back(item);
        }
    }
    pub fn pop(&mut self) -> Option<T> {
        let item = self.order.pop_front()?;
        self.present.remove(&item);
        Some(item)
    }
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
}

impl<T: Copy + Eq + Hash> Extend<T> for Queue<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, items: I) {
        for item in items {
            self.push(item);
        }
    }
}

/// One work list per rule that has one.
#[derive(Clone, Debug, Default)]
pub struct Queues {
    /// Columns whose bounds leave a single value.
    pub fixed_column: Queue<usize>,
    /// Rows with no coefficients left.
    pub empty_row: Queue<usize>,
    /// Columns with no coefficients left.
    pub empty_column: Queue<usize>,
    /// Rows with a single coefficient left.
    pub singleton_row: Queue<usize>,
    /// Columns with a single coefficient left and no cost.
    pub singleton_column: Queue<usize>,
    /// Columns with a single coefficient left and a cost.
    pub implied_free_singleton: Queue<usize>,
    /// Equality rows with two coefficients left.
    pub doubleton_equation: Queue<usize>,
    /// Rows, and the activity end, worth propagating from.
    pub activity: Queue<(usize, BoundDirection)>,
    /// Rows whose numbers changed, and so may have fractions to clear.
    pub row_scaling: Queue<usize>,
}

impl Queues {
    /// Whether every list has been drained.
    pub fn are_empty(&self) -> bool {
        self.fixed_column.is_empty()
            && self.empty_row.is_empty()
            && self.empty_column.is_empty()
            && self.singleton_row.is_empty()
            && self.singleton_column.is_empty()
            && self.implied_free_singleton.is_empty()
            && self.doubleton_equation.is_empty()
            && self.activity.is_empty()
            && self.row_scaling.is_empty()
    }
}
