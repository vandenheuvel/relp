//! # Rotate a range
//!
//! Move an element to the back of a range, shifting the other elements in the range a step in the
//! other direction.
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::Permutation;
use std::collections::HashSet;

pub struct RotateLeft(Rotate);
pub struct RotateRight(Rotate);

struct Rotate {
    /// Start of the rotated range.
    ///
    /// Value at this index gets moved to the back.
    start: usize,
    /// End of the rotated range, inclusive.
    end: usize,
    /// Size of the permutation.
    len: usize,
}

impl Rotate {
    /// Create a new instance.
    ///
    /// The range length (`end - start`) must have value at least one, so we need `start < end`.
    pub fn new(start: usize, end: usize, len: usize) -> Self {
        debug_assert_ne!(len, 0);
        debug_assert!(start < len);
        debug_assert!(end < len);
        debug_assert!(start < end, "Range should have length at least one");

        Self {
            start,
            end,
            len,
        }
    }

    fn rotate_left(&self, i: &mut usize) {
        debug_assert!(*i < self.len);

        if *i < self.start {
            // Left of the range, do nothing
        } else if *i == self.start {
            *i = self.end - 1;
        } else if *i < self.end {
            *i -= 1;
        } else {
            // Right of the range, do nothing
        }
    }

    fn rotate_right(&self, i: &mut usize) {
        debug_assert!(*i < self.len);

        if *i < self.start {
            // Left of the range, do nothing
        } else if *i < self.end - 1 {
            *i += 1;
        } else if *i == self.end - 1 {
            *i = self.start;
        } else {
            // Right of the range, do nothing
        }
    }

    fn rotate_left_sorted<T>(&self, items: &mut [(usize, T)]) {
        self.debug_assert_is_valid_sparse_slice(items);

        let end_index = items.binary_search_by_key(&self.end, |&(i, _)| i).into_ok_or_err();

        let shift_range = match items[..end_index].binary_search_by_key(&self.start, |&(i, _)| i) {
            Ok(start_index) => {
                items[start_index].0 = self.end - 1;
                items[start_index..end_index].rotate_left(1);

                start_index..(end_index - 1)
            }
            Err(start_index) => start_index..end_index,
        };

        for (i, _) in &mut items[shift_range] {
            *i -= 1;
        }

        self.debug_assert_is_valid_sparse_slice(items);
    }

    fn rotate_right_sorted<T>(&self, items: &mut [(usize, T)]) {
        self.debug_assert_is_valid_sparse_slice(items);

        let start_index = items.binary_search_by_key(&self.start, |&(i, _)| i).into_ok_or_err();
        let from_start = &mut items[start_index..];
        let shift_range = match from_start.binary_search_by_key(&(self.end - 1), |&(i, _)| i) {
            Ok(relative_end_index) => {
                from_start[relative_end_index].0 = self.start;
                from_start[..=relative_end_index].rotate_right(1);

                1..(relative_end_index + 1)
            }
            Err(relative_end_index) => 0..relative_end_index
        };

        for (i, _) in &mut from_start[shift_range] {
            *i += 1;
        }

        self.debug_assert_is_valid_sparse_slice(items);
    }

    fn debug_assert_is_valid_sparse_slice<T>(&self, items: &[(usize, T)]) {
        debug_assert!(items.iter().map(|&(i, _)| i).is_sorted());
        debug_assert!(items.iter().map(|&(i, _)| i).max().map_or(true, |v| v < self.len));
        debug_assert_eq!(items.iter().map(|&(i, _)| i).collect::<HashSet<_>>().len(), items.len());
    }
}

impl RotateLeft {
    /// Create a new instance.
    ///
    /// The range length (`end - start`) must have value at least one, so we need `start < end`.
    pub fn new(start: usize, end: usize, len: usize) -> Self {
        Self(Rotate::new(start, end, len))
    }
}

impl Permutation for RotateLeft {
    fn forward(&self, i: &mut usize) {
        self.0.rotate_left(i);
    }

    fn backward(&self, i: &mut usize) {
        self.0.rotate_right(i);
    }

    fn forward_sorted<T>(&self, items: &mut [(usize, T)]) {
        self.0.rotate_left_sorted(items)
    }

    fn backward_sorted<T>(&self, items: &mut [(usize, T)]) {
        self.0.rotate_right_sorted(items)
    }

    fn len(&self) -> usize {
        self.0.len
    }
}

impl RotateRight {
    /// Create a new instance.
    ///
    /// The range length (`end - start`) must have value at least one, so we need `start < end`.
    pub fn new(start: usize, end: usize, len: usize) -> Self {
        Self(Rotate::new(start, end, len))
    }
}

impl Permutation for RotateRight {
    fn forward(&self, i: &mut usize) {
        self.0.rotate_right(i);
    }

    fn backward(&self, i: &mut usize) {
        self.0.rotate_left(i);
    }

    fn forward_sorted<T>(&self, items: &mut [(usize, T)]) {
        self.0.rotate_right_sorted(items)
    }

    fn backward_sorted<T>(&self, items: &mut [(usize, T)]) {
        self.0.rotate_left_sorted(items)
    }

    fn len(&self) -> usize {
        self.0.len
    }
}

#[cfg(test)]
mod test {
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::rotate::{RotateLeft, RotateRight};
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::Permutation;

    #[test]
    fn rotate_left() {
        let size = 5;
        let rotation = RotateLeft::new(1, 3, size);
        let mut tasks = (0..size).collect::<Vec<_>>();
        for task in &mut tasks {
            rotation.forward(task);
        }
        assert_eq!(tasks, [0, 2, 1, 3, 4]);
    }

    #[test]
    fn rotate_right() {
        let size = 5;
        let rotation = RotateLeft::new(1, 3, size);
        let mut tasks = (0..size).collect::<Vec<_>>();
        for task in &mut tasks {
            rotation.backward(task);
        }
        assert_eq!(tasks, [0, 2, 1, 3, 4]);
    }

    fn test_swap<T: Permutation>(rotation: T, start: usize, end: usize, len: usize) {
        // All present
        let mut task = (0..len).map(|i| (i, i)).collect::<Vec<_>>();
        rotation.forward_sorted(&mut task);
        assert_eq!(task, [(0, 0), (1, 2), (2, 1), (3, 3), (4, 4)]);

        // Start not present
        let mut task = (0..len).filter(|&i| i != start).map(|i| (i, i)).collect::<Vec<_>>();
        rotation.forward_sorted(&mut task);
        assert_eq!(task, [(0, 0), (1, 2), (3, 3), (4, 4)]);

        // End not present
        let mut task = (0..len).filter(|&i| i != end).map(|i| (i, i)).collect::<Vec<_>>();
        rotation.forward_sorted(&mut task);
        assert_eq!(task, [(0, 0), (1, 2), (2, 1), (4, 4)]);

        // Before end not present
        let mut task = (0..len).filter(|&i| i != end - 1).map(|i| (i, i)).collect::<Vec<_>>();
        rotation.forward_sorted(&mut task);
        assert_eq!(task, [(0, 0), (2, 1), (3, 3), (4, 4)]);

        // Start and end not present
        let mut task = (0..len)
            .filter(|&i| i != start).filter(|&i| i != end)
            .map(|i| (i, i)).collect::<Vec<_>>();
        rotation.forward_sorted(&mut task);
        assert_eq!(task, [(0, 0), (1, 2), (4, 4)]);

        // Start and before end not present
        let mut task = (0..len)
            .filter(|&i| i != start).filter(|&i| i != end - 1)
            .map(|i| (i, i)).collect::<Vec<_>>();
        rotation.forward_sorted(&mut task);
        assert_eq!(task, [(0, 0), (3, 3), (4, 4)]);
    }

    #[test]
    fn rotate_left_sorted() {
        let (start, end, len) = (1, 3, 5);
        let rotation = RotateLeft::new(start, end, len);

        test_swap(rotation, start, end, len);
    }

    #[test]
    fn rotate_right_sorted() {
        let (start, end, len) = (1, 3, 5);
        let rotation = RotateRight::new(start, end, len);

        test_swap(rotation, start, end, len);
    }
}
