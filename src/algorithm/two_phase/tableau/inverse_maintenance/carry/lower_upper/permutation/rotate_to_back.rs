//! # Rotate to back
//!
//! Move an element to the back, shifting the other elements a step.
use std::cmp::Ordering;
use std::fmt;

use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::Permutation;

/// Rotate a part of the range.
///
/// That means that one element (the only public field `index`) is moved to the back, while all
/// elements it "jumps over" are moved one step to the front. All indices lower than `index` will
/// not be changed.
#[derive(Eq, PartialEq, Clone, Debug)]
pub struct RotateToBack {
    /// Value at this index gets moved to the back.
    pub index: usize,
    /// Size of the permutation.
    ///
    /// `self.index` gets moved to `len - 1`.
    len: usize,
}

impl RotateToBack {
    /// Create a new instance.
    ///
    /// # Arguments
    ///
    /// * `index`: Permutation will move this index to the back.
    /// * `len`: Size of the permutation, `index` gets moved to `len - 1`.
    pub fn new(index: usize, len: usize) -> Self {
        debug_assert!(index < len);

        Self {
            index,
            len,
        }
    }
}

impl Permutation for RotateToBack {
    fn forward(&self, i: usize) -> usize {
        debug_assert!(i < self.len());

        match i.cmp(&self.index) {
            Ordering::Less => i,
            Ordering::Equal => self.len() - 1,
            Ordering::Greater => i - 1,
        }
    }

    fn backward(&self, i: usize) -> usize {
        debug_assert!(i < self.len());

        if i < self.index {
            i
        } else if i < self.len() - 1 {
            i + 1
        } else {
            // i == self.len - 1
            self.index
        }
    }

    fn forward_sorted<T>(&self, items: &mut [(usize, T)]) {
        debug_assert!(items.iter().all(|&(i, _)| i < self.len()));
        debug_assert!(items.windows(2).all(|w| w[0].0 < w[1].0));

        let has_pivot_row = items.binary_search_by_key(&self.index, |&(i, _)| i);

        match has_pivot_row {
            Ok(data_index) => {
                items[data_index].0 = self.len() - 1;
                for (i, _) in &mut items[(data_index + 1)..] {
                    *i -= 1;
                }
                items[data_index..].rotate_left(1);
            }
            Err(data_index) => {
                for (i, _) in &mut items[data_index..] {
                    *i -= 1;
                }
            }
        }

        debug_assert!(items.windows(2).all(|w| w[0].0 < w[1].0));
    }

    fn backward_sorted<T>(&self, items: &mut [(usize, T)]) {
        debug_assert!(items.iter().all(|&(i, _)| i < self.len()));
        debug_assert!(items.windows(2).all(|w| w[0].0 < w[1].0));

        if items.is_empty() {
            return;
        }

        let data_index = match items.binary_search_by_key(&self.index, |&(i, _)| i) {
            Ok(index) | Err(index) => index,
        };
        if data_index == items.len() {
            // No values in the range being rotated
            return;
        }

        let increment_start_index = if items.last().unwrap().0 == self.len() - 1 {
            items.last_mut().unwrap().0 = self.index;
            items[data_index..].rotate_right(1);
            data_index + 1
        } else {
            data_index
        };
        for (i, _) in &mut items[increment_start_index..] {
            *i += 1;
        }

        debug_assert!(items.windows(2).all(|w| w[0].0 < w[1].0));
    }

    fn len(&self) -> usize {
        self.len
    }
}

impl fmt::Display for RotateToBack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "index: {}", self.index)
    }
}

#[cfg(test)]
mod test {
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::Permutation;
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::rotate_to_back::RotateToBack;

    #[test]
    fn no_change() {
        let mut items = vec![(0, 0), (1, 1), (2, 2)];
        RotateToBack::new(2, 3).forward_sorted(&mut items);
        let expected = vec![(0, 0), (1, 1), (2, 2)];
        assert_eq!(items, expected);

        let mut items = vec![(0, 0), (1, 1), (2, 2)];
        RotateToBack::new(2, 3).backward_sorted(&mut items);
        let expected = vec![(0, 0), (1, 1), (2, 2)];
        assert_eq!(items, expected);
    }

    #[test]
    fn switch() {
        let mut items = vec![(0, 0), (1, 1), (2, 2)];
        RotateToBack::new(1, 3).forward_sorted(&mut items);
        let expected = vec![(0, 0), (1, 2), (2, 1)];
        assert_eq!(items, expected);

        let mut items = vec![(0, 0), (1, 1), (2, 2)];
        RotateToBack::new(1, 3).backward_sorted(&mut items);
        let expected = vec![(0, 0), (1, 2), (2, 1)];
        assert_eq!(items, expected);
    }

    #[test]
    fn switch_with_empty() {
        let mut items = vec![(0, 0), (1, 1), (3, 3)];
        RotateToBack::new(2, 4).forward_sorted(&mut items);
        let expected = vec![(0, 0), (1, 1), (2, 3)];
        assert_eq!(items, expected);

        let mut items = vec![(0, 0), (1, 1), (3, 3)];
        RotateToBack::new(2, 4).backward_sorted(&mut items);
        let expected = vec![(0, 0), (1, 1), (2, 3)];
        assert_eq!(items, expected);
    }

    #[test]
    fn rotate_with_empty_in_between() {
        let mut items = vec![(0, 0), (1, 1), (3, 3)];
        RotateToBack::new(1, 4).forward_sorted(&mut items);
        let expected = vec![(0, 0), (2, 3), (3, 1)];
        assert_eq!(items, expected);

        let mut items = vec![(0, 0), (1, 1), (3, 3)];
        RotateToBack::new(1, 4).backward_sorted(&mut items);
        let expected = vec![(0, 0), (1, 3), (2, 1)];
        assert_eq!(items, expected);
    }

    #[test]
    fn back_and_forth() {
        let n = 4;
        let original = (0..n).map(|i| (i, i)).collect::<Vec<_>>();
        let mut test = original.clone();

        let permutation = RotateToBack::new(0, n);
        permutation.forward_unsorted(&mut test);
        permutation.backward_unsorted(&mut test);
        assert_eq!(test, original);

        let permutation = RotateToBack::new(1, n);
        permutation.forward_unsorted(&mut test);
        permutation.backward_unsorted(&mut test);
        assert_eq!(test, original);

        let permutation = RotateToBack::new(3, n);
        permutation.forward_unsorted(&mut test);
        assert_eq!(test, original);
        permutation.backward_unsorted(&mut test);
        assert_eq!(test, original);
    }
}
