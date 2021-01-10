//! # Swapping
//!
//! Exchange two values.
use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::Permutation;
use std::cmp::Ordering;

/// Exchange two values.
pub struct Swap {
    indices: (usize, usize),
    len: usize,
}
impl Swap {
    pub fn new(indices: (usize, usize), len: usize) -> Self {
        debug_assert!(indices.0 < len && indices.1 < len);

        Self {
            indices,
            len,
        }
    }
}

impl Permutation for Swap {
    fn forward(&self, i: &mut usize) {
        debug_assert!(*i < self.len());

        if *i == self.indices.0 {
            *i = self.indices.1
        } else if *i == self.indices.1 {
            *i = self.indices.0
        }
    }

    fn backward(&self, i: &mut usize) {
        debug_assert!(*i < self.len());

        self.forward(i);
    }

    fn forward_sorted<T>(&self, items: &mut [(usize, T)]) {
        debug_assert!(items.iter().all(|&(i, _)| i < self.len()));
        debug_assert!(items.windows(2).all(|w| w[0].0 < w[1].0));

        let has_0 = items.binary_search_by_key(&self.indices.0, |&(i, _)| i);
        let has_1 = items.binary_search_by_key(&self.indices.1, |&(i, _)| i);

        let mut rotate = |found: usize, not_found: usize, new_index: usize| {
            items[found].0 = new_index;
            match found.cmp(&not_found) {
                Ordering::Less => items[found..not_found].rotate_left(1),
                Ordering::Equal => {},
                Ordering::Greater => items[not_found..=found].rotate_right(1),
            };
        };

        match (has_0, has_1) {
            (Err(_), Err(_)) => {}
            (Err(index_0), Ok(index_1)) => rotate(index_1, index_0, self.indices.0),
            (Ok(index_0), Err(index_1)) => rotate(index_0, index_1, self.indices.1),
            (Ok(index_0), Ok(index_1)) => {
                items.swap(index_0, index_1);
                items[index_0].0 = self.indices.0;
                items[index_1].0 = self.indices.1;
            }
        }

        debug_assert!(items.windows(2).all(|w| w[0].0 < w[1].0));
    }

    fn backward_sorted<T>(&self, items: &mut [(usize, T)]) {
        debug_assert!(items.iter().all(|&(i, _)| i < self.len()));
        debug_assert!(items.windows(2).all(|w| w[0].0 < w[1].0));

        self.forward_sorted(items);
        debug_assert!(items.windows(2).all(|w| w[0].0 < w[1].0));
    }

    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod test {
    use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::permutation::{SwapPermutation, Permutation};

    #[test]
    fn empty() {
        let mut items: Vec<(usize, usize)> = vec![];
        SwapPermutation::new((4, 2), 10).forward_sorted(&mut items);
        assert_eq!(items, vec![]);
    }

    #[test]
    fn empty_same() {
        let mut items: Vec<(usize, usize)> = vec![];
        SwapPermutation::new((5, 4), 10).forward_sorted(&mut items);
        assert_eq!(items, vec![]);
    }

    #[test]
    fn none_present() {
        let mut items = vec![(0, 1), (4, 5)];
        SwapPermutation::new((5, 2), 10).forward_sorted(&mut items);
        assert_eq!(items, vec![(0, 1), (4, 5)]);

        let mut items = vec![(1, 1), (4, 5)];
        SwapPermutation::new((5, 0), 10).forward_sorted(&mut items);
        assert_eq!(items, vec![(1, 1), (4, 5)]);

        let mut items = vec![(1, 1), (4, 5)];
        SwapPermutation::new((3, 2), 10).forward_sorted(&mut items);
        assert_eq!(items, vec![(1, 1), (4, 5)]);
    }

    #[test]
    fn one_present() {
        let mut items = vec![(1, 1), (4, 5)];
        SwapPermutation::new((3, 1), 10).forward_sorted(&mut items);
        assert_eq!(items, vec![(3, 1), (4, 5)]);

        let mut items = vec![(1, 1), (4, 5)];
        SwapPermutation::new((5, 1), 10).forward_sorted(&mut items);
        assert_eq!(items, vec![(4, 5), (5, 1)]);

        let mut items = vec![(1, 1), (4, 5)];
        SwapPermutation::new((4, 0), 10).forward_sorted(&mut items);
        assert_eq!(items, vec![(0, 5), (1, 1)]);

        let mut items = vec![(1, 1), (4, 5)];
        SwapPermutation::new((4, 3), 10).forward_sorted(&mut items);
        assert_eq!(items, vec![(1, 1), (3, 5)]);
    }

    #[test]
    fn both_present() {
        let mut items = vec![(0, 1), (4, 5)];
        SwapPermutation::new((4, 0), 10).forward_sorted(&mut items);
        assert_eq!(items, vec![(0, 5), (4, 1)]);

        let mut items = vec![(0, 1), (4, 5), (6, 10)];
        SwapPermutation::new((4, 0), 10).forward_sorted(&mut items);
        assert_eq!(items, vec![(0, 5), (4, 1), (6, 10)]);

        let mut items = vec![(0, 1), (4, 5), (6, 10)];
        SwapPermutation::new((6, 0), 10).forward_sorted(&mut items);
        assert_eq!(items, vec![(0, 10), (4, 5), (6, 1)]);

        let mut items = vec![(0, 1), (4, 5), (6, 10)];
        SwapPermutation::new((6, 4), 10).forward_sorted(&mut items);
        assert_eq!(items, vec![(0, 1), (4, 10), (6, 5)]);
    }

    #[test]
    fn one_present_multiple_between() {
        let mut items = vec![(1, 1), (2, 2), (3, 3), (6, 5)];
        SwapPermutation::new((5, 1), 10).forward_sorted(&mut items);
        assert_eq!(items, vec![(2, 2), (3, 3), (5, 1), (6, 5)]);

        let mut items = vec![(2, 2), (3, 3), (5, 5), (6, 6)];
        SwapPermutation::new((5, 1), 10).forward_sorted(&mut items);
        assert_eq!(items, vec![(1, 5), (2, 2), (3, 3), (6, 6)]);
    }
}
