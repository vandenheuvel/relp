use crate::algorithm::two_phase::tableau::inverse_maintenance::carry::lower_upper::WorkSpace;
use crate::data::linear_algebra::SparseTuple;
use num_traits::Zero;
use crate::algorithm::two_phase::tableau::inverse_maintenance::ops;
use core::mem;

pub(in super) fn triangle_discover<'a, I: Iterator<Item=SparseTuple<G>>, F, G, T: 'a>(
    rhs: I,
    get_outgoing: impl Fn(usize) -> Option<&'a Vec<SparseTuple<T>>>,
    work: &mut WorkSpace<F>,
)
where
    F: From<G> + Zero,
{
    debug_assert!(work.marked.iter().all(|&v| v == false));
    debug_assert!(work.discovered.is_empty());

    for (mut column, value) in rhs {
        work.scattered[column] = value.into();

        if !work.marked[column] {
            work.marked[column] = true;

            let mut data_index = 0;
            'outer: loop {
                if let Some(outgoing) = get_outgoing(column) {
                    for step in data_index..outgoing.len() {
                        let (i, _) = outgoing[step];
                        if !work.marked[i] {
                            work.marked[i] = true;
                            work.scattered[i].set_zero();

                            let next_data_index = step + 1;
                            work.stack.push((column, next_data_index));

                            column = i;
                            data_index = 0;
                            continue 'outer;
                        }
                    }
                }

                work.discovered.push(column);

                match work.stack.pop() {
                    None => break,
                    Some((next_column, next_data_index)) => {
                        column = next_column;
                        data_index = next_data_index;
                    }
                }
            }
        }
    }

    debug_assert!(work.stack.is_empty());
}

pub(in super) fn triangle_process<'a, F: 'a>(
    get_outgoing: impl Fn(usize) -> Option<&'a Vec<SparseTuple<F>>>,
    divide_by_diagonal: impl Fn(F, usize) -> F,
    work: &mut WorkSpace<F>,
) -> Vec<SparseTuple<F>>
where
    F: ops::Field + ops::FieldHR,
{
    let result = work.discovered.drain(..).rev()
        .filter_map(|column| {
            work.marked[column] = false;

            if work.scattered[column].is_not_zero() {
                let result = divide_by_diagonal(mem::take(&mut work.scattered[column]), column);

                if let Some(outgoing) = get_outgoing(column) {
                    for (row, coefficient) in outgoing {
                        work.scattered[*row] -= &result * coefficient;
                    }
                }

                Some((column, result))
            } else {
                None
            }
        })
        .collect();

    debug_assert!(work.discovered.is_empty());
    debug_assert!(work.marked.iter().all(|&v| v == false));

    result
}