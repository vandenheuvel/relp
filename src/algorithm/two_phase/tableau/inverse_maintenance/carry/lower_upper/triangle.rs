use std::marker::PhantomData;

struct Triangle<F, Ordering> {
    row_major: Vec<Vec<X<F>>>,
    column_major: Vec<Vec<Y>>,
    len: usize,

    ordering: PhantomData<Ordering>,
}

struct Lower;
struct Upper;

type LowerTriangle<F> = Triangle<F, Lower>;
type UpperTriangle<F> = Triangle<F, Upper>;

#[derive(Clone)]
struct X<F> {
    value: F,
    minor_index: usize,
    minor_data_index_in_other: usize,
}

#[derive(Clone)]
struct Y {
    minor_index: usize,
    minor_data_index_in_other: usize,
}
