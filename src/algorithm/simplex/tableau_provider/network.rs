use data::linear_algebra::matrix::SparseMatrix;

pub struct MaxFlowNetwork  {
    adjacency_matrix: SparseMatrix,
    s: usize,
    t: usize,

}

impl MaxFlowNetwork {
    pub fn new(adjacency_matrix: SparseMatrix, s: usize, t: usize) -> MaxFlowNetwork {
        MaxFlowNetwork { adjacency_matrix, s, t, }
    }
}
