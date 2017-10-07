/// A `Row` is either a cost row or has one of the three equation types.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum RowType {
    Cost,
    Equal,
    Greater,
    Less,
}

/// A variable is either continuous or integer.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum VariableType {
    Continuous,
    Integer,
}

/// An LP either has a finite optimum, is unbounded or has no basic feasible solution.
#[derive(Debug, Copy, Clone)]
pub enum LPCategory {
    FiniteOptimum(f64),
    Unbounded,
    Infeasible,
}