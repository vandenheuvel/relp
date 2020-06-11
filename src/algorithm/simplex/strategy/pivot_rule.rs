//! # Pivot rules
//!
//! Strategies for moving from basis to basis, whether primal or dual.
use crate::algorithm::simplex::tableau::kind::Kind;
use crate::algorithm::simplex::tableau::Tableau;
use crate::data::linear_algebra::SparseTuple;
use crate::data::linear_algebra::traits::SparseElementZero;
use crate::data::number_types::traits::{OrderedField, OrderedFieldRef};

/// Deciding how to pivot.
///
/// During the Simplex method, one needs to decide how to move from basic solution to basic
/// solution. The pivot rule describes that behavior.
///
/// Once the column has been selected for a primal pivot (or the row for a dual pivot), a row
/// (column) needs to be found. This decision is currently made independent of the strategy.
pub trait PivotRule {
    /// Create a new instance.
    fn new() -> Self;
    /// Column selection rule for the primal Simplex method.
    fn select_primal_pivot_column<OF, OFZ, K>(
        &mut self,
        tableau: &Tableau<OF, OFZ, K>,
    ) -> Option<SparseTuple<OF>>
        where
            OF: OrderedField,
            for<'r> &'r OF: OrderedFieldRef<OF>,
            OFZ: SparseElementZero<OF>,
            K: Kind<OF, OFZ>,
    ;
}

/// Simply pivot on the first column, which has a negative relative cost.
///
/// The behavior is the same for both the primal and dual simplex method.
pub struct FirstProfitable;
impl PivotRule for FirstProfitable {
    fn new() -> Self {
        Self
    }

    fn select_primal_pivot_column<OF, OFZ, K>(
        &mut self,
        tableau: &Tableau<OF, OFZ, K>,
    ) -> Option<SparseTuple<OF>>
    where
        OF: OrderedField,
        for<'r> &'r OF: OrderedFieldRef<OF>,
        OFZ: SparseElementZero<OF>,
        K: Kind<OF, OFZ>,
    {
        (tableau.kind.nr_artificial_variables()..tableau.nr_columns())
            .filter(|column| !tableau.is_in_basis(column))
            .map(|column| (column, tableau.relative_cost(column)))
            .find(|(_, cost)| cost < &OF::zero())
    }
}

/// Small modification w.r.t. the `FirstProfitable` rule; it starts the search from the last
/// column selected.
pub struct FirstProfitableWithMemory {
    last_selected: Option<usize>,
}
impl PivotRule for FirstProfitableWithMemory {
    fn new() -> Self {
        Self { last_selected: None }
    }

    fn select_primal_pivot_column<OF, OFZ, K>(
        &mut self,
        tableau: &Tableau<OF, OFZ, K>,
    ) -> Option<SparseTuple<OF>>
        where
            OF: OrderedField,
            for<'r> &'r OF: OrderedFieldRef<OF>,
            OFZ: SparseElementZero<OF>,
            K: Kind<OF, OFZ>,
    {
        let to_consider = if let Some(last) = self.last_selected {
            (last..tableau.nr_columns())
                .chain(tableau.kind.nr_artificial_variables()..last)
        } else {
            (tableau.kind.nr_artificial_variables()..tableau.nr_columns()).chain(0..0)
        };

        let potential = to_consider
            .filter(|column| !tableau.is_in_basis(column))
            .map(|column| (column, tableau.relative_cost(column)))
            .find(|(_, cost)| cost < &OF::zero());

        self.last_selected = potential.clone().map(|(i, _)| i);
        potential
    }
}

#[cfg(test)]
mod test {
    use crate::algorithm::simplex::strategy::pivot_rule::{FirstProfitable, PivotRule};
    use crate::data::linear_algebra::vector::Sparse as SparseVector;
    use crate::data::linear_algebra::vector::test::TestVector;
    use crate::tests::problem_2;

    #[test]
    fn find_profitable_column() {
        let (constraints, b) = problem_2::create_matrix_data_data();
        let matrix_data = problem_2::matrix_data_form(&constraints, &b);
        let artificial_tableau = problem_2::artificial_tableau_form(&matrix_data);
        let mut rule = <FirstProfitable as PivotRule>::new();
        if let Some((value, _)) = rule.select_primal_pivot_column(&artificial_tableau) {
            assert_eq!(value, 3);
        } else { assert!(false); }

        let tableau = problem_2::tableau_form(&matrix_data);
        let mut rule = <FirstProfitable as PivotRule>::new();
        assert_eq!(rule.select_primal_pivot_column(&tableau), None);
    }

    #[test]
    fn find_pivot_row() {
        let (constraints, b) = problem_2::create_matrix_data_data();
        let matrix_data = problem_2::matrix_data_form(&constraints, &b);
        let artificial_tableau = problem_2::artificial_tableau_form(&matrix_data);
        let column = SparseVector::from_test_data(vec![3, 5, 2]);
        assert_eq!(artificial_tableau.select_primal_pivot_row(&column), Some(0));

        let column = SparseVector::from_test_data(vec![2, 1, 5]);
        assert_eq!(artificial_tableau.select_primal_pivot_row(&column), Some(0));

        let tableau = problem_2::tableau_form(&matrix_data);
        let column = SparseVector::from_test_data(vec![3, 2, -1]);
        assert_eq!(tableau.select_primal_pivot_row(&column), Some(0));

        let column = SparseVector::from_test_data(vec![2, -1, 3]);
        assert_eq!(tableau.select_primal_pivot_row(&column), Some(0));
    }
}
