//! # Pivot rules
//!
//! Strategies for moving from basis to basis, whether primal or dual.
use crate::algorithm::simplex::data::{Artificial, NonArtificial, Tableau, TableauType};
use crate::algorithm::simplex::matrix_provider::MatrixProvider;
use crate::data::linear_algebra::vector::Vector;
use crate::data::number_types::traits::OrderedField;

/// Deciding how to pivot.
///
/// During the Simplex method, one needs to decide how to move from basic solution to basic
/// solution. The pivot rule describes that behavior.
///
/// Once the column has been selected for a primal pivot (or the row for a dual pivot), a row
/// (column) needs to be found. This decision is currently made independent of the strategy.
pub trait PivotRule<TT: TableauType> {
    fn new() -> Self;
    /// Column selection rule for the primal Simplex method.
    fn select_primal_pivot_column<OF: OrderedField, MP: MatrixProvider<OF>>(
        &mut self,
        tableau: &Tableau<OF, TT, MP>,
    ) -> Option<(usize, OF)>;
    /// Row selection rule for the dual Simplex method.
    fn select_dual_pivot_row<OF: OrderedField, MP: MatrixProvider<OF>>(
        &mut self,
        tableau: &Tableau<OF, TT, MP>,
    ) -> Option<(usize, OF)>;
}

/// Simply pivot on the first column, which has a negative relative cost.
///
/// The behavior is the same for both the primal and dual simplex method.
pub struct FirstProfitable;
impl PivotRule<Artificial> for FirstProfitable {
    fn new() -> Self {
        Self
    }

    fn select_primal_pivot_column<OF: OrderedField, MP: MatrixProvider<OF>>(
        &mut self,
        tableau: &Tableau<OF, Artificial, MP>,
    ) -> Option<(usize, OF)> {
        (tableau.nr_artificial_variables()..tableau.nr_columns())
            .filter(|column| !tableau.is_in_basis(column))
            .map(|column| (column, tableau.relative_cost(column)))
            .find(|&(_, cost)| cost < OF::additive_identity())
    }

    fn select_dual_pivot_row<OF: OrderedField, MP: MatrixProvider<OF>>(
        &mut self,
        _tableau: &Tableau<OF, Artificial, MP>,
    ) -> Option<(usize, OF)> {
        unimplemented!();
    }
}
impl PivotRule<NonArtificial> for FirstProfitable {
    fn new() -> Self {
        Self
    }

    fn select_primal_pivot_column<OF: OrderedField, MP: MatrixProvider<OF>>(
        &mut self,
        tableau: &Tableau<OF, NonArtificial, MP>,
    ) -> Option<(usize, OF)> {
        (0..tableau.nr_columns())
            .filter(|column| !tableau.is_in_basis(column))
            .map(|column| (column, tableau.relative_cost(column)))
            .find(|&(_, cost)| cost < OF::additive_identity())
    }

    fn select_dual_pivot_row<OF: OrderedField, MP: MatrixProvider<OF>>(
        &mut self,
        tableau: &Tableau<OF, NonArtificial, MP>,
    ) -> Option<(usize, OF)> {
        tableau.constraint_values()
            .iter_values()
            .enumerate()
            .find(|&(_, &constraint)| constraint < OF::additive_identity())
            .map(|(row, &constraint)| (row, constraint))
    }
}

// TODO
pub struct FirstProfitableWithMemory {
    last_selected: Option<usize>,
}
impl PivotRule<Artificial> for FirstProfitableWithMemory {
    fn new() -> Self {
        Self { last_selected: None }
    }

    fn select_primal_pivot_column<OF: OrderedField, MP: MatrixProvider<OF>>(
        &mut self,
        tableau: &Tableau<OF, Artificial, MP>,
    ) -> Option<(usize, OF)> {
        let to_consider = if let Some(last) = self.last_selected {
            (last..tableau.nr_columns())
                .chain(tableau.nr_artificial_variables()..last)
        } else {
            (tableau.nr_artificial_variables()..tableau.nr_columns()).chain(0..0)
        };

        let potential = to_consider
            .filter(|column| !tableau.is_in_basis(column))
            .map(|column| (column, tableau.relative_cost(column)))
            .find(|&(_, cost)| cost < OF::additive_identity());

        self.last_selected = potential.map(|(i, _)| i);
        potential
    }
    fn select_dual_pivot_row<OF: OrderedField, MP: MatrixProvider<OF>>(
        &mut self,
        _tableau: &Tableau<OF, Artificial, MP>,
    ) -> Option<(usize, OF)> {
        unimplemented!();
    }
}
impl PivotRule<NonArtificial> for FirstProfitableWithMemory {
    fn new() -> Self {
        Self { last_selected: None, }
    }

    fn select_primal_pivot_column<OF: OrderedField, MP: MatrixProvider<OF>>(
        &mut self,
        tableau: &Tableau<OF, NonArtificial, MP>,
    ) -> Option<(usize, OF)> {
        let to_consider = if let Some(last) = self.last_selected {
            (last..tableau.nr_columns())
                .chain(0..last)
        } else {
            (0..tableau.nr_columns()).chain(0..0)
        };

        let potential = to_consider
            .filter(|column| !tableau.is_in_basis(column))
            .map(|column| (column, tableau.relative_cost(column)))
            .find(|&(_, cost)| cost < OF::additive_identity());

        self.last_selected = potential.map(|(i, _)| i);
        potential
    }
    fn select_dual_pivot_row<OF: OrderedField, MP: MatrixProvider<OF>>(
        &mut self,
        _tableau: &Tableau<OF, NonArtificial, MP>,
    ) -> Option<(usize, OF)> {
        unimplemented!()
    }
}

#[cfg(test)]
mod test {
    use num::rational::Ratio;

    use crate::algorithm::simplex::data::{Artificial, NonArtificial};
    use crate::algorithm::simplex::strategy::pivot_rule::{FirstProfitable, PivotRule};
    use crate::data::linear_algebra::vector::SparseVector;
    use crate::data::linear_algebra::vector::test::TestVector;
    use crate::tests::problem_2;

    #[test]
    fn test_find_profitable_column() {
        let data = problem_2::matrix_data_form::<Ratio<i32>>();
        let artificial_tableau = problem_2::artificial_tableau_form(&data);
        let mut rule = <FirstProfitable as PivotRule<Artificial>>::new();
        if let Some((value, _)) = rule.select_primal_pivot_column(&artificial_tableau) {
            assert_eq!(value, 3);
        } else { assert!(false); }

        let tableau = problem_2::tableau_form(&data);
        let mut rule = <FirstProfitable as PivotRule<NonArtificial>>::new();
        assert_eq!(rule.select_primal_pivot_column(&tableau), None);
    }

    #[test]
    fn test_find_pivot_row() {
        let data = problem_2::matrix_data_form::<Ratio<i32>>();
        let artificial_tableau = problem_2::artificial_tableau_form(&data);
        let column = SparseVector::from_test_data(vec![3f64, 5f64, 2f64]);
        assert_eq!(artificial_tableau.select_primal_pivot_row(&column), Some(0));

        let column = SparseVector::from_test_data(vec![2f64, 1f64, 5f64]);
        assert_eq!(artificial_tableau.select_primal_pivot_row(&column), Some(0));

        let tableau = problem_2::tableau_form(&data);
        let column = SparseVector::from_test_data(vec![3f64, 2f64, -1f64]);
        assert_eq!(tableau.select_primal_pivot_row(&column), Some(0));

        let column = SparseVector::from_test_data(vec![2f64, -1f64, 3f64]);
        assert_eq!(tableau.select_primal_pivot_row(&column), Some(0));
    }
}
