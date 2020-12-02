//! # Basis initially completely from artificial variables
//!
//! When no initial pivots can be derived, this module is used. It is slightly quicker than the
//! partially artificial tableau kind.
use std::marker::PhantomData;

use crate::algorithm::two_phase::matrix_provider::MatrixProvider;
use crate::algorithm::two_phase::tableau::inverse_maintenance::InverseMaintenance;
use crate::algorithm::two_phase::tableau::kind::artificial::{Artificial, IdentityColumn};
use crate::algorithm::two_phase::tableau::kind::Kind;
use crate::algorithm::two_phase::tableau::Tableau;
use crate::data::linear_algebra::traits::SparseElementZero;
use crate::data::number_types::traits::{Field, FieldRef};

/// All variables are artificial.
pub struct Fully<'a, F, FZ, MP> {
    /// Values that can be referred to when unsized constants need to be returned.
    ///
    /// TODO(ARCHITECTURE): Replace with values that are Copy, or an enum?
    /// TODO(ARCHITECTURE): Rename the (single) method that uses these to shift the the relevant
    ///  value to be able to remove these fields.
    ONE: F,
    ZERO: F,
    /// Supplies data about the problem.
    ///
    /// This data doesn't change throughout the lifetime of this `Tableau`, and it is independent of
    /// the current basis as described by the `carry` and `basis_columns` attributes.
    provider: &'a MP,
    phantom_zero: PhantomData<FZ>,
}
impl<'a, F: 'static, FZ, MP> Kind<F, FZ> for Fully<'a, F, FZ, MP>
where
    F: Field,
    FZ: SparseElementZero<F>,
    MP: MatrixProvider<F, FZ, Column: IdentityColumn<F>>,
{
    type Column = MP::Column;

    fn initial_cost_value(&self, j: usize) -> &F {
        if j < self.nr_rows() {
            &self.ONE
        } else {
            &self.ZERO
        }
    }

    fn original_column(&self, j: usize) -> Self::Column {
        if j < self.nr_rows() {
            // TODO(ENHANCEMENT): Would it be possible to specialize the code where this identity
            //  column is used?
            <Self::Column as IdentityColumn<F>>::identity(j, self.nr_rows())
        } else {
            self.provider.column(j - self.nr_rows())
        }
    }

    fn nr_rows(&self) -> usize {
        self.provider.nr_rows()
    }

    fn nr_columns(&self) -> usize {
        self.provider.nr_columns() + self.nr_rows()
    }
}

impl<'provider, F: 'static, FZ, MP> Artificial<F, FZ> for Fully<'provider, F, FZ, MP>
where
    F: Field + 'provider,
    for<'r> &'r F: FieldRef<F>,
    FZ: SparseElementZero<F>,
    MP: MatrixProvider<F, FZ, Column: IdentityColumn<F>>,
{
    fn nr_artificial_variables(&self) -> usize {
        self.nr_rows()
    }

    fn pivot_row_from_artificial(&self, artificial_index: usize) -> usize {
        debug_assert!(artificial_index < self.nr_artificial_variables());

        artificial_index
    }
}

impl<'provider, F: 'static, FZ, IM, MP> Tableau<F, FZ, IM, Fully<'provider, F, FZ, MP>>
where
    F: Field + 'provider,
    for<'r> &'r F: FieldRef<F>,
    FZ: SparseElementZero<F>,
    IM: InverseMaintenance<F, FZ>,
    MP: MatrixProvider<F, FZ>,
{
    /// Create a `Tableau` augmented with artificial variables.
    ///
    /// The tableau is then in a basic feasible solution having only the artificial variables in the
    /// basis.
    ///
    /// # Arguments
    ///
    /// * `provider`: Provides the problem to find a basic feasible solution for.
    ///
    /// # Return value
    ///
    /// The tableau.
    pub(crate) fn new(provider: &'provider MP) -> Self {
        let m = provider.nr_rows();

        Tableau {
            inverse_maintainer: IM::create_for_fully_artificial(provider.constraint_values()),
            basis_indices: (0..m).collect(),
            basis_columns: (0..m).collect(),

            // TODO: Make a special `Artificial` variant with this trivial basis
            kind: Fully {
                ONE: F::one(),
                ZERO: F::zero(),

                provider,
                phantom_zero: PhantomData,
            },

            phantom: PhantomData,
        }
    }
}
