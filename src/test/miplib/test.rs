use algorithm::simplex::logic::solve;
use algorithm::simplex::tableau_provider::matrix_data::MatrixData;
use data::linear_program::canonical_form::CanonicalForm;
use data::linear_program::general_form::GeneralForm;
use io::import;
use test::miplib::get_test_file_path;

#[test]
/// Testing problem 50v-10
fn test_50v() {
    let name = String::from("50v-10");
    let path = get_test_file_path(&name);
    let result = import(&path).unwrap();

    let general: GeneralForm = result.into();
    let canonical: CanonicalForm = general.into();
    let data = MatrixData::from(canonical);

    let result = solve(&data);
    let result = result.ok().unwrap();

    assert_approx_eq!(result.1, 2879.065687f64);
}
