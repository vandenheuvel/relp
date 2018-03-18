use std::path::Path;
use io::read;
use algorithm::simplex::logic::solve;
use test::miplib::get_test_file_path;
use data::linear_program::general_form::GeneralForm;

#[test]
fn test_50v() {
    /// Testing problem 50v-10
    let name = String::from("50v-10");
    let path = get_test_file_path(&name);
    let result = read(&path).ok().unwrap();

    let mut general: GeneralForm = result.into();
    let canonical = general.to_canonical();

    let result = solve(&canonical);
    let result = result.ok().unwrap();

    assert_approx_eq!(result.1, 2879.065687f64);
}
