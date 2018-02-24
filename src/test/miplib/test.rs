use std::path::Path;
use io::read;
use algorithm::simplex::logic::solve;
use test::miplib::get_test_file_path;

#[test]
fn test_50v() {
    /// Testing problem 50v-10
    let name = String::from("50v-10");
    let path = get_test_file_path(&name);
    let result = read(&path).ok().unwrap();

    let mut general = result.to_general_lp();
    let canonical = general.to_canonical();

    assert_approx_eq!(solve(&canonical).ok().unwrap().1, 2879.065687f64);
}
