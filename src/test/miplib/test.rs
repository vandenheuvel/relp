use algorithm::simplex::logic::solve;
use algorithm::simplex::tableau_provider::matrix_data::MatrixData;
use data::linear_program::canonical_form::CanonicalForm;
use data::linear_program::general_form::GeneralForm;
use io::import;
use test::miplib::get_test_file_path;

macro_rules! generate_test {
    ($name:ident, $file:expr, $obj:expr) => {
        /// Testing problem $name
        #[test]
        fn $name() {
            let name = String::from($file);
            let path = get_test_file_path(&name);
            let result = import(&path).unwrap();

            let general: GeneralForm = result.into();
            let canonical: CanonicalForm = general.into();
            let data = MatrixData::from(canonical);

            let result = solve(&data);
            let result = result.ok().unwrap();

            assert_relative_eq!(result.1, $obj);
        }
    }
}

generate_test!(test_50v, "50v-10", 2879.065687f64);
