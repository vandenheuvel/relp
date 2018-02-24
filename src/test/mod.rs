/// The tests in this module are only ran when the `miplib_tests` feature is enabled as these tests
/// take a long time (minutes) to run.
#[cfg(all(test, feature="miplib_tests"))]
mod miplib;
