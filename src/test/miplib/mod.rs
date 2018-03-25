use std::path::{Path, PathBuf};


const PROBLEM_FILE_DIRECTORY: &str = "./src/test/miplib/";
fn get_test_file_path(name: &str) -> PathBuf {
    let test_directory = Path::new(PROBLEM_FILE_DIRECTORY);
    test_directory.join(name).with_extension("mps")
}

#[cfg(test)]
mod test;
