# rust-lp
A Linear Program solver written in Rust.

This is an implementation of the methods described in the second chapter of Papadimitriou's Combinatorial Optimization.

## Usage

```
extern crate rust_lp as lp;

use std::path::Path;

use lp::algorithm::simplex::logic::solve;
use lp::io::read;

fn main() {
    let path = Path::new("./path/to/mps/myprogram.mps");
    let result = read(path).unwrap();

    let general: GeneralForm = result.into();
    let canonical: CanonicalForm = general.into();
    let data = MatrixData::from(canonical);

    let result = solve(&data).unwrap();

    println!("{:?}", result.1);
}
```
