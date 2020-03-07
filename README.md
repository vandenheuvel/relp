[![Build Status](https://travis-ci.com/vandenheuvel/rust-lp.svg?branch=master)](https://travis-ci.com/vandenheuvel/rust-lp) [![codecov](https://codecov.io/gh/vandenheuvel/rust-lp/branch/master/graph/badge.svg)](https://codecov.io/gh/vandenheuvel/rust-lp)



# rust-lp
A Linear Program solver written in Rust.

This is an implementation of the methods described in the second chapter of Papadimitriou's Combinatorial Optimization.

## Usage

```
extern crate rust_lp as lp;

use std::path::Path;

use lp::algorithm::simplex::logic::solve;
use lp::io::import;

fn main() {
    let path = Path::new("./path/to/mps/myprogram.mps");
    let result = import(path).unwrap();

    let general: GeneralForm = result.into();
    let canonical: CanonicalForm = general.into();
    let data = MatrixData::from(canonical);

    let result = solve(&data).unwrap();

    println!("{:?}", result.1);
}
```
