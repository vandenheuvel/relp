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
    let result = read(path).ok().unwrap();

    let general = result.to_general_lp();
    let canonical = general.to_canonical();

    let result = solve(&canonical).ok().unwrap().1;
    println!("{:?}", result);
}
```
