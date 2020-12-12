[![Build Status](https://travis-ci.com/vandenheuvel/rust-lp.svg?branch=master)](https://travis-ci.com/vandenheuvel/rust-lp) [![codecov](https://codecov.io/gh/vandenheuvel/rust-lp/branch/master/graph/badge.svg)](https://codecov.io/gh/vandenheuvel/rust-lp)

# rust-lp
A linear program solver written in Rust.

This project is a work in progress. Integer variables are not supported.

## Priorities

- Correctness and safety 
- Performance
- Maintainability and flexibility

## Roadmap

- [x] Simplex algorithm
- [x] Lazy column generation
- [x] Arbitrary ordered fields
- [x] Arbitrary precision
- [x] Presolving framework
- [x] Reading MPS files
- [ ] Smart matrix inverse maintenance
- [ ] Floating point support
- [ ] A convenient API
- [ ] Dual algorithm
- [ ] Integration into branch and bound frameworks

## Usage

The binary can be compiled with `cargo build --bin rust-lp --features="binaries"`. You can then call it like `target/release/rust-lp tests/burkardt/problem_files/adlittle.mps`. If you would like to use the library, read through the `tests/` folder for examples.
