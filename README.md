[![Build Status](https://github.com/vandenheuvel/relp/actions/workflows/main.yml/badge.svg?branch=master)](https://github.com/vandenheuvel/relp/actions) [![codecov](https://codecov.io/gh/vandenheuvel/relp/branch/master/graph/badge.svg)](https://codecov.io/gh/vandenheuvel/relp)

# RELP: Rust Exact Linear Programming

A linear program solver written in Rust.

This project is a work in progress. Integer variables are not (yet) supported.

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
- [x] Smart matrix inverse maintenance
- [ ] Dual algorithm
- [ ] Integer variables through Gomory cuts
- [ ] A convenient API
- [ ] Floating point support
- [ ] Integration into branch and bound frameworks

## Usage

The binary can be compiled with `cargo build --bin relp --features="binaries"`. You can then call it like `target/release/relp tests/burkardt/problem_files/adlittle.mps`. If you would like to use the library, read through the `tests/` folder for examples.

