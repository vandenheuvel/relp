[![crate](https://img.shields.io/crates/v/relp.svg)](https://crates.io/crates/relp)
[![documentation](https://docs.rs/relp/badge.svg)](https://docs.rs/relp)
[![build status](https://github.com/vandenheuvel/relp/actions/workflows/main.yml/badge.svg?branch=master)](https://github.com/vandenheuvel/relp) [![codecov](https://codecov.io/gh/vandenheuvel/relp/branch/master/graph/badge.svg)](https://codecov.io/gh/vandenheuvel/relp)

# RELP: Rust Exact Linear Programming

A framework to describe and solve linear programs exactly, written in Rust. [Linear programming](https://en.wikipedia.org/wiki/Linear_programming) is a modelling technique to solve the wide range of optimization problems that can be represented by linear constraints and a linear target function.

This repository houses the library. Do you just want to solve a linear program stored in a file from the command line? See [`relp-bin`](https://github.com/vandenheuvel/relp-bin).

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
relp = "0.2.1"
```

You'll probably want to use the number types provided by the [`relp-num`](https://github.com/vandenheuvel/relp-num) crate, so include that dependency as well:

```toml
relp-num = "0.1.2"
```

You can now use both crates, for example to read a linear program in the classic [MPS format](https://en.wikipedia.org/wiki/MPS_(format)) in arbitrary precision:

```rust
fn main() {
    let path = std::path::Path::new("my_program.mps");
    let my_program = relp::io::import::<relp_num::RationalBig>(path);
}
```

## Features

### Describing LPs

Linear programs are described through a constraint matrix, right-hand side vector and cost function. You can provide these explicitly, but Relp also offers the [`MatrixProvider`](https://docs.rs/relp/0.2.1/relp/algorithm/two_phase/matrix_provider/trait.MatrixProvider.html) trait which can be implemented to represent your problem. You can find an example implementation of this trait for the shortest path and max flow problems in [the `examples/` directory](https://github.com/vandenheuvel/relp/tree/master/examples). You can capture the specific properties of your problem by implementing additional traits, such as the [`PartialInitialBasis`](https://docs.rs/relp/0.2.1/relp/algorithm/two_phase/phase_one/trait.PartialInitialBasis.html) trait in case you know a partial initial basis for your problem.

### Solving LPs

Relp provides standard solve procedures for implementors of the aforementioned [`MatrixProvider`](https://docs.rs/relp/0.2.1/relp/algorithm/two_phase/matrix_provider/trait.MatrixProvider.html) trait through the [`SolveRelaxation`](https://docs.rs/relp/0.2.1/relp/algorithm/trait.SolveRelaxation.html) trait: simply call `problem.solve_relaxation()` with that trait in scope. When additional traits are implemented, will the problem-specific properties they represent be exploited by faster solve procedures, seemlessly hidden behind that same call.

For more advanced use cases, it is also possible to replace some of the solve default algorithms. You can, for example, implement your own [pivot rule](https://docs.rs/relp/0.2.1/relp/algorithm/two_phase/strategy/pivot_rule/trait.PivotRule.html) or [basis factorization](https://docs.rs/relp/0.2.1/relp/algorithm/two_phase/tableau/inverse_maintenance/carry/trait.BasisInverse.html).

## Roadmap

- [x] Two-phase simplex algorithm
- [x] Lazy column generation
- [x] LU decomposition
- [x] Steepest descent pricing
- [x] Presolving framework
- [x] Prescaling framework
- [ ] Integer variables through Gomory cuts
- [ ] Floating point support
- [ ] Branch and bound
