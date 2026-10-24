# Nuclear Pasta: Periodic Geometry and Checkpointing

**Status: M1 implementation in progress.**

This repository contains my implementation of the first milestone of a nuclear-pasta simulation project, developed under the guidance of **Dr. Soham Pal at the University of Illinois Urbana-Champaign**.

The current goal is to establish a correct C++ CPU implementation of periodic geometry and checkpointing. This will provide a reproducible starting point and a correctness reference for later CUDA development.

## Current milestone

M1 focuses on:

- Representing particle positions, velocities, and species using separate arrays.
- Deriving the cubic box length from particle count and number density.
- Wrapping positions and calculating minimum-image separations under periodic boundaries.
- Validating state consistency and the cutoff condition.
- Saving and restoring particle data, metadata, and random-number-generator state.

Implementation and validation are ongoing. The checks below describe the acceptance targets for this milestone.

## Validation targets

- Face, edge, corner, and multiple-box cases agree with hand calculations.
- Every periodic pair distance is at most `L * sqrt(3) / 2`, within the stated numerical tolerance.
- Positions remain in `[0, L)` and the cutoff satisfies `0 < r_c <= L / 2`.
- A checkpoint restores floating-point values bit-for-bit, along with species, metadata, and the complete RNG state.

Lengths are expressed in fm, number density in fm^-3, time in fm/c, and velocities in units of c. The box length is derived as `L = cbrt(N / density)`.

## Build and run

Requirements: a C++20 compiler and Make. The development Makefile enables AddressSanitizer.

From the repository root:

```bash
make exec CXX=g++
./bin/exec
```

## Repository organization

| Path | Purpose |
| --- | --- |
| `src/` | C++ implementation files |
| `includes/` | Header files |
| `bin/` | Generated executables |
| `Makefile` | Build targets and compiler options |

Generated checkpoints are intended to live in `checkpoints/`. Relative checkpoint paths are resolved from the program's working directory. Small, fixed checkpoint fixtures can be tracked with tests when those tests are added.

## License

Licensed under the [MIT License](LICENSE).