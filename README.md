# Nuclear Pasta: CPU Reference Simulation

**Status: M1–M3 CPU implementation passes local checks. ICRN GPU verification is still needed for G1.**

This repository contains my classical nuclear-pasta simulation, developed under the guidance of **Dr. Soham Pal at the University of Illinois Urbana-Champaign**.

The C++ CPU code is a correctness reference for later CUDA development.

## Implemented CPU milestones

- **M1:** Periodic box geometry, minimum-image separations, state validation, and lossless checkpoints.
- **M2:** A parameterized two-Gaussian plus screened-Coulomb potential, its analytic shifted force, and finite-difference checks for nn, np, and pp pairs.
- **M3:** Seeded initial conditions, NVE velocity Verlet, energy and momentum logging, checkpoint resume, and timestep/reversibility checks.

The CPU development program limits `N` to 1000. Its short cutoff at `N=200` is for correctness work, not for a pasta-phase claim.

## Validation

- `make tests` checks M1 geometry and checkpoint round trips, 60 M2 force gradients against central differences, cutoff continuity, Newton's third law, the screening-length reference table, velocity initialization, momentum, and bit-identical restart.
- `make tests-slow` runs the fixed-duration N=200 timestep sweep and 10,000-step forward/reverse integration checks, including a periodic-boundary crossing. It writes `checkpoints/m3_energy_scaling.csv`. The fit uses `dt <= 1`; the coarser `dt = 2` result remains in the CSV. Locally, the fitted slope was 1.984, momentum drift was below `10^-14` relative, and reversal error was below `10^-9` fm.
- `make plot-m3` reruns the slow check and renders the log-log sweep as `checkpoints/m3_energy_scaling.pdf`. This optional target requires LaTeX with pgfplots.

Lengths are in fm, density in fm^-3, energy and temperature in MeV, time in fm/c, and velocity in units of c. The box length is derived as `L = cbrt(N / density)`, with `cutoff <= L/2`.

The pair table selects nn, np, pn, or pp interactions. The reported potential and integrated force use the same shifted-force cutoff. `--lambda` accepts a screening length in fm or `tf` for the Thomas-Fermi value at the chosen density and proton fraction.

## Build and run

Requirements: a C++20 compiler and Make. The default development flags enable AddressSanitizer.

From the repository root:

```bash
make tests
make tests-slow
make plot-m3
```

`make`, `make exec`, and both test commands create `bin/` and `checkpoints/` when needed. Override the compiler with `make tests CXX=...`. Timings from the default AddressSanitizer and `-O0` build are not a performance baseline.

## Run and resume NVE

```bash
./bin/exec simulate --n 200 --density 0.05 --yp 0.3 \
  --temperature 1 --dt 0.5 --cutoff 7 --lambda tf \
  --seed 20261117 --steps 1000 --log-every 10 \
  --checkpoint checkpoints/run.txt --csv checkpoints/energy.csv

./bin/exec resume --input checkpoints/run.txt --steps 1000 \
  --checkpoint checkpoints/resumed.txt --csv checkpoints/resumed_energy.csv
```

The energy CSV records every model and run parameter, the step and time, kinetic and potential energy, total energy, and momentum. Each step uses one new CPU force evaluation. `PASTA_TEXT_V2` checkpoints store the full state, model, NVE mode, RNG seed and state, and code version. The CPU reader also accepts `M1_TEXT_V1` using its original fixed model defaults. The format may change when thermostat modes are added.

The V2 text layout is: magic string; `N` and proton count; density, derived `L`, and proton fraction; cutoff, timestep, time, step, and seed; `a b c width charge lambda mass initial_temperature`; quoted thermostat and code version; serialized `mt19937_64` state; six lines of `x y z vx vy vz` arrays; then species labels. Floating-point values use the classic locale and `max_digits10` precision. Bit-identical restart is checked on the same CPU build; cross-platform trajectories should be compared through short-run energies and later statistics.

## Repository organization

| Path | Purpose |
| --- | --- |
| `src/pasta_cpu.cpp` | CPU model, NVE simulation, checkpointing, and checks |
| `src/pasta_g1.cu` | Standalone G1 force prototype |
| `plots/m3_energy_scaling.tex` | Reproducible M3 energy-error plot |
| `bin/` | Generated executables |
| `checkpoints/` | Generated checkpoints and CSV output |
| `Makefile` | Build targets and compiler options |

The G1 checkpoint reader accepts both formats and the V2 screening length, while rejecting V2 model constants unsupported by its current kernel. CUDA compilation, GPU force agreement, and profiling remain for ICRN after transfer through GitHub. The local Mac tests exercise CPU paths only.

## License

Licensed under the [MIT License](LICENSE).
