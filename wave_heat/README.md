# LFCN time integration for a coupled 1D wave/heat system

Source code and a version-pinned environment for the three numerical studies of
the LFCN integrator.

The model problem lives on the interval (0, 1) with homogeneous Dirichlet data
at both ends and couples a wave equation to a heat equation:

```
x1      = d/dt x2                                (wave velocity)
d/dt x1 = kappa * x2_xx - c_wave * theta_x       (wave momentum)
d/dt th = alpha * theta_xx - c_heat * x1_x       (heat)
```

The right-hand sides are manufactured: the analytic solutions in
[Data/Solutions.py](Data/Solutions.py) carry the `f` and `g` that make them
exact solutions of the continuous problem.
[Data/CalcDiffs.py](Data/CalcDiffs.py) is the sympy script they were derived
with.

Errors are reported in the energy norm of the system,

```
sqrt( ||x1 - x1_ex||_L2^2 + |x2 - x2_ex|_H1^2 + ||theta - theta_ex||_L2^2 )
```

## Environment

Everything runs in a container built on the official DOLFINx image, pinned by
manifest digest to v0.11.0:

- DOLFINx 0.11.0, PETSc/petsc4py 3.25.1, mpi4py 4.1.2
- basix 0.11.0, UFL 2026.1.0, FFCx 0.11.0
- gmsh 4.15.2
- numpy 2.4.6, scipy 1.17.1, matplotlib 3.10.9, sympy 1.14.0
- Ubuntu 24.04, Python 3.12.3

The system is real valued, so the default real-scalar PETSc build of the image
is the right one - do **not** `source dolfinx-complex-mode`.

In VSCode, open the repository and choose "Reopen in Container"; the definition
is in [.devcontainer/](.devcontainer/).

Without VSCode, use the image directly:

```bash
docker run -it --rm -v "$PWD":/root/shared -w /root/shared \
    dolfinx/dolfinx@sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8 \
    bash

# inside the container:
pip install -r .devcontainer/requirements.txt
```

## Running the studies

Each of the three scripts is run from the repository root with no arguments and
writes its own output directory. They are serial; the parameter sets are not
sized for MPI. Parameters are set in each script's `main()`; the values below
are the ones the reported results were produced with. All runs are on uniform
meshes of (0, 1) with `kappa`, `alpha` and both coupling strengths at 1.0, and
integrate to `T = 1`.

```bash
python3 HPlot.py        # ~45 s   -> output_h_plot/
python3 TauPlot.py      # ~3 min  -> output_tau_plot/
python3 TraceCFL.py     # ~4 min  -> output_trace_cfl/
```

### HPlot.py

Error at `T = 1` against mesh size, at a fixed time step.

- `h`: 40 geometrically spaced values from 1e-1 to 1e-2
- `tau`: 1e-3
- degrees: two sweeps - wave 1, 2, 3, 4 with the heat space at the same degree,
  and wave 2, 3, 4 with the heat space one degree below
- exact solution: `MonExpSolution(3*pi, 4*pi)`

Artifacts in `output_h_plot/`, the `miss_matched_` prefix marking the second
sweep:

- `deg_{p}_data.txt`, `miss_matched_deg_{p}_data.txt` - CSV, columns `h,err`
- `error.png`, `miss_matched_error.png` - one curve per degree, log-log

### TauPlot.py

Largest error over all steps against time step size, on several meshes.

- `tau`: 40 geometrically spaced values from 1e-1 to 1e-3
- `h`: 8 geometrically spaced values from 5e-1 to 5e-3
- degrees: 2 in all three spaces
- exact solution: `ExpSolution(3.73*pi, 2.3*pi)`
- runs with an error above 10 are treated as unstable and dropped before
  plotting and before the order is reported

Artifacts in `output_tau_plot/`:

- `{h}_data.txt` - CSV, columns `tau,err`, one per mesh size
- `error_plot.png` - one curve per mesh size, log-log

The estimated order in `tau` is printed per mesh size.

### TraceCFL.py

Largest stable time step against mesh size, found by bisection.

- `h`: 20 linearly spaced values from 1e-2 to 1e-3
- degrees: 1 in all three spaces
- bracket: `tau = h/10` (stable) to `tau = h` (unstable), bisected until the
  bracket is narrower than 1e-5
- stability test: energy-norm distance to a run of the same `tau` on a one-cell
  P2 reference mesh, against the threshold `eps = 1e-1`
- exact solution: `ExpSolution(3.73*pi, 2.3*pi)`

Artifacts in `output_trace_cfl/`:

- `cfl_data_deg_{deg}.txt` - CSV, columns `h,tau`
- `cfl_plot.png` - critical `tau` against `h`

The order in `h` and the mean slope are printed at the end.

