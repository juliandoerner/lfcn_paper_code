# LFCN time integration for the 2D thermo-elastic wave-heat coupling

Source code and a version-pinned environment for the two numerical studies of the
LFCN integrator in 2D.

The model problem lives on the unit square, with the wave part clamped (`v = 0`,
imposed weakly through the flux) and homogeneous Dirichlet data for the heat part
on all four edges:

```
rho d/dt v     = div sigma + alpha grad theta + rho f    (momentum)
    d/dt sigma = C eps(v)                                (constitutive law)
    d/dt theta = kappa lap(theta) + alpha div v + g      (heat)
```

`v` is the velocity, `sigma` the symmetric stress, `theta` the temperature and
`C eps = 2 mu eps + lambda tr(eps) I`. The wave part `(v, sigma)` is discretised
with DG_k and an upwind flux, the heat part `theta` with CG_l; the time stepping
is leapfrog for the wave part and Crank-Nicolson for the heat part.

The right-hand sides are manufactured:
[Data/ManufacturedSolution.py](Data/ManufacturedSolution.py) builds
`v = V(x) a(t)`, `sigma = C eps(V(x)) A(t)` (with `A' = a`) and
`theta = Th(x) c(t)`, and carries the `f` and `g` that make them exact solutions
of the continuous problem. The two spatial profiles are
[Data/TrigSolution.py](Data/TrigSolution.py) and
[Data/PolySolution.py](Data/PolySolution.py); both vanish on the boundary.

Errors are reported at the final time in the total L2 norm of the system,

```
sqrt( ||v - v_ex||_L2^2 + ||sigma - sigma_ex||_L2^2 + ||theta - theta_ex||_L2^2 )
```

## Environment

Everything runs in a container built on the official DOLFINx image, pinned by
manifest digest to v0.11.0:

- DOLFINx 0.11.0, PETSc/petsc4py 3.25.1, mpi4py 4.1.2
- basix 0.11.0, UFL 2026.1.0, FFCx 0.11.0
- numpy 2.4.6, matplotlib 3.10.9
- Ubuntu 24.04, Python 3.12.3

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


If a run is interrupted, the FFCx JIT cache or the MPICH shared-memory segments
can be left in a state that makes the next run fail ("JIT compilation timed out",
or a bus error). [clean_caches.sh](clean_caches.sh) clears both; run it while no
MPI job is active.

## Running the studies

The two experiments are run from the repository root and write their own output
directory. `NP` sets the number of MPI ranks (4 by default), further arguments
are forwarded to the underlying driver, which takes every parameter as a
command-line argument (see its `parse_args`).

```bash
./run_h_plot.sh             # -> output_h_plot/
NP=8 ./run_tau_plot.sh      # -> output_tau_plot/
```

Both use the material `rho = 1.1`, `mu = 2.32`, `lam = 1.42`, `kappa = 4.2`,
`alpha = 1` and the end time `T = 1`.

### run_h_plot.sh

Error against mesh size, at a fixed time step
([driver_mesh_experiment.py](driver_mesh_experiment.py)). The time profiles are
`a = 1`, `c = t`, for which LFCN is temporally exact, so the error is purely
spatial.

- profile: `V = (P, P)`, `Th = P` with `P = sin(2 pi x1) sin(2 pi x2)`
- meshes: 15 geometrically spaced target mesh sizes from 0.2 to 0.02, crossed
  triangles
- degrees: `k = l = p` for `p` in 1, 2, 3, and `k = l + 1 = p` for `p` in 2, 3
- `tau`: fixed at `cfl * nu_lim(p_max) * h_min / c_P` with `cfl = 0.8`.
  `nu_lim(p)` is the measured maximum Courant number of the leapfrog step: 0.118
  (p=1), 0.062 (p=2), 0.040 (p=3), 0.12/p beyond.


Artifacts in `output_h_plot/`:

- `data.csv` - CSV, column `h` plus one `total_k{k}_l{l}_w{wavenumber}` per curve
- `plot.png` - one curve per degree pair, log-log, with `h^{min(k,l)+1}` guides
  and the fitted EOC in the legend
- `call_string.txt` - the invocation the directory was produced with

### run_tau_plot.sh

Error against time step size, on ten meshes
([driver_timestep_experiment.py](driver_timestep_experiment.py)). `P` lies in
`Q_2` and is resolved exactly by the second-order ansatz spaces, so the error is
purely temporal.

- profile: `V = (P, P)`, `Th = P` with `P = x1(1-x1) x2(1-x2)`,
  `a = cos(omega t)`, `c = sin(eta t)`, `omega = eta = 2 pi`
- meshes: 10 geometrically spaced target mesh sizes from 0.2 to 0.02,
  quadrilaterals
- degree: `k = l = 2` in all three spaces
- `tau`: 20 geometrically spaced values from 1e-4 to 3e-2

Runs with an error above 1 count as unstable. They are not dropped but recorded
as 10 and drawn off the top of the clipped axis, so the CFL threshold shows up as
a vertical rise; after three of them in a row, the remaining larger `tau` of that
mesh are marked unstable without being computed.

Artifacts in `output_tau_plot/`:

- `data.csv` - CSV, column `tau` plus one `n{n}` per mesh, unstable entries as 10
- `curves.csv` - CSV, the `n`, `h`, `omega` and `eta` behind each column
- `plot.png` - one curve per mesh, log-log, y-axis clipped at 1, with a `tau^2`
  reference
