# Maxwell Interface

## Environment

Everything runs in a container built on the official deal.II image, pinned to
the `v9.5.0-jammy` tag:

- deal.II 9.5.0, GCC 11.4.0, clang 14.0.0
- CMake 3.22.1, Ninja 1.10.1
- FFTW 3.3.8, gmsh 4.8.4
- gdb 12.1, doxygen 1.9.1
- Ubuntu 22.04

The container definition is vendored in [.devcontainer/](.devcontainer/).


### TiMaxdG

[timaxdg/](timaxdg/) is **a shipped copy of TiMaxdG**, not a checkout of it.
TiMaxdG is a C++ library for discontinuous Galerkin simulations of the Maxwell
equations in 2D and 3D, built on deal.II.

- upstream repository: <https://git.scc.kit.edu/dg-maxwell/timaxdg>

### Reproduction Env

In VSCode, open the repository and choose "Reopen in Container". Without
VSCode, use the image directly:

```bash
docker run -it --rm -v "$PWD":/home/dealii/shared -w /home/dealii/shared \
    dealii/dealii:v9.5.0-jammy bash
```


## Building

Release is strongly recommended; the experiments are compute-heavy.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Binaries land in `build/src/` and write their output into the working directory
they are started from.

## Running the studies

Both drivers take no arguments; parameters are set hard in their `main()`. Each
sweep is threaded over a Boost asio pool - check `num_threads` before starting
one, it is sized for a large workstation.

```bash
cd build/src
./InterfaceSolution_Prototype
./InterfaceSolutionPolynomial_Prototype
```

Both write `degree_{p}/mesh_size_{h}/steps_{n}/error_steps.txt`, a tab
separated table of the L2 field error and the interface errors in `K` and `J`
over time. VTU output of the computed and exact fields is off during a sweep;
enable it with `set_vtu_output` / `set_exact_output` for a single configuration.

### InterfaceSolution_Prototype

Mesh convergence against the trigonometric solution. The bulk is source free
and everything is driven by the surface current on `F`. Sweeps 20 target mesh
sizes from 5e-2 to 5e-3, at degree 3 and 10000 time steps. The mesh is
distorted and bulk cells are randomly refined, which breaks the central-flux
parity a Cartesian mesh would otherwise have.

### InterfaceSolutionPolynomial_Prototype

Time convergence against the polynomial solution. Being piecewise polynomial in
space, it is represented exactly by a DG space of degree >= 2 on an affine
mesh, so the spatial error vanishes and only the time discretisation error is
left. The mesh is therefore kept Cartesian and undistorted while `dt` is swept
over 50 step counts from 10 to 10000, across 10 mesh sizes. Coarse `dt`
violates the CFL condition; those runs are detected, flagged `unstable` in the
error table and stopped early.
