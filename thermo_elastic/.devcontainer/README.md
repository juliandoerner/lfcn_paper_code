# Development container

VSCode development container for this project, built on the official
[DOLFINx](https://hub.docker.com/r/dolfinx/dolfinx) image.

The base image is pinned by **manifest digest** (DOLFINx v0.11.0), not by the
`stable` tag. `stable` moves with every release and this code targets the 0.11
API, so an unpinned build would eventually break. The digest is the multi
architecture index for `v0.11.0`, so it resolves on both amd64 and arm64.

`dolfinx`, `mpi4py`, `petsc4py`, `ufl`, `basix` and `ffcx` come from the base
image; `requirements.txt` only pins the pure-Python additions (numpy,
matplotlib).
