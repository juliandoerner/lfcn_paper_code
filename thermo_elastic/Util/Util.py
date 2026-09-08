
import numpy as np
from mpi4py import MPI

class TimeLoop:

    def __init__(self, tau: float, t_start: float, t_end: float):
        self.tau = tau
        self.t_end = t_end
        self.t_start = t_start

        if t_end < t_start:
            raise ValueError("t_end < t_start is forbidden")

        self.num_steps = int((self.t_end - self.t_start) / self.tau) + 1

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        self.n = 0
        self.tn = self.t_start
        return self

    def __next__(self):
        if self.n >= self.num_steps:
            raise StopIteration
        ret_n = self.n
        ret_tn = self.tn
        self.n += 1
        self.tn += self.tau
        return int(ret_n), ret_tn

def _owned_hs(mesh):
    tdim = mesh.topology.dim
    n_owned = mesh.topology.index_map(tdim).size_local
    return mesh.h(tdim, np.arange(n_owned, dtype=np.int32))

def get_h_max(mesh):
    hs = _owned_hs(mesh)
    local = float(np.max(hs)) if hs.size else 0.0
    return mesh.comm.allreduce(local, op=MPI.MAX)

def get_h_min(mesh):
    hs = _owned_hs(mesh)
    local = float(np.min(hs)) if hs.size else np.inf
    return mesh.comm.allreduce(local, op=MPI.MIN)
