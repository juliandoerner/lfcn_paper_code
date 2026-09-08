"""Time-loop iterators and mesh-size queries."""

import numpy as np

class TimeLoop:
    """Iterate over `(n, t_n)` with fixed step tau; also indexable and sliceable.

    Yields the time *before* each step, so after the loop the state lives one
    tau past the last yielded t_n.
    """

    def __init__(self, tau: float, t_start: float, t_end: float):
        self.tau = tau
        self.t_end = t_end
        self.t_start = t_start

        if t_end < t_start:
            raise ValueError("t_end < t_start is forbidden")

        self.num_steps = int((self.t_end - self.t_start) / self.tau) + 1

    def __len__(self):
        return self.num_steps

    def __getitem__(self, key):
        if isinstance(key, int):
            if key == -1:
                return self[self.num_steps-1]
            if key < 0 or key >= self.num_steps:
                raise KeyError(f"Invalid index {key}")
            ret_n = key
            ret_tn = self.t_start + self.tau * key
            if ret_tn > self.t_end:
                ret_tn = self.t_end
            return int(ret_n), ret_tn

        elif isinstance(key, slice):
            return [self[i] for i in range(*key.indices(self.num_steps))]

        else:
            raise TypeError(f"Invalid argument type: {type(key)}")

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

class EndTimeLoop(TimeLoop):
    """`TimeLoop` yielding `(n, t_n, tau_n)`, last step trimmed to land on t_end."""

    def __init__(self, tau: float, t_start: float, t_end: float):
        super().__init__(tau, t_start, t_end)
        self.parent_stopped = False

    def __len__(self):
        return super().__len__() + 1
    
    def __getitem__(self, key):
        if key < super().__len__():
            return *super().__getitem__(key), self.tau
        elif key == super().__len__():
            return (int(key), self.t_end, self.t_end - super()[-1][1])
        else:
            TypeError(f"Invalid argument type: {type(key)}") 
        
    def __iter__(self):
        self.parent_stopped = False
        super().__iter__()
        return self
    
    def __next__(self):
        if self.parent_stopped:
            raise StopIteration
        try:
            ret_super = super().__next__()
            return *ret_super, self.tau
        except StopIteration:
            self.parent_stopped = True
            if np.isclose(self.t_end - super().__iter__()[-1][1], 1e-9):
                raise StopIteration
            return super().__len__(), self.t_end, self.t_end - super().__iter__()[-1][1]

def apply_on_hs(mesh, func):
    """Apply `func` to the array of cell diameters of `mesh` (local + ghosts)."""
    tdim = mesh.topology.dim
    c_map = mesh.topology.index_map(tdim)
    num_cells_local = c_map.size_local + c_map.num_ghosts
    cells = np.arange(num_cells_local, dtype=np.int32)
    hs = mesh.h(tdim,cells)
    return func(hs)

def get_h_max(mesh):
    """Largest cell diameter."""
    return apply_on_hs(mesh, np.max)

def get_h_min(mesh):
    """Smallest cell diameter -- the h reported by HPlot.py."""
    return apply_on_hs(mesh, np.min)

def get_h_median(mesh):
    """Median cell diameter."""
    return apply_on_hs(mesh, np.median)

def get_h_mean(mesh):
    """Mean cell diameter."""
    return apply_on_hs(mesh, np.mean)

if __name__ == "__main__":

    print("TimeLoop(0.1, 0, 1)")
    for (n, tn) in TimeLoop(0.09, 0, 1):
        print(f"n = {n}, tn = {tn}")

    print("EndTimeLoop(0.1, 0, 1)")
    for (n, tn, tau) in EndTimeLoop(0.09, 0, 1):
        print(f"n = {n}, tn = {tn}, tau = {tau}")

