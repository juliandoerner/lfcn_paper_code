"""Unit-square meshes with the four edges tagged separately, so that boundary
conditions can be assigned per edge in the DG wave operator and the CG heat part.
"""

from enum import IntEnum
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI
import dolfinx as dfx


class MESH_ID(IntEnum):
    # facet (boundary) markers, one per edge of the square
    LEFT_BC = 10
    RIGHT_BC = 20
    BOTTOM_BC = 30
    TOP_BC = 40


@dataclass
class MeshGeometry:
    """Axis-aligned rectangle ``[x0, x1] x [y0, y1]``."""
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 1.0
    y1: float = 1.0
    n: int = 16                       # cells per direction
    cell_type: str = "triangle"      # "triangle" or "quadrilateral"
    diagonal: str = "crossed"        # triangle diagonal pattern


class MeshData:
    """Container bundling a mesh with its facet tags."""

    def __init__(self, mesh, facet_tags=None):
        self.mesh = mesh
        self.facet_tags = facet_tags


def _tag_square_facets(mesh, geom: MeshGeometry) -> dfx.mesh.MeshTags:
    """Tag the four boundary edges of the rectangle with ``MESH_ID`` values."""
    fdim = mesh.topology.dim - 1
    markers = {
        MESH_ID.LEFT_BC:   lambda x: np.isclose(x[0], geom.x0),
        MESH_ID.RIGHT_BC:  lambda x: np.isclose(x[0], geom.x1),
        MESH_ID.BOTTOM_BC: lambda x: np.isclose(x[1], geom.y0),
        MESH_ID.TOP_BC:    lambda x: np.isclose(x[1], geom.y1),
    }
    facet_indices, facet_markers = [], []
    for mid, marker in markers.items():
        facets = dfx.mesh.locate_entities_boundary(mesh, fdim, marker)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), int(mid), dtype=np.int32))

    facet_indices = np.concatenate(facet_indices).astype(np.int32)
    facet_markers = np.concatenate(facet_markers)
    sort = np.argsort(facet_indices)
    return dfx.mesh.meshtags(mesh, fdim, facet_indices[sort], facet_markers[sort])


def generate_square_mesh(geom: MeshGeometry, comm=MPI.COMM_WORLD) -> MeshData:
    """Rectangle mesh with the four edges tagged.

    ``GhostMode.shared_facet`` is required for the DG interior-facet (``dS``)
    integrals to be computable across process boundaries under MPI.
    """
    cell_type = {
        "triangle": dfx.mesh.CellType.triangle,
        "quadrilateral": dfx.mesh.CellType.quadrilateral,
    }[geom.cell_type]

    kwargs = {}
    if cell_type == dfx.mesh.CellType.triangle:
        kwargs["diagonal"] = {
            "right": dfx.mesh.DiagonalType.right,
            "left": dfx.mesh.DiagonalType.left,
            "crossed": dfx.mesh.DiagonalType.crossed,
        }[geom.diagonal]

    mesh = dfx.mesh.create_rectangle(
        comm,
        [np.array([geom.x0, geom.y0]), np.array([geom.x1, geom.y1])],
        [geom.n, geom.n],
        cell_type=cell_type,
        ghost_mode=dfx.mesh.GhostMode.shared_facet,
        **kwargs,
    )
    return MeshData(mesh, facet_tags=_tag_square_facets(mesh, geom))
