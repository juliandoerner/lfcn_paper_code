"""Gmsh geometry for the 1D interval, split and tagged at the midpoint."""

import gmsh
from mpi4py import MPI

from enum import IntEnum
from dataclasses import dataclass

class MESH_ID(IntEnum):
    """Physical group tags written into the gmsh model."""

    MATERIAL_LEFT   = 1
    MATERIAL_RIGHT  = 2
    LEFT_BC         = 10
    RIGHT_BC        = 20
    MID_IF          = 30

@dataclass
class MeshGeometry:
    """Interval geometry and target cell sizes at the midpoint and the ends.

    With `gmsh = False` only `lc_far` is used, as the uniform cell count.
    """

    left_point: float = 0.0
    right_point: float = 1.0
    mid_point: float = 0.5
    lc_near: float = 0.01
    lc_far: float = 0.01
    gmsh: bool = False


def generate_mesh(mesh_data: MeshGeometry) -> gmsh.model:
    """Build the tagged, possibly graded 1D gmsh model for `mesh_data`."""

    if not gmsh.is_initialized():
        gmsh.initialize()
    
    name = "1D_line"
    model = gmsh.model()
    model.add(name)
    model.setCurrent(name)

    # Points
    p0 = model.geo.addPoint(mesh_data.left_point, 0.0, 0.0, mesh_data.lc_far)
    p1 = model.geo.addPoint(mesh_data.mid_point, 0.0, 0.0, mesh_data.lc_near)
    p2 = model.geo.addPoint(mesh_data.right_point, 0.0, 0.0, mesh_data.lc_far)

    # Lines (left and right of x_ref)
    line_left = model.geo.addLine(p0, p1)
    line_right = model.geo.addLine(p1, p2)

    # Physical groups for materials
    model.geo.synchronize()
    model.addPhysicalGroup(1, [line_left], tag=MESH_ID.MATERIAL_LEFT)  # Material 1
    model.setPhysicalName(1, MESH_ID.MATERIAL_LEFT, "MESH_ID.MATERIAL_LEFT")

    model.addPhysicalGroup(1, [line_right], tag=MESH_ID.MATERIAL_RIGHT)  # Material 2
    model.setPhysicalName(1, MESH_ID.MATERIAL_RIGHT, "MESH_ID.MATERIAL_RIGHT")

    # tag the mid point
    model.addPhysicalGroup(0, [p1], tag=MESH_ID.MID_IF)
    model.setPhysicalName(0, MESH_ID.MID_IF, "MESH_ID.MID_IF")
    # tag the left boundary
    model.addPhysicalGroup(0, [p0], tag=MESH_ID.LEFT_BC)
    model.setPhysicalName(0, MESH_ID.LEFT_BC, "MESH_ID.LEFT_BC")
    # tag the right boundary
    model.addPhysicalGroup(0, [p2], tag=MESH_ID.RIGHT_BC)
    model.setPhysicalName(0, MESH_ID.RIGHT_BC, "MESH_ID.RIGHT_BC")


    # Generate 1D mesh
    model.mesh.generate(1)

    return model

if __name__ == '__main__':

    import dolfinx as dfx

    gmsh.initialize()

    model = generate_mesh(MeshGeometry())
    mesh_data = dfx.io.gmsh.model_to_mesh(model, MPI.COMM_WORLD, 0, 1)

    print("cell_tags.values: ", mesh_data.cell_tags.values)
    print("cell_tags.indices: ", mesh_data.cell_tags.indices)
    print("facet_tags.values: ", mesh_data.facet_tags.values)
    print("facet_tags.indices: ", mesh_data.facet_tags.indices)

    print("facet_tags.find(MID_IF): ", mesh_data.facet_tags.find(MESH_ID.MID_IF))
    
    gmsh.finalize()
