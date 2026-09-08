// Grid generation for the split domain. Everywhere below the geometry is given
// by three points:
//
//   ---------p1-------p2
//   |        |        |
//   |        |        |     interface at x = p1[0]
//   |        |        |
//   p0-----------------
//
// The interface must stay on cell faces and the two sides must carry different
// material ids, otherwise the interface extraction cannot work.

#ifndef GENERATE_DOMAIN_H_
#define GENERATE_DOMAIN_H_

#include <tuple>

#include "deal.II/grid/tria.h"
#include "deal.II/grid/grid_generator.h"
#include "deal.II/grid/grid_tools.h"
#include "deal.II/base/point.h"

#include "Distort.h"

const std::tuple<const dealii::Point<2>, const dealii::Point<2>, const dealii::Point<2>>
    translate_coordinates_2d(
        const double a_1_minus, const double a_1_plus,
        const double a_2_minus, const double a_2_plus,
        const double zeta);

const std::tuple<const dealii::Point<3>, const dealii::Point<3>, const dealii::Point<3>>
    translate_coordinates_3d(
        const double a_1_minus, const double a_1_plus,
        const double a_2_minus, const double a_2_plus,
        const double a_3_minus, const double a_3_plus,
        const double zeta);

// side lengths l1, l2, l3 (and l4 for the height in 3D)
inline const std::tuple<double, double, double>
    get_side_ratio(const dealii::Point<2> p0, const dealii::Point<2> p1, const dealii::Point<2> p2)
{
    return {p1[0] - p0[0], p1[1] - p0[1], p2[0] - p1[0]};
}

inline const std::tuple<double, double, double, double>
    get_side_ratio(const dealii::Point<3> p0, const dealii::Point<3> p1, const dealii::Point<3> p2)
{
    return {p1[0] - p0[0], p1[1] - p0[1], p2[0] - p1[0], p1[3] - p0[3]};
}

// Distorts in x and y but pins the interface; the midpoint is not preserved.
// Keep distort_factor well below 0.5, and at 0.0 the mesh stays Cartesian.
void assemble_grid_xy_distortion(
    dealii::Triangulation<2>& triangulation,
    const dealii::Point<2>& p0,
    const dealii::Point<2>& p1,
    const dealii::Point<2>& p2,
    const double mesh_size,
    const double distort_factor);

void assemble_grid_xy_distortion(
    dealii::Triangulation<3>& triangulation,
    const dealii::Point<3>& p0,
    const dealii::Point<3>& p1,
    const dealii::Point<3>& p2,
    const double mesh_size,
    const double distort_factor);

// As above, but the midpoint survives the distortion.
void assemble_gird_preserve_midpoint(
    dealii::Triangulation<2>& triangulation,
    const dealii::Point<2>& p0,
    const dealii::Point<2>& p1,
    const dealii::Point<2>& p2,
    const double mesh_size,
    const double distort_factor);

void assemble_grid_via_triangles(
    dealii::Triangulation<2>& triangulation,
    const dealii::Point<2>& p0,
    const dealii::Point<2>& p1,
    const dealii::Point<2>& p2,
    const double mesh_size,
    const double distort_factor = 0.0);

// void assemble_grid_via_gmsh(
//     dealii::Triangulation<2>& triangulation,
//     const dealii::Point<2>& p0,
//     const dealii::Point<2>& p1,
//     const dealii::Point<2>& p2,
//     const double mesh_size,
//     const double mesh_size_interface);

// void assemble_grid_via_gmsh_via_triangles(
//     dealii::Triangulation<2>& triangulation,
//     const dealii::Point<2>& p0,
//     const dealii::Point<2>& p1,
//     const dealii::Point<2>& p2,
//     const double mesh_size,
//     const double mesh_size_interface);

void assemble_grid_via_8_triangles(
    dealii::Triangulation<2>& triangulation,
    const dealii::Point<2>& p0,
    const dealii::Point<2>& p1,
    const dealii::Point<2>& p2,
    const double mesh_size,
    const double distort_factor = 0.0);

// Refines towards the interface. Expects the cells to be marked already.
void refine_interface(
    dealii::Triangulation<2>& triangulation,
    const unsigned int refinements,
    const dealii::types::material_id material_left = 0,
    const dealii::types::material_id material_right = 1,
    const double x_mid = 1.);

// Material id by side of the interface at x = zeta.
template <int dim>
void mark_cells(
    dealii::Triangulation<dim>& triangulation,
    const dealii::types::material_id material_left = 0,
    const dealii::types::material_id material_right = 1,
    const double zeta = 1.) {
    for (const auto& cell : triangulation.active_cell_iterators())
    {
        const auto center = cell->center();
        if (center[0] < zeta)
        {
            cell->set_material_id(material_left);
        }
        else if (center[0] > zeta)
        {
            cell->set_material_id(material_right);
        }
        else
        {
            throw dealii::StandardExceptions::ExcInternalError("Cell center on Interface?");
        }
    }
}

void refine_down_to_mesh_size(
    dealii::Triangulation<2>& triangulation,
    const double mesh_size);

// Randomly refines bulk cells, leaving the interface conforming. Used to break
// the central-flux parity of a Cartesian mesh.
void random_refiner(
    dealii::Triangulation<2>& triangulation,
    const double probability);

#endif //GENERATE_DOMAIN_H_
