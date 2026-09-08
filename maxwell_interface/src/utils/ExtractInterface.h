// Builds a codim-1 triangulation from the volume faces where material_id
// jumps, together with the maps back into the volume mesh.

#ifndef EXTRACT_INTERFACE_H
#define EXTRACT_INTERFACE_H

#include <deal.II/base/point.h>
#include <deal.II/base/table.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out.h>

#include <vector>
#include <array>
#include <fstream>
#include <iostream>
#include <map>

template <template <int, int> class MeshType>
std::map<typename MeshType<1, 2>::cell_iterator,
         typename MeshType<2, 2>::face_iterator>
extract_interface_mesh_2d(const MeshType<2, 2> &volume_mesh,
                          MeshType<1, 2> &interface_mesh)
{
  constexpr int dim = 2;
  constexpr int spacedim = 2;
  const unsigned int interface_dim = dim - 1; // 1

  // The interface is the set of internal faces whose two adjacent cells carry
  // different material ids. Serial 2D meshes only.

  // level 0: (face iterator, face number)
  std::vector<
      std::pair<typename MeshType<dim, spacedim>::face_iterator, unsigned int>>
      temporary_mapping_level0;

  // keeps interface vertices unique
  std::vector<bool> touched(volume_mesh.get_triangulation().n_vertices(),
                            false);

  std::vector<dealii::CellData<interface_dim>> cells;
  dealii::SubCellData subcell_data; // unused in 2D case
  std::vector<dealii::Point<spacedim>> vertices;

  // volume vertex index -> interface vertex index
  std::map<unsigned int, unsigned int> map_vert_index;

  // swap_matrix[face_number][j] is where the j-th face vertex ends up in the
  // interface cell, which is what keeps the orientation consistent.
  dealii::Table<2, unsigned int> swap_matrix(
      dealii::GeometryInfo<spacedim>::faces_per_cell,
      dealii::GeometryInfo<interface_dim>::vertices_per_cell);

  for (unsigned int i1 = 0; i1 < dealii::GeometryInfo<spacedim>::faces_per_cell; ++i1)
    for (unsigned int i2 = 0;
         i2 < dealii::GeometryInfo<interface_dim>::vertices_per_cell;
         ++i2)
      swap_matrix[i1][i2] = i2;

  // faces 1 and 2 need swapping to get outward normals
  std::swap(swap_matrix[1][0], swap_matrix[1][1]);
  std::swap(swap_matrix[2][0], swap_matrix[2][1]);

  for (typename MeshType<dim, spacedim>::cell_iterator cell =
           volume_mesh.begin(0);
       cell != volume_mesh.end(0);
       ++cell)
    for (const unsigned int f : cell->reference_cell().face_indices())
    {
      const typename MeshType<dim, spacedim>::face_iterator face =
          cell->face(f);

      if (face->at_boundary())
        continue;

      const auto neighbor = cell->neighbor(f);

      if (cell->material_id() != neighbor->material_id())
      {
        // take each face from one side only
        if (cell->material_id() < neighbor->material_id())
          continue;

        dealii::CellData<interface_dim> c_data;
        for (const unsigned int j :
             dealii::GeometryInfo<interface_dim>::vertex_indices())
        {
          const unsigned int v_index = face->vertex_index(j);
          if (!touched[v_index])
          {
            vertices.push_back(face->vertex(j));
            map_vert_index[v_index] = vertices.size() - 1;
            touched[v_index] = true;
          }
          c_data.vertices[swap_matrix[f][j]] = map_vert_index[v_index];
        }

        c_data.material_id = cell->material_id();
        c_data.manifold_id = face->manifold_id();

        cells.emplace_back(std::move(c_data));
        temporary_mapping_level0.emplace_back(face, f);
      }
    }

  Assert(!cells.empty(), dealii::ExcMessage("No interface faces found."));

  const_cast<dealii::Triangulation<interface_dim, spacedim> &>(
      interface_mesh.get_triangulation())
      .create_triangulation(vertices, cells, subcell_data);

  // endpoints of the interface segments
  for (const auto &cell : interface_mesh.active_cell_iterators())
    for (unsigned int vertex = 0; vertex < 2; ++vertex)
      if (cell->face(vertex)->at_boundary())
        cell->face(vertex)->set_boundary_id(0);

  // Mirror the volume refinement so every interface cell keeps matching an
  // active volume face.
  std::vector<std::pair<
      const typename MeshType<interface_dim, spacedim>::cell_iterator,
      std::pair<typename MeshType<dim, spacedim>::face_iterator, unsigned int>>>
      temporary_map_boundary_cell_face;

  for (const auto &cell : interface_mesh.active_cell_iterators())
    temporary_map_boundary_cell_face.emplace_back(
        cell,
        temporary_mapping_level0.at(cell->index()));

  unsigned int index_cells_deepest_level = 0;

  while (true)
  {
    bool changed = false;
    std::vector<unsigned int> cells_refined;

    for (unsigned int cell_n = index_cells_deepest_level;
         cell_n < temporary_map_boundary_cell_face.size();
         ++cell_n)
    {
      const auto &face = temporary_map_boundary_cell_face[cell_n]
                             .second.first;

      if (face->has_children())
      {
        Assert(face->refinement_case() ==
                   dealii::RefinementCase<interface_dim>::isotropic_refinement,
               dealii::ExcNotImplemented());

        temporary_map_boundary_cell_face[cell_n].first->set_refine_flag();
        cells_refined.push_back(cell_n);
        changed = true;
      }
    }

    if (!changed)
      break;

    const_cast<dealii::Triangulation<interface_dim, spacedim> &>(
        interface_mesh.get_triangulation())
        .execute_coarsening_and_refinement();

    index_cells_deepest_level = temporary_map_boundary_cell_face.size();

    for (const auto refined_cell_n : cells_refined)
    {
      const auto refined_cell =
          temporary_map_boundary_cell_face[refined_cell_n].first;
      const auto refined_face =
          temporary_map_boundary_cell_face[refined_cell_n].second.first;
      const unsigned int refined_face_number =
          temporary_map_boundary_cell_face[refined_cell_n].second.second;

      for (unsigned int child_n = 0; child_n < refined_cell->n_children();
           ++child_n)
        temporary_map_boundary_cell_face.emplace_back(
            refined_cell->child(
                swap_matrix[refined_face_number][child_n]),
            std::make_pair(refined_face->child(child_n),
                           refined_face_number));
    }
  }

  std::map<typename MeshType<interface_dim, spacedim>::cell_iterator,
           typename MeshType<dim, spacedim>::face_iterator>
      interface_to_volume_mapping;

  for (const auto &entry : temporary_map_boundary_cell_face)
    interface_to_volume_mapping[entry.first] = entry.second.first;

  const auto attached_mids =
      interface_mesh.get_triangulation().get_manifold_ids();
  for (const auto i : volume_mesh.get_triangulation().get_manifold_ids())
    if (i != dealii::numbers::flat_manifold_id &&
        std::find(attached_mids.begin(), attached_mids.end(), i) ==
            attached_mids.end())
      const_cast<dealii::Triangulation<interface_dim, spacedim> &>(
          interface_mesh.get_triangulation())
          .set_manifold(i, dealii::FlatManifold<interface_dim, spacedim>());

  return interface_to_volume_mapping;
}

std::map<dealii::Triangulation<2>::face_iterator,
         std::pair<dealii::Triangulation<2>::active_cell_iterator,
                   dealii::Triangulation<2>::active_cell_iterator>>
build_face_to_cells_sorted_by_material(
    const dealii::Triangulation<2> &tria,
    const std::map<dealii::Triangulation<1, 2>::cell_iterator,
                   dealii::Triangulation<2>::face_iterator> &interface_to_face)
{
  using Tria = dealii::Triangulation<2>;
  using FaceIter = Tria::face_iterator;
  using CellIter = Tria::active_cell_iterator;
  using CellPair = std::pair<CellIter, CellIter>;

  std::map<FaceIter, CellPair> face_to_cells;

  auto find_cells_for_face = [&tria](const FaceIter &face)
      -> std::vector<CellIter>
  {
    std::vector<CellIter> result;

    // any vertex of the face is enough to reach both adjacent cells
    const unsigned int v_index = face->vertex_index(0);

    std::vector<CellIter> adjacent_cells =
        dealii::GridTools::find_cells_adjacent_to_vertex(tria, v_index);

    for (const auto &cell : adjacent_cells)
    {
      for (unsigned int f = 0; f < dealii::GeometryInfo<2>::faces_per_cell; ++f)
      {
        //     << "\t face->index(): " << face->index() << "," << face->has_children() << "\n";
        if (cell->face(f) == face)
        {
          result.push_back(cell);
          if (result.size() == 2)
            return result; // we found both sides
        }
      }
    }

    return result;
  };

  for (const auto &entry : interface_to_face)
  {
    const FaceIter face = entry.second;

    // the map also holds coarser-level faces, skip those
    if (face->has_children())
      continue;

    if (face_to_cells.find(face) != face_to_cells.end())
      continue;

    std::vector<CellIter> cells = find_cells_for_face(face);

    if (cells.empty())
    {
      AssertThrow(false, dealii::ExcMessage("No cell found for face on interface."));
    }

    CellIter c0, c1;

    if (cells.size() == 1)
    {
      // boundary face: only one adjacent cell
      c0 = cells[0];
      c1 = CellIter(); // invalid
    }
    else
    {
      // sort by material id: c0 is Omega_-, c1 is Omega_+
      CellIter a = cells[0];
      CellIter b = cells[1];

      if (a->material_id() <= b->material_id())
      {
        c0 = a;
        c1 = b;
      }
      else
      {
        c0 = b;
        c1 = a;
      }
    }

    face_to_cells[face] = std::make_pair(c0, c1);
  }

  return face_to_cells;
}

#endif //EXTRACT_INTERFACE_H