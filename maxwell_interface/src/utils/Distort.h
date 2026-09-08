// GridTools::distort_random, adapted so that the shift is restricted to a
// single coordinate direction and so that vertices on a material interface
// can be pinned along with the boundary.

#ifndef DISTORT_H_
#define DISTORT_H_

#include "deal.II/grid/tria.h"
#include "deal.II/grid/grid_tools.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

// Shifts every vertex randomly along distort_dim by up to `factor` times the
// shortest adjacent edge. keep_interface additionally pins vertices sitting on
// a material jump.
template <int dim, int spacedim>
void
distort_random(
  const double                  factor,
  const unsigned int distort_dim,
  dealii::Triangulation<dim, spacedim>& triangulation,
  const bool                    keep_boundary,
  const bool                    keep_interface,
  const unsigned int            seed)
{
  Assert(spacedim == dim, dealii::ExcNotImplemented());
  Assert((distort_dim <= dim), dealii::ExcNotImplemented());

  // Upper bound for the shortest adjacent edge, so the min below always wins.
  double almost_infinite_length = 0;
  for (typename dealii::Triangulation<dim, spacedim>::cell_iterator cell =
    triangulation.begin(0);
    cell != triangulation.end(0);
    ++cell)
    almost_infinite_length += cell->diameter();

  std::vector<double> minimal_length(triangulation.n_vertices(),
    almost_infinite_length);

  std::vector<bool> at_boundary(keep_boundary ? triangulation.n_vertices() :
    0,
    false);
  // shared triangulations need all vertices, not just the locally owned ones
  const bool is_parallel_shared =
    (dynamic_cast<
    dealii::parallel::shared::Triangulation<dim, spacedim> *>(
      &triangulation) != nullptr);
  for (const auto& cell : triangulation.active_cell_iterators())
    if (is_parallel_shared || cell->is_locally_owned())
    {
      if (dim > 1)
      {
        for (unsigned int i = 0; i < cell->n_lines(); ++i)
        {
          const typename dealii::Triangulation<dim, spacedim>::line_iterator
            line = cell->line(i);

          if (keep_boundary && line->at_boundary())
          {
            at_boundary[line->vertex_index(0)] = true;
            at_boundary[line->vertex_index(1)] = true;
          }

          minimal_length[line->vertex_index(0)] =
            std::min(line->diameter(),
              minimal_length[line->vertex_index(0)]);
          minimal_length[line->vertex_index(1)] =
            std::min(line->diameter(),
              minimal_length[line->vertex_index(1)]);
        }
        if(keep_interface) {
          const auto cell_mat_id = cell->material_id();
          for (unsigned int i = 0; i < cell->n_faces(); ++i)
          {
            const auto face = cell->face(i);
            if (face->at_boundary()) continue;
            const auto neighbor_cell = cell->neighbor(i);
            const auto neighbor_cell_mat_id = neighbor_cell->material_id();

            if  (cell_mat_id != neighbor_cell_mat_id) {
              for (unsigned int j = 0; j < face->n_vertices(); ++j) {
                at_boundary[face->vertex_index(j)] = true;
              }
            }
          }
        }
      }
      else // dim==1
      {
        if (keep_boundary)
          for (unsigned int vertex = 0; vertex < 2; ++vertex)
            if (cell->at_boundary(vertex) == true)
              at_boundary[cell->vertex_index(vertex)] = true;

        minimal_length[cell->vertex_index(0)] =
          std::min(cell->diameter(),
            minimal_length[cell->vertex_index(0)]);
        minimal_length[cell->vertex_index(1)] =
          std::min(cell->diameter(),
            minimal_length[cell->vertex_index(1)]);
      }
    }

  boost::random::mt19937                     rng(seed);
  boost::random::uniform_real_distribution<> uniform_distribution(-1, 1);

  // distributed meshes have to exchange the moved vertices
  if (auto distributed_triangulation =
    dynamic_cast<dealii::parallel::DistributedTriangulationBase<dim, spacedim>*>(
      &triangulation))
  {
    const std::vector<bool> locally_owned_vertices =
      dealii::GridTools::get_locally_owned_vertices(triangulation);
    std::vector<bool> vertex_moved(triangulation.n_vertices(), false);

    // Next move vertices on locally owned cells
    for (const auto& cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        for (const unsigned int vertex_no : cell->vertex_indices())
        {
          const unsigned global_vertex_no =
            cell->vertex_index(vertex_no);

          if ((keep_boundary && at_boundary[global_vertex_no]) ||
            vertex_moved[global_vertex_no] ||
            !locally_owned_vertices[global_vertex_no])
            continue;

          dealii::Point<spacedim> shift_vector;
          for (unsigned int d = 0; d < spacedim; ++d)
            shift_vector(d) = (d == distort_dim) ? uniform_distribution(rng) : 0.0;

          shift_vector *= factor * minimal_length[global_vertex_no] /
            std::sqrt(shift_vector.square());

          cell->vertex(vertex_no) += shift_vector;
          vertex_moved[global_vertex_no] = true;
        }
      }

    distributed_triangulation->communicate_locally_moved_vertices(
      locally_owned_vertices);
  }
  else
  {
    const unsigned int           n_vertices = triangulation.n_vertices();
    std::vector<dealii::Point<spacedim>> new_vertex_locations(n_vertices);
    const std::vector<dealii::Point<spacedim>>& old_vertex_locations =
      triangulation.get_vertices();

    for (unsigned int vertex = 0; vertex < n_vertices; ++vertex)
    {
      if (keep_boundary && at_boundary[vertex])
        new_vertex_locations[vertex] = old_vertex_locations[vertex];
      else
      {
        dealii::Point<spacedim> shift_vector;
        for (unsigned int d = 0; d < spacedim; ++d)
          shift_vector(d) = (d == distort_dim) ? uniform_distribution(rng) : 0.0;

        shift_vector *= factor * minimal_length[vertex] /
          std::sqrt(shift_vector.square());

        new_vertex_locations[vertex] =
          old_vertex_locations[vertex] + shift_vector;
      }
    }

    for (const auto& cell : triangulation.active_cell_iterators())
      for (const unsigned int vertex_no : cell->vertex_indices())
        cell->vertex(vertex_no) =
        new_vertex_locations[cell->vertex_index(vertex_no)];
  }

  // Move hanging nodes back onto the (now shifted) parent face. Artificial
  // cells are skipped because their refinement is unknown here.
  if (dim >= 2)
  {
    typename dealii::Triangulation<dim, spacedim>::active_cell_iterator
      cell = triangulation.begin_active(),
      endc = triangulation.end();
    for (; cell != endc; ++cell)
      if (!cell->is_artificial())
        for (const unsigned int face : cell->face_indices())
          if (cell->face(face)->has_children() &&
            !cell->face(face)->at_boundary())
          {
            if (dim == 2)
              cell->face(face)->child(0)->vertex(1) =
              (cell->face(face)->vertex(0) +
                cell->face(face)->vertex(1)) /
              2;
            else if (dim == 3)
            {
              cell->face(face)->child(0)->vertex(1) =
                .5 * (cell->face(face)->vertex(0) +
                  cell->face(face)->vertex(1));
              cell->face(face)->child(0)->vertex(2) =
                .5 * (cell->face(face)->vertex(0) +
                  cell->face(face)->vertex(2));
              cell->face(face)->child(1)->vertex(3) =
                .5 * (cell->face(face)->vertex(1) +
                  cell->face(face)->vertex(3));
              cell->face(face)->child(2)->vertex(3) =
                .5 * (cell->face(face)->vertex(2) +
                  cell->face(face)->vertex(3));

              // face centre
              cell->face(face)->child(0)->vertex(3) =
                .25 * (cell->face(face)->vertex(0) +
                  cell->face(face)->vertex(1) +
                  cell->face(face)->vertex(2) +
                  cell->face(face)->vertex(3));
            }
          }
  }
}

// Overload without interface pinning.
template <int dim, int spacedim>
void
distort_random(
  const double                  factor,
  const unsigned int distort_dim,
  dealii::Triangulation<dim, spacedim>& triangulation,
  const bool                    keep_boundary,
  const unsigned int            seed)
{
  Assert(spacedim == dim, dealii::ExcNotImplemented());
  Assert((0 <= distort_dim && distort_dim <= dim), dealii::ExcNotImplemented());

  // Upper bound for the shortest adjacent edge, so the min below always wins.
  double almost_infinite_length = 0;
  for (typename dealii::Triangulation<dim, spacedim>::cell_iterator cell =
    triangulation.begin(0);
    cell != triangulation.end(0);
    ++cell)
    almost_infinite_length += cell->diameter();

  std::vector<double> minimal_length(triangulation.n_vertices(),
    almost_infinite_length);

  std::vector<bool> at_boundary(keep_boundary ? triangulation.n_vertices() :
    0,
    false);
  // shared triangulations need all vertices, not just the locally owned ones
  const bool is_parallel_shared =
    (dynamic_cast<
    dealii::parallel::shared::Triangulation<dim, spacedim> *>(
      &triangulation) != nullptr);
  for (const auto& cell : triangulation.active_cell_iterators())
    if (is_parallel_shared || cell->is_locally_owned())
    {
      if (dim > 1)
      {
        for (unsigned int i = 0; i < cell->n_lines(); ++i)
        {
          const typename dealii::Triangulation<dim, spacedim>::line_iterator
            line = cell->line(i);

          if (keep_boundary && line->at_boundary())
          {
            at_boundary[line->vertex_index(0)] = true;
            at_boundary[line->vertex_index(1)] = true;
          }

          minimal_length[line->vertex_index(0)] =
            std::min(line->diameter(),
              minimal_length[line->vertex_index(0)]);
          minimal_length[line->vertex_index(1)] =
            std::min(line->diameter(),
              minimal_length[line->vertex_index(1)]);
        }
      }
      else // dim==1
      {
        if (keep_boundary)
          for (unsigned int vertex = 0; vertex < 2; ++vertex)
            if (cell->at_boundary(vertex) == true)
              at_boundary[cell->vertex_index(vertex)] = true;

        minimal_length[cell->vertex_index(0)] =
          std::min(cell->diameter(),
            minimal_length[cell->vertex_index(0)]);
        minimal_length[cell->vertex_index(1)] =
          std::min(cell->diameter(),
            minimal_length[cell->vertex_index(1)]);
      }
    }

  boost::random::mt19937                     rng(seed);
  boost::random::uniform_real_distribution<> uniform_distribution(-1, 1);

  // distributed meshes have to exchange the moved vertices
  if (auto distributed_triangulation =
    dynamic_cast<dealii::parallel::DistributedTriangulationBase<dim, spacedim>*>(
      &triangulation))
  {
    const std::vector<bool> locally_owned_vertices =
      dealii::GridTools::get_locally_owned_vertices(triangulation);
    std::vector<bool> vertex_moved(triangulation.n_vertices(), false);

    // Next move vertices on locally owned cells
    for (const auto& cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        for (const unsigned int vertex_no : cell->vertex_indices())
        {
          const unsigned global_vertex_no =
            cell->vertex_index(vertex_no);

          if ((keep_boundary && at_boundary[global_vertex_no]) ||
            vertex_moved[global_vertex_no] ||
            !locally_owned_vertices[global_vertex_no])
            continue;

          dealii::Point<spacedim> shift_vector;
          for (unsigned int d = 0; d < spacedim; ++d)
            shift_vector(d) = (d == distort_dim) ? uniform_distribution(rng) : 0.0;

          shift_vector *= factor * minimal_length[global_vertex_no] /
            std::sqrt(shift_vector.square());

          cell->vertex(vertex_no) += shift_vector;
          vertex_moved[global_vertex_no] = true;
        }
      }

    distributed_triangulation->communicate_locally_moved_vertices(
      locally_owned_vertices);
  }
  else
  {
    const unsigned int           n_vertices = triangulation.n_vertices();
    std::vector<dealii::Point<spacedim>> new_vertex_locations(n_vertices);
    const std::vector<dealii::Point<spacedim>>& old_vertex_locations =
      triangulation.get_vertices();

    for (unsigned int vertex = 0; vertex < n_vertices; ++vertex)
    {
      if (keep_boundary && at_boundary[vertex])
        new_vertex_locations[vertex] = old_vertex_locations[vertex];
      else
      {
        dealii::Point<spacedim> shift_vector;
        for (unsigned int d = 0; d < spacedim; ++d)
          shift_vector(d) = (d == distort_dim) ? uniform_distribution(rng) : 0.0;

        shift_vector *= factor * minimal_length[vertex] /
          std::sqrt(shift_vector.square());

        new_vertex_locations[vertex] =
          old_vertex_locations[vertex] + shift_vector;
      }
    }

    for (const auto& cell : triangulation.active_cell_iterators())
      for (const unsigned int vertex_no : cell->vertex_indices())
        cell->vertex(vertex_no) =
        new_vertex_locations[cell->vertex_index(vertex_no)];
  }

  // Move hanging nodes back onto the (now shifted) parent face. Artificial
  // cells are skipped because their refinement is unknown here.
  if (dim >= 2)
  {
    typename dealii::Triangulation<dim, spacedim>::active_cell_iterator
      cell = triangulation.begin_active(),
      endc = triangulation.end();
    for (; cell != endc; ++cell)
      if (!cell->is_artificial())
        for (const unsigned int face : cell->face_indices())
          if (cell->face(face)->has_children() &&
            !cell->face(face)->at_boundary())
          {
            if (dim == 2)
              cell->face(face)->child(0)->vertex(1) =
              (cell->face(face)->vertex(0) +
                cell->face(face)->vertex(1)) /
              2;
            else if (dim == 3)
            {
              cell->face(face)->child(0)->vertex(1) =
                .5 * (cell->face(face)->vertex(0) +
                  cell->face(face)->vertex(1));
              cell->face(face)->child(0)->vertex(2) =
                .5 * (cell->face(face)->vertex(0) +
                  cell->face(face)->vertex(2));
              cell->face(face)->child(1)->vertex(3) =
                .5 * (cell->face(face)->vertex(1) +
                  cell->face(face)->vertex(3));
              cell->face(face)->child(2)->vertex(3) =
                .5 * (cell->face(face)->vertex(2) +
                  cell->face(face)->vertex(3));

              // face centre
              cell->face(face)->child(0)->vertex(3) =
                .25 * (cell->face(face)->vertex(0) +
                  cell->face(face)->vertex(1) +
                  cell->face(face)->vertex(2) +
                  cell->face(face)->vertex(3));
            }
          }
  }
}

#endif // DISTORT_H_