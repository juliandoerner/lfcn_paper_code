#include <random>
#include <filesystem>
#include <thread>

#include <deal.II/grid/grid_in.h>

// #include <gmsh.h>

#include "GenerateDomain.h"

// std::tuple<const dealii::Point<2>, const dealii::Point<2>, const dealii::Point<2>> 
// translate_coordinates_2d(
//     const double a_1_minus, const double a_1_plus, 
//     const double a_2_minus, const double a_2_plus, 
//     const double zeta)
// {
//     if (!(a_1_minus < zeta && zeta < a_1_plus ))
//         throw dealii::ExcMessage("Interface is located outside domain! Check zeta1");

//     return {
//         {a_1_minus, a_2_minus},
//         {zeta, a_2_plus},
//         {a_1_plus, a_2_plus}
//         };
// }

// std::tuple<const dealii::Point<3>, const dealii::Point<3>, const dealii::Point<3>> 
// translate_coordinates_3d(const double a_1_minus, const double a_1_plus, const double a_2_minus, const double a_2_plus, const double a_3_minus, const double a_3_plus, const double zeta)
// {
//     if (!(a_1_minus < zeta && zeta < a_1_plus ))
//         throw dealii::ExcMessage("Interface is located outside domain! Check zeta1");

//     return {
//         {a_1_minus, a_2_minus, a_3_minus},
//         {zeta, a_2_plus, a_3_plus},
//         {a_1_plus, a_2_plus, a_3_plus}
//         };
// }

void assemble_grid_xy_distortion(
    dealii::Triangulation<2>& triangulation,
    const dealii::Point<2>& p0,
    const dealii::Point<2>& p1,
    const dealii::Point<2>& p2,
    const double mesh_size,
    const double distort_factor)
{
    dealii::Triangulation<2> triangulation_left;
    dealii::Triangulation<2> triangulation_right;

    // p3 is the lower left corner of the second rectangle
    dealii::Point<2> p3;
    p3[0] = p1[0];
    p3[1] = p0[1];

    const unsigned int blocks_per_unit_square = std::round(std::abs(std::sqrt(2) / mesh_size));

    std::vector<unsigned int> repetitions{ blocks_per_unit_square, blocks_per_unit_square };
    dealii::GridGenerator::subdivided_hyper_rectangle(
        triangulation_left, repetitions, p0, p1);
    dealii::GridGenerator::subdivided_hyper_rectangle(
        triangulation_right, repetitions, p3, p2);

    dealii::GridGenerator::merge_triangulations(
        triangulation_left, triangulation_right, triangulation);
    
    std::mt19937 gen(671679);
    std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());

    // Mark before distorting: distort_random(keep_interface=true) finds the
    // interface by the material jump and pins those vertices.
    mark_cells(triangulation, 0, 1, p1[0]);

    distort_random(distort_factor, 0, triangulation, true, true, dist(gen));
    distort_random(distort_factor, 1, triangulation, true, true, dist(gen));
}

void assemble_grid_xy_distortion(
    dealii::Triangulation<3>& triangulation,
    const dealii::Point<3>& p0,
    const dealii::Point<3>& p1,
    const dealii::Point<3>& p2,
    const double mesh_size,
    const double distort_factor)
{
    dealii::Triangulation<3> triangulation_left;
    dealii::Triangulation<3> triangulation_right;

    // p3 is the lower left corner of the second box
    dealii::Point<3> p3;
    p3[0] = p1[0];
    p3[1] = p0[1];
    p3[2] = p0[3];

    const unsigned int blocks_per_unit_square = std::round(std::abs(std::sqrt(2) / mesh_size));

    auto[l1, l2, l3, l4] = get_side_ratio(p0, p1, p2);
    {
        std::vector<unsigned int> repetitions{ 
            static_cast<unsigned int>(std::lround(std::abs(l1*blocks_per_unit_square))), 
            static_cast<unsigned int>(std::lround(std::abs(l2*blocks_per_unit_square))), 
            static_cast<unsigned int>(std::lround(std::abs(l4*blocks_per_unit_square)))};
        dealii::GridGenerator::subdivided_hyper_rectangle(
            triangulation_left, repetitions, p0, p1);
    }
    {
        std::vector<unsigned int> repetitions{ 
            static_cast<unsigned int>(std::lround(std::abs(l1*blocks_per_unit_square))), 
            static_cast<unsigned int>(std::lround(std::abs(l3*blocks_per_unit_square))), 
            static_cast<unsigned int>(std::lround(std::abs(l4*blocks_per_unit_square)))};
        dealii::GridGenerator::subdivided_hyper_rectangle(
            triangulation_right, repetitions, p0, p1);
    }
    
    dealii::GridGenerator::merge_triangulations(
        triangulation_left, triangulation_right, triangulation);
    
    std::mt19937 gen(671679);
    std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());

    // Mark before distorting: distort_random(keep_interface=true) finds the
    // interface by the material jump and pins those vertices.
    mark_cells(triangulation, 0, 1, p1[0]);

    distort_random(distort_factor, 0, triangulation, true, true, dist(gen));
    distort_random(distort_factor, 1, triangulation, true, true, dist(gen));
}

void assemble_gird_preserve_midpoint(
    dealii::Triangulation<2>& triangulation,
    const dealii::Point<2>& p0,
    const dealii::Point<2>& p1,
    const dealii::Point<2>& p2,
    const double mesh_size,
    const double distort_factor)
{
    Assert(p1[1] == p2[1],
        dealii::StandardExceptions::ExcMessage("Only rectangular domain allowed."));

    dealii::Triangulation<2> triangulation_lower_left;
    dealii::Triangulation<2> triangulation_upper_left;
    dealii::Triangulation<2> triangulation_lower_right;
    dealii::Triangulation<2> triangulation_upper_right;

    // The four quarters meet at b1, which is the point kept fixed.
    dealii::Point<2> b0{ p0[0], p1[1] / 2. };
    dealii::Point<2> b1{ p1[0], p1[1] / 2. };
    dealii::Point<2> b2{ p1[0], p0[0] };
    dealii::Point<2> b3{ p2[0], p2[0] / 2. };

    const unsigned int blocks_per_unit_square = std::round(std::abs(std::sqrt(2) / mesh_size));

    // two unit squares!
    std::vector<unsigned int> repetitions{ blocks_per_unit_square, blocks_per_unit_square / 2 };
    dealii::GridGenerator::subdivided_hyper_rectangle(
        triangulation_lower_left, repetitions, p0, b1);
    dealii::GridGenerator::subdivided_hyper_rectangle(
        triangulation_upper_left, repetitions, b0, p1);
    dealii::GridGenerator::subdivided_hyper_rectangle(
        triangulation_lower_right, repetitions, b2, b3);
    dealii::GridGenerator::subdivided_hyper_rectangle(
        triangulation_upper_right, repetitions, b1, p2);

    std::mt19937 gen(671679);
    std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());

    distort_random(distort_factor, 0, triangulation_lower_left, true, dist(gen));
    distort_random(distort_factor, 0, triangulation_upper_left, true, dist(gen));
    distort_random(distort_factor, 0, triangulation_lower_right, true, dist(gen));
    distort_random(distort_factor, 0, triangulation_upper_right, true, dist(gen));

    dealii::Triangulation<2> triangulation_top;
    dealii::Triangulation<2> triangulation_down;
    dealii::GridGenerator::merge_triangulations(
        triangulation_lower_left, triangulation_lower_right, triangulation_down);
    dealii::GridGenerator::merge_triangulations(
        triangulation_upper_left, triangulation_upper_right, triangulation_top);

    distort_random(distort_factor, 1, triangulation_top, true, dist(gen));
    distort_random(distort_factor, 1, triangulation_down, true, dist(gen));

    dealii::GridGenerator::merge_triangulations(
        triangulation_top, triangulation_down, triangulation);
}

void assemble_grid_via_triangles(
    dealii::Triangulation<2>& triangulation,
    const dealii::Point<2>& p0,
    const dealii::Point<2>& p1,
    const dealii::Point<2>& p2,
    const double mesh_size,
    const double distort_factor)
{
    const std::vector<dealii::Point<2>> 
        verts_triangle_1{{p0[0], p0[1]}, {p1[0], p1[1]}, {p0[0], p1[1]}};
    const std::vector<dealii::Point<2>> 
        verts_triangle_2{{p0[0], p0[1]}, {p1[0],p0[1]}, {p1[0], p1[1]}};
    const std::vector<dealii::Point<2>>
        verts_triangle_3{{p1[0], p1[1]}, {p1[0], p0[1]}, {p2[0], p0[1]}};
    const std::vector<dealii::Point<2>>
        verts_triangle_4{{p1[0], p1[1]}, {p2[0], p0[1]}, {p2[0], p2[1]}};
    
    dealii::Triangulation<2> tria_triangle_1;
    dealii::Triangulation<2> tria_triangle_2;
    dealii::Triangulation<2> tria_triangle_3;
    dealii::Triangulation<2> tria_triangle_4;

    dealii::GridGenerator::simplex(tria_triangle_1, verts_triangle_1);
    dealii::GridGenerator::simplex(tria_triangle_2, verts_triangle_2);
    dealii::GridGenerator::simplex(tria_triangle_3, verts_triangle_3);
    dealii::GridGenerator::simplex(tria_triangle_4, verts_triangle_4);

    std::mt19937 gen(671679);
    std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());

    distort_random(distort_factor, 0, tria_triangle_1, true, dist(gen));
    distort_random(distort_factor, 0, tria_triangle_2, true, dist(gen));
    distort_random(distort_factor, 0, tria_triangle_3, true, dist(gen));
    distort_random(distort_factor, 0, tria_triangle_4, true, dist(gen));

    dealii::Triangulation<2> tria_left;
    dealii::Triangulation<2> tria_right;

    dealii::GridGenerator::merge_triangulations(tria_triangle_1, tria_triangle_2, tria_left);
    dealii::GridGenerator::merge_triangulations(tria_triangle_3, tria_triangle_4, tria_right);

    distort_random(distort_factor, 1, tria_left, true, dist(gen));
    distort_random(distort_factor, 1, tria_right, true, dist(gen));

    dealii::GridGenerator::merge_triangulations(tria_left, tria_right, triangulation);

    refine_down_to_mesh_size(triangulation, mesh_size);
}

// void assemble_grid_via_gmsh(
//     dealii::Triangulation<2>& triangulation,
//     const dealii::Point<2>& p0,
//     const dealii::Point<2>& p1,
//     const dealii::Point<2>& p2,
//     const double mesh_size,
//     const double mesh_size_interface)
// {
//     std::string tmpf = std::tmpnam(nullptr);
//     std::stringstream tmpfile; tmpfile  << tmpf << ".msh";
//     std::cout << "tmp file name: " << tmpfile.str() << std::endl;
//     //const auto tmpfile = std::filesystem::temp_directory_path() / "gmsh_tmpfile.msh";

//     gmsh::initialize();
    
//     gmsh::option::setNumber("General.NumThreads", std::thread::hardware_concurrency());

//     gmsh::model::add(tmpfile.str());

//     auto gmsh_p1 = gmsh::model::geo::addPoint(p0[0],p0[1],0, mesh_size);
//     auto gmsh_p2 = gmsh::model::geo::addPoint(p1[0],p0[1],0, mesh_size_interface);
//     auto gmsh_p3 = gmsh::model::geo::addPoint(p1[0],p1[1],0, mesh_size_interface);
//     auto gmsh_p4 = gmsh::model::geo::addPoint(p0[0],p1[1],0, mesh_size);
//     auto gmsh_p5 = gmsh::model::geo::addPoint(p2[0],p0[1],0, mesh_size);
//     auto gmsh_p6 = gmsh::model::geo::addPoint(p2[0],p1[1],0, mesh_size);

//     auto l1 = gmsh::model::geo::addLine(gmsh_p1, gmsh_p2);
//     auto l2 = gmsh::model::geo::addLine(gmsh_p2, gmsh_p3);
//     auto l3 = gmsh::model::geo::addLine(gmsh_p3, gmsh_p4);
//     auto l4 = gmsh::model::geo::addLine(gmsh_p4, gmsh_p1);
//     auto l5 = gmsh::model::geo::addLine(gmsh_p3, gmsh_p6);
//     auto l6 = gmsh::model::geo::addLine(gmsh_p6, gmsh_p5);
//     auto l7 = gmsh::model::geo::addLine(gmsh_p5, gmsh_p2);

//     auto cl1 = gmsh::model::geo::addCurveLoop({l1,l2,l3,l4});
//     auto cl2 = gmsh::model::geo::addCurveLoop({l2,l5,l6,l7});

//     auto s1 = gmsh::model::geo::addPlaneSurface({cl1});
//     auto s2 = gmsh::model::geo::addPlaneSurface({cl2});

//     gmsh::model::geo::synchronize();

//     gmsh::model::addPhysicalGroup(2, {s1}, 0);
//     gmsh::model::addPhysicalGroup(2, {s2}, 1);
//     gmsh::model::setPhysicalName(2, 0, "MaterialID: 0");
//     gmsh::model::setPhysicalName(2, 1, "MaterialID: 1");

//     gmsh::option::setNumber("Mesh.Algorithm", 6);
//     gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 1);
//     gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 1);
//     gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 1);
//     //gmsh::option::setNumber("Mesh.MeshSizeMin", mesh_size*0.5);
//     gmsh::option::setNumber("Mesh.MeshSizeMax", mesh_size);
//     gmsh::option::setNumber("Mesh.MeshSizeFactor", 0.5);
//     gmsh::option::setNumber("Mesh.RecombinationAlgorithm", 3);

//     gmsh::model::geo::synchronize();
//     gmsh::model::mesh::generate();
//     gmsh::model::mesh::recombine();
//     gmsh::model::mesh::optimize("", false, 20);
//     gmsh::write(tmpfile.str());

//     gmsh::finalize();

//     dealii::GridIn<2>(triangulation).read(tmpfile.str());

//     //std::filesystem::remove(tmpfile.str());

// }

// void assemble_grid_via_gmsh_via_triangles(
//     dealii::Triangulation<2>& triangulation,
//     const dealii::Point<2>& p0,
//     const dealii::Point<2>& p1,
//     const dealii::Point<2>& p2,
//     const double mesh_size,
//     const double mesh_size_interface)
// {
//     const auto tmpfile = std::filesystem::temp_directory_path() / "gmsh_tmpfile.msh";

//     gmsh::initialize();
//     gmsh::option::setNumber("General.NumThreads", std::thread::hardware_concurrency());

//     gmsh::model::add(tmpfile);

//     auto gmsh_p1 = gmsh::model::geo::addPoint(p0[0],p0[1],0, mesh_size);
//     auto gmsh_p2 = gmsh::model::geo::addPoint(p1[0],p0[1],0, mesh_size_interface);
//     auto gmsh_p3 = gmsh::model::geo::addPoint(p1[0],p1[1],0, mesh_size_interface);
//     auto gmsh_p4 = gmsh::model::geo::addPoint(p0[0],p1[1],0, mesh_size);
//     auto gmsh_p5 = gmsh::model::geo::addPoint(p2[0],p0[1],0, mesh_size);
//     auto gmsh_p6 = gmsh::model::geo::addPoint(p2[0],p1[1],0, mesh_size);

//     auto t1_l1 = gmsh::model::geo::addLine(gmsh_p1, gmsh_p3);
//     auto t1_l2 = gmsh::model::geo::addLine(gmsh_p3, gmsh_p4);
//     auto t1_l3 = gmsh::model::geo::addLine(gmsh_p4, gmsh_p1);

//     auto t2_l1 = gmsh::model::geo::addLine(gmsh_p1, gmsh_p2);
//     auto t2_l2 = gmsh::model::geo::addLine(gmsh_p2, gmsh_p3);
//     auto t2_l3 = gmsh::model::geo::addLine(gmsh_p3, gmsh_p1);

//     auto t3_l1 = gmsh::model::geo::addLine(gmsh_p2, gmsh_p6);
//     auto t3_l2 = gmsh::model::geo::addLine(gmsh_p6, gmsh_p3);
//     auto t3_l3 = gmsh::model::geo::addLine(gmsh_p3, gmsh_p2);

//     auto t4_l1 = gmsh::model::geo::addLine(gmsh_p2, gmsh_p5);
//     auto t4_l2 = gmsh::model::geo::addLine(gmsh_p5, gmsh_p6);
//     auto t4_l3 = gmsh::model::geo::addLine(gmsh_p6, gmsh_p2);

//     auto ct1 = gmsh::model::geo::addCurveLoop({t1_l1, t1_l2, t1_l3});
//     auto ct2 = gmsh::model::geo::addCurveLoop({t2_l1, t2_l2, t2_l3});
//     auto ct3 = gmsh::model::geo::addCurveLoop({t3_l1, t3_l2, t3_l3});
//     auto ct4 = gmsh::model::geo::addCurveLoop({t4_l1, t4_l2, t4_l3});

//     auto st1 = gmsh::model::geo::addPlaneSurface({ct1});
//     auto st2 = gmsh::model::geo::addPlaneSurface({ct2});
//     auto st3 = gmsh::model::geo::addPlaneSurface({ct3});
//     auto st4 = gmsh::model::geo::addPlaneSurface({ct4});

//     gmsh::model::geo::synchronize();

//     auto pg1 = gmsh::model::addPhysicalGroup(2, {st1, st2}, -1);
//     auto pg2 = gmsh::model::addPhysicalGroup(2, {st3, st4}, -1);
    
//     gmsh::model::setPhysicalName(2, pg1, "MaterialID: 0");
//     gmsh::model::setPhysicalName(2, pg2, "MaterialID: 1");

//     gmsh::model::geo::synchronize();
//     gmsh::model::mesh::generate(2);
//     gmsh::model::mesh::recombine();
//     gmsh::model::mesh::optimize("Laplace2D", false, 20);
//     gmsh::write(tmpfile);

//     gmsh::finalize();

//     dealii::GridIn<2>(triangulation).read(tmpfile);

//     std::filesystem::remove(tmpfile);

// }

void assemble_grid_via_8_triangles(
    dealii::Triangulation<2>& triangulation,
    const dealii::Point<2>& p0,
    const dealii::Point<2>& p1,
    const dealii::Point<2>& p2,
    const double mesh_size,
    const double distort_factor)
{
    const dealii::Point<2> point_0{p0[0], p0[1]};
    const dealii::Point<2> point_1{p1[0], p0[1]};
    const dealii::Point<2> point_2{0.5*(p1[0]-p0[0]) + p0[0], 0.5*(p1[1]-p0[1]) + p0[1]};
    const dealii::Point<2> point_3{p1[0], p1[1]};
    const dealii::Point<2> point_4{p0[0], p1[1]};
    const dealii::Point<2> point_5{p2[0], p0[1]};
    const dealii::Point<2> point_6{0.5*(p2[0]-p1[0]) + p1[0], 0.5*(p1[1]-p0[1]) + p0[1]};
    const dealii::Point<2> point_7{p2[0], p1[1]};

    const std::vector<dealii::Point<2>>
        verts_triangle_1{point_0, point_1, point_2};
    const std::vector<dealii::Point<2>>
        verts_triangle_2{point_1, point_3, point_2};
    const std::vector<dealii::Point<2>>
        verts_triangle_3{point_3, point_4, point_2};
    const std::vector<dealii::Point<2>>
        verts_triangle_4{point_4, point_0, point_2};
    const std::vector<dealii::Point<2>>
        verts_triangle_5{point_1, point_5, point_6};
    const std::vector<dealii::Point<2>>
        verts_triangle_6{point_5, point_7, point_6};
    const std::vector<dealii::Point<2>>
        verts_triangle_7{point_7, point_3, point_6};
    const std::vector<dealii::Point<2>>
        verts_triangle_8{point_3, point_1, point_6};
    
    dealii::Triangulation<2> tria_triangle_1;
    dealii::Triangulation<2> tria_triangle_2;
    dealii::Triangulation<2> tria_triangle_3;
    dealii::Triangulation<2> tria_triangle_4;
    dealii::Triangulation<2> tria_triangle_5;
    dealii::Triangulation<2> tria_triangle_6;
    dealii::Triangulation<2> tria_triangle_7;
    dealii::Triangulation<2> tria_triangle_8;

    dealii::GridGenerator::simplex(tria_triangle_1, verts_triangle_1);
    dealii::GridGenerator::simplex(tria_triangle_2, verts_triangle_2);
    dealii::GridGenerator::simplex(tria_triangle_3, verts_triangle_3);
    dealii::GridGenerator::simplex(tria_triangle_4, verts_triangle_4);
    dealii::GridGenerator::simplex(tria_triangle_5, verts_triangle_5);
    dealii::GridGenerator::simplex(tria_triangle_6, verts_triangle_6);
    dealii::GridGenerator::simplex(tria_triangle_7, verts_triangle_7);
    dealii::GridGenerator::simplex(tria_triangle_8, verts_triangle_8);

    std::mt19937 gen(671679);
    std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());

    //distort_random(distort_factor, 0, tria_triangle_1, true, dist(gen));
    //distort_random(distort_factor, 0, tria_triangle_2, true, dist(gen));
    //distort_random(distort_factor, 0, tria_triangle_3, true, dist(gen));
    //distort_random(distort_factor, 0, tria_triangle_4, true, dist(gen));
    //distort_random(distort_factor, 0, tria_triangle_5, true, dist(gen));
    //distort_random(distort_factor, 0, tria_triangle_6, true, dist(gen));
    //distort_random(distort_factor, 0, tria_triangle_7, true, dist(gen));
    //distort_random(distort_factor, 0, tria_triangle_8, true, dist(gen));

    dealii::Triangulation<2> tria_left;
    dealii::Triangulation<2> tria_right;

    dealii::GridGenerator::merge_triangulations(
        {&tria_triangle_1, &tria_triangle_2, &tria_triangle_3, &tria_triangle_4}, 
        tria_left);
    dealii::GridGenerator::merge_triangulations(
        {&tria_triangle_5, &tria_triangle_6, &tria_triangle_7, &tria_triangle_8}, 
        tria_right);

    //distort_random(distort_factor, 1, tria_left, true, dist(gen));
    // distort_random(distort_factor, 1, tria_right, true, dist(gen));
    dealii::GridGenerator::merge_triangulations(tria_left, tria_right, triangulation);

    mark_cells(triangulation);
    refine_down_to_mesh_size(triangulation, mesh_size);
    distort_random(distort_factor, 1, triangulation, true, false, dist(gen));
    distort_random(distort_factor, 0, triangulation, true, true, dist(gen));
}

void refine_interface(
    dealii::Triangulation<2>& triangulation,
    const unsigned int refinements,
    const dealii::types::material_id material_left,
    const dealii::types::material_id material_right,
    const double x_mid)
{
    Assert(material_left != material_right, 
        dealii::StandardExceptions::ExcMessage("Two identical material numbers"));

    for (unsigned int i = 0; i < refinements; i++)
    {
        for (const auto& cell : triangulation.active_cell_iterators())
        {
            auto own_material_id = cell->material_id();
            if (own_material_id != material_left) continue;

            for (unsigned int face_num = 0; 
                face_num < dealii::GeometryInfo<2>::faces_per_cell; 
                ++face_num)
            {
                if (!cell->at_boundary(face_num))
                {
                    auto neighbor = cell->neighbor(face_num);
                    if (neighbor->material_id() == material_right)
                    {
                        cell->set_refine_flag();
                        neighbor->set_refine_flag();
                    }
                }
            }
        }

        triangulation.prepare_coarsening_and_refinement();
        triangulation.execute_coarsening_and_refinement();
        mark_cells(triangulation, material_left, material_right, x_mid);
    }
}

void refine_down_to_mesh_size(
    dealii::Triangulation<2>& triangulation,
    const double mesh_size) {

    bool any_cell_was_refined = true;
    int ref_round = 0;
    while (any_cell_was_refined) {
        any_cell_was_refined = false;
        std::cout << "ref round: " << ref_round++ << "\n";
        for (const auto& cell : triangulation.active_cell_iterators())
        {
            const double cell_diameter = cell->diameter();
            if (mesh_size <  cell_diameter) {
                cell->set_refine_flag();
                any_cell_was_refined = true;
            }
        }
        triangulation.prepare_coarsening_and_refinement();
        triangulation.execute_coarsening_and_refinement();
    }
}

void random_refiner(
    dealii::Triangulation<2>& triangulation,
    const double probability)
{
    boost::random::mt19937  rng;
    boost::random::uniform_real_distribution<> uniform_distribution(0, 1);

    for (const auto& cell : triangulation.active_cell_iterators())
    {
        if (uniform_distribution(rng) > probability) continue;

        bool is_at_interface = false;
        const auto cell_mat_id = cell->material_id();
        for (unsigned int i = 0; i < cell->n_faces(); ++i)
        {
          const auto face = cell->face(i);
          if (face->at_boundary()) continue;
          const auto neighbor_cell = cell->neighbor(i);
          const auto neighbor_cell_mat_id = neighbor_cell->material_id();

          if  (cell_mat_id != neighbor_cell_mat_id)
            is_at_interface = true;
        }
        if(!is_at_interface)
            cell->set_refine_flag();
    }
    triangulation.prepare_coarsening_and_refinement();
    triangulation.execute_coarsening_and_refinement();
}