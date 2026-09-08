#ifndef ODE_SYSTEM_H_
#define ODE_SYSTEM_H_

#include "deal.II/lac/full_matrix.h"
#include "deal.II/lac/vector.h"

// Local interface ODE  d_t u + L u = c * E_par + g,
// with L = system_matrix and c = coupling_vector.
struct ODESystem
{
    const unsigned int system_size;
    dealii::FullMatrix<double> system_matrix;
    dealii::Vector<double> coupling_vector;

    ODESystem(const unsigned int system_size) : 
        system_size{system_size},
        system_matrix(system_size),
        coupling_vector(system_size)
        {};

};

#endif //ODE_SYSTEM_H_