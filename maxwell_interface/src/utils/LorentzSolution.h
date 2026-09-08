// Exact solution of Exact_Solution.tex: 2D Maxwell on Omega = (-1,1) x (0,1)
// split at F = {0} x (0,1), driven by a dispersive surface current on F.
// Parameters: mode n (k = n*pi), longitudinal beta (needs cos(beta) != 0),
// oscillator gamma/omega; nu = sqrt(beta^2 + k^2). The bulk is source free.

#ifndef LFCN_PAPER_EXACT_SOLUTION_H
#define LFCN_PAPER_EXACT_SOLUTION_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

namespace lfcn {

// H jumps across F, so interpolating with `automatic` would average the two
// sides on the interface. The forced variants let the caller pick the branch
// per cell via the material id.
enum class Side { automatic, minus, plus };

// Fields (H, E_1, E_2). Evaluating at t = 0 gives the initial data.
class ExactSolution : public dealii::Function<2, double> {
 public:
  ExactSolution(unsigned int n, double beta, double initial_time = 0.0,
                Side side = Side::automatic);

  double value(const dealii::Point<2> &p,
               const unsigned int component = 0) const override;

  void vector_value(const dealii::Point<2> &p,
                    dealii::Vector<double> &values) const override;

 private:
  double h(double x1) const;   // piecewise profile, discontinuous across F
  double dh(double x1) const;

  const unsigned int n;
  const double beta;
  const Side side;

  const double k;
  const double nu;
};

// Trace of the surface current J_F on F; x1 is ignored.
class SurfaceCurrent : public dealii::Function<2, double> {
 public:
  SurfaceCurrent(unsigned int n, double beta, double initial_time = 0.0);

  double value(const dealii::Point<2> &p,
               const unsigned int component = 0) const override;

 private:
  const unsigned int n;
  const double beta;

  const double k;
  const double nu;
};

// Interface ODE unknowns u = (K, J_F) with K = omega * int_0^t J_F, in the
// component order of the interface FESystem. Used to measure the ODE error.
class AuxiliaryVariables : public dealii::Function<2, double> {
 public:
  AuxiliaryVariables(unsigned int n, double beta, double omega,
                     double initial_time = 0.0);

  double value(const dealii::Point<2> &p,
               const unsigned int component = 0) const override;

  void vector_value(const dealii::Point<2> &p,
                    dealii::Vector<double> &values) const override;

 private:
  const unsigned int n;
  const double beta;
  const double omega;

  const double k;
  const double nu;
};

// Forcing g of the second order oscillator equation. Not used by LFCN, which
// takes IntegratedForcing instead.
class InterfaceForcing : public dealii::Function<2, double> {
 public:
  InterfaceForcing(unsigned int n, double beta, double gamma, double omega,
                   double initial_time = 0.0);

  double value(const dealii::Point<2> &p,
               const unsigned int component = 0) const override;

 private:
  const unsigned int n;
  const double beta;
  const double gamma;
  const double omega;

  const double k;
  const double nu;
};

// Forcing G of the first-order (K, J) system actually solved by LFCN:
// d_t u + L u = c E_par + (0, G)^T. This is the time integral of g.
class IntegratedForcing : public dealii::Function<2, double> {
 public:
  IntegratedForcing(unsigned int n, double beta, double gamma, double omega,
                    double initial_time = 0.0);

  double value(const dealii::Point<2> &p,
               const unsigned int component = 0) const override;

 private:
  const unsigned int n;
  const double beta;
  const double gamma;
  const double omega;

  const double k;
  const double nu;
};

}  // namespace lfcn

#endif  // LFCN_PAPER_EXACT_SOLUTION_H
