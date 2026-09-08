#ifndef HORNER_SCHEME_H_
#define HORNER_SCHEME_H_

#include <vector>

namespace MaxwellProblem::Tools {
/**
 * @brief Horner scheme to evaluate a polynomial.
 * 
 * The polynomial is in the form
 * p(x) = c[0] + c[1]*x + c[2]*x^2 + ... c[n]*x^n
 * where c is the coefficient vector.
 * 
 * @param coefficients Coefficient vector
 * @param x Point to evaluate at
 * @return double
 */
double inline horner_scheme(const std::vector<double> &coefficients, double x) {
  const int deg = coefficients.size() - 1;

  double ret = coefficients[deg];

  for (int idx = deg - 1; idx >= 0; idx--) {
	ret = ret * x + coefficients[idx];
  }

  return ret;
}

/**
 * @brief Horner scheme to evaluate the derivative of a polynomial.
 * 
 * The polynomial is in the form
 * p(x) = c[1]*x + 2*c[2]*x^1 + ... n*c[n]*x^(n-1)
 * where c is the coefficient vector.
 * 
 * @param coefficients Coefficient vector
 * @param x Point to evaluate at
 * @return double 
 */
double inline horner_scheme_derivative(
	const std::vector<double> &coefficients,
	double x) {
  const int deg = coefficients.size() - 1;

  double ret = deg * coefficients[deg];

  for (int idx = deg - 1; idx > 0; idx--) {
	ret = ret * x + idx * coefficients[idx];
  }

  return ret;
}

/**
 * @brief Horner scheme to evaluate the derivative of a polynomial.
 * 
 * The polynomial is in the form
 * p(x) = 2*c[2] + 3*2 c[3]*x^1 + ... n*(n-1)*c[n]*x^(n-2)
 * where c is the coefficient vector.
 * 
 * @param coefficients Coefficient vector
 * @param x Point to evaluate at
 * @return double 
 */
double inline horner_scheme_second_derivative(
	const std::vector<double> &coefficients,
	double x) {
  const int deg = coefficients.size() - 1;

  double ret = deg * (deg-1) * coefficients[deg];

  for (int idx = deg - 1; idx > 1; idx--) {
	ret = ret * x + idx * (idx - 1) * coefficients[idx];
  }

  return ret;
}

}// namespace MaxwellProblem::Tools

#endif// HORNER_SCHEME_H_