#ifndef FLOQUET
#define FLOQUET

#define ERR_TOL 1e-6
#define ERR_EIGEN_TOL 1e-4

#include "diffEqSolvers.h"
#include <gsl/gsl_eigen.h>

/*******************************************************************************
 * \brief Function which checks if the function is Floquet Stable or unstable
 *
 * Naive implementation with the elements of the \f$ X(T) \f$ matrix calculated to precision ERR_TOL
 * @param n Number of elements of vector that \f$ A(t) \f$ operates on
 * @param A \f$ A(t) \f$ matrix corresponding to the equation \f$ \dot{x} = Ax \f$
 * @param params Parameters to be passed to \f$ A(t) \f$
 * @param T Period of the evolution function \f$ A(t) \f$ s.t. \f$ A(t + T) = A(t)\ \forall t \in \mathbb{R} \f$
 * @returns 1 if Stable, -1 if unstable, 0 if periodic or indeterminate to accuracy ERR_EIGEN_TOL. In rare cases, 2 would be returned if none of the computed floquet multipliers lead to instability, but not all of multipliers could be computed.
 ******************************************************************************/


int floquet_get_stability_reals(int n, void (*A)(double, gsl_matrix*, void*), void* params, double T);
#endif