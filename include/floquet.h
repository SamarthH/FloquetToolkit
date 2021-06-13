/// \file
#ifndef FLOQUET
#define FLOQUET

#define ERR_TOL 1e-6 ///< Maximum total error to which the differential equation would be solved
#define ERR_EIGEN_TOL 1e-5 ///< Maximum value of \f$ ||\rho_\text{max}| - 1| \f$ to differentiate between stable and unstable system

#include "diffEqSolvers.h"
#include <gsl/gsl_eigen.h>
#include <omp.h>

/** @brief Function which checks if the function is Floquet Stable or unstable for a general function
 *
 * Naive implementation with the elements of the \f$ X(T) \f$ matrix calculated to precision ERR_TOL
 * @param n Number of elements of vector that \f$ A(t) \f$ operates on
 * @param A \f$ A(t) \f$ matrix corresponding to the equation \f$ \dot{x} = Ax \f$
 * @param params Parameters to be passed to \f$ A(t) \f$
 * @param T Period of the evolution function \f$ A(t) \f$ s.t. \f$ A(t + T) = A(t)\ \forall t \in \mathbb{R} \f$
 * @param largest_multiplier Pointer to store the largest (by abs) computed Floquet multiplier into. No multiplier will be stored if NULL is passed.
 * @param largest_multiplier_abs Pointer to store the absolute value of the largest (by abs) computed Floquet multiplier into. No multiplier will be stored if NULL is passed.
 * @returns 1 if Stable, -1 if unstable, 0 if periodic or indeterminate to accuracy ERR_EIGEN_TOL. In rare cases, 2 would be returned if none of the computed floquet multipliers lead to instability, but not all of multipliers could be computed.
 ******************************************************************************/
int floquet_get_stability_reals_general(int n, void (*A)(double, gsl_matrix*, void*), void* params, double T, gsl_complex* largest_multiplier, double* largest_multiplier_abs);

/** @brief CPU Parallelized Function which iterates over a range of a parameter and checks if the function is Floquet Stable or unstable for a general function
 *
 * Run floquet_get_stability_reals_general on a range of a parameter and stores the stability, floquet multiplier corresponding to the largest absolute value, and the absolute value of the largest floquet multiplier.
 * This is a memory naive implementation.
 * @param n Number of elements of vector that \f$ A(t) \f$ operates on
 * @param A \f$ A(t) \f$ matrix corresponding to the equation \f$ \dot{x} = Ax \f$. The void* should be resolved to a double inside the function because a double* with a single double would be passed.
 * @param T Period of the evolution function \f$ A(t) \f$ s.t. \f$ A(t + T) = A(t)\ \forall t \in \mathbb{R} \f$
 * @param start Starting value of the parameter
 * @param end Ending value of the parameter
 * @param nstep Number of steps to take inclusive of the first and last values. Should be at least 2. Behaviour not defined otherwise.
 * @param largest_multiplier Array to store the largest (by abs) computed Floquet multipliers into. No multiplier will be stored if NULL is passed.
 * @param largest_multiplier_abs Array to store the absolute value of the largest (by abs) computed Floquet multipliers into. No multiplier will be stored if NULL is passed.
 ******************************************************************************/
void floquet_get_stability_array_real_single_param_general(int n, void (*A)(double, gsl_matrix*, void*), double T, double start, double end, int nstep, int* stability, gsl_complex* largest_multiplier, double* largest_multiplier_abs);

/** @brief CPU Parallelized Function which iterates over the ranges of 2 parameters and checks if the function is Floquet Stable or unstable for a general function
 *
 * Run floquet_get_stability_reals_general on the ranges of 2 parameters and stores the stability, floquet multiplier corresponding to the largest absolute value, and the absolute value of the largest floquet multiplier.
 * This is a memory naive implementation.
 * @param n Number of elements of vector that \f$ A(t) \f$ operates on
 * @param A \f$ A(t) \f$ matrix corresponding to the equation \f$ \dot{x} = Ax \f$. The void* should be resolved to a double inside the function because a double* with 2 doubles would be passed.
 * @param T Period of the evolution function \f$ A(t) \f$ s.t. \f$ A(t + T) = A(t)\ \forall t \in \mathbb{R} \f$
 * @param start Starting values of the parameters
 * @param end Ending values of the parameters
 * @param nstep Number of steps to take inclusive of the first and last values keeping the other parameter constant. Should be at least 2. Behaviour not defined otherwise.
 * @param largest_multiplier Matrix to store the largest (by abs) computed Floquet multipliers into. No multiplier will be stored if NULL is passed.
 * @param largest_multiplier_abs Matrix to store the absolute value of the largest (by abs) computed Floquet multipliers into. No multiplier will be stored if NULL is passed.
 ******************************************************************************/
void floquet_get_stability_array_real_double_param_general(int n, void (*A)(double, gsl_matrix*, void*), double T, double* start, double* end, int* nstep, int** stability, gsl_complex** largest_multiplier, double** largest_multiplier_abs);

#endif