#ifndef DIFFEQ_SOLVERS
#define DIFFEQ_SOLVERS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

#define RK4_MAX_SCALE 5
#define RK4_MIN_SCALE 0.2

/*******************************************************************************
 * \brief Fixed Step RK4 for vectors for general \f$ \dot{x} = f_\lambda(x,t) \f$
 *
 * Implemented according to Computational Physics, Mark Newman (2013)
 * @param ndim Dimensionality of x
 * @param x_i Initial Vector \f$ x_i \in  \mathbb{R}^{n} \f$
 * @param t_i Time when x_i is specified
 * @param H Interval after which final x is required
 * @param h Step size
 * @param evol_func Function that computes \f$ \frac{dx}{dt}(x,t) \f$. The function should be of the form void evol_func(double* x, double t, double* x_dot, void* params)
 * @param x_f Array to store the final x into. This should be preallocated
 * @param params Parameters to be passed to evol_func
 ******************************************************************************/
void rk4_fixed_final_vector_real(int ndim, double* x_i, double t_i, double H, double h, void (*evol_func)(double*, double, double*, void*), double* x_f, void* params);

/*******************************************************************************
 * \brief Fixed Step RK4 for real matrices for evolution of the form \f$ \dot{X} = A_\lambda(t)X \f$
 *
 * Implemented according to Computational Physics, Mark Newman (2013)
 * @param x_i Initial Matrix \f$ x_i \in  \mathbb{R}^{n\cross m} \f$
 * @param t_i Time when x_i is specified
 * @param H Interval after which final x is required
 * @param h Step size
 * @param evol_func Function that computes \f$ A_\lambda(t) \f$. The function should be of the form void A(double t, gsl_matrix* out, void* params)
 * @param x_f Array to store the final x into. This should be preallocated
 * @param params Parameters to be passed to A(t)
 ******************************************************************************/

void rk4_fixed_final_matrix_floquet_type_real(gsl_matrix* x_i, double t_i, double H, double h, void (*A)(double, gsl_matrix*, void*), gsl_matrix* x_f, void* params);

/*******************************************************************************
 * \brief Fixed Step RK4 for complex matrices for evolution of the form \f$ \dot{X} = A_\lambda(t)X \f$
 *
 * Implemented according to Computational Physics, Mark Newman (2013)
 * @param x_i Initial Matrix \f$ x_i \in  \mathbb{C}^{n\cross m} \f$
 * @param t_i Time when x_i is specified
 * @param H Interval after which final x is required
 * @param h Step size
 * @param evol_func Function that computes \f$ A_\lambda(t) \f$. The function should be of the form void A(double t, gsl_matrix_complex* out, void* params)
 * @param x_f Array to store the final x into. This should be preallocated
 * @param params Parameters to be passed to A(t)
 ******************************************************************************/

void rk4_fixed_final_matrix_floquet_type_complex(gsl_matrix_complex* x_i, double t_i, double H, double h, void (*A)(double, gsl_matrix_complex*, void*), gsl_matrix_complex* x_f, void* params);

/*******************************************************************************
 * \brief Adaptive Step RK4 for real matrices for evolution of the form \f$ \dot{X} = A_\lambda(t)X \f$
 *
 * Implemented according to Computational Physics, Mark Newman (2013)
 * @param x_i Initial Matrix \f$ x_i \in  \mathbb{R}^{n\cross m} \f$
 * @param t_i Time when x_i is specified
 * @param H Interval after which final x is required
 * @param delta Maximum error allowed per unit time (The error is taken to be the maximum of the error of each element of the matrix)
 * @param evol_func Function that computes \f$ A_\lambda(t) \f$. The function should be of the form void A(double t, gsl_matrix* out, void* params)
 * @param x_f Array to store the final x into. This should be preallocated
 * @param params Parameters to be passed to A(t)
 ******************************************************************************/

void rk4_adaptive_final_matrix_floquet_type_real(gsl_matrix* x_i, double t_i, double H, double delta, void (*A)(double, gsl_matrix*, void*), gsl_matrix* x_f, void* params);


#endif
