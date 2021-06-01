#include <stdio.h>
#include <gsl/gsl_math.h>
#include <math.h>
#include "diffEqSolvers.h"

void Vdot(double* x, double t, double* xdot, void* param)
{
	xdot[0] = x[0];
}

void A(double t, gsl_matrix* A_val, void* param)
{
	gsl_matrix_set_identity(A_val);
	gsl_matrix_set(A_val,1,1,t);
}

int main()
{
	double t_i = 0;
	double H = 1;
	double x_i[1];
	double x_f[1];
	double h = 1e-6;
	x_i[0] = 1.;
	rk4_fixed_final_vector_real(1, x_i, t_i, H, h, Vdot, x_f,NULL);
	printf("%0.9lf\n",x_f[0]);
	
	gsl_matrix* x_i_mat = gsl_matrix_alloc(2,2);
	gsl_matrix* x_f_mat = gsl_matrix_alloc(2,2);
	gsl_matrix_set_identity(x_i_mat);

	rk4_fixed_final_matrix_floquet_type_real(x_i_mat, t_i, H, h, A, x_f_mat, NULL);
	for (int i = 0; i < 2; ++i)
	{
		for (int j = 0; j < 2; ++j)
		{
			printf("%lf ",gsl_matrix_get(x_f_mat,i,j));
		}
		printf("\n");
	}

	rk4_adaptive_final_matrix_floquet_type_real(x_i_mat, t_i, H, 1e-6, A, x_f_mat, NULL);
	for (int i = 0; i < 2; ++i)
	{
		for (int j = 0; j < 2; ++j)
		{
			printf("%0.9lf ",gsl_matrix_get(x_f_mat,i,j));
		}
		printf("\n");
	}
	gsl_matrix_free(x_i_mat);
	gsl_matrix_free(x_f_mat);
	return 0;
}