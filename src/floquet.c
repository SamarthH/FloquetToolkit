/*! \file */ 
#include "floquet.h"

int floquet_get_stability_reals_general(int n, void (*A)(double, gsl_matrix*, void*), void* params, double T, gsl_complex* largest_multiplier, double* largest_multiplier_abs)
{
	gsl_matrix* X = gsl_matrix_alloc(n,n);
	gsl_matrix_set_identity(X);

	gsl_matrix* B = gsl_matrix_alloc(n,n);
	bulsto_final_matrix_floquet_type_real(X, 0., T, ERR_TOL, A, B, params);

	gsl_vector_complex* eigenvals = gsl_vector_complex_alloc(n);

	gsl_eigen_nonsymm_workspace* w = gsl_eigen_nonsymm_alloc(n);

	int n_eigenvals_evaluated = n;
	
	int err_code = gsl_eigen_nonsymm(B,eigenvals,w);
	if(err_code)
	{
		n_eigenvals_evaluated = w->n_evals;
	}

	double mult_max_abs = -HUGE_VAL;
	gsl_complex mult_max;
	double ev_test;
	for (int i = 0; i < n_eigenvals_evaluated; ++i)
	{
		ev_test = gsl_complex_logabs(gsl_vector_complex_get(eigenvals,i));
		if (mult_max_abs < ev_test)
		{
			mult_max_abs = ev_test;
			mult_max = gsl_vector_complex_get(eigenvals,i);
		}
	}

	if (largest_multiplier != NULL)
	{
		*largest_multiplier = mult_max;
	}
	if (largest_multiplier_abs != NULL)
	{
		*largest_multiplier_abs = gsl_complex_abs(mult_max);
	}

	gsl_eigen_nonsymm_free(w);
	gsl_vector_complex_free(eigenvals);
	gsl_matrix_free(B);
	gsl_matrix_free(X);

	if (mult_max_abs > ERR_EIGEN_TOL)
	{
		return 1;
	}
	else if (mult_max_abs < (-ERR_EIGEN_TOL))
	{
		if (n_eigenvals_evaluated < n)
		{
			return 2;
		}
		return -1;
	}
	else
	{
		return 0;
	}
}

void floquet_get_stability_array_real_single_param_general(int n, void (*A)(double, gsl_matrix*, void*), double T, double start, double end, int nstep, int* stability, gsl_complex* largest_multiplier, double* largest_multiplier_abs)
{
	gsl_complex* mult_temp;
	double* mult_abs_temp;

	if (largest_multiplier)
	{
		mult_temp = largest_multiplier;
	}
	else
	{
		mult_temp = (gsl_complex*) malloc(nstep*sizeof(gsl_complex));
	}

	if (largest_multiplier_abs)
	{
		mult_abs_temp = largest_multiplier_abs;
	}
	else
	{
		mult_abs_temp = (double*) malloc(nstep*sizeof(double));
	}

	double step = (end-start)/(nstep-1);
	#pragma omp parallel for
	for (int i = 0; i < nstep; ++i)
	{
		double param = start + step*i;
		stability[i] = floquet_get_stability_reals_general(n,A,&param,T,mult_temp+i,mult_abs_temp+i);
	}

	if(!largest_multiplier)
	{
		free(mult_temp);
	}
	if(!largest_multiplier_abs)
	{
		free(mult_abs_temp);
	}
}

void floquet_get_stability_array_real_double_param_general(int n, void (*A)(double, gsl_matrix*, void*), double T, double* start, double* end, int* nstep, int** stability, gsl_complex** largest_multiplier, double** largest_multiplier_abs)
{
	gsl_complex** mult_temp;
	double** mult_abs_temp;

	if (largest_multiplier)
	{
		mult_temp = largest_multiplier;
	}
	else
	{
		mult_temp = (gsl_complex**) malloc(nstep[0]*sizeof(gsl_complex*));
		for (int i = 0; i < nstep[0]; ++i)
		{
			mult_temp[i] = (gsl_complex*) malloc(nstep[1]*sizeof(gsl_complex));
		}
	}

	if (largest_multiplier_abs)
	{
		mult_abs_temp = largest_multiplier_abs;
	}
	else
	{
		mult_abs_temp = (double**) malloc(nstep[0]*sizeof(double*));
		for (int i = 0; i < nstep[0]; ++i)
		{
			mult_abs_temp[i] = (double*) malloc(nstep[1]*sizeof(double));
		}
	}

	double step[2] = {(end[0]-start[0])/(nstep[0]-1), (end[1]-start[1])/(nstep[1]-1)};
	#pragma omp parallel for collapse(2) schedule(guided)
	for (int i = 0; i < nstep[0]; ++i)
	{
		for (int j = 0; j < nstep[1]; ++j)
		{
			double param[2] = {start[0] + step[0]*i, start[1] + step[1]*j };
			stability[i][j] = floquet_get_stability_reals_general(n,A,param,T,&mult_temp[i][j],&mult_abs_temp[i][j]);
		}
	}

	if(!largest_multiplier)
	{
		for (int i = 0; i < nstep[0]; ++i)
		{
			free(mult_temp[i]);
		}
		free(mult_temp);
	}

	if(!largest_multiplier_abs)
	{
		for (int i = 0; i < nstep[0]; ++i)
		{
			free(mult_abs_temp[i]);
		}
		free(mult_abs_temp);
	}
}