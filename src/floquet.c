#include "floquet.h"

int floquet_get_stability_reals(int n, void (*A)(double, gsl_matrix*, void*), void* params, double T)
{
	gsl_matrix* X = gsl_matrix_alloc(n,n);
	gsl_matrix_set_identity(X);

	gsl_matrix* B = gsl_matrix_alloc(n,n);

	rk4_adaptive_final_matrix_floquet_type_real(X, 0., T, ERR_TOL, A, B, params);

	gsl_matrix* eigenvectors = gsl_matrix_alloc(n,n);
	gsl_vector_complex* eigenvals = gsl_vector_complex_alloc(n);

	gsl_eigen_nonsymm_workspace* w = gsl_eigen_nonsymm_alloc(n);

	int n_eigenvals_evaluated = n;
	
	if(gsl_eigen_nonsymm(eigenvectors,eigenvals,w))
	{
		n_eigenvals_evaluated = w->n_evals;
	}

	gsl_eigen_nonsymm_free(w);

	double mult_max_abs = -HUGE_VAL;
	double ev_test;
	for (int i = 0; i < n_eigenvals_evaluated; ++i)
	{
		ev_test = gsl_complex_logabs(eigenvals[i]);
		if (mult_max_abs < ev_test)
		{
			mult_max_abs = ev_test;
		}
	}

	if (mult_max_abs > ERR_EIGEN_TOL)
	{
		return 1;
	}
	else if (mult_max_abs < ERR_EIGEN_TOL)
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