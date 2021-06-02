#include <stdio.h>
#include <gsl/gsl_math.h>
#include <math.h>
#include "floquet.h"

void mathieu(double t, gsl_matrix* A_val, void* param)
{
	double* par_temp = (double*) param;
	gsl_matrix_set_zero(A_val);
	gsl_matrix_set(A_val,0,1,1.);
	gsl_matrix_set(A_val,1,0,-(par_temp[1] + par_temp[0]*cos(2.*t)));
}

int main()
{
	int n = 2;
	double T = M_PI;

	double start[2] = {0.,-5.};
	double end[2] = {60., 20.};
	int nstep[2] = {80,80};
	int** stability = (int**) malloc(nstep[0]*sizeof(int*));
	gsl_complex** largest_multiplier = NULL;
	double** largest_multiplier_abs = (double**) malloc(nstep[0]*sizeof(double*));

	for (int i = 0; i < nstep[0]; ++i)
	{
		stability[i] = (int*) malloc(nstep[1]*sizeof(int));
		largest_multiplier_abs[i] = (double*) malloc(nstep[1]*sizeof(double));
	}


	floquet_get_stability_array_real_double_param_general(n, mathieu, T, start, end, nstep, stability, largest_multiplier, largest_multiplier_abs);

	
	for (int i = 0; i < nstep[0]; ++i)
	{
		for (int j = 0; j < nstep[1]; ++j)
		{
			printf("%d ",stability[i][j]);
		}
		printf("\n");
	}

	printf("\n");
	printf("\n");

	for (int i = 0; i < nstep[0]; ++i)
	{
		for (int j = 0; j < nstep[1]; ++j)
		{
			printf("%lf ",largest_multiplier_abs[i][j]);
		}
		printf("\n");
	}

	for (int i = 0; i < nstep[0]; ++i)
	{
		free(stability[i]);
		free(largest_multiplier_abs[i]);
	}

	free(stability);
	free(largest_multiplier_abs);
	return 0;
}