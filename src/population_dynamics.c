/*! \file */
#include <stdio.h>
#include <gsl/gsl_math.h>
#include <math.h>
#include "floquet.h"

void print_int_matrix_to_file_csv(int n1, int n2, int** m, FILE* file)
{
	for (int i = 0; i < n1; ++i)
	{
		for (int j = 0; j < n2-1; ++j)
		{
			fprintf(file,"%d,",m[i][j]);
		}
		fprintf(file,"%d\n",m[i][n2-1]);
	}
}

void print_double_matrix_to_file_csv(int n1, int n2, double** m, FILE* file)
{
	for (int i = 0; i < n1; ++i)
	{
		for (int j = 0; j < n2-1; ++j)
		{
			fprintf(file,"%lf,",m[i][j]);
		}
		fprintf(file,"%lf\n",m[i][n2-1]);
	}
}


void fitness_periodic(double t, gsl_matrix* A_val, void* param)
{
	double d = *((double*) param);
	double sint = sin(2.*M_PI*t);
	gsl_matrix_set(A_val,0,0,sint-d);
	gsl_matrix_set(A_val,0,1,d);
	gsl_matrix_set(A_val,1,0,d);
	gsl_matrix_set(A_val,1,1,-sint-d);
}

int main()
{
	int n = 2;
	double T = M_PI;

	double start = 0.;
	double end = 100.;
	int nstep = 1024;
	int* stability = (int*) malloc(nstep*sizeof(int));
	gsl_complex* largest_multiplier = NULL;
	double* largest_multiplier_abs = (double*) malloc(nstep*sizeof(double));

	floquet_get_stability_array_real_single_param_general(n, fitness_periodic, T, start, end, nstep, stability, largest_multiplier, largest_multiplier_abs);
	
	FILE* file = fopen("stability.csv","w");
	for (int i = 0; i < nstep-1; ++i)
	{
		fprintf(file,"%d,",stability[i]);
	}
	fprintf(file,"%d",stability[nstep-1]);
	fclose(file);

	file = fopen("largest_multiplier_abs.csv","w");
	for (int i = 0; i < nstep-1; ++i)
	{
		fprintf(file,"%lf,",largest_multiplier_abs[i]);
	}
	fprintf(file,"%lf",largest_multiplier_abs[nstep-1]);
	fclose(file);

	file = fopen("extraparams.txt","w");
	fprintf(file," Dominant Floquet Multiplier as a function of Dispersal rate $d$\n");
	fprintf(file,"$d$\n");
	fprintf(file,"$\\max(|\\rho|)$\n");
	fprintf(file,"%lf\n",start);
	fprintf(file,"%lf\n",end);
	fprintf(file,"%d\n",nstep);
	fclose(file);

	free(stability);
	free(largest_multiplier_abs);
	return 0;
}