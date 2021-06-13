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


void hill_meissner(double t, gsl_matrix* A_val, void* param)
{
	double* par_temp = (double*) param;
	gsl_matrix_set_zero(A_val);
	gsl_matrix_set(A_val,0,1,1.);
	gsl_matrix_set(A_val,1,0,-(par_temp[1] + par_temp[0]*GSL_SIGN(t-M_PI)));
}

int main()
{
	int n = 2;
	double T = 2*M_PI;

	double start[2] = {9.,-1.};
	double end[2] = {0., 9.};
	int nstep[2] = {64,64};
	int** stability = (int**) malloc(nstep[0]*sizeof(int*));
	gsl_complex** largest_multiplier = NULL;
	double** largest_multiplier_abs = (double**) malloc(nstep[0]*sizeof(double*));

	for (int i = 0; i < nstep[0]; ++i)
	{
		stability[i] = (int*) malloc(nstep[1]*sizeof(int));
		largest_multiplier_abs[i] = (double*) malloc(nstep[1]*sizeof(double));
	}


	floquet_get_stability_array_real_double_param_general(n, hill_meissner, T, start, end, nstep, stability, largest_multiplier, largest_multiplier_abs);
	
	FILE* file = fopen("stability.csv","w");
	print_int_matrix_to_file_csv(nstep[0],nstep[1],stability,file);
	fclose(file);

	file = fopen("largest_multiplier_abs.csv","w");
	print_double_matrix_to_file_csv(nstep[0],nstep[1],largest_multiplier_abs,file);
	fclose(file);

	file = fopen("extraparams.txt","w");
	fprintf(file,"Hill-Meissner Stability Plot\n");
	fprintf(file,"$\\omega^2$\n");
	fprintf(file,"$\\alpha^2$\n");
	fprintf(file,"%lf %lf\n",start[0],start[1]);
	fprintf(file,"%lf %lf\n",end[0],end[1]);
	fprintf(file,"%d %d\n",nstep[0], nstep[1]);
	fclose(file);
	
	for (int i = 0; i < nstep[0]; ++i)
	{
		free(stability[i]);
		free(largest_multiplier_abs[i]);
	}

	free(stability);
	free(largest_multiplier_abs);
	return 0;
}