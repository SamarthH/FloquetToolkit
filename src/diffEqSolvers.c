/*! \file */ 
#include "diffEqSolvers.h"

void __rk4_single_vector(int n, double* x, double t, double h, void (*evol_func)(double*, double, double*, void*), double* x_f, void* params)
{
	double k1[n], k2[n], k3[n], k4[n], k2_in[n], k3_in[n], k4_in[n];
	evol_func(x,t,k1,params);
	cblas_dscal(n,h,k1,sizeof(k1[0]));

	cblas_dcopy(n,x,sizeof(x[0]),k2_in,sizeof(k2_in[0]));
	cblas_daxpy(n,0.5,k1,sizeof(k1[0]),k2_in,sizeof(k2_in[0]));
	evol_func(k2_in,t+0.5*h,k2,params);
	cblas_dscal(n,h,k2,sizeof(k2[0]));

	cblas_dcopy(n,x,sizeof(x[0]),k3_in,sizeof(k3_in[0]));
	cblas_daxpy(n,0.5,k2,sizeof(k2[0]),k3_in,sizeof(k3_in[0]));
	evol_func(k3_in,t+0.5*h,k3,params);
	cblas_dscal(n,h,k3,sizeof(k3[0]));

	cblas_dcopy(n,x,sizeof(x[0]),k4_in,sizeof(k4_in[0]));
	cblas_daxpy(n,1.,k3,sizeof(k3[0]),k4_in,sizeof(k4_in[0]));
	evol_func(k4_in,t+h,k4,params);
	cblas_dscal(n,h,k4,sizeof(k4[0]));

	for (int i = 0; i < n; ++i)
	{
		x_f[i] = x[i] + (1./6.)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
	}
}

void rk4_fixed_final_vector_real(int ndim, double* x_i, double t_i, double H, double h, void (*evol_func)(double*, double, double*, void*), double* x_f, void* params)
{
	double t = t_i;
	double t_f = t_i + H;

	cblas_dcopy(ndim,x_i,sizeof(x_i[0]),x_f,sizeof(x_f[0]));
	while(t<t_f)
	{
		__rk4_single_vector(ndim,x_f,t,h,evol_func,x_f, params);
		t += h;
	}
}

void rk4_fixed_final_matrix_floquet_type_real(gsl_matrix* x_i, double t_i, double H, double h, void (*A)(double, gsl_matrix*, void*), gsl_matrix* x_f, void* params)
{
	int nr = x_i->size1;
	int nc = x_i->size2;
	gsl_matrix* k[4];
	gsl_matrix* k_in[4];
	gsl_matrix* A_val = gsl_matrix_alloc(nr,nc);
	for (int i = 0; i < 4; ++i)
	{
		k[i] = gsl_matrix_calloc(nr,nc);
		k_in[i] = gsl_matrix_calloc(nr,nc);
	}

	double t = t_i;
	double t_f = t_i + H;
	gsl_matrix_memcpy(x_f,x_i);
	while (t<t_f)
	{
		A(t,A_val,params);
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, h, A_val, x_f, 0., k[0]);

		gsl_matrix_memcpy(k_in[0], k[0]);
		gsl_matrix_scale(k_in[0],0.5);
		gsl_matrix_add(k_in[0],x_f);

		A(t+0.5*h,A_val,params);
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, h, A_val, k_in[0], 0., k[1]);

		gsl_matrix_memcpy(k_in[1], k[1]);
		gsl_matrix_scale(k_in[1],0.5);
		gsl_matrix_add(k_in[1],x_f);

		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, h, A_val, k_in[1], 0., k[2]);

		gsl_matrix_memcpy(k_in[2], k[2]);
		gsl_matrix_add(k_in[2],x_f);

		A(t+h,A_val,params);
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, h, A_val, k_in[2], 0., k[3]);

		gsl_matrix_memcpy(k_in[3],k[3]);
		gsl_matrix_add(k_in[3],k[2]);
		gsl_matrix_add(k_in[3],k[2]);
		gsl_matrix_add(k_in[3],k[1]);
		gsl_matrix_add(k_in[3],k[1]);
		gsl_matrix_add(k_in[3],k[0]);
		gsl_matrix_scale(k_in[3], 1./6.);
		gsl_matrix_add(x_f,k_in[3]);
		t += h;
	}

	for (int i = 0; i < 4; ++i)
	{
		gsl_matrix_free(k[i]);
		gsl_matrix_free(k_in[i]);
	}
}

void rk4_fixed_final_matrix_floquet_type_complex(gsl_matrix_complex* x_i, double t_i, double H, double h_, void (*A)(double, gsl_matrix_complex*, void*), gsl_matrix_complex* x_f, void* params)
{
	int nr = x_i->size1;
	int nc = x_i->size2;
	gsl_matrix_complex* k[4];
	gsl_matrix_complex* k_in[4];
	gsl_matrix_complex* A_val = gsl_matrix_complex_alloc(nr,nc);
	for (int i = 0; i < 4; ++i)
	{
		k[i] = gsl_matrix_complex_alloc(nr,nc);
		k_in[i] = gsl_matrix_complex_alloc(nr,nc);
	}

	gsl_complex h = gsl_complex_rect(h_,0.);
	gsl_complex zero = gsl_complex_rect(0.,0.);
	gsl_complex half = gsl_complex_rect(0.5,0.);
	gsl_complex one_sixth = gsl_complex_rect(1./6.,0.);

	double t = t_i;
	double t_f = t_i + H;
	gsl_matrix_complex_memcpy(x_f,x_i);
	while (t<t_f)
	{
		A(t,A_val,params);
		gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, h, A_val, x_f, zero, k[0]);

		gsl_matrix_complex_memcpy(k_in[0], k[0]);
		gsl_matrix_complex_scale(k_in[0],half);
		gsl_matrix_complex_add(k_in[0],x_f);

		A(t+0.5*h_,A_val,params);
		gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, h, A_val, k_in[0], zero, k[1]);

		gsl_matrix_complex_memcpy(k_in[1], k[1]);
		gsl_matrix_complex_scale(k_in[1],half);
		gsl_matrix_complex_add(k_in[1],x_f);

		gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, h, A_val, k_in[1], zero, k[2]);

		gsl_matrix_complex_memcpy(k_in[2], k[2]);
		gsl_matrix_complex_add(k_in[2],x_f);

		A(t+h_,A_val,params);
		gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, h, A_val, k_in[2], zero, k[3]);

		gsl_matrix_complex_memcpy(k_in[3],k[3]);
		gsl_matrix_complex_add(k_in[3],k[2]);
		gsl_matrix_complex_add(k_in[3],k[2]);
		gsl_matrix_complex_add(k_in[3],k[1]);
		gsl_matrix_complex_add(k_in[3],k[1]);
		gsl_matrix_complex_add(k_in[3],k[0]);
		gsl_matrix_complex_scale(k_in[3], one_sixth);
		gsl_matrix_complex_add(x_f,k_in[3]);
		t += h_;
	}

	for (int i = 0; i < 4; ++i)
	{
		gsl_matrix_complex_free(k[i]);
		gsl_matrix_complex_free(k_in[i]);
	}
}

void rk4_adaptive_final_matrix_floquet_type_real(gsl_matrix* x_i, double t_i, double H, double delta, void (*A)(double, gsl_matrix*, void*), gsl_matrix* x_f, void* params)
{
	double h_min = H/RK4_MAX_SLICES;
	int nr = x_i->size1;
	int nc = x_i->size2;
	gsl_matrix* k[4];
	gsl_matrix* k_in[4];
	gsl_matrix* A_val = gsl_matrix_alloc(nr,nc);
	for (int i = 0; i < 4; ++i)
	{
		k[i] = gsl_matrix_calloc(nr,nc);
		k_in[i] = gsl_matrix_calloc(nr,nc);
	}

	double t = t_i;
	double t_f = t_i + H;
	double h = H/10.; // Initial h. This is a bit conservative, but will be refined further.
	gsl_matrix_memcpy(x_f,x_i);

	gsl_matrix* x1 = gsl_matrix_alloc(nr,nc);
	gsl_matrix* x2 = gsl_matrix_alloc(nr,nc);

	while (t<t_f)
	{
		// Evaluate x1
		gsl_matrix_memcpy(x1,x_f);

		A(t,A_val,params);
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, h, A_val, x_f, 0., k[0]);

		gsl_matrix_memcpy(k_in[0], k[0]);
		gsl_matrix_scale(k_in[0],0.5);
		gsl_matrix_add(k_in[0],x_f);

		A(t+0.5*h,A_val,params);
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, h, A_val, k_in[0], 0., k[1]);

		gsl_matrix_memcpy(k_in[1], k[1]);
		gsl_matrix_scale(k_in[1],0.5);
		gsl_matrix_add(k_in[1],x_f);

		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, h, A_val, k_in[1], 0., k[2]);

		gsl_matrix_memcpy(k_in[2], k[2]);
		gsl_matrix_add(k_in[2],x_f);

		A(t+h,A_val,params);
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, h, A_val, k_in[2], 0., k[3]);

		gsl_matrix_memcpy(k_in[3],k[3]);
		gsl_matrix_add(k_in[3],k[2]);
		gsl_matrix_add(k_in[3],k[2]);
		gsl_matrix_add(k_in[3],k[1]);
		gsl_matrix_add(k_in[3],k[1]);
		gsl_matrix_add(k_in[3],k[0]);
		gsl_matrix_scale(k_in[3], 1./6.);
		gsl_matrix_add(x1,k_in[3]);

		t += h;

		A(t,A_val,params);
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, h, A_val, x_f, 0., k[0]);

		gsl_matrix_memcpy(k_in[0], k[0]);
		gsl_matrix_scale(k_in[0],0.5);
		gsl_matrix_add(k_in[0],x1);

		A(t+0.5*h,A_val,params);
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, h, A_val, k_in[0], 0., k[1]);

		gsl_matrix_memcpy(k_in[1], k[1]);
		gsl_matrix_scale(k_in[1],0.5);
		gsl_matrix_add(k_in[1],x1);

		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, h, A_val, k_in[1], 0., k[2]);

		gsl_matrix_memcpy(k_in[2], k[2]);
		gsl_matrix_add(k_in[2],x1);

		A(t+h,A_val,params);
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, h, A_val, k_in[2], 0., k[3]);

		gsl_matrix_memcpy(k_in[3],k[3]);
		gsl_matrix_add(k_in[3],k[2]);
		gsl_matrix_add(k_in[3],k[2]);
		gsl_matrix_add(k_in[3],k[1]);
		gsl_matrix_add(k_in[3],k[1]);
		gsl_matrix_add(k_in[3],k[0]);
		gsl_matrix_scale(k_in[3], 1./6.);
		gsl_matrix_add(x1,k_in[3]);

		t -= h;

		// Evaluate x2
		gsl_matrix_memcpy(x2,x_f);

		A(t,A_val,params);
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 2*h, A_val, x_f, 0., k[0]);

		gsl_matrix_memcpy(k_in[0], k[0]);
		gsl_matrix_scale(k_in[0],0.5);
		gsl_matrix_add(k_in[0],x_f);

		A(t+h,A_val,params);
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 2*h, A_val, k_in[0], 0., k[1]);

		gsl_matrix_memcpy(k_in[1], k[1]);
		gsl_matrix_scale(k_in[1],0.5);
		gsl_matrix_add(k_in[1],x_f);

		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 2*h, A_val, k_in[1], 0., k[2]);

		gsl_matrix_memcpy(k_in[2], k[2]);
		gsl_matrix_add(k_in[2],x_f);

		A(t+2*h,A_val,params);
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 2*h, A_val, k_in[2], 0., k[3]);

		gsl_matrix_memcpy(k_in[3],k[3]);
		gsl_matrix_add(k_in[3],k[2]);
		gsl_matrix_add(k_in[3],k[2]);
		gsl_matrix_add(k_in[3],k[1]);
		gsl_matrix_add(k_in[3],k[1]);
		gsl_matrix_add(k_in[3],k[0]);
		gsl_matrix_scale(k_in[3], 1./6.);
		gsl_matrix_add(x2,k_in[3]);

		// Evaluate Error
		gsl_matrix_sub(x2,x1);
		double rho_to_the_fourth = (30.*h*delta)/GSL_MAX(gsl_matrix_max(x2), -1.*gsl_matrix_min(x2));
		rho_to_the_fourth = pow(rho_to_the_fourth,0.25);
		if (rho_to_the_fourth>1)
		{
			gsl_matrix_memcpy(x_f,x1);
			gsl_matrix_scale(x2, -1./15.);
			gsl_matrix_add(x_f,x2);
			t += 2*h;
			h = h*(GSL_MIN(rho_to_the_fourth, RK4_MAX_SCALE));
			h = GSL_MIN(0.5*(t_f-t),h);
		}
		else if(h > h_min)
		{
			h = h*(GSL_MAX(rho_to_the_fourth, RK4_MIN_SCALE));
		}
		else
		{
			gsl_matrix_memcpy(x_f,x1);
			gsl_matrix_scale(x2, -1./15.);
			gsl_matrix_add(x_f,x2);
			t += 2*h;
			h = h_min;
		}
	}

	gsl_matrix_free(A_val);

	for (int i = 0; i < 4; ++i)
	{
		gsl_matrix_free(k[i]);
		gsl_matrix_free(k_in[i]);
	}

	gsl_matrix_free(x1);
	gsl_matrix_free(x2);
}

void __midpoint_method(gsl_matrix* x, double t_i, double H, double h, void (*A)(double, gsl_matrix*, void*), gsl_matrix* x_f, void* params, gsl_matrix* y, gsl_matrix* eval)
{
	double t = t_i;
	double t_f = t_i + H;
	double h_2 = h/2.;
	gsl_matrix_memcpy(x_f,x);
	
	A(t,eval,params);
	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,h_2,eval,x_f,0.,y);
	gsl_matrix_add(y,x_f);

	A(t+h_2,eval,params);
	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,h,eval,y,1.,x_f);
	
	t+=h;

	while(t<t_f)
	{
		A(t,eval,params);
		gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,h,eval,x_f,1.,y);

		A(t+h_2,eval,params);
		gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,h,eval,y,1.,x_f);

		t+= h;
	}

	A(t,eval,params);
	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,h_2,eval,x_f,1.,y);
	gsl_matrix_add(x_f,y);
	gsl_matrix_scale(x_f,0.5);

}

double __bulsto_final_matrix_floquet_type_real_main(gsl_matrix* x_i, double t_i, double H, double delta, void (*A)(double, gsl_matrix*, void*), gsl_matrix* x_f, void* params, gsl_matrix* y, gsl_matrix* eval, gsl_matrix** R1, gsl_matrix** R2, gsl_matrix* epsilon)
{
	//int ndim = x_i->size1;
	

	//printf("0 ERR %e %e %e %e\n", gsl_matrix_get(x_i,0,0), gsl_matrix_get(x_i,0,1), gsl_matrix_get(x_i,1,0), gsl_matrix_get(x_i,1,1));

	int n = 1;
	double h = H;
	__midpoint_method(x_i, t_i, H, h, A, R1[0], params, y, eval);
	double error = HUGE_VAL;

	gsl_matrix** temp;

	//printf("%d %e %e %e %e %e\n", n, error, gsl_matrix_get(R1[0],0,0), gsl_matrix_get(R1[0],0,1), gsl_matrix_get(R1[0],1,0), gsl_matrix_get(R1[0],1,1));
	while (error > H*delta && n<BULSTO_STEP_MAX)
	{
		n++;
		h = H/n;

		// Swapping the arrays of matrices to save space
		temp = R2;
		R2 = R1;
		R1 = temp;

		double scaler = n/(n-1.);
		scaler *= scaler;

		double scale = 1.;
		__midpoint_method(x_i, t_i, H, h, A, R1[0], params, y, eval);
		for (int m = 1; m < n; ++m)
		{
			scale *= scaler;
			gsl_matrix_memcpy(epsilon,R1[m-1]);
			gsl_matrix_sub(epsilon,R2[m-1]);
			gsl_matrix_scale(epsilon, 1./(scale-1.));

			gsl_matrix_memcpy(R1[m],R1[m-1]);
			gsl_matrix_add(R1[m],epsilon);
		}
		error = GSL_MAX(gsl_matrix_max(epsilon), -1.*gsl_matrix_min(epsilon));
		//printf("%d %e %e %e %e %e %e\n", n, H, error, gsl_matrix_get(R1[0],0,0), gsl_matrix_get(R1[0],0,1), gsl_matrix_get(R1[0],1,0), gsl_matrix_get(R1[0],1,1));
	}

	gsl_matrix_memcpy(x_f,R1[n-1]);

	return error;
}

double __bulsto_final_matrix_floquet_type_real_runner(int nlayer, gsl_matrix* x_i, double t_i, double H, double delta, void (*A)(double, gsl_matrix*, void*), gsl_matrix* x_f, void* params, gsl_matrix* y, gsl_matrix* eval, gsl_matrix** R1, gsl_matrix** R2, gsl_matrix* epsilon)
{
	int ndim = x_i->size1;
	gsl_matrix* xfin = gsl_matrix_alloc(ndim,ndim);
	double error = __bulsto_final_matrix_floquet_type_real_main(x_i, t_i, H, delta, A, xfin, params, y, eval, R1, R2, epsilon);
	nlayer++;
	if (error > H*delta && nlayer < BULSTO_MAX_LAYERS)
	{
		error = __bulsto_final_matrix_floquet_type_real_runner(nlayer, x_i, t_i, H/2., delta, A, xfin, params, y, eval, R1, R2, epsilon);
		error += __bulsto_final_matrix_floquet_type_real_runner(nlayer, xfin, t_i+(H/2.), H/2., delta, A, xfin, params, y, eval, R1, R2, epsilon);
	}
	gsl_matrix_memcpy(x_f,xfin);
	gsl_matrix_free(xfin);
	return error;
}

void bulsto_final_matrix_floquet_type_real(gsl_matrix* x_i, double t_i, double H, double delta, void (*A)(double, gsl_matrix*, void*), gsl_matrix* x_f, void* params)
{
	// This program only initializes and provides and frees temp variables
	int ndim = x_i->size1;
	gsl_matrix** R1 = (gsl_matrix**) malloc((BULSTO_STEP_MAX+1)*sizeof(gsl_matrix*));
	gsl_matrix** R2 = (gsl_matrix**) malloc((BULSTO_STEP_MAX+1)*sizeof(gsl_matrix*));

	for (int i = 0; i <= BULSTO_STEP_MAX; ++i)
	{
		R1[i] = gsl_matrix_alloc(ndim,ndim);
		R2[i] = gsl_matrix_alloc(ndim,ndim);
	}

	gsl_matrix* y = gsl_matrix_alloc(ndim,ndim);
	gsl_matrix* eval = gsl_matrix_alloc(ndim,ndim);
	gsl_matrix* epsilon = gsl_matrix_calloc(ndim,ndim);

	__bulsto_final_matrix_floquet_type_real_runner(0, x_i, t_i, H, delta, A, x_f, params, y, eval, R1, R2, epsilon);
	//printf("%e %e %e\n",error, H, error/H);
	for (int i = 0; i <= BULSTO_STEP_MAX; ++i)
	{
		gsl_matrix_free(R1[i]);
		gsl_matrix_free(R2[i]);
	}
	free(R1);
	free(R2);
	gsl_matrix_free(y);
	gsl_matrix_free(eval);
	gsl_matrix_free(epsilon);
}