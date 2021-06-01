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

		//printf("%lf %lf %e %lf %lf %lf %lf ",gsl_matrix_get(x_f,0,0), gsl_matrix_get(x_f,1,1), h,gsl_matrix_get(x1,0,0), gsl_matrix_get(x1,1,1),gsl_matrix_get(x2,0,0), gsl_matrix_get(x2,1,1)  );
		// Evaluate Error
		gsl_matrix_sub(x2,x1);
		double rho_to_the_fourth = (30.*h*delta)/GSL_MAX(gsl_matrix_max(x2), -1.*gsl_matrix_min(x2));
		//printf("%e %e\n",GSL_MAX(gsl_matrix_max(x2), -1.*gsl_matrix_min(x2)),rho_to_the_fourth );
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
		else
		{
			h = h*(GSL_MAX(rho_to_the_fourth, RK4_MIN_SCALE));
		}
	}

	for (int i = 0; i < 4; ++i)
	{
		gsl_matrix_free(k[i]);
		gsl_matrix_free(k_in[i]);
	}

	gsl_matrix_free(x1);
	gsl_matrix_free(x2);
}
