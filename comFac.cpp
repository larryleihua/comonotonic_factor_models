/*---------------------------------------------*/
/* number of factors can be arbitrary (cuhre)  */
/* adjusts needs if use Gaussian-Quandrature   */
/*---------------------------------------------*/

#include <iostream>
#include <fstream>
#include <RcppArmadillo.h>
#include <RcppGSL.h>
#include <vector>
#include <complex>
#include <string>
#include <numeric>
#include <algorithm>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>

#define qMAX 1e+100
#define qMIN 1e-100
#define parMAX 100
#define parMIN 0.01
#define TOL 1e-6
#define MAXIT 200

// [[Rcpp::depends(RcppGSL)]]

// [[Rcpp::depends(RcppArmadillo)]]



// Comonotonic Factor BB7 copulas

using namespace std;

struct G_params
  {
    double u;
    double de;
    double th;
  };

struct psi_params
{
    double u;
    Rcpp::NumericVector de;
    Rcpp::NumericVector th;
};


///////////////////////////////////////////////////////////////////////////////////////
///  Martin Ridout                                                                   //
///  Generating random numbers from a distribution specified by its Laplace transform//
///////////////////////////////////////////////////////////////////////////////////////

typedef std::complex<double> (* LTfunc)(std::complex<double> s,  double de, double th);

//only used for getting G from LTI, there is another version called psi()
std::complex<double> LTI(std::complex<double> s, double de, double th)
{
    std::complex<double> tmp;
    std::complex<double> psii;
    std::complex<double> out(1.0, 0);

    tmp = 1.0 - std::pow( (1.0 + s), (-1/de));
    psii = 1.0 - std::pow(tmp, (1/th));
    out *= psii;
    return out;
}


double psi(double s, Rcpp::NumericVector de, Rcpp::NumericVector th) // LT of independent sum is the product of LTs
{
    double tmp, psii;
    double out = 1.0;
    int i;
    for(i=0; i<de.size(); ++i)
    {
        if( de[i] != 0  &&  th[i] != 0 )
        {
            tmp = 1.0 - std::pow( (1 + s), (-1 / de[i]) );
            psii = 1.0 - std::pow( tmp, (1/th[i]) );
            out *= psii;
        }
    }
    return out;
}

double psi_univ(double s, double de, double th)
{
    double tmp, out;
        tmp = std::pow( (1+s), (-1 / de) );
        out = 1 - std::pow( (1 - tmp), (1/th) );
    return(out);
}


double qGfromLT(double u, LTfunc ltpdf, double de, double th)
{
    const double tol = TOL;
    const double x0 = 1;
    const double xinc = 2;
    const int m = 11;
    const int L = 1;
    const int A = 19;
    const int nburn = 38;
    
    std::complex<double> I(0.0, 1.0);
   
    double x, A2L, expxt;
    double cdf = 0;
    double pdf = 0;
    double lower, upper;
    double ltx_re, t, pdfsum, cdfsum;
    std::complex<double> ltx;
    
    int i = 0;
    int maxiter = MAXIT;
    int nterms = nburn + m*L;
   
    std::vector<std::complex<double> > y(nterms);
    std::vector<std::complex<double> > z(nterms);
    std::vector<std::complex<double> > expy(nterms);
    std::vector<std::complex<double> > ltpvec(nterms);
    std::vector<std::complex<double> > ltzexpy(nterms);
    std::vector<double> ltzexpy_re(nterms);
    std::vector<double> ltzexpy2_re(nterms);
    std::vector<double> sum11, sum22;
    std::vector<double> par_sum(nterms);
    std::vector<double> par_sum2(nterms);
    std::vector<double> coef(m+1);
    std::complex<double> tmp;
    
    for(i=0; i<nterms; ++i)
    {   
        tmp.real((i+1) / L);
        tmp.imag(0.0);
        y[i] = M_PI * I * tmp;
        expy[i] = std::exp(y[i]);
    }
    
    A2L = 0.5 * A / L;
    expxt = std::exp(A2L) / L;
    double pow2m = std::pow(2, m);
    
	
	// for(i = 0; i < m+1; ++i){coef[i] = gsl_sf_choose(m, i) / pow2m;}
    double tem=1.;
    for(i = 0; i < m+1; ++i) 
    {
		coef[i] = tem/ pow2m; 
		tem*=(m-i)/(i+1.);
    }
	
    ////////////////////////////////////   
    int kount = 0;
    
    t = x0 / xinc;

    /*-------------------------------------------------------------------*
    while (kount0 < maxiter && cdf < umax)
    {
        t = xinc * t;
        kount0 = kount0 + 1;
        x = A2L / t;
        pdfsum = 0;
        cdfsum = 0;
               
        for(i=0; i<nterms; ++i)
        {
            z[i] = x + y[i] / t;
            ltpvec[i] = ltpdf(z[i], de, th);
            ltzexpy[i] = ltpvec[i] * expy[i];
            ltzexpy_re[i] = ltzexpy[i].real();
            ltzexpy2_re[i] = (ltzexpy[i] / z[i]).real();
        }

        sum1.clear();
        sum2.clear();

        ltx = ltpdf(x, de, th);
        ltx_re = ltx.real();
        std::partial_sum( ltzexpy_re.begin(), ltzexpy_re.end(), std::back_inserter(sum1));
        std::partial_sum( ltzexpy2_re.begin(), ltzexpy2_re.end(), std::back_inserter(sum2));

        for(i =0; i < nterms; ++i)
        {
            par_sum[i] = 0.5 * ltx_re  + sum1[i];
            par_sum2[i] = 0.5 * ltx_re / x + sum2[i];
        }
        
        for(i=0; i < m+1; i+=L)
        {
            pdfsum = pdfsum + coef[i] * par_sum[nburn+i-1];
            cdfsum = cdfsum + coef[i] * par_sum2[nburn+i-1];
        }
        pdf = pdfsum * expxt / t;
        cdf = cdfsum * expxt / t;

        
        if (!set1st && cdf > umin)
        {
            cdf1 = cdf;
            pdf1 = pdf;
            t1 = t;
            set1st = true;
        }
    }
    
    upplim = t;
    
    *-------------------------------------------------------------------*/

    /*--------------------------------
    # Now use modified Newton-Raphson
    #--------------------------------*/

    lower = 0;
    upper = qMAX;
    t = 1;
    cdf = 1;
    pdf = 10000;
        
    while ( kount < maxiter && std::abs(u - cdf) > tol )
    {
        kount += 1;
        t = t - (cdf-u) / pdf; 
        
        if (t < lower || t > upper)
        {
            t = 0.5 * (lower + upper);
        }

        x = A2L / t;
        pdfsum = 0;
        cdfsum = 0;

        for(i=0; i<nterms; ++i)
        {
            z[i] = x + y[i] / t;
            ltpvec[i] = ltpdf(z[i], de, th);
            ltzexpy[i] = ltpvec[i] * expy[i];
            ltzexpy_re[i] = ltzexpy[i].real();
            ltzexpy2_re[i] = (ltzexpy[i] / z[i]).real();
        }

        sum11.clear();
        sum22.clear();
        
        ltx = ltpdf(x, de, th);
        ltx_re = ltx.real();
        std::partial_sum( ltzexpy_re.begin(), ltzexpy_re.end(), std::back_inserter(sum11));
        std::partial_sum( ltzexpy2_re.begin(), ltzexpy2_re.end(), std::back_inserter(sum22));
        

        for(i =0; i < nterms; ++i)
        {
            par_sum[i] = 0.5 * ltx_re  + sum11[i];
            par_sum2[i] = 0.5 * ltx_re / x + sum22[i];
        }
        
        for(i=0; i < m+1; i+=L)
        {
            pdfsum = pdfsum + coef[i] * par_sum[nburn+i-1];
            cdfsum = cdfsum + coef[i] * par_sum2[nburn+i-1];
        }
        pdf = pdfsum * expxt / t;
        cdf = cdfsum * expxt / t;

        if ( cdf <= u )
        {
            lower = t;
        }
        else
        {
            upper = t;
        }
    }
    return t;
}

double qG(double u, double de, double th)
{
    if(u == 0)
    {
        return 0.0;
    }
    else
    {
        return qGfromLT(u, (LTfunc)LTI, de, th);
    }
}


double psi1(double s, Rcpp::NumericVector de, Rcpp::NumericVector th)
{
    double out = 0;
    int d = de.size();
    double psi10_i, psi0_i;
    int i;
    
    double psiprod = psi(s, de, th);   // LT of sum of indep rv is the prod
    
        for(i = 0; i < d; ++i)
        {
            if( de[i] != 0  &&  th[i] != 0 )
            {
                psi10_i = ( -1 / (de[i]*th[i]) ) * (std::pow(( 1-std::pow((1+s), (-1/de[i])) ), (1/th[i] - 1))) * (std::pow((1+s), (-1/de[i]-1))); 
                psi0_i = psi_univ(s, de[i], th[i]);
                out += psiprod / psi0_i * psi10_i;
            }
        }
    
    return out;
}

double invpsi(double u, Rcpp::NumericVector de, Rcpp::NumericVector th)
{
    const double tol = TOL;
    const double x0 = 1;
    const double xinc = 2;
    
    double LT = 1;
    double LT1 = 0;
    double lower, upper;
    double t;
    
    int maxiter = MAXIT;
    
    ////////////////////////////////////   
    int kount = 0;
    
    t = x0 / xinc;
    
    /*-------------------------------------------------------------------*
    while (kount0 < maxiter && LT > tmin)
    {
        t = xinc * t;
        kount0 = kount0 + 1;
        
        LT1 = psi1(t, de, th);
        LT = psi(t, de, th);
        
        if (!set1st && LT <= 1)
        {
            LT0 = LT;
            LT10 = LT1;
            t1 = t;
            set1st = true;
        }
    }
    
    upplim = t;
    
    *-------------------------------------------------------------------*/

    /*--------------------------------
    # Now use modified Newton-Raphson
    #--------------------------------*/

    lower = 0;
    upper = qMAX;
    t = 1;
    LT = 0.5;
    LT1 = -1;
    
    while ( kount < maxiter && std::abs(u - LT) > tol )
    {
        kount += 1;
        t = t - (LT-u) / LT1; 
        
        if (t < lower || t > upper)
        {
            t = 0.5 * (lower + upper);
        }

        LT1 = psi1(t, de, th);
        LT = psi(t, de, th);

        if ( LT > u )
        {
            lower = t;
        }
        else
        {
            upper = t;
        }
    }
    return t;
}

double psi1inv(double t, Rcpp::NumericVector de, Rcpp::NumericVector th)  // \psi'(\psi^{-1})
{
    double inv = invpsi(t, de, th);
    double out = psi1(inv, de, th);
    return out;
}

/*
double G(double x, double de, double th, int M)  // Fixed Talbot for geeting cdf of G based on LT of G
{
  double Vk, cotVk, out;
  std::complex<double> sk, sigk, wk, expsk, tmp2;
  std::complex<double> I(0,1);
	double sumtmp = 0.0;
	int k;
	
	double tem1 = 0.4 * M;
	double cc = exp(tem1) / (2*M);
	
  for(k = 1; k < M; k++)
	{
            Vk = k * PI / M;
            cotVk = 1 / tan(Vk);
            sk = tem1 * Vk * (cotVk + I);
            sigk = 1.0 + I * ( Vk + ( Vk * cotVk - 1.0) * cotVk);
            expsk = std::exp(sk);
            wk = expsk * sigk / sk;
            tmp2 = wk * x * LTI( sk / x, de, th);
            sumtmp = sumtmp + tmp2.real();
  }
	out = cc * (LTI(tem1/x, de, th).real()) + (0.4 / x) * sumtmp;
  return(out);
}

double Groot(double x, void *params) // G function for getting G^{-1}
{
  struct G_params *p = (struct G_params *) params;
  double u = p->u;
  double de = p->de;
  double th = p->th;
  int M = p->M;
  return ( G(x, de, th, M) - u);
}


double qG(double u, double de, double th, int M)
{
    if(u == 0)
    {
        return 0;
    }else if(u == 1)
    {
        return qMAX;
    }else if( (G(qMIN, de, th, M) > u)  &&  (G(qMAX, de, th, M) > u) )
    {
        return qMIN;
    }else if( (G(qMIN, de, th, M) < u)  &&  (G(qMAX, de, th, M) < u) )
    {
        return qMAX;
    }else
    {  
        int status;
        int iter = 0, max_iter = 10000;
        const gsl_root_fsolver_type *T;
        gsl_root_fsolver *ss;
        double r = 0.0;
        double out = 0;
        double x_lo = qMIN;
        double x_hi = qMAX;
        gsl_function F;
        struct G_params params = {u, de, th, M};

        gsl_set_error_handler_off();

        F.function = &Groot;
        F.params = &params;

        T = gsl_root_fsolver_brent;
        ss = gsl_root_fsolver_alloc(T);
        gsl_root_fsolver_set (ss, &F, x_lo, x_hi);

        do
        {
        iter++;
        status = gsl_root_fsolver_iterate (ss);
        r = gsl_root_fsolver_root (ss);
        x_lo = gsl_root_fsolver_x_lower (ss);
        x_hi = gsl_root_fsolver_x_upper (ss);
        status = gsl_root_test_interval (x_lo, x_hi, TOL, TOL);
        }
        while (status == GSL_CONTINUE && iter < max_iter);

        gsl_root_fsolver_free(ss);	

        if (status == GSL_SUCCESS)
        {
            out = r;
        }	
        else
        {
            out = (x_lo+x_hi)/2;        
        }
        return out;
    }	
}
*/



/*
double psiroot(double s, void *params)
{
    struct psi_params *p = (struct psi_params *) params;
    double u = p->u;
    Rcpp::NumericVector de = p->de;
    Rcpp::NumericVector th = p->th;
    double out = psi(s, de, th) - u;
    return(out);
}


double invpsi_0(double u, Rcpp::NumericVector de, Rcpp::NumericVector th)
{
    if(u == 0)
    {
        return qMAX;
    }else if(u == 1)
    {
        return 0;
    }else if( (psi(qMIN, de, th) > u)  &&  (psi(qMAX, de, th) > u) )
    {
        return qMAX;
    }else if( (psi(qMIN, de, th) < u)  &&  (psi(qMAX, de, th) < u) )
    {
        return qMIN;
    }else
    {  
        int status;
        int iter = 0, max_iter = 10000;
        double out = 0;
        const gsl_root_fsolver_type *T;
        gsl_root_fsolver *ss;
        double r = 0.0;
        double x_lo = qMIN;
        double x_hi = qMAX;
        gsl_function F;
        struct psi_params params = {u, de, th};

        gsl_set_error_handler_off();

        F.function = &psiroot;
        F.params = &params;

        T = gsl_root_fsolver_brent;
        ss = gsl_root_fsolver_alloc(T);
        gsl_root_fsolver_set (ss, &F, x_lo, x_hi);

        do
        {
          iter++;
          status = gsl_root_fsolver_iterate (ss);
          r = gsl_root_fsolver_root (ss);
          x_lo = gsl_root_fsolver_x_lower (ss);
          x_hi = gsl_root_fsolver_x_upper (ss);
          status = gsl_root_test_interval (x_lo, x_hi, TOL, TOL);
        }
        while (status == GSL_CONTINUE && iter < max_iter);

        gsl_root_fsolver_free(ss);	

        if (status == GSL_SUCCESS)
        {
            // Rcpp::Rcout << "invpsi:u/de/th/r => " << u << "==" << de  << "==" <<  th  << "==" <<  r << std::endl;      
            out = r;
        }	
        else
        {
            // Rcpp::Rcout << "invpsi:u/de/th/x_lo/x_hi => " << u << "==" << de  << "==" <<  th  << "==" <<  x_lo  << "==" <<  x_hi << std::endl;
            out = (x_lo+x_hi)/2;
        }
        return out;
    }	
}
*/


///////////////////////////
// Gaussian Quadrature   //
///////////////////////////

#define EPS 3.0e-11
// xq: nodes;  wq: weights
void gauleg(int nq, Rcpp::NumericVector& xq, Rcpp::NumericVector& wq)
{
    int m,j,i,n;
    double z1,z,xm,xl,pp,p3,p2,p1;
    n = nq; 
    m = (n+1)/2;
  
    // boundary points 
    double x1 = 0;
    double x2 = 1;
    
    xm = 0.5*(x2 + x1);
    xl = 0.5*(x2 - x1);
    for(i=1;i<=m;++i)
    {
        z = cos(3.14159265358979323846*(i-0.25)/(n+0.5));
        do
        {
            p1=1.0;
            p2=0.0;
            for(j=1;j<=n;j++)
            {
                p3=p2;
                p2=p1;
                p1=((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j;
            }
            pp=n*(z*p1-p2)/(z*z-1.0);
            z1=z;
            z=z1-p1/pp;
        }while (fabs(z-z1) > EPS);
        
        xq[i] = xm-xl*z;
        xq[n+1-i]=xm+xl*z;
        wq[i]=2.0*xl/((1.0-z*z)*pp*pp);
        wq[n+1-i]=wq[i];
  }
}
#undef EPS

/////////////////////////////////////
// bivariate normal copula density //
/////////////////////////////////////
double denB1(double u, double v, double r)
{
    double tem0, tem1, tem2, tem3, tem4, x, y;
    
    x = gsl_cdf_ugaussian_Pinv(u);
    y = gsl_cdf_ugaussian_Pinv(v);
    
    tem0 = (1- std::pow(r,2));
    tem1 = std::pow( tem0, -0.5 );
    tem2 = std::pow(x,2) +  std::pow(y,2);
    tem3 = std::exp( -0.5 / tem0 * ( tem2 - 2*r*x*y  ) );
    tem4 = std::exp( tem2 / 2 );
    
    return tem1 * tem3 * tem4;
}

//////////////////////////////////
// multivariate Gaussian copula //
//////////////////////////////////
  
const double log2pi = std::log(2.0 * M_PI);

double dmvnrm_arma(arma::rowvec x, arma::rowvec mean, arma::mat sigma, bool logd = false)
{ 
    int xdim = x.n_cols;
    double out;
    arma::mat rooti = arma::trans(arma::inv(trimatu(arma::chol(sigma))));
    double rootisum = arma::sum(log(rooti.diag()));
    double constants = -(static_cast<double>(xdim)/2.0) * log2pi;

    arma::vec z = rooti * arma::trans( x - mean );
    out = constants - 0.5 * arma::sum(z%z) + rootisum;     

    if(logd == false)
    {
        out = std::exp(out);
    }
    return out;
}

// this is the density of the copula for connecting those factors

double denFAC(Rcpp::NumericVector uvec, Rcpp::NumericVector parFAC)
{
    // as an experiment, use multivariate Gaussian copula first   
    int udim = uvec.length();
    bool logd = false;
    Rcpp::NumericVector Pinv(udim);
    Rcpp::NumericVector duniv(udim);
    Rcpp::NumericVector mean0(udim);
    Rcpp::NumericMatrix sigma0(udim, udim);
    
    int i,j;   
    
    for(i = 0; i < udim; ++i)
    {
        Pinv[i] = gsl_cdf_ugaussian_Pinv(uvec[i]);
        duniv[i] =  gsl_ran_ugaussian_pdf(Pinv[i]);
    }

    arma::rowvec x(Pinv.begin(), Pinv.size(), false);
    arma::rowvec mean(mean0.begin(), mean0.size(), false);
    arma::mat sigma(sigma0.begin(), udim, udim, false);
    
    int k = 0;
    for(i = 0; i < udim; ++i)
    {
        sigma(i,i) = 1.0; // the diagonal of the correlation matrix
        for(j = i+1; j < udim; ++j)
        {
            sigma(i, j) = parFAC[k];
            sigma(j, i) = parFAC[k];
            ++k;
        }
    }
    
    double dmvn = dmvnrm_arma(x, mean, sigma, logd);
    double duniv_all = 1;
    for(i =0; i < udim; ++i)
    {
        duniv_all *= duniv[i];
    }
    return dmvn / duniv_all;
}



///////////////////////////
// bivariate BB1 density //
///////////////////////////
double denBB1(double u, double v, double de, double th)
{
    double t1, t2, t3, t4, t5, t6, t7, t9, t10, t12, t13, t14, t16;
    double t18, t21, t22, t23, t25, t28, t29, t30, t33, t35, t40, t52, pdf;
    t1 = std::pow(u,-1.0*th);
    t2 = t1-1.0;
    t3 = std::pow(t2,1.0*de);
    t4 = std::pow(v,-1.0*th);
    t5 = t4-1.0;
    t6 = std::pow(t5,1.0*de);
    t7 = t3+t6;
    t9 = std::pow(t7,1.0/de);
    t10 = 1.0+t9;
    t12 = std::pow(t10,-1.0/th);
    t13 = t9*t9;
    t14 = t12*t13;
    t16 = 1/v;
    t18 = 1/t5;
    t21 = t7*t7;
    t22 = 1/t21;
    t23 = t10*t10;
    t25 = t22/t23;
    t28 = t1/u;
    t29 = 1/t2;
    t30 = t28*t29;
    t33 = t12*t9;
    t35 = t4*th;
    t40 = 1/t10;
    t52 = t35*t16*t18;
    pdf = t14*t6*t4*t16*t18*t25*t3*t30-t33*t6*t35*t16*t18*t22*t3*t28*t29*t40+t33*t3*t28*t29*t22*t40*t6*de*t52+t14*t3*t30*t25*t6*t52;
    return pdf;
}


///////////////////////////
// bivariate BB7 density //
///////////////////////////
double denBB7(double u, double v, double de, double th)
{
	double t1, t2, t3, t4, t5, t6, t7, t8, t9, t11, t12, t14, t15, t16, t18, t20;
	double t23, t24, t25, t27, t30, t31, t32, t35, t37, t42, t54, pdf;
	t1 = 1.0-u;
	t2 = std::pow(t1,1.0*th);
	t3 = 1.0-t2;
	t4 = std::pow(t3,-1.0*de);
	t5 = 1.0-v;
	t6 = std::pow(t5,1.0*th);
	t7 = 1.0-t6;
	t8 = std::pow(t7,-1.0*de);
	t9 = t4+t8-1.0;
	t11 = std::pow(t9,-1.0/de);
	t12 = 1.0-t11;
	t14 = std::pow(t12,1.0/th);
	t15 = t11*t11;
	t16 = t14*t15;
	t18 = 1/t5;
	t20 = 1/t7;
	t23 = t9*t9;
	t24 = 1/t23;
	t25 = t12*t12;
	t27 = t24/t25;
	t30 = t2/t1;
	t31 = 1/t3;
	t32 = t30*t31;
	t35 = t14*t11;
	t37 = t6*th;
	t42 = 1/t12;
	t54 = t37*t18*t20;
    pdf = -t16*t8*t6*t18*t20*t27*t4*t32+t35*t8*t37*t18*t20*t24*t4*t30*t31*t42+t35*t4*t30*t31*t24*t42*t8*de*t54+t16*t4*t32*t27*t8*t54;
	return pdf;
}




///////////////////////////////
//       CBB7intgrad         //
///////////////////////////////
// [[Rcpp::export]]
double CBB7intgrad(Rcpp::NumericVector uvec, Rcpp::NumericVector tvec, 
                   Rcpp::NumericMatrix DM, Rcpp::NumericVector devec, 
                   Rcpp::NumericVector thvec, Rcpp::NumericVector parFAC, int parMode)
{
    int i, j, k = 0;
    int f = DM.ncol();
    int d = DM.nrow();
	
    Rcpp::NumericMatrix demat(d,f);
    Rcpp::NumericMatrix thmat(d,f);
    Rcpp::NumericMatrix invG(d,f);
    
    Rcpp::NumericVector invG_i(d);
    Rcpp::NumericVector invpsi_i(d);
    Rcpp::NumericVector psi1inv_i(d);
    double tem1 = 0;
    double tem2 = 0;
    double out =0;

    for(j = 0; j < f; ++j)  // i: row,   j: col
    {
        for(i = 0; i < d; ++i)
        {
            if(DM(i, j) == 1)
            {
                demat(i, j) = devec[k];
                thmat(i, j) = thvec[k];
                invG(i, j) = qG(uvec[j], demat(i, j), thmat(i, j));
                invG_i[i] += invG(i, j);
                ++k;
            }
        }
    }

    Rcpp::NumericVector demat_i(f); 
    Rcpp::NumericVector thmat_i(f);
    
    for(i = 0; i < d; ++i)
    {
        demat_i = demat(i,Rcpp::_);
        thmat_i = thmat(i,Rcpp::_);
        invpsi_i[i] = invpsi(tvec[i], demat_i, thmat_i);
        psi1inv_i[i] = psi1inv(tvec[i], demat_i, thmat_i);
    }
        
    for(i=0; i<d; ++i)
    {
        tem1 = tem1 + invG_i[i] * invpsi_i[i]; 
        tem2 = tem2 + std::log(invG_i[i]) - std::log(-psi1inv_i[i]);
    }

    
    switch( parMode )
    {
        case 0: 
            out = std::exp( tem2 - tem1 );
            break;
        case 1:
            out = std::exp( tem2 - tem1 ) * denFAC(uvec, parFAC);
            break;
        default:
            out = std::exp( tem2 - tem1 );
            break;
    } 
    
    
    /*
    if(parMode == 0)
    {
        out = std::exp( tem2 - tem1 );
    }
    else if(parMode == 1)
    {
        out = std::exp( tem2 - tem1 ) * denFAC(uvec, parFAC);
    }
    else
    {
        out = std::exp( tem2 - tem1 );
    }
    */
    
    if( !R_finite(out) )
    {
	out = 0;
    }
    return out;
}

/////////////////////////////
////////   density    ///////
/////////////////////////////


double nllk_f(int kk, Rcpp::NumericVector par, Rcpp::NumericVector parM, 
              Rcpp::NumericMatrix dat, Rcpp::NumericMatrix DM, int nq,
              Rcpp::NumericVector parFAC, int parMode)
{
// parMode: 0: fitting only clusters
//          1: fitting dependence between clusters, dependence in clusters are given
//          2: fitting all
    double nllk=0;
    int nf = DM.ncol();
    int len = std::pow(nq, nf);

    if( std::exp(Rcpp::min(par)) < parMIN  || std::exp(Rcpp::max(par)) > parMAX )
    {
        nllk = 0;
    }
    else
    {    
        int i, j;
        double den = 0;
        double intg = 0;
        Rcpp::NumericVector uvec(nf);
        int len_par = par.length();
        Rcpp::NumericVector parvec;

        Rcpp::NumericVector tvec = dat(kk-1, Rcpp::_);

        /// setup Gaussian quadrature
        Rcpp::NumericVector xl(nq), wl(nq);
        gauleg(nq, xl, wl);
        Rcpp::NumericVector x1 = Rcpp::rep( Rcpp::rep_each(xl, std::pow(nq,0)), std::pow(nq, nf-1) );
        Rcpp::NumericVector x2 = Rcpp::rep( Rcpp::rep_each(xl, std::pow(nq,1)), std::pow(nq, nf-2) );
        
        Rcpp::NumericVector w1 = Rcpp::rep( Rcpp::rep_each(wl, std::pow(nq,0)), std::pow(nq, nf-1) );
        Rcpp::NumericVector w2 = Rcpp::rep( Rcpp::rep_each(wl, std::pow(nq,1)), std::pow(nq, nf-2) );
        
        
        /// setup parameters
        //double de = std::exp(par[len_par-2]);
        //double th = std::exp(par[len_par-1])+1;
        
        for(i=0; i < len_par; ++i)  // the last two are for copula between U1 and U2
        {
            for(j=0; j < parM[i]; ++j)
            {
                parvec.push_back(par[i]);
            }
        }
        
        int NN0 = parvec.length();   
        Rcpp::NumericVector devec(NN0/2);
        Rcpp::NumericVector thvec(NN0/2);
        
        for(i=0;i<NN0/2;++i)
        {
            devec[i] = std::exp(parvec[i]);            // de > 0
            thvec[i] = std::exp(parvec[NN0/2+i]) + 1;  // th >= 1
        }

        
        for(i=0;i < len; ++i)
        {
            uvec[0] = x1[i];
            uvec[1] = x2[i];
            
            intg = CBB7intgrad(uvec, tvec, DM, devec, thvec, parFAC, parMode);
            den = den + w1[i]*w2[i] * intg;
        }

        nllk = - std::log(den);
        
        if( !R_finite(nllk) )
        {
            nllk = 0;
        }
    }
    return nllk;
}

