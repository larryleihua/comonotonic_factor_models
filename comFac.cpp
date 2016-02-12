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
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_hyperg.h>
#include <boost/math/special_functions/bessel.hpp> // for real arguments

#ifdef COMPLEX_BESSEL
#include <complex_bessel.h> // for complex arguments, https://github.com/valandil/complex_bessel
#endif


#include "comFac.hpp"

#define qMAX 1e+100
#define qMIN 1e-100
#define pMAX 0.999999
#define pMIN 0.000001
#define parMAX 100
#define parMIN 0.01
#define TOL 1e-6
#define MAXIT 1000

// [[Rcpp::depends(RcppGSL)]]

// [[Rcpp::depends(RcppArmadillo)]]

using std::pow; using std::log;
using std::exp; using std::vector;
using std::complex; using std::size_t;
using Rcpp::NumericVector;
using Rcpp::NumericMatrix;

const double log2pi = log(2.0 * M_PI);

// to make functions coherent, all parameter inputs are of the type vector<double>
// unless 'R' is used to pass parameters when Rcpp::NumericVector is used. 

typedef complex<double> (* LTfunc_complex)(complex<double>,  vector<double>);
typedef double (* LTfunc_univariate)(double, vector<double>);
typedef double (* LTfunc_vector)(double, vector<double>);


// B6, Gumbel
complex<double> LTA_complex(complex<double> s, vector<double> par)
{
    complex<double> tmp;
    complex<double> psii;
    complex<double> out(1.0, 0);
    double de = par[0];

    tmp = pow(s, (1.0/de));
    psii = exp(-tmp);
    out *= psii;
    return out;
}

// B4, Clayton
complex<double> LTB_complex(complex<double> s, vector<double> par)
{
    complex<double> tmp;
    complex<double> psii;
    complex<double> out(1.0, 0);
    double de = par[0];

    tmp = 1.0 + s;
    psii = pow(tmp, -1.0/de);
    out *= psii;
    return out;
}

// for BB7
complex<double> LTI_complex(complex<double> s, vector<double> par)
{
    complex<double> tmp;
    complex<double> psii;
    complex<double> out(1.0, 0);
    double de = par[0];
    double th = par[1];

    tmp = 1.0 - pow( (1.0 + s), (-1.0/de));
    psii = 1.0 - pow(tmp, (1.0/th));
    out *= psii;
    return out;
}

// for BB1
complex<double> LTE_complex(complex<double> s, vector<double> par)
{
    complex<double> tmp;
    complex<double> psii;
    complex<double> out(1.0, 0);
    double de = par[0];
    double th = par[1];

    tmp = 1.0 + pow(s, (1.0/de));
    psii = pow(tmp, (-1.0/th));
    out *= psii;
    return out;
}

complex<double> LTA_LTE_complex(complex<double> s, vector<double> par)
{
    complex<double> out(1.0, 0);
    vector<double> par1;
    vector<double> par2;
    par1.push_back(par[0]);
    par2.push_back(par[1]);
    par2.push_back(par[2]);
    
    out = out * LTA_complex(s, par1) * LTE_complex(s, par2);
    return out;
}

complex<double> LTB_LTB_complex(complex<double> s, vector<double> par)
{
    complex<double> out(1.0, 0);
    vector<double> par1;
    vector<double> par2;
    par1.push_back(par[0]);
    par2.push_back(par[1]);

    out = out * LTB_complex(s, par1) * LTB_complex(s, par2);
    return out;
}

complex<double> LTE_LTA_complex(complex<double> s, vector<double> par)
{
    complex<double> out(1.0, 0);
    vector<double> par1;
    vector<double> par2;
    par1.push_back(par[0]);
    par1.push_back(par[1]);
    par2.push_back(par[2]);
    
    out = out * LTE_complex(s, par1) * LTA_complex(s, par2);
    return out;
}

// for LT of Generalized Inverse Gaussian   // to-do
#ifdef COMPLEX_BESSEL
// for LT of Inverse Gamma
complex<double> LTIG_complex(complex<double> s, vector<double> par)
{
  complex<double> tmp;
  complex<double> psii;
  complex<double> out(1.0, 0);
  double al = par[0];
  double gal = gsl_sf_gamma(al);
  complex<double> sroot = sqrt(s);
  
  tmp = sp_bessel::besselK(al, 2.0*sroot);
  psii = 2.0 * pow(s, (al/2)) * tmp / gal;
  out *= psii;
  return out;
}

complex<double> LTGIG_complex(complex<double> s, vector<double> par)
{
    complex<double> tem1, tem2, tem3;
    complex<double> psii;
    complex<double> out(1.0, 0);
    double th = par[0];
    double la = par[1];
    double xi = par[2];
	
    tem1 = pow(la / (la + 2.0*s), th / 2);
    tem2 = sp_bessel::besselK(th, pow( (la+2.0*s)*xi, 0.5));
    tem3 = sp_bessel::besselK(th, pow( la*xi, 0.5));
    psii = tem1 * tem2 / tem3;
    out *= psii;
    return out;
}
#endif

double LTGIG_vector(double s, vector<double> par)
{
    double tem1, tem2, tem3, psii;
    double out = 1.0;
    size_t i;
    
    size_t d = par.size() / 3;
    vector<double> th;
    vector<double> la;
    vector<double> xi;

    for(i=0;i<d;++i)
    {
        th.push_back(par[i]);
        la.push_back(par[i+d]);
        xi.push_back(par[i+2*d]);
    }
    
    for(i=0; i<d; ++i)
    {
        tem1 = pow(la[i] / (la[i] + 2*s), th[i] / 2);
        tem2 = boost::math::cyl_bessel_k(th[i], pow( (la[i]+2*s)*xi[i], 0.5));
        tem3 = boost::math::cyl_bessel_k(th[i], pow( la[i]*xi[i], 0.5));
        psii = tem1 * tem2 / tem3;
        out *= psii;
    }
    return out;
}

double IML_vector(double s, vector<double> par)
{
    double tmp, psii;
    double out = 1.0;
    size_t i;
    
    size_t d = par.size() / 2;
    vector<double> de;
    vector<double> va;

    for(i=0;i<d;++i)
    {
        de.push_back(par[i]);
        va.push_back(par[i+d]);
    }
    
    for(i=0; i<d; ++i)
    {
        if( de[i] != 0  &&  va[i] != 0 )
        {
            tmp = pow(s, 1.0/de[i]);
            psii = 1.0 - gsl_sf_beta_inc(de[i], 1.0/va[i], tmp / (1.0+tmp));
            out *= psii;
        }
    }
    return out;
}

// LT of independent sum is the product of LTs
double LTI_vector(double s, vector<double> par) 
{
    double tmp, psii;
    double out = 1.0;
    size_t i;
    
    size_t d = par.size() / 2;
    vector<double> de;
    vector<double> th;

    for(i=0;i<d;++i)
    {
        de.push_back(par[i]);
        th.push_back(par[i+d]);
    }
    
    for(i=0; i<d; ++i)
    {
        if( de[i] != 0  &&  th[i] != 0 )
        {
            tmp = 1.0 - pow( (1.0 + s), (-1.0 / de[i]) );
            psii = 1.0 - pow( tmp, (1.0/th[i]) );
            out *= psii;
        }
    }
    return out;
}

// LT of independent sum is the product of LTs
double LTA_vector(double s, vector<double> par)
{
    double tmp, psii;
    double out = 1.0;
    size_t i;
    
    size_t d = par.size();
    vector<double> de;

    for(i=0;i<d;++i)
    {
        de.push_back(par[i]);
    }

    for(i=0; i<d; ++i)
    {
        if( de[i] != 0 )
        {
            tmp = pow(s, (1.0/de[i]));
            psii = exp(-tmp);
            out *= psii;
        }
    }
    return out;
}

// LT of independent sum is the product of LTs
double LTB_vector(double s, vector<double> par)
{
    double tmp, psii;
    double out = 1.0;
    size_t i;
    
    size_t d = par.size();
    vector<double> de;

    for(i=0;i<d;++i)
    {
        de.push_back(par[i]);
    }

    for(i=0; i<d; ++i)
    {
        if( de[i] != 0 )
        {
            tmp = 1.0+s;
            psii = pow(tmp, -1.0/de[i]);
            out *= psii;
        }
    }
    return out;
}

double LTE_vector(double s, vector<double> par) // LT of independent sum is the product of LTs
{
    double tmp, psii;
    double out = 1.0;
    size_t i;
    
    size_t d = par.size() / 2;
    vector<double> de;
    vector<double> th;

    for(i=0;i<d;++i)
    {
        de.push_back(par[i]);
        th.push_back(par[i+d]);
    }

    for(i=0; i<d; ++i)
    {
        if( de[i] != 0  &&  th[i] != 0 )
        {
            tmp = 1.0 + pow(s, (1.0/de[i]));
            psii = pow(tmp, (-1.0/th[i]));
            out *= psii;
        }
    }
    return out;
}

//  note the difference between LTA and LTA_vector
double LTA(double s, vector<double> par) 
{
    double tmp;
    double out;
    double de = par[0];
    tmp = pow(s, (1.0/de));
    out = exp(-tmp);
    return out;
}

double LTB(double s, vector<double> par) 
{
    double tmp;
    double out;
    double de = par[0];
    tmp = 1.0 + s;
    out = pow(tmp, -1.0 / de);
    return out;
}

double IML(double s, vector<double> par)
{
    double tmp, out;
    double de = par[0];
    double va = par[1];
    tmp = pow(s, 1.0/de);
    out = 1.0 - gsl_sf_beta_inc(de, 1.0/va, tmp / (1.0+tmp));
    return out;
}

double LTI(double s, vector<double> par)
{
    double tmp, out;
    double de = par[0];
    double th = par[1];
    tmp = pow( (1.0+s), (-1.0 / de) );
    out = 1.0 - pow( (1.0 - tmp), (1.0/th) );
    return out;
}

double LTE(double s, vector<double> par)
{
    double tmp, out;
    double de = par[0];
    double th = par[1];
    tmp = 1.0 + pow(s, (1.0/de));
    out = pow(tmp, (-1.0/th));
    return out;
}


double LTGIG(double s, vector<double> par) 
{
    double tem1, tem2, tem3;
    double out;
    double th = par[0];
    double la = par[1];
    double xi = par[2];
    
    tem1 = pow(la / (la + 2*s), th / 2);
    tem2 = boost::math::cyl_bessel_k(th, pow( (la+2*s)*xi, 0.5));
    tem3 = boost::math::cyl_bessel_k(th, pow( la*xi, 0.5));
    out = tem1 * tem2 / tem3;
    return out;
}

double LTA_LTE(double s, vector<double> par1, vector<double> par2)
{
    double out = LTA(s, par1) * LTE(s, par2);
    return out;
}

double LTE_LTA(double s, vector<double> par1, vector<double> par2)
{
    double out = LTE(s, par1) * LTA(s, par2);
    return out;
}

double LTB_LTB(double s, vector<double> par1, vector<double> par2)
{
    double out = LTB(s, par1) * LTB(s, par2);
    return out;
}

/////////////////////////////////////////////////////////////////////////////////////////
///  Martin Ridout                                                                     //
///  Generating random numbers from a distribution specified by its Laplace transform  //
///  modified from r codes at                                                          //
///  http://www.kent.ac.uk/smsas/personal/msr/rlaptrans.html                           //
/////////////////////////////////////////////////////////////////////////////////////////

double qGfromLT(double u, LTfunc_complex ltpdf, vector<double> par, int& err_msg)
{
    const double tol = TOL;
    const double x0 = 1.0;
    const double xinc = 2;
    const int m = 11;
    const int L = 1;
    const int A = 19;
    const int nburn = 38;
    complex<double> I(0.0, 1.0);

    double x, A2L, expxt;
    double cdf = 0;
    double pdf = 0;
    double lower, upper;
    double ltx_re, t, pdfsum, cdfsum;
    complex<double> ltx;
    
    size_t i = 0;
    size_t maxiter = MAXIT;
    size_t nterms = nburn + m*L;
   
    vector<complex<double> > y(nterms);
    vector<complex<double> > z(nterms);
    vector<complex<double> > expy(nterms);
    vector<complex<double> > ltpvec(nterms);
    vector<complex<double> > ltzexpy(nterms);
    vector<double> ltzexpy_re(nterms);
    vector<double> ltzexpy2_re(nterms);
    vector<double> sum11, sum22;
    vector<double> par_sum(nterms);
    vector<double> par_sum2(nterms);
    vector<double> coef(m+1);
    complex<double> tmp;
    
    for(i=0; i<nterms; ++i)
    {   
        tmp.real((i+1.0) / L);
        tmp.imag(0.0);
        y[i] = M_PI * I * tmp;
        expy[i] = exp(y[i]);
    }
    
    A2L = 0.5 * A / L;
    expxt = exp(A2L) / L;
    double pow2m = pow(2, m);
    	
    // for(i = 0; i < m+1; ++i){coef[i] = gsl_sf_choose(m, i) / pow2m;}
    // suggested by harry
    double tem=1.0;
    for(i = 0; i < m+1; ++i) 
    {
        coef[i] = tem/ pow2m; 
        tem*=(m-i)/(i+1.0);
    }
	
    size_t kount = 0;
    t = x0 / xinc; // omit the steps to find the potential maximum, refer to Ridout

    /*--------------------------------
    # Now use modified Newton-Raphson
    #--------------------------------*/

    lower = 0;
    upper = qMAX;
    t = 1.0;
    cdf = 1.0;
    pdf = 100.0;
        
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
            ltpvec[i] = ltpdf(z[i], par);
            ltzexpy[i] = ltpvec[i] * expy[i];
            ltzexpy_re[i] = ltzexpy[i].real();
            ltzexpy2_re[i] = (ltzexpy[i] / z[i]).real();
        }

        sum11.clear();
        sum22.clear();
        
        ltx = ltpdf(x, par);
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
    
    if(kount < maxiter && std::abs(u - cdf) < tol )
    {
        err_msg = 1;
    }
    else if(kount >= maxiter)
    {
        err_msg = 2;
    }
    return t;
}

double qG(double u, LTfunc_complex LT, vector<double> par, int& err_msg)
{
    if(u == 0)
    {
        return 0.0;
    }
    else
    {
        return qGfromLT(u, LT, par, err_msg);
    }
}

double IML1_vector(double s, vector<double> par)
{
    double out = 0;
    double ze, psi10_i, psi0_i;
    size_t i;
    
    size_t d = par.size() / 2;
    vector<double> de;
    vector<double> va;

    for(i=0;i<d;++i)
    {
        de.push_back(par[i]);
        va.push_back(par[i+d]);
    }
    
    double psiprod = IML_vector(s, par);   // LT of sum of indep rv is the prod
    vector<double> par_i;
    for(i = 0; i < d; ++i)
    {
        if( de[i] != 0  &&  va[i] != 0 )
        {
            ze = 1.0 / va[i] + de[i];
            psi10_i = - pow((1.0 + pow(s, 1.0/de[i])), -ze) / (de[i] * (gsl_sf_beta(de[i], 1.0/va[i])));
            par_i.clear();
            par_i.push_back(de[i]);
            par_i.push_back(va[i]);
            psi0_i = IML(s, par_i);
            out += psiprod / psi0_i * psi10_i;
        }
    }
    return out;
}

double LTI1_vector(double s, vector<double> par)
{
    double out = 0;
    double psi10_i, psi0_i;
    size_t i;
    vector<double> par_i;
    
    size_t d = par.size() / 2;
    vector<double> de;
    vector<double> th;

    for(i=0;i<d;++i)
    {
        de.push_back(par[i]);
        th.push_back(par[i+d]);
    }
    
    double psiprod = LTI_vector(s, par);   // LT of sum of indep rv is the prod
    
    for(i = 0; i < d; ++i)
    {
        if( de[i] != 0  &&  th[i] != 0 )
        {
            psi10_i = ( -1.0 / (de[i]*th[i]) ) * (pow(( 1.0-pow((1.0+s), (-1.0/de[i])) ), (1.0/th[i] - 1.0))) * (pow((1.0+s), (-1.0/de[i]-1.0))); 
            par_i.clear();
            par_i.push_back(de[i]);
            par_i.push_back(th[i]);
            psi0_i = LTI(s, par_i);
            out += psiprod / psi0_i * psi10_i;
        }
    }
    return out;
}

double LTE1_vector(double s, vector<double> par)
{
    double out = 0;
    double psi10_i, psi0_i;
    size_t i;

    size_t d = par.size() / 2;
    vector<double> de;
    vector<double> th;
    vector<double> par_i;
   
    for(i=0;i<d;++i)
    {
        de.push_back(par[i]);
        th.push_back(par[i+d]);
    }

    double psiprod = LTE_vector(s, par);   // LT of sum of indep rv is the prod
    for(i = 0; i < d; ++i)
    {
        if( de[i] != 0  &&  th[i] != 0 )
        {
            psi10_i = ( -1.0 / (de[i]*th[i]) ) * (pow(1.0+pow(s,1.0/de[i]),-1.0/th[i]-1.0)) * (pow(s,1.0/de[i]-1.0)); 
            par_i.clear();
            par_i.push_back(de[i]);
            par_i.push_back(th[i]);
            psi0_i = LTE(s, par_i);
            out += psiprod / psi0_i * psi10_i;
        }
    }
    return out;
}

double LTA1_vector(double s, vector<double> par)
{
    double out = 0;
    double psi10_i, psi0_i;
    size_t i;

    size_t d = par.size();
    vector<double> de;
    vector<double> par_i;

    for(i=0;i<d;++i)
    {
        de.push_back(par[i]);
    }
    
    double psiprod = LTA_vector(s, par);   // LT of sum of indep rv is the prod
    for(i = 0; i < d; ++i)
    {
        if( de[i] != 0)
        {
            psi10_i = exp(-pow(s,1.0/de[i])) / (-de[i]) * (pow(s, 1.0/de[i] -1.0 )); 
            par_i.clear();
            par_i.push_back(de[i]);
            psi0_i = LTA(s, par_i);
            out += psiprod / psi0_i * psi10_i;
        }
    }
    return out;
}

double LTB1_vector(double s, vector<double> par)
{
    double out = 0;
    double psi10_i, psi0_i;
    size_t i;

    size_t d = par.size();
    vector<double> de;
    vector<double> par_i;

    for(i=0;i<d;++i)
    {
        de.push_back(par[i]);
    }
    
    double psiprod = LTB_vector(s, par);   // LT of sum of indep rv is the prod
    for(i = 0; i < d; ++i)
    {
        if( de[i] != 0)
        {
            psi10_i = - pow((1.0+s), -1.0/de[i] -1.0) / de[i]; 
            par_i.clear();
            par_i.push_back(de[i]);
            psi0_i = LTB(s, par_i);
            out += psiprod / psi0_i * psi10_i;
        }
    }
    return out;
}

double LTGIG1_vector(double s, vector<double> par)
{
    double out = 0;
    double psi10_i, psi0_i;
    double tem1, tem2, tmp1, tmp2, K1, K2, K3, K4;
    size_t i;

    size_t d = par.size() / 3;
    vector<double> th;
    vector<double> la;
    vector<double> xi;

    vector<double> par_i;
    
    for(i=0;i<d;++i)
    {
        th.push_back(par[i]);
        la.push_back(par[i+d]);
        xi.push_back(par[i+2*d]);
    }
    
    double psiprod = LTGIG_vector(s, par);   // LT of sum of indep rv is the prod
    for(i = 0; i < d; ++i)
    {
        tem1 = pow( (la[i] + 2*s)*xi[i], 0.5 );
        tem2 = pow( la[i]*xi[i], 0.5 );

        K1 = boost::math::cyl_bessel_k(th[i], tem1);
        K2 = boost::math::cyl_bessel_k(th[i], tem2);
        K3 = boost::math::cyl_bessel_k(th[i]-1, tem1);
        K4 = boost::math::cyl_bessel_k(th[i]+1, tem1);

        tmp1 = -(th[i]*la[i]) * (pow(la[i], th[i]/2 -1)) * pow(la[i]+2*s, -th[i]/2 -1) * K1 / K2;
        tmp2 = - (0.5)* pow(la[i] / (la[i] + 2*s), th[i]/2) * xi[i] / tem1 / K2 * (K3 + K4);
        psi10_i = tmp1 + tmp2;
        
        par_i.clear();
        par_i.push_back(th[i]);
        par_i.push_back(la[i]);
        par_i.push_back(xi[i]);
        psi0_i = LTGIG(s, par_i);
        out += psiprod / psi0_i * psi10_i;
    }
    return out;
}



double LTA1_LTE1(double s, vector<double> par_1, vector<double> par_2)
{
    double de = par_1[0];
    double de1 = par_2[0];
    double th1 = par_2[1];

    double psi1_LTA = exp(-pow(s,1.0/de)) * (pow(s, 1.0/de -1.0 )) / (- de); 
    double psi_LTA = LTA(s, par_1);
    double psi1_LTE = ( -1.0 / (de1*th1) ) * (pow(1.0+pow(s,1.0/de1),-1.0/th1-1.0)) * (pow(s,1.0/de1-1.0)); 
    double psi_LTE = LTE(s, par_2);
    double out = psi1_LTA*psi_LTE + psi1_LTE*psi_LTA;
    return out;
}

// invpsi can handle sum of independent nonnegative random variables if the same family
double invpsi(double u, vector<double> par, int LTfamily)
{
    const double tol = TOL;
    const double x0 = 1.0;
    const double xinc = 2.0;
     
    double LT = 1.0;
    double LT1 = 0;
    double lower, upper;
    double t;
    
    size_t maxiter = MAXIT;
    
    LTfunc_vector LT1_vector;
    LTfunc_vector LT_vector;
    
    switch( LTfamily )
    {
        case 1:
            LT1_vector = &LTE1_vector;
            LT_vector = &LTE_vector;
            break;
        case 4:
            LT1_vector = &LTB1_vector;
            LT_vector = &LTB_vector;
            break;
        case 7:
            LT1_vector = &LTI1_vector;
            LT_vector = &LTI_vector;
            break;
        case 2: // integrated Mittag-Leffler LT, the complex LT hasn't been implemented
            LT1_vector = &IML1_vector;
            LT_vector = &IML_vector;
            break;
        case 6:
            LT1_vector = &LTA1_vector;
            LT_vector = &LTA_vector;
            break;
        case 100:
            LT1_vector = &LTGIG1_vector;
            LT_vector = &LTGIG_vector;
            break;
        default:
            LT1_vector = &LTI1_vector;
            LT_vector = &LTI_vector;
            break;
    }
    
    size_t kount = 0;
    t = x0 / xinc;
    
    /*--------------------------------
    # Now use modified Newton-Raphson
    #--------------------------------*/

    lower = 0;
    upper = qMAX;
    t = 1.0;
    LT = 0.5;
    LT1 = -1.0;
    
    while ( kount < maxiter && std::abs(u - LT) > tol )
    {
        kount += 1;
        t = t - (LT-u) / LT1; 
        
        if (t < lower || t > upper)
        {
            t = 0.5 * (lower + upper);
        }

        LT1 = LT1_vector(t, par);
        LT = LT_vector(t, par);

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

// to-do
double invpsi_LTA_LTE(double u, vector<double> par_1, vector<double> par_2, int& err_msg)
{
    const double tol = TOL;
    const double x0 = 1.0;
    const double xinc = 2.0;
     
    double LT = 1.0;
    double LT1 = 0;
    double lower, upper;
    double t;
    
    size_t maxiter = MAXIT;
    size_t kount = 0;
    t = x0 / xinc;

    /*--------------------------------
    # Now use modified Newton-Raphson
    #--------------------------------*/

    lower = 0;
    upper = qMAX;
    LT = std::min(u+2*tol, pMAX);
    t = LT;
    LT1 = -1.0;
    
    while ( kount < maxiter && std::abs(u - LT) > tol )
    {
        kount += 1;
        t = t - (LT-u) / LT1; 
        
        if (t < lower || t > upper)
        {
            t = 0.5 * (lower + upper);
        }

        LT1 = LTA1_LTE1(t, par_1, par_2);
        LT = LTA_LTE(t, par_1, par_2);

        if ( LT > u )
        {
            lower = t;
        }
        else
        {
            upper = t;
        }
    }
    
    if(kount < maxiter && std::abs(u - LT) < tol )
    {
        err_msg = 1;
    }
    else if(kount >= maxiter)
    {
        err_msg = 2;
    }
    
    return t;
}

/////////////////////////////////////
//   bivariate BB1 copula density  //
/////////////////////////////////////

double dBB1(double u, double v, vector<double> par)
{
    double de, th, de1, th1, ut, vt, x, y, sm, smd, tem, out;
    de = par[0];
    th = par[1];
    de1 = 1.0/de;
    th1 = 1.0/th;
    ut = pow(u,(-th))-1.0; 
    vt = pow(v,(-th))-1.0;
    x = pow(ut,de); 
    y = pow(vt,de);
    sm = x+y; 
    smd = pow(sm, de1);
    tem = pow((1.0+smd),(-th1-2.0)) * (th*(de-1.0)+(th*de+1.0)*smd);
    out = tem*smd*x*y*(ut+1.0)*(vt+1.0)/sm/sm/ut/vt/u/v;
    return out;
}



/////////////////////////////////////
// bivariate Gumbel copula density //
/////////////////////////////////////
double denB6(double u, double v, vector<double> par)
{
    double t1, t2, t3, t4, t5, t7, t9, t10, t11, t13, t14, t18, t19, t31, out;
    double de = par[0];
    t1 = log(u);
    t2 = pow(-t1,1.0*de);
    t3 = log(v);
    t4 = pow(-t3,1.0*de);
    t5 = t2+t4;
    t7 = pow(t5,1.0/de);
    t9 = 1.0/v;
    t10 = 1.0/t3;
    t11 = t9*t10;
    t13 = t5*t5;
    t14 = 1.0/t13;
    t18 = 1.0/u/t1;
    t19 = exp(-t7);
    t31 = t7*t7;
    out = -t7*t4*t11*t14*t2*t18*t19+t7*t2*t18*t14*t19*t4*de*t9*t10+t31*t2*t18*t14*t4*t11*t19;
    if(R_finite(out))
    {
        return out;
    }
    else
    {
        return 0;
    }
}

/////////////////////////////////////
// 1-factor Gumbel copula density //
/////////////////////////////////////
double denF1B6(vector<double> uvec, vector<double> par, int nq)
{
    double out = 0;
    double den_i = 1.0;
    vector<double> par_i(1);
    int i, j;
    int udim = uvec.size();
    vector<double> xl(nq), wl(nq);
    gauleg(nq, xl, wl);
    for(j=0; j<nq; ++j)
    {
        for(i =0; i<udim; ++i)
        {
            par_i.clear();
            par_i.push_back(par[i]);
            den_i *= denB6(uvec[i], xl[j], par_i);            
        }
        out += den_i * wl[j];
        den_i = 1; 
    }
    return out;
}


/////////////////////////////////////
//   1-factor BB1 copula density   //
/////////////////////////////////////
double denF1BB1(vector<double> uvec, vector<double> par, int nq)
{
    double out = 0;
    double den_i = 1;
    vector<double> par_i(2);
    int i, j;
    int udim = uvec.size();
    vector<double> xl(nq), wl(nq);
    gauleg(nq, xl, wl);
    for(j=0; j<nq; ++j)
    {
        for(i =0; i<udim; ++i)
        {
            par_i.clear();
            par_i.push_back(par[i]);
            par_i.push_back(par[udim+i]);
            den_i *= dBB1(uvec[i], xl[j], par_i);            
        }
        out += den_i * wl[j];
        den_i = 1; 
    }
    return out;
}

/////////////////////////////////////
// bivariate normal copula density //
/////////////////////////////////////
double denB1(double u, double v, vector<double> par)
{
    double tem0, tem1, tem2, tem3, tem4, x, y;
    double r = par[0];
    
    x = gsl_cdf_ugaussian_Pinv(u);
    y = gsl_cdf_ugaussian_Pinv(v);
    
    tem0 = (1- pow(r,2));
    tem1 = pow( tem0, -0.5 );
    tem2 = pow(x,2) +  pow(y,2);
    tem3 = exp( -0.5 / tem0 * ( tem2 - 2*r*x*y  ) );
    tem4 = exp( tem2 / 2 );
    // Rcpp::Rcout << "tem1 tem2 tem3 " << tem1 << " " << tem2 << " " << tem4 << std::endl;
    return tem1 * tem3 * tem4;
}

//////////////////////////////////
// multivariate Gaussian copula //
//////////////////////////////////
double dmvnrm_arma(arma::rowvec x, arma::rowvec mean, arma::mat sigma, bool logd = false)
{ 
    size_t xdim = x.n_cols;
    double out;
    arma::mat rooti = arma::trans(arma::inv(trimatu(arma::chol(sigma))));
    double rootisum = arma::sum(log(rooti.diag()));
    double constants = -(static_cast<double>(xdim)/2.0) * log2pi;

    arma::vec z = rooti * arma::trans( x - mean );
    out = constants - 0.5 * arma::sum(z%z) + rootisum;     

    if(logd == false)
    {
        out = exp(out);
    }
    return out;
}

// this is the density of the copula for connecting those factors
// here parFAC interacts with R directly, so use NumericVector type
double denFAC(vector<double> uvec, NumericVector parFAC)
{
    // as an experiment, use multivariate Gaussian copula first   
    size_t udim = uvec.size();
    bool logd = false;
    NumericVector Pinv(udim);
    NumericVector duniv(udim);
    NumericVector mean0(udim);
    NumericMatrix sigma0(udim, udim);
    
    size_t i,j;   
    
    for(i = 0; i < udim; ++i)
    {
        Pinv[i] = gsl_cdf_ugaussian_Pinv(uvec[i]);
        duniv[i] =  gsl_ran_ugaussian_pdf(Pinv[i]);
    }

    arma::rowvec x(Pinv.begin(), Pinv.size(), false);
    arma::rowvec mean(mean0.begin(), mean0.size(), false);
    arma::mat sigma(sigma0.begin(), udim, udim, false);
    
    size_t k = 0;
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

// this version of denFAC allows different Gfamily, copula family for among-groups
double denFAC1(vector<double> uvec, NumericVector parFAC, int Gfamily)
{
    int udim = uvec.size();
    int i;
    double out;
    
    if(Gfamily == 6)
    {
        vector<double> par;
        for(i=0; i<udim; ++i)
        {
            par.push_back(parFAC[i]);
        }
        out = denF1B6(uvec, par, 31); // here use nq = 31 for integration
    }
    else if(Gfamily == 11) // BB1
    {
        vector<double> par;
        for(i=0; i<udim*2; ++i)
        {
            par.push_back(parFAC[i]);
        }
        out = denF1BB1(uvec, par, 31); // here use nq = 31 for integration	
    }
    else if(Gfamily == 1)
    {
        bool logd = false;
        NumericVector Pinv(udim);
        NumericVector duniv(udim);
        NumericVector mean0(udim);
        NumericMatrix sigma0(udim, udim);

        size_t i,j;   

        for(i = 0; i < udim; ++i)
        {
            Pinv[i] = gsl_cdf_ugaussian_Pinv(uvec[i]);
            duniv[i] =  gsl_ran_ugaussian_pdf(Pinv[i]);
        }

        arma::rowvec x(Pinv.begin(), Pinv.size(), false);
        arma::rowvec mean(mean0.begin(), mean0.size(), false);
        arma::mat sigma(sigma0.begin(), udim, udim, false);

        size_t k = 0;
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
        out = dmvn / duniv_all;
    }
    else
    {
        Rcpp::Rcout << "Among-group copula missing!!" << std::endl;
        out = 1;
    }    
    return out;
}


// based on CopulaModel R package
// xq: nodes;  wq: weights
#define EPS 3.0e-11
void gauleg(int nq, vector<double>& xq, vector<double>& wq)
{
    size_t m,j,i,n;
    double z1,z,xm,xl,pp,p3,p2,p1;
    n = nq; 
    m = (n+1)/2;
  
    // boundary points 
    double x1 = 0;
    double x2 = 1;
    
    xm = 0.5*(x2 + x1);
    xl = 0.5*(x2 - x1);
    for(i=1;i<=m;++i) // yes, i starts from 1
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
        
        xq[i-1] = xm-xl*z;
        xq[n-i]=xm+xl*z;
        wq[i-1]=2.0*xl/((1.0-z*z)*pp*pp);
        wq[n-i]=wq[i-1];
        
  }
}
#undef EPS

///////////////////////////////////////////////////////
//       comonotonic factor copula densities         //
///////////////////////////////////////////////////////

// density for using the same kind of LT for all
// using Gaussian quadratures for numerical integration 
// nq: number of quadratures

/*
double denCF(NumericVector tvec, NumericMatrix DM, NumericVector parCluster, 
            NumericVector parFAC, int parMode, int LTfamily, int nq)
{
    int i, j, m, m1, m2, m3;
    int f = DM.ncol();           // to-do: f = 3 is currently supported
    int d = DM.nrow();
    vector< vector<double> > par_i(d); // parameter vector for each row

    if( LTfamily == 1 || LTfamily == 7)
    {
        for(i = 0; i < d; ++i)
        {
            // to-do: to support multiple non-zero entries in a single row
            par_i[i].push_back(parCluster[i]); // de for i
            par_i[i].push_back(parCluster[i+d]); // th for i
        }
    }
    else if( LTfamily == 6 || LTfamily == 4)
    {
        for(i = 0; i < d; ++i)
        {
            par_i[i].push_back(parCluster[i]); // de for i
        }
    }
    else
    {
        std::cout << "The LT family is not supported." << std::endl;
    }

    NumericMatrix invG(d,f);
    NumericVector invG_i(d);
    NumericVector invpsi_i(d);
    NumericVector psi1inv_i(d);
    double tem1 = 0;
    double tem2 = 0;
    double out =0;
    double den_m =0;
    double den = 0;

    LTfunc_complex LT;
    LTfunc_vector LT1_vector;
    switch( LTfamily )
    {
        case 1:
            LT = &LTE_complex;
            LT1_vector = &LTE1_vector;
            break;
        case 7:
            LT = &LTI_complex;
            LT1_vector = &LTI1_vector;
            break;
        case 6:
            LT = &LTA_complex;
            LT1_vector = &LTA1_vector;
            break;
        default:
            LT = &LTE_complex;
            LT1_vector = &LTE1_vector;
            break;
    }
    
    vector<double> xl(nq), wl(nq);
    gauleg(nq, xl, wl);
    vector<double> uvec;

    NumericMatrix qvec(d, nq);
    int err_msg;

    // calculate qvec for reuse, improving speed
    for(i=0; i < d; ++i)
    {
        for(m=0; m < nq; ++m)
        {
            qvec(i,m) = qG(xl[m], LT, par_i[i], err_msg);
#ifdef DEBUG
            Rcpp::Rcout << "(i,m): " <<i<<m << " qvec(i,m)=" << qvec(i,m) << " err= " << err_msg << std::endl;
#endif            
        }
    }
    
    for(i = 0; i < d; ++i)
    {
        invpsi_i[i] = invpsi(tvec[i], par_i[i], LTfamily);
        psi1inv_i[i] = LT1_vector(invpsi_i[i], par_i[i]);
        //Rcpp::Rcout << "invpsi_i[i] / psi1inv_i[i]: " << invpsi_i[i] << " / " << psi1inv_i[i] << std::endl;
    }
    
    // to-do: only support f=3 for now
    for(m1=0;m1<nq;++m1)
    {    
        for(m2=0;m2<nq;++m2)
        {	    
            for(m3=0;m3<nq;++m3)
            {    
                uvec.clear();
                uvec.push_back(xl[m1]);
                uvec.push_back(xl[m2]);
                uvec.push_back(xl[m3]);
                
                for(j=0;j<f;++j)  // i: row,   j: col
                {
                    for(i=0;i<d;++i)
                    {
                        if(DM(i, j) == 1)
                        {
                            switch( j )
                            {
                                case 0:
                                        invG(i, j) = qvec(i,m1);
                                        break;
                                case 1:
                                        invG(i, j) = qvec(i,m2);
                                        break;
                                case 2:
                                        invG(i, j) = qvec(i,m3);
                                        break;
                                default:
                                        break;
                            }
                            invG_i[i] += invG(i, j);
                            // Rcpp::Rcout << "(i,j): " << i << " / " << j << " / " << invG(i, j) << " / " << invG_i[i] << " / " << std::endl;
                        }
                    }
                }

                for(i=0; i<d; ++i)
                {
                    tem1 = tem1 + invG_i[i] * invpsi_i[i]; 
                    tem2 = tem2 + log(invG_i[i]) - log(-psi1inv_i[i]);
                    invG_i[i] = 0;
                }

                switch( parMode )
                {
                    case 0: 
                        out = exp( tem2 - tem1 );
                        break;
                    case 1:
                        out = exp( tem2 - tem1 ) * denFAC(uvec, parFAC);
                        break;
                    default:
                        out = exp( tem2 - tem1 );
                        break;
                } 


                // Rcpp::Rcout << "m / den_m: " << m << " / " << den_m << " / " << tem1 << " / " << tem2 <<  " / " << x1[m] << " / " << w1[m]*w2[m]*w3[m]*w4[m]*den_m << std::endl;

                tem1 = 0;
                tem2 = 0;

                if(R_finite(den_m))
                {
                  den += wl[m1]*wl[m2]*wl[m3]*den_m;
                }

            }

        }

    }
    return den;
}
*/

// denCF1 using factor Gumbel copula for dependence among groups
// [[Rcpp::export]]
// nf: number of sets of comonotonic factors
double denCF1(NumericVector tvec, NumericMatrix DM, NumericVector parCluster, 
            NumericVector parFAC, int parMode, int LTfamily, int Gfamily, int nq, int nf) 
{
    int i, j, m, m1, m2, m3;
    int f = DM.ncol();           // to-do: f = 2,3 is currently supported only
    int d = DM.nrow();
    vector< vector<double> > par_i(d); // parameter vector for each row

    if( LTfamily == 1 || LTfamily == 7)
    {
        for(i = 0; i < d; ++i)
        {
            // to-do: to support multiple non-zero entries in a single row
            par_i[i].push_back(parCluster[i]); // de for i
            par_i[i].push_back(parCluster[i+d]); // th for i
        }
    }
    else if( LTfamily == 6 || LTfamily == 4)
    {
        for(i = 0; i < d; ++i)
        {
            par_i[i].push_back(parCluster[i]); // de for i
        }
    }
    else if(LTfamily == 100)  // LTGIG
    {
        for(i = 0; i < d; ++i)
        {
            par_i[i].push_back(parCluster[i]); // th for i
            par_i[i].push_back(parCluster[i+d]); // la for i
            par_i[i].push_back(parCluster[i+2*d]); // xi for i
        }
    }
    else
    {
        std::cout << "The LT family is not supported." << std::endl;
    }

    NumericMatrix invG(d,f);
    NumericVector invG_i(d);
    NumericVector invpsi_i(d);
    NumericVector psi1inv_i(d);
    double tem1 = 0;
    double tem2 = 0;
    double den_m =0;
    double den = 0;
    double denF = 0;

    LTfunc_complex LT;
    LTfunc_vector LT1_vector;
    switch( LTfamily )
    {
        case 1:
            LT = &LTE_complex;
            LT1_vector = &LTE1_vector;
            break;
        case 4:
            LT = &LTB_complex;
            LT1_vector = &LTB1_vector;
            break;			
        case 7:
            LT = &LTI_complex;
            LT1_vector = &LTI1_vector;
            break;
        case 6:
            LT = &LTA_complex;
            LT1_vector = &LTA1_vector;
            break;
#ifdef COMPLEX_BESSEL        
        case 100:
            LT = &LTGIG_complex;
            LT1_vector = &LTGIG1_vector;
            break;
#endif            
        default:
            LT = &LTE_complex;
            LT1_vector = &LTE1_vector;
            break;
    }
    
    vector<double> xl(nq), wl(nq);
    gauleg(nq, xl, wl);
    vector<double> uvec;

    NumericMatrix qvec(d, nq);
    int err_msg;

    // calculate qvec for reuse, improving speed
    for(i=0; i < d; ++i)
    {
        for(m=0; m < nq; ++m)
        {
            qvec(i,m) = qG(xl[m], LT, par_i[i], err_msg);
#ifdef DEBUG
            Rcpp::Rcout << "(i,m): " <<i<<m << " qvec(i,m)=" << qvec(i,m) << " err= " << err_msg << std::endl;
#endif            
        }
    }
    
    for(i = 0; i < d; ++i)
    {
        invpsi_i[i] = invpsi(tvec[i], par_i[i], LTfamily);
        psi1inv_i[i] = LT1_vector(invpsi_i[i], par_i[i]);
#ifdef DEBUG
        Rcpp::Rcout << "tvec[i]/invpsi_i[i]/psi1inv_i[i]: " << " / " << tvec[i] << " / " << invpsi_i[i] << " / " << psi1inv_i[i] << std::endl;
#endif
    }
    
    // to-do: only support f=1,2,3 for now
    
    if(nf == 1)
    {
        for(m1=0;m1<nq;++m1)
        {    
            uvec.clear();
            uvec.push_back(xl[m1]);

            for(j=0;j<nf;++j)  // i: row,   j: col
            {
                for(i=0;i<d;++i)
                {
                    if(DM(i, j) == 1)
                    {
                        invG(i, j) = qvec(i,m1);
                        invG_i[i] += invG(i, j);
                    }
                }
            }

            for(i=0; i<d; ++i)
            {
                tem1 = tem1 + invG_i[i] * invpsi_i[i]; 
                tem2 = tem2 + log(invG_i[i]) - log(-psi1inv_i[i]);
                invG_i[i] = 0;
            }

            switch( parMode )
            {
                case 0: 
                        den_m = exp( tem2 - tem1 );
                        break;
                case 1:
                        denF = denFAC1(uvec, parFAC, Gfamily);
                        den_m = exp( tem2 - tem1 ) * denF;
                        break;
                default:
                        den_m = exp( tem2 - tem1 );
                        break;
            } 

#ifdef DEBUG
            Rcpp::Rcout << "m / denF / den_m: " << m1 << m2  << " / " << denF << " / " << den_m << " / " << tem1 << " / " << tem2 <<  " / " << wl[m1]*wl[m2]*den_m << std::endl;
#endif
            tem1 = 0;
            tem2 = 0;

            if(R_finite(den_m))
            {
              den += wl[m1]*den_m;
            }

        }
    }        
    else if(nf == 2)
	{
		for(m1=0;m1<nq;++m1)
		{    
			for(m2=0;m2<nq;++m2)
			{	    
					uvec.clear();
					uvec.push_back(xl[m1]);
					uvec.push_back(xl[m2]);
					
					for(j=0;j<nf;++j)  // i: row,   j: col
					{
						for(i=0;i<d;++i)
						{
							if(DM(i, j) == 1)
							{
								switch( j )
								{
									case 0:
											invG(i, j) = qvec(i,m1);
											break;
									case 1:
											invG(i, j) = qvec(i,m2);
											break;
									default:
											break;
								}
								invG_i[i] += invG(i, j);
							}
						}
					}

					for(i=0; i<d; ++i)
					{
						tem1 = tem1 + invG_i[i] * invpsi_i[i]; 
						tem2 = tem2 + log(invG_i[i]) - log(-psi1inv_i[i]);
						invG_i[i] = 0;
					}

					switch( parMode )
					{
						case 0: 
							den_m = exp( tem2 - tem1 );
							break;
						case 1:
							denF = denFAC1(uvec, parFAC, Gfamily);
							den_m = exp( tem2 - tem1 ) * denF;
							break;
						default:
							den_m = exp( tem2 - tem1 );
							break;
					} 

#ifdef DEBUG
					Rcpp::Rcout << "m / denF / den_m: " << m1 << m2  << " / " << denF << " / " << den_m << " / " << tem1 << " / " << tem2 <<  " / " << wl[m1]*wl[m2]*den_m << std::endl;
#endif
					tem1 = 0;
					tem2 = 0;

					if(R_finite(den_m))
					{
					  den += wl[m1]*wl[m2]*den_m;
					}

			}
		}
	}
	else if(nf == 3)
	{
		for(m1=0;m1<nq;++m1)
		{    
			for(m2=0;m2<nq;++m2)
			{	    
				for(m3=0;m3<nq;++m3)
				{    
					uvec.clear();
					uvec.push_back(xl[m1]);
					uvec.push_back(xl[m2]);
					uvec.push_back(xl[m3]);
					
					for(j=0;j<f;++j)  // i: row,   j: col
					{
						for(i=0;i<d;++i)
						{
							if(DM(i, j) == 1)
							{
								switch( j )
								{
									case 0:
											invG(i, j) = qvec(i,m1);
											break;
									case 1:
											invG(i, j) = qvec(i,m2);
											break;
									case 2:
											invG(i, j) = qvec(i,m3);
											break;
									default:
											break;
								}
								invG_i[i] += invG(i, j);
								// Rcpp::Rcout << "(i,j): " << i << " / " << j << " / " << invG(i, j) << " / " << invG_i[i] << " / " << std::endl;
							}
						}
					}

					for(i=0; i<d; ++i)
					{
						tem1 = tem1 + invG_i[i] * invpsi_i[i]; 
						tem2 = tem2 + log(invG_i[i]) - log(-psi1inv_i[i]);
						invG_i[i] = 0;
					}

					switch( parMode )
					{
						case 0: 
							den_m = exp( tem2 - tem1 );
							break;
						case 1:
							denF = denFAC1(uvec, parFAC, Gfamily);
							den_m = exp( tem2 - tem1 ) * denF;
							break;
						default:
							den_m = exp( tem2 - tem1 );
							break;
					} 

#ifdef DEBUG
					Rcpp::Rcout << "m / denF / den_m: " << m1 << m2 << m3 << " / " << denF << " / " << den_m << " / " << tem1 << " / " << tem2 <<  " / " << wl[m1]*wl[m2]*wl[m3]*den_m << std::endl;
#endif
					tem1 = 0;
					tem2 = 0;

					if(R_finite(den_m))
					{
					  den += wl[m1]*wl[m2]*wl[m3]*den_m;
					}

				}

			}

		}		
	}

    return den;
    // return exp( tem2 - tem1 );
}

// to-do
// denCF2 is based on denCF1, adding support on overlap comonotonic factors that belong to the same family
// nf: number of sets of comonotonic factors
/*
double denCF2(NumericVector tvec, NumericMatrix DM, NumericVector parCluster, 
            NumericVector parFAC, int parMode, int LTfamily, int Gfamily, int nq, int nf) 
{
    int i, j, m, m1, m2, m3;
    int nDM = 0; // number of 1 in the DM matrix
	int f = DM.ncol();           // to-do: f = 2,3 is currently supported only
    int d = DM.nrow();
    vector< vector<double> > par_i(d); // parameter vector for each row

	for(i = 0; i<d; ++i)
	{
		for(j=0; j<f; ++j)
		{
			if(DM(i,j) == 1)
			{
				++nDM;	
			}
		}
	}
	
    if( LTfamily == 1 || LTfamily == 7)
    {
        for(i = 0; i < d; ++i)
        {
            
			DM(Rcpp::_)
			
			par_i[i].push_back(parCluster[i]); // de for i
            par_i[i].push_back(parCluster[i+nDM]); // th for i
        }
    }
    else if( LTfamily == 6 || LTfamily == 4)
    {
        for(i = 0; i < d; ++i)
        {
            par_i[i].push_back(parCluster[i]); // de for i
        }
    }
    else
    {
        std::cout << "The LT family is not supported." << std::endl;
    }

    NumericMatrix invG(d,f);
    NumericVector invG_i(d);
    NumericVector invpsi_i(d);
    NumericVector psi1inv_i(d);
    double tem1 = 0;
    double tem2 = 0;
    double den_m =0;
    double den = 0;
    double denF = 0;

    LTfunc_complex LT;
    LTfunc_vector LT1_vector;
    switch( LTfamily )
    {
        case 1:
            LT = &LTE_complex;
            LT1_vector = &LTE1_vector;
            break;
        case 4:
            LT = &LTB_complex;
            LT1_vector = &LTB1_vector;
            break;			
        case 7:
            LT = &LTI_complex;
            LT1_vector = &LTI1_vector;
            break;
        case 6:
            LT = &LTA_complex;
            LT1_vector = &LTA1_vector;
            break;
        default:
            LT = &LTE_complex;
            LT1_vector = &LTE1_vector;
            break;
    }
    
    vector<double> xl(nq), wl(nq);
    gauleg(nq, xl, wl);
    vector<double> uvec;

    NumericMatrix qvec(d, nq);
    int err_msg;

    // calculate qvec for reuse, improving speed
    for(i=0; i < d; ++i)
    {
        for(m=0; m < nq; ++m)
        {
            qvec(i,m) = qG(xl[m], LT, par_i[i], err_msg);
#ifdef DEBUG
            Rcpp::Rcout << "(i,m): " <<i<<m << " qvec(i,m)=" << qvec(i,m) << " err= " << err_msg << std::endl;
#endif            
        }
    }
    
    for(i = 0; i < d; ++i)
    {
        invpsi_i[i] = invpsi(tvec[i], par_i[i], LTfamily);
        psi1inv_i[i] = LT1_vector(invpsi_i[i], par_i[i]);
#ifdef DEBUG
        Rcpp::Rcout << "tvec[i]/invpsi_i[i]/psi1inv_i[i]: " << " / " << tvec[i] << " / " << invpsi_i[i] << " / " << psi1inv_i[i] << std::endl;
#endif
    }
    
    // to-do: only support f=2,3 for now
    if(nf == 2)
	{
		for(m1=0;m1<nq;++m1)
		{    
			for(m2=0;m2<nq;++m2)
			{	    
					uvec.clear();
					uvec.push_back(xl[m1]);
					uvec.push_back(xl[m2]);
					
					for(j=0;j<nf;++j)  // i: row,   j: col
					{
						for(i=0;i<d;++i)
						{
							if(DM(i, j) == 1)
							{
								switch( j )
								{
									case 0:
											invG(i, j) = qvec(i,m1);
											break;
									case 1:
											invG(i, j) = qvec(i,m2);
											break;
									default:
											break;
								}
								invG_i[i] += invG(i, j);
							}
						}
					}

					for(i=0; i<d; ++i)
					{
						tem1 = tem1 + invG_i[i] * invpsi_i[i]; 
						tem2 = tem2 + log(invG_i[i]) - log(-psi1inv_i[i]);
						invG_i[i] = 0;
					}

					switch( parMode )
					{
						case 0: 
							den_m = exp( tem2 - tem1 );
							break;
						case 1:
							denF = denFAC1(uvec, parFAC, Gfamily);
							den_m = exp( tem2 - tem1 ) * denF;
							break;
						default:
							den_m = exp( tem2 - tem1 );
							break;
					} 

#ifdef DEBUG
					Rcpp::Rcout << "m / denF / den_m: " << m1 << m2  << " / " << denF << " / " << den_m << " / " << tem1 << " / " << tem2 <<  " / " << wl[m1]*wl[m2]*den_m << std::endl;
#endif
					tem1 = 0;
					tem2 = 0;

					if(R_finite(den_m))
					{
					  den += wl[m1]*wl[m2]*den_m;
					}

			}
		}
	}
	else if(nf == 3)
	{
		for(m1=0;m1<nq;++m1)
		{    
			for(m2=0;m2<nq;++m2)
			{	    
				for(m3=0;m3<nq;++m3)
				{    
					uvec.clear();
					uvec.push_back(xl[m1]);
					uvec.push_back(xl[m2]);
					uvec.push_back(xl[m3]);
					
					for(j=0;j<f;++j)  // i: row,   j: col
					{
						for(i=0;i<d;++i)
						{
							if(DM(i, j) == 1)
							{
								switch( j )
								{
									case 0:
											invG(i, j) = qvec(i,m1);
											break;
									case 1:
											invG(i, j) = qvec(i,m2);
											break;
									case 2:
											invG(i, j) = qvec(i,m3);
											break;
									default:
											break;
								}
								invG_i[i] += invG(i, j);
								// Rcpp::Rcout << "(i,j): " << i << " / " << j << " / " << invG(i, j) << " / " << invG_i[i] << " / " << std::endl;
							}
						}
					}

					for(i=0; i<d; ++i)
					{
						tem1 = tem1 + invG_i[i] * invpsi_i[i]; 
						tem2 = tem2 + log(invG_i[i]) - log(-psi1inv_i[i]);
						invG_i[i] = 0;
					}

					switch( parMode )
					{
						case 0: 
							den_m = exp( tem2 - tem1 );
							break;
						case 1:
							denF = denFAC1(uvec, parFAC, Gfamily);
							den_m = exp( tem2 - tem1 ) * denF;
							break;
						default:
							den_m = exp( tem2 - tem1 );
							break;
					} 

#ifdef DEBUG
					Rcpp::Rcout << "m / denF / den_m: " << m1 << m2 << m3 << " / " << denF << " / " << den_m << " / " << tem1 << " / " << tem2 <<  " / " << wl[m1]*wl[m2]*wl[m3]*den_m << std::endl;
#endif
					tem1 = 0;
					tem2 = 0;

					if(R_finite(den_m))
					{
					  den += wl[m1]*wl[m2]*wl[m3]*den_m;
					}

				}

			}

		}		
	}

    return den;
    // return exp( tem2 - tem1 );
}

*/



/////////////////////////////////
// comonotonic bi-factor model //
/////////////////////////////////

// [[Rcpp::export]]
double den_LTA_LTE(NumericVector tvec, NumericMatrix DM, NumericVector par, int nq)
{
    // devec are paramters for positive stable LT (LTA)
    // devec1 and thevec1 are parameters for the Mittag-Leffler LT
    
    int i, j, m, m1, m2, m3, m4;
    int f = DM.ncol();
    int d = DM.nrow();
	
    NumericMatrix invG(d,f);
   
    NumericVector invG_i(d);
    NumericVector invpsi_i(d);
    NumericVector psi1inv_i(d);
    double tem1 = 0;
    double tem2 = 0;
    double den_m =0;
    double den = 0;

    vector< vector<double> > par_1_i(d); // for the 1st type of LT
    vector< vector<double> > par_2_i(d); // for the 2nd type of LT
    
    for(i = 0; i < d; ++i)
    {
        par_1_i[i].push_back(par[i]);
        par_2_i[i].push_back(par[i+d]);
        par_2_i[i].push_back(par[i+2*d]);
    }
    
    LTfunc_complex LT_1, LT_2;
    LT_1 = &LTA_complex;
    LT_2 = &LTE_complex;
    
    /// setup Gaussian quadrature
    vector<double> xl(nq), wl(nq);
    gauleg(nq, xl, wl);

#ifdef DEBUG
    Rcpp::Rcout << "xl / wl: " << xl[0] << ", " << xl[1] << " // " << wl[0] << ", " << wl[1] << std::endl;
#endif
    
    NumericMatrix qvec(d, nq);
    NumericMatrix hvec(d, nq);
    int err_msg_1, err_msg_2;
	
    for(i=0; i < d; ++i)
    {
        for(m=0; m < nq; ++m)
        {
            qvec(i,m) = qG(xl[m], LT_1, par_1_i[i], err_msg_1);
            hvec(i,m) = qG(xl[m], LT_2, par_2_i[i], err_msg_2);
#ifdef DEBUG
            Rcpp::Rcout<<"(i,m)= "<<i<<","<<m<<" par_1= "<< par_1_i[i][0] <<" par_2= "<< par_2_i[i][0] <<","<<par_2_i[i][1] <<  " xl[m]= " << xl[m] <<  " qvec(i,m)= " << qvec(i,m) << " hvec(i,m) = " << hvec(i,m) << " err="<< err_msg_1 << "," << err_msg_2 << std::endl;
#endif
        }
    }
    
    int err_msg;
    
    for(i = 0; i < d; ++i)
    {
        invpsi_i[i] = invpsi_LTA_LTE(tvec[i], par_1_i[i], par_2_i[i], err_msg);
        psi1inv_i[i] = LTA1_LTE1(invpsi_i[i], par_1_i[i], par_2_i[i]);
#ifdef DEBUG
        Rcpp::Rcout << "err: " << err_msg << " invpsi_i[i] / psi1inv_i[i]: " << invpsi_i[i] << " / " << psi1inv_i[i] << " par "  << par_1_i[i][0] << "," << par_2_i[i][0] << "," << par_2_i[i][1]  << "," << std::endl;
#endif
    }
    
    for(m1=0;m1<nq;++m1)
    {    
        for(m2=0;m2<nq;++m2)
        {	    
            for(m3=0;m3<nq;++m3)
            {    
                for(m4=0;m4<nq;++m4)
                {		    
                    for(j=0;j<f;++j)  // i: row,   j: col
                    {
                        for(i=0;i<d;++i)
                        {
                            if(DM(i, j) == 1)
                            {
                                switch( j )
                                {
                                    case 0:
                                            invG(i, j) = qvec(i,m1);
                                            break;
                                    case 1:
                                            invG(i, j) = qvec(i,m2);
                                            break;
                                    case 2:
                                            invG(i, j) = qvec(i,m3);
                                            break;
                                    case 3:
                                            invG(i, j) = hvec(i,m4);
                                            break;
                                    default:
                                            break;
                                }
                                invG_i[i] += invG(i, j);
#ifdef DEBUG
                                //Rcpp::Rcout << m1<<","<<m2<<","<<m3 <<","<< m4 <<","<< i << "," << j << " invG(i,j)= " << invG(i,j) << std::endl;
#endif
                            }
                        }
                    }

                    for(i=0; i<d; ++i)
                    {
                        tem1 = tem1 + invG_i[i] * invpsi_i[i]; 
                        tem2 = tem2 + log(invG_i[i]) - log(-psi1inv_i[i]);
                        invG_i[i] = 0;
                    }

                    den_m = exp( tem2 - tem1 );
#ifdef DEBUG
                    Rcpp::Rcout << "m/den_m/tem1/tem2/: " << m1<<m2<<m3<<m4 << " / " << den_m << " / " << tem1 << " / " << tem2 <<  " / " << std::endl;
#endif
                    tem1 = 0;
                    tem2 = 0;

                    if(R_finite(den_m))
                    {
                      den += wl[m1]*wl[m2]*wl[m3]*wl[m4]*den_m;
                    }
                }

            }

        }

    }
    return den;
}


// [[Rcpp::export]]
double den_LTE_LTA(NumericVector tvec, NumericMatrix DM, NumericVector par, int nq)
{
    // devec thevec are parameters for the Mittag-Leffler LT (LTE)
    // devec1 are parameters for positive stable LT (LTA)
    
    int i, j, m, m1, m2, m3, m4;
    int f = DM.ncol();
    int d = DM.nrow();
	
    NumericMatrix invG(d,f);
   
    NumericVector invG_i(d);
    NumericVector invpsi_i(d);
    NumericVector psi1inv_i(d);
    double tem1 = 0;
    double tem2 = 0;
    double den_m =0;
    double den = 0;

    vector< vector<double> > par_1_i(d); // for the 1st type of LT
    vector< vector<double> > par_2_i(d); // for the 2nd type of LT
    
    for(i = 0; i < d; ++i)
    {
        par_1_i[i].push_back(par[i]);
        par_1_i[i].push_back(par[i+d]);
        par_2_i[i].push_back(par[2*d+i]);  
    }
    
    LTfunc_complex LT_1, LT_2;
    LT_1 = &LTE_complex;
    LT_2 = &LTA_complex;
    
    /// setup Gaussian quadrature
    vector<double> xl(nq), wl(nq);
    gauleg(nq, xl, wl);

#ifdef DEBUG
    Rcpp::Rcout << "xl / wl: " << xl[0] << ", " << xl[1] << " // " << wl[0] << ", " << wl[1] << std::endl;
#endif
    
    NumericMatrix qvec(d, nq);
    NumericMatrix hvec(d, nq);
    int err_msg_1, err_msg_2;
	
    for(i=0; i < d; ++i)
    {
        for(m=0; m < nq; ++m)
        {
            qvec(i,m) = qG(xl[m], LT_1, par_1_i[i], err_msg_1);
            hvec(i,m) = qG(xl[m], LT_2, par_2_i[i], err_msg_2);
#ifdef DEBUG
            Rcpp::Rcout<<"(i,m)= "<<i<<","<<m<<" par_1= "<< par_1_i[i][0] <<" par_2= "<< par_2_i[i][0] <<","<<par_2_i[i][1] <<  " xl[m]= " << xl[m] <<  " qvec(i,m)= " << qvec(i,m) << " hvec(i,m) = " << hvec(i,m) << " err="<< err_msg_1 << "," << err_msg_2 << std::endl;
#endif
        }
    }
    
    int err_msg;
    
    for(i = 0; i < d; ++i)
    {
        invpsi_i[i] = invpsi_LTA_LTE(tvec[i], par_2_i[i], par_1_i[i], err_msg); // exchange LTA and LTE with their parameters
        psi1inv_i[i] = LTA1_LTE1(invpsi_i[i], par_2_i[i], par_1_i[i]); // exchange LTA and LTE with their parameters
#ifdef DEBUG
        Rcpp::Rcout << "err: " << err_msg << " invpsi_i[i] / psi1inv_i[i]: " << invpsi_i[i] << " / " << psi1inv_i[i] << " par "  << par_1_i[i][0] << "," << par_2_i[i][0] << "," << par_2_i[i][1]  << "," << std::endl;
#endif
    }
    
    for(m1=0;m1<nq;++m1)
    {    
        for(m2=0;m2<nq;++m2)
        {	    
            for(m3=0;m3<nq;++m3)
            {    
                for(m4=0;m4<nq;++m4)
                {		    
                    for(j=0;j<f;++j)  // i: row,   j: col
                    {
                        for(i=0;i<d;++i)
                        {
                            if(DM(i, j) == 1)
                            {
                                switch( j )
                                {
                                    case 0:
                                            invG(i, j) = qvec(i,m1);
                                            break;
                                    case 1:
                                            invG(i, j) = qvec(i,m2);
                                            break;
                                    case 2:
                                            invG(i, j) = qvec(i,m3);
                                            break;
                                    case 3:
                                            invG(i, j) = hvec(i,m4);
                                            break;
                                    default:
                                            break;
                                }
                                invG_i[i] += invG(i, j);
#ifdef DEBUG
                                //Rcpp::Rcout << m1<<","<<m2<<","<<m3 <<","<< m4 <<","<< i << "," << j << " invG(i,j)= " << invG(i,j) << std::endl;
#endif
                            }
                        }
                    }

                    for(i=0; i<d; ++i)
                    {
                        tem1 = tem1 + invG_i[i] * invpsi_i[i]; 
                        tem2 = tem2 + log(invG_i[i]) - log(-psi1inv_i[i]);
                        invG_i[i] = 0;
                    }

                    den_m = exp( tem2 - tem1 );
#ifdef DEBUG
                    Rcpp::Rcout << "m/den_m/tem1/tem2/: " << m1<<m2<<m3<<m4 << " / " << den_m << " / " << tem1 << " / " << tem2 <<  " / " << std::endl;
#endif
                    tem1 = 0;
                    tem2 = 0;

                    if(R_finite(den_m))
                    {
                      den += wl[m1]*wl[m2]*wl[m3]*wl[m4]*den_m;
                    }
                }

            }

        }

    }
    return den;
}

// pairwise Spearman's rho
// [[Rcpp::export]]
NumericVector srho_LTE_LTA(NumericMatrix DM, NumericVector par, int nq)
{
    // devec thevec are parameters for the Mittag-Leffler LT (LTE)
    // devec1 are parameters for positive stable LT (LTA)
    int i, j, i1, i2, m, m1, m2, mm, mn, mk1, mk2;
    int d = DM.nrow();
    int p = DM.ncol();
    int err_msg;
    
    NumericVector out;
    NumericMatrix qvec1(d, nq);
    NumericMatrix qvec2(d, nq);
    
    double tem1 = 0;
    double tem2 = 0;
    double intg = 0;
    double srho = 0;
    vector<int> group(d);

    vector< vector<double> > par_1_i(d);
    vector< vector<double> > par_2_i(d);
    
    // assigning index for non overlapping groups
    for(i = 0; i<d; ++i)
    {
        for(j = 0; j<p-1; ++j)
        {
            if(DM(i,j)==1)
            {
                group[i] = j; 
                // Rcpp::Rcout << group[i] << std::endl;
                break;
            }
        }
    }
   
    for(i = 0; i < d; ++i)
    {
        par_1_i[i].push_back(par[i]);
        par_1_i[i].push_back(par[i+d]);
        par_2_i[i].push_back(par[i+2*d]);  
    }
    
    LTfunc_complex LT_complex_1, LT_complex_2;
    LT_complex_1 = &LTE_complex;
    LT_complex_2 = &LTA_complex;
    
    /// setup Gaussian quadrature
    vector<double> xl(nq), wl(nq);
    gauleg(nq, xl, wl);
    
    for(i=0; i < d; ++i)
    {
        for(m=0; m < nq; ++m)
        {
            qvec1(i,m) = qG(xl[m], LT_complex_1, par_1_i[i], err_msg);
            qvec2(i,m) = qG(xl[m], LT_complex_2, par_2_i[i], err_msg);
            // Rcpp::Rcout<<"(i,m)= "<<i<<","<<m<<" par_1_i= "<< par_1_i[i][0] << " " << par_1_i[i][1] << " " << par_2_i[i][0] <<  " xl[m]= " << xl[m] <<  " qvec1(i,m)= " << qvec1(i,m) << " qvec2(i,m)= " << qvec2(i,m) << " err= "<< err_msg << std::endl;
        }
    }
    
    for(i1 = 0; i1 < d-1; ++i1)
    {
        for(i2 = i1 + 1; i2 < d; ++i2)
        {
            if(group[i1] == group[i2])
            {
                for(mk1=0;mk1<nq;++mk1)
                {
                    for(mk2=0;mk2<nq;++mk2)
                    {
                        for(m1=0;m1<nq;++m1)
                        {
                            for(m2=0;m2<nq;++m2)
                            {
                                        tem1 = LTE_LTA( - log(1-xl[m1]) / (qvec1(i1,mk1)+qvec2(i1,mk2)), par_1_i[i1], par_2_i[i1]);
                                        tem2 = LTE_LTA( - log(1-xl[m2]) / (qvec1(i2,mk1)+qvec2(i2,mk2)), par_1_i[i2], par_2_i[i2]);
                                        intg += wl[m1] * wl[m2] * wl[mk1] * wl[mk2] * tem1 * tem2;
                                        //Rcpp::Rcout<< "mu,m1,m2,tem1,tem2,intg " << mu <<" "<<  m1 <<" "<<  m2 <<" "<<  tem1 <<" "<<  tem2  <<" "<<  intg << std::endl;
                            }
                        }
                    }
                }
            }
            else
            {
                for(mn=0;mn<nq;++mn)
                {
                    for(mm=0;mm<nq;++mm)
                    {
                        for(mk2=0;mk2<nq;++mk2)
                        {
                            for(m1=0;m1<nq;++m1)
                            {
                                for(m2=0;m2<nq;++m2)
                                {
                                    tem1 = LTE_LTA( - log(1-xl[m1]) / (qvec1(i1,mn)+qvec2(i1,mk2)), par_1_i[i1], par_2_i[i1]);
                                    tem2 = LTE_LTA( - log(1-xl[m2]) / (qvec1(i2,mm)+qvec2(i2,mk2)), par_1_i[i2], par_2_i[i2]);
                                    intg += wl[m1] * wl[m2] * wl[mn] * wl[mm] * wl[mk2] * tem1 * tem2;
                                    //Rcpp::Rcout<< "mu,m1,m2,tem1,tem2,intg " << mu <<" "<<  m1 <<" "<<  m2 <<" "<<  tem1 <<" "<<  tem2  <<" "<<  intg << std::endl;
                                }
                            }
                        }
                    }
                }
                
            }
            srho = 12 * intg - 3;
            intg = 0;
            out.push_back(srho);
        }
    }
    return out; 
}


// pairwise Spearman's rho
// [[Rcpp::export]]
NumericVector srho_LTE_Gaussian(NumericMatrix DM, NumericVector par, int nq)
{
    // devec thevec are parameters for the Mittag-Leffler LT (LTE)
    // devec1 are parameters for positive stable LT (LTA)
    int i, j, i1, i2, m, m1, m2, mk1, mk2;
    int d = DM.nrow();
    int p = DM.ncol();
    int err_msg;
    
    NumericVector out;
    NumericMatrix qvec(d, nq);
    
    double tem1 = 0;
    double tem2 = 0;
    double tem3 = 0;
    double intg = 0;
    double srho = 0;
    double rho_ij = 0;
    vector<int> group(d);
    vector<double> par_g(3);
    vector<double> rhovec(1);

    vector< vector<double> > par_i(d);
    
    // assigning index for non overlapping groups
    for(i = 0; i<d; ++i)
    {
        for(j = 0; j<p; ++j)
        {
            if(DM(i,j)==1)
            {
                group[i] = j;
                // Rcpp::Rcout << group[i] <<  " d=" << d <<  " p=" << p << std::endl;
                break;
            }
        }
    }
    
    for(i = 0; i < d; ++i)
    {
        par_i[i].push_back(par[i]);
        par_i[i].push_back(par[i+d]);
    }
    
    par_g.clear();
    for(i=0; i<3; ++i)
    {
        par_g.push_back(par[2*d + i]);
    }
    
    LTfunc_complex LT_complex;
    LT_complex = &LTE_complex;
    
    /// setup Gaussian quadrature
    vector<double> xl(nq), wl(nq);
    gauleg(nq, xl, wl);
    
    for(i=0; i < d; ++i)
    {
        for(m=0; m < nq; ++m)
        {
            qvec(i,m) = qG(xl[m], LT_complex, par_i[i], err_msg);
            // Rcpp::Rcout<<"(i,m)= "<<i<<","<<m<<" par_i= "<< par_i[i][0] << " " << par_i[i][1] << " " << par_g[0] << " " << par_g[1] << " " << par_g[2] <<  " xl[m]= " << xl[m] <<  " qvec1(i,m)= " << qvec(i,m) << " err= "<< err_msg << std::endl;
        }
    }
    
    for(i1 = 0; i1 < d-1; ++i1)
    {
        for(i2 = i1 + 1; i2 < d; ++i2)
        {
            if(group[i1] == group[i2])
            {
                for(mk1=0;mk1<nq;++mk1)
                {
                    for(m1=0;m1<nq;++m1)
                    {
                        for(m2=0;m2<nq;++m2)
                        {
                            tem1 = LTE( - log(1-xl[m1]) / (qvec(i1,mk1)), par_i[i1]);
                            tem2 = LTE( - log(1-xl[m2]) / (qvec(i2,mk1)), par_i[i2]);
                            intg += wl[m1] * wl[m2] * wl[mk1] * tem1 * tem2;
                            //Rcpp::Rcout<< "mu,m1,m2,tem1,tem2,intg " << mu <<" "<<  m1 <<" "<<  m2 <<" "<<  tem1 <<" "<<  tem2  <<" "<<  intg << std::endl;
                        }
                    }
                }
            }
            else
            {
                for(mk1=0;mk1<nq;++mk1)
                {
                    for(mk2=0;mk2<nq;++mk2)
                    {
                        for(m1=0;m1<nq;++m1)
                        {
                            for(m2=0;m2<nq;++m2)
                            {
                                tem1 = LTE( - log(1-xl[m1]) / (qvec(i1,mk1)), par_i[i1]);
                                tem2 = LTE( - log(1-xl[m2]) / (qvec(i2,mk2)), par_i[i2]);
                                
                                if( ((group[i1] == 0) && (group[i2] == 1))  || ((group[i1] == 1) && (group[i2] == 0))   )
                                {
                                    rho_ij = par_g[0];
                                }
                                else if( ((group[i1] == 0) && (group[i2] == 2)) || ((group[i1] == 2) && (group[i2] == 0)) )
                                {
                                    rho_ij = par_g[1];
                                }
                                else if( ((group[i1] == 1) && (group[i2] == 2)) || ((group[i1] == 2) && (group[i2] == 1)) )
                                {
                                    rho_ij = par_g[2];
                                }
                                else
                                {
                                    rho_ij = 0;
                                }
                                
                                rhovec.clear();
                                rhovec.push_back(rho_ij);
                                
                                tem3 = denB1(xl[mk1], xl[mk2], rhovec);
                                
                                intg += wl[m1] * wl[m2] * wl[mk1] * wl[mk2] * tem1 * tem2 * tem3;
                                // Rcpp::Rcout<< "m1,m2,mk1,mk2,tem1,tem2,tem3 " << " " << i1 << " " << i2 << " " << group[i1] << " " << group[i2] << " " << m1 <<" "<<  m2 <<" "<<  mk1 <<" "<< mk2  <<" "<<  tem1 <<" "<<  tem2  <<" "<<  tem3 << std::endl;
                                // Rcpp::Rcout << "par_g, rho_ij" << group[i1] << " " << group[i2] << " " << par_g[0] << " " << par_g[1] << " " <<  par_g[2] << " " << rho_ij << " " << par[2*d + 0] << " " << par[2*d + 1] << " " << par[2*d + 2] << std::endl;
                            }
                        }
                    }
                }                
            }
            srho = 12 * intg - 3;
            intg = 0;
            out.push_back(srho);
        }
    }
    return out; 
}

// ad-hoc case for overlap LTB and LTB
// [[Rcpp::export]]
double den_LTB_LTB(NumericVector tvec, NumericMatrix DM, NumericVector par, int nq)
{
    // devec are paramters for positive stable LT (LTA)
    // devec1 and thevec1 are parameters for the Mittag-Leffler LT
    
    int i, j, m, m1, m2;
    int f = DM.ncol();
    int d = DM.nrow();
	int nCol1 = 0;
	
	for(i=0; i<d; ++i)
	{
		if(DM(i,0)==1)
		{
			++nCol1;
		}
	}
	
    NumericMatrix invG(d,f);
   
    NumericVector invG_i(d);
    NumericVector invpsi_i(d);
    NumericVector psi1inv_i(d);
    double tem1 = 0;
    double tem2 = 0;
    double den_m =0;
    double den = 0;  

    vector< vector<double> > par_1_i(d); // for the 1st column of LT
    vector< vector<double> > par_2_i(d); // for the 2nd column of LT
    
    j = 0;
	for(i = 0; i < d; ++i)
    {
        if(DM(i,0)==1)
		{
			par_1_i[i].push_back(par[i]);
		}
		if(DM(i,1)==1)
		{
			par_2_i[i].push_back(par[j+nCol1]);
			++j;	
		}
    }
    
    LTfunc_complex LT_1, LT_2;
    LT_1 = &LTB_complex;
    LT_2 = &LTB_complex;
    
    /// setup Gaussian quadrature
    vector<double> xl(nq), wl(nq);
    gauleg(nq, xl, wl);

#ifdef DEBUG
    Rcpp::Rcout << "xl / wl: " << xl[0] << ", " << xl[1] << " // " << wl[0] << ", " << wl[1] << std::endl;
#endif
    
    NumericMatrix qvec(d, nq);
    NumericMatrix hvec(d, nq);
    int err_msg_1, err_msg_2;
	
    for(i=0; i < d; ++i)
    {
        for(m=0; m < nq; ++m)
        {
            if(DM(i,0)==1)
			{
				qvec(i,m) = qG(xl[m], LT_1, par_1_i[i], err_msg_1);
			}
			if(DM(i,1)==1)
			{
				hvec(i,m) = qG(xl[m], LT_2, par_2_i[i], err_msg_2);
			}
            
#ifdef DEBUG
            Rcpp::Rcout<<"(i,m)= "<<i<<","<<m<<" par_1= "<< par_1_i[i][0] <<" par_2= "<< par_2_i[i][0] <<","<<par_2_i[i][1] <<  " xl[m]= " << xl[m] <<  " qvec(i,m)= " << qvec(i,m) << " hvec(i,m) = " << hvec(i,m) << " err="<< err_msg_1 << "," << err_msg_2 << std::endl;
#endif
        }
    }
    
    int err_msg;
    
    for(i = 0; i < 2; ++i)
    {
        invpsi_i[i] = invpsi(tvec[i], par_1_i[i], 4);
        psi1inv_i[i] = LTB1_vector(invpsi_i[i], par_1_i[i]);
#ifdef DEBUG
        Rcpp::Rcout << "err: " << err_msg << " invpsi_i[i] / psi1inv_i[i]: " << invpsi_i[i] << " / " << psi1inv_i[i] << " par "  << par_1_i[i][0] << "," << par_2_i[i][0] << "," << par_2_i[i][1]  << "," << std::endl;
#endif
    }
	
	vector<double> par_1_2_vec;
	
	for(i = 2; i < 4; ++i)
    {
        par_1_2_vec.push_back(par_1_i[i][0]);
		par_1_2_vec.push_back(par_2_i[i][0]);
		invpsi_i[i] = invpsi(tvec[i], par_1_2_vec, 4);
        psi1inv_i[i] = LTB1_vector(invpsi_i[i], par_1_2_vec);
		par_1_2_vec.clear();
#ifdef DEBUG
        Rcpp::Rcout << "err: " << err_msg << " invpsi_i[i] / psi1inv_i[i]: " << invpsi_i[i] << " / " << psi1inv_i[i] << " par "  << par_1_i[i][0] << "," << par_2_i[i][0] << "," << par_2_i[i][1]  << "," << std::endl;
#endif
    }
	
	for(i = 4; i < 6; ++i)
    {
        invpsi_i[i] = invpsi(tvec[i], par_2_i[i], 4);
        psi1inv_i[i] = LTB1_vector(invpsi_i[i], par_2_i[i]);
#ifdef DEBUG
        Rcpp::Rcout << "err: " << err_msg << " invpsi_i[i] / psi1inv_i[i]: " << invpsi_i[i] << " / " << psi1inv_i[i] << " par "  << par_1_i[i][0] << "," << par_2_i[i][0] << "," << par_2_i[i][1]  << "," << std::endl;
#endif
    }
    
    for(m1=0;m1<nq;++m1)
    {    
        for(m2=0;m2<nq;++m2)
        {	    
			for(j=0;j<f;++j)  // i: row,   j: col
			{
				for(i=0;i<d;++i)
				{
					if(DM(i, j) == 1)
					{
						switch( j )
						{
							case 0:
									invG(i, j) = qvec(i,m1);
									break;
							case 1:
									invG(i, j) = hvec(i,m2);
									break;
							default:
									break;
						}
						invG_i[i] += invG(i, j);
#ifdef DEBUG
						//Rcpp::Rcout << m1<<","<<m2<<","<<m3 <<","<< m4 <<","<< i << "," << j << " invG(i,j)= " << invG(i,j) << std::endl;
#endif
					}
				}
			}

			for(i=0; i<d; ++i)
			{
				tem1 = tem1 + invG_i[i] * invpsi_i[i]; 
				tem2 = tem2 + log(invG_i[i]) - log(-psi1inv_i[i]);
				invG_i[i] = 0;
			}

			den_m = exp( tem2 - tem1 );
#ifdef DEBUG
			Rcpp::Rcout << "m/den_m/tem1/tem2/: " << m1<<m2 << " / " << den_m << " / " << tem1 << " / " << tem2 <<  " / " << std::endl;
#endif
			tem1 = 0;
			tem2 = 0;

			if(R_finite(den_m))
			{
			  den += wl[m1]*wl[m2]*den_m;
			}
        }

    }
    return den;
}
