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

#define qMAX 1e+100
#define qMIN 1e-100
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

    tmp = pow(s, (1/de));
    psii = exp(-tmp);
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

    tmp = 1.0 - pow( (1.0 + s), (-1/de));
    psii = 1.0 - pow(tmp, (1/th));
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

    tmp = 1.0 + pow(s, (1/de));
    psii = pow(tmp, (-1/th));
    out *= psii;
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
            tmp = pow(s, 1/de[i]);
            psii = 1.0 - gsl_sf_beta_inc(de[i], 1/va[i], tmp / (1+tmp));
            out *= psii;
        }
    }
    return out;
}

double LTI_vector(double s, vector<double> par) // LT of independent sum is the product of LTs
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
            tmp = 1.0 - pow( (1 + s), (-1 / de[i]) );
            psii = 1.0 - pow( tmp, (1/th[i]) );
            out *= psii;
        }
    }
    return out;
}

double LTA_vector(double s, vector<double> par) // LT of independent sum is the product of LTs
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
            tmp = pow(s, (1/de[i]));
            psii = exp(-tmp);
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
            tmp = 1.0 + pow(s, (1/de[i]));
            psii = pow(tmp, (-1/th[i]));
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
    tmp = pow(s, (1/de));
    out = exp(-tmp);
    return out;
}

double IML(double s, vector<double> par)
{
    double tmp, out;
    double de = par[0];
    double va = par[1];
    tmp = pow(s, 1/de);
    out = 1.0 - gsl_sf_beta_inc(de, 1/va, tmp / (1+tmp));
    return out;
}

double LTI(double s, vector<double> par)
{
    double tmp, out;
    double de = par[0];
    double th = par[1];
    tmp = pow( (1+s), (-1 / de) );
    out = 1 - pow( (1 - tmp), (1/th) );
    return out;
}

double LTE(double s, vector<double> par)
{
    double tmp, out;
    double de = par[0];
    double th = par[1];
    tmp = 1.0 + pow(s, (1/de));
    out = pow(tmp, (-1/th));
    return out;
}

// one for LTA and one for LTE
double LTA_LTE(double s, vector<double> par_1, vector<double> par_2)
{
    double out = LTA(s, par_1) * LTE(s, par_2);
    return out;
}


/////////////////////////////////////////////////////////////////////////////////////////
///  Martin Ridout                                                                     //
///  Generating random numbers from a distribution specified by its Laplace transform  //
///  modified from r codes at                                                          //
///  http://www.kent.ac.uk/smsas/personal/msr/rlaptrans.html                           //
/////////////////////////////////////////////////////////////////////////////////////////

double qGfromLT(double u, LTfunc_complex ltpdf, vector<double> par)
{
    const double tol = TOL;
    const double x0 = 1;
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
        tmp.real((i+1) / L);
        tmp.imag(0.0);
        y[i] = M_PI * I * tmp;
        expy[i] = exp(y[i]);
    }
    
    A2L = 0.5 * A / L;
    expxt = exp(A2L) / L;
    double pow2m = pow(2, m);
    	
    // for(i = 0; i < m+1; ++i){coef[i] = gsl_sf_choose(m, i) / pow2m;}
    // suggested by harry
    double tem=1.;
    for(i = 0; i < m+1; ++i) 
    {
        coef[i] = tem/ pow2m; 
        tem*=(m-i)/(i+1.);
    }
	
    size_t kount = 0;
    t = x0 / xinc; // omit the steps to find the potential maximum, refer to Ridout

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
    return t;
}

double qG(double u, LTfunc_complex LT, vector<double> par)
{
    if(u == 0)
    {
        return 0.0;
    }
    else
    {
        return qGfromLT(u, LT, par);
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
            ze = 1 / va[i] + de[i];
            psi10_i = - pow((1 + pow(s, 1/de[i])), -ze) / (de[i] * (gsl_sf_beta(de[i], 1/va[i])));
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
            psi10_i = ( -1 / (de[i]*th[i]) ) * (pow(( 1-pow((1+s), (-1/de[i])) ), (1/th[i] - 1))) * (pow((1+s), (-1/de[i]-1))); 
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
            psi10_i = ( -1 / (de[i]*th[i]) ) * (pow(1+pow(s,1/de[i]),-1/th[i]-1)) * (pow(s,1/de[i]-1)); 
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
            psi10_i = exp(-pow(s,1/de[i])) / (-de[i]) * (pow(s, 1/de[i] -1 )); 
            par_i.clear();
            par_i.push_back(de[i]);
            psi0_i = LTA(s, par_i);
            out += psiprod / psi0_i * psi10_i;
        }
    }
    return out;
}

double LTA1_LTE1(double s, vector<double> par_1, vector<double> par_2)
{
    double de = par_1[0];
    double de1 = par_2[0];
    double th1 = par_2[0];

    double psi1_LTA = exp(-pow(s,1/de)) * (pow(s, 1/de -1 )) / (- de); 
    double psi_LTA = LTA(s, par_1);
    double psi1_LTE = ( -1 / (de1*th1) ) * (pow(1+pow(s,1/de1),-1/th1-1)) * (pow(s,1/de1-1)); 
    double psi_LTE = LTE(s, par_2);
    double out = psi1_LTA*psi_LTE + psi1_LTE*psi_LTA;
    return out;
}

double invpsi(double u, vector<double> par, int LTfamily)
{
    const double tol = TOL;
    const double x0 = 1;
    const double xinc = 2;
     
    double LT = 1;
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
double invpsi_LTA_LTE(double u, vector<double> par_1, vector<double> par_2)
{
    const double tol = TOL;
    const double x0 = 1;
    const double xinc = 2;
     
    double LT = 1;
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
    return t;
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

// [[Rcpp::export]]
double denCF(NumericVector tvec, NumericMatrix DM, NumericVector parCluster, 
            NumericVector parFAC, int parMode, int LTfamily, int nq)
{
    size_t i, j, m, m1, m2, m3;
    const int f = DM.ncol();           // to-do: f = 3 is currently supported
    const int d = DM.nrow();
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
    else if( LTfamily == 6)
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

    // calculate qvec for reuse, improving speed
    for(i=0; i < d; ++i)
    {
        for(m=0; m < nq; ++m)
        {
            qvec(i,m) = qG(xl[m], LT, par_i[i]);
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

/////////////////////////////////
// comonotonic bi-factor model //
/////////////////////////////////

// [[Rcpp::export]]
double den_LTA_LTE(NumericVector tvec, NumericMatrix DM, NumericVector par, int nq)
{
    // devec are paramters for positive stable LT (LTA)
    // devec1 and thevec1 are parameters for the Mittag-Leffler LT
    
    size_t i, j, m, m1, m2, m3, m4;
    size_t f = DM.ncol();
    size_t d = DM.nrow();
	
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
	
    for(i=0; i < d; ++i)
    {
        for(m=0; m < nq; ++m)
        {
            qvec(i,m) = qG(xl[m], LT_1, par_1_i[i]);
            hvec(i,m) = qG(xl[m], LT_2, par_2_i[i]);
#ifdef DEBUG
            Rcpp::Rcout << "(i,m)" << i << "," << m <<    " par_1= " << par_1_i[i][0] <<  " xl[m]= " << xl[m] <<  " qvec(i,m)= " << qvec(i,m) << " hvec(i,m) = " << hvec(i,m) << std::endl;
#endif
        }
    }
    
    for(i = 0; i < d; ++i)
    {
        invpsi_i[i] = invpsi_LTA_LTE(tvec[i], par_1_i[i], par_2_i[i]);
        psi1inv_i[i] = LTA1_LTE1(invpsi_i[i], par_1_i[i], par_2_i[i]);
#ifdef DEBUG
        //Rcpp::Rcout << "invpsi_i[i] / psi1inv_i[i]: " << invpsi_i[i] << " / " << psi1inv_i[i] << std::endl;
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
                    //Rcpp::Rcout << "m / den_m: " << m1<<m2<<m3<<m4 << " / " << den_m << " / " << tem1 << " / " << tem2 <<  " / " << wl[m1]*wl[m2]*wl[m3]*wl[m4]*den_m << std::endl;
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
