### comonotonic bi-factor model: LTA (B6, PS) + LTA (B6, PS) for the boco data
### comonotonic bi-factor model: LTE (BB1, ML) + LTE (BB1, ML) for the boco data
### use different scale parameters to handle within and among group dependence
rm(list = setdiff(ls(), lsf.str()))
require(Rcpp)
require(RcppGSL)
require(RcppArmadillo)
require(CopulaModel)
sourceCpp("comFac.cpp", verbose = T, rebuild = T)

nllk_LTE_LTE_s = function(par, parM, grp, dat, nq=31)
{
  dat = as.matrix(dat)
  nd = sum(grp)
  parvec = NULL
  for(i in 1:length(par))
  {
    parvec = append(parvec, rep(par[i], parM[i]))
  }
  parvec[1:nd] = exp(parvec[1:nd])+1
  parvec[(nd+1):(2*nd)] = exp(parvec[(nd+1):(2*nd)])
  parvec[(2*nd+1):(3*nd)] = 1 / (1 + exp( -parvec[(2*nd+1):(3*nd)]))
  parvec[parvec > 30] = 30
  nllk = -den_LTE_LTE_s(dat, grp, parvec, nq)
  if( is.na(nllk) || is.infinite(nllk) ) { nllk = 0 }
  return(nllk)
}

nllk_LTA_LTA_s = function(par, parM, grp, dat, nq=31)
{
  dat = as.matrix(dat)
  nd = sum(grp)
  parvec = NULL
  for(i in 1:length(par))
  {
    parvec = append(parvec, rep(par[i], parM[i]))
  }
  parvec[1:nd] = exp(parvec[1:nd])+1
  parvec[(nd+1):(2*nd)] = 1 / (1 + exp( -parvec[(nd+1):(2*nd)]))
  parvec[parvec > 30] = 30
  nllk = -den_LTA_LTA_s(dat, grp, parvec, nq)
  if( is.na(nllk) || is.infinite(nllk) ) { nllk = 0 }
  return(nllk)
}

############## data #############
source("data.r")
dat = UU
grp = c(5,3,4)
######################## PS+PS ###############
parM = rep(1, sum(grp)*2)
par0 = c( log(c(3.46, 2.98, 2.96, 1.80, 4.06, 2.35, 2.45, 2.21, 1.72,2.16,3.16,2.46)-1) , rep(0.85, sum(grp)))
fit = nlm(nllk_LTA_LTA_s, p=par0, parM=parM, grp=grp, dat=dat, nq=21, hessian = F, print.level = 2, iterlim = 10000)
PAR = fit$estimate
nllk = fit$minimum
cat("bi-factor LTA/LTA, convergence: ", fit$code, "\n", file = filep, append = T)
cat("bi-factor LTA/LTA, estimates: ", PAR, "\n", file = filep, append = T)
cat("bi-factor LTA/LTA, nllk: ", nllk, "\n", file = filep, append = T)

######################## ML+ML ###############
parM = rep(1, sum(grp)*2)
par0 = c(rep(-1, sum(grp)*2), rep(0.85, sum(grp)))
fit = nlm(nllk_LTE_LTE_s, p=par0, parM=parM, grp=grp, dat=dat, nq=21, hessian = F, print.level = 2, iterlim = 10000)
PAR = fit$estimate
nllk = fit$minimum
cat("bi-factor LTA/LTA, convergence: ", fit$code, "\n")
cat("bi-factor LTA/LTA, estimates: ", PAR, "\n")
cat("bi-factor LTA/LTA, nllk: ", nllk, "\n")
