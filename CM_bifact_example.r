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
source("data_example.r")
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
fit2 = nlm(nllk_LTE_LTE_s, p=par0, parM=parM, grp=grp, dat=dat, nq=21, hessian = F, print.level = 2, iterlim = 10000)
PAR = fit$estimate
nllk = fit$minimum
cat("bi-factor LTE/LTE, convergence: ", fit$code, "\n")
cat("bi-factor LTE/LTE, estimates: ", PAR, "\n")
cat("bi-factor LTE/LTE, nllk: ", nllk, "\n")

### checking model-based Spearman's rho
DM = t( matrix(c(
    1,0,0,1,
    1,0,0,1,
    1,0,0,1,
    1,0,0,1,
    1,0,0,1,
    0,1,0,1,
    0,1,0,1,
    0,1,0,1,
    0,0,1,1,
    0,0,1,1,
    0,0,1,1,
    0,0,1,1)
    ,4,12))
  
  ############
  ## PS+PS ###
  ############
  par = fit$estimate
  nd = dim(DM)[1]
  parvec = NULL
  
  parvec[1:nd] = exp(par[1:nd])+1
  parvec[(nd+1):(2*nd)] = 1 / (1 + exp( -par[(nd+1):(2*nd)]))
  rhovec = srho_LTA_LTA_s(DM, parvec, nq=21)
  source("data.r")
  emprirho = rep(NA, 12*11/2)
  
  k = 1
  for(i in 1:11)
  {
    for(j in (i+1):12)
    {
      emprirho[k] = cor(x = UU[,i], y = UU[,j], method = "spearman")
      k=k+1
    }
  }
  
  nam = NULL
  for(i in 1:11)
  {
    for(j in (i+1):12)
    {
      nam = c(nam, paste(format(i), format(j),sep="."))
    }
  }
  
  names(rhovec) = names(emprirho) = nam
  pdf("CM-bifact2-PSPS.pdf", width = 15, height = 15)
  plot(emprirho, rhovec, xlim=c(0,1), ylim=c(0,1), main = "CM-bifact2-PSPS")
  text(emprirho, rhovec, nam, cex=0.6, pos=4, col="red")
  abline(c(0,0), c(1,1))
  dev.off()
  
  ############
  ## ML+ML ###
  ############
  par = fit2$estimate
  nd = dim(DM)[1]
  parvec = NULL
  
  parvec[1:nd] = exp(par[1:nd])+1
  parvec[(nd+1):(2*nd)] = exp(par[(nd+1):(2*nd)])
  parvec[(2*nd+1):(3*nd)] = 1 / (1 + exp( -par[(2*nd+1):(3*nd)]))
  rhovec = srho_LTE_LTE_s(DM, parvec, nq=21)
  source("data.r")
  emprirho = rep(NA, 12*11/2)
  
  k = 1
  for(i in 1:11)
  {
    for(j in (i+1):12)
    {
      emprirho[k] = cor(x = UU[,i], y = UU[,j], method = "spearman")
      k=k+1
    }
  }
  
  nam = NULL
  for(i in 1:11)
  {
    for(j in (i+1):12)
    {
      nam = c(nam, paste(format(i), format(j),sep="."))
    }
  }
  
names(rhovec) = names(emprirho) = nam
pdf("CM-bifact2-MLML.pdf", width = 15, height = 15)
plot(emprirho, rhovec, xlim=c(0,1), ylim=c(0,1), main = "CM-bifact2-MLML")
text(emprirho, rhovec, nam, cex=0.6, pos=4, col="red")
abline(c(0,0), c(1,1))
dev.off()
