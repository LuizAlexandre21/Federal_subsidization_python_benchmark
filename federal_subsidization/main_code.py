# Bibliotecas 
import numpy as np 
import scipy as sc 
import agent_params
from scipy.io import loadmat

# Crescimento da População 
pgr = 0.012 # Taxa de crescimento da população
mu2 = np.zeros(1,T)
mu2[1] = 1 
for i1 in range(2,T):
    mu2[i1] = (surv[1,1,i1]/(1+pgr))*mu2[i1-1]

mu2 = np.matmul((1/sum(mu2))),mu2[i1-1]
mu3 = np.matmul(np.squeeze(1-surv[1,1,:]),mu2)

gdp2 = 1.2 
population = 1
retiredpop = sum(mu2[Tr:])
ss_should_be = ss*retiredpop/(gdp2)
percapgdp = (gdp2)/population
cfloor_should_be = percapgdp*(27/440)
Gf_target = .12*(1-.45)
Gs_target = .06*(1-.25)
Gf = gdp2*Gf_target
Gs = gdp2*Gs_target
FMAP=.6
threshgrid = [0,0.565751862694302]
nthresh = length(threshgrid)
threshdev = .035
i1 = 2  %1:nthresh
thresh1 = threshgrid[i1]
welfss,welf1,welf2 = transeqm(thresh1,threshdev)

