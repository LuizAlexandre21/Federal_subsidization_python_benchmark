import numpy as np 
import scipy as sc 
import agent_params
from scipy.io import loadmat

# Importando dados externos 
insdata4 = loadmat('/home/alexandre/Documents/Mestrado/Artigos de dissertação/Replication Files/Benchmark Economy/insdata4.mat')
wage_med_data = loadmat('/home/alexandre/Documents/Mestrado/Artigos de dissertação/Replication Files/Benchmark Economy/wage_med_data1.mat')

# Crescimento da População 
pgr = 0.12
mu2 = np.zeros(1,T)
mu2[1] = 1 
for i1 in range(2,T):
    mu2[i1] = surv[1,1,i1]/(1+pgr)*mu2[i1-1]

mu2 = np.matmul((1/sum(mu2)),mu2)
mu3 = np.matmul(np.squeeze(1-surv[1,1,:]),np.transpose(mu2))
gdp2 = 1.2 
population = 1
retiredpop = sum(mu2[Tr:])
ss_should_be = ss*retiredpop/(gdp2)  # ss as a percentage of GDP should be about 4.5
percapgdp = (gdp2)/population
cfloor_should_be = percapgdp*(27/440)
Gf_target = .12*(1-.45)    # Percentage of federal budget not allocated to Medicaid, Medicare, or SS
Gs_target = .06*(1-.25)    # Percentage of state budget not allocated to Medicaid
Gf = gdp2*Gf_target
Gs = gdp2*Gs_target
FMAP=.6
