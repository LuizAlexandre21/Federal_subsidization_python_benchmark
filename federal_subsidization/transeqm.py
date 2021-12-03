import numpy as np 
import scipy as sc 
import agent
import head
import tail 
import sseqm1
from scipy.io import loadmat

def transeqm(thresh,threshdev):
    
    # Detemina o estado estacionario de equilibrio
    dist1,dist2,rho,tau,KPR,welfss = sseqm1(thresh)
     
    # Parametros 
    rho0 = rho    # define K / L no valor de estado estacionário na etapa anterior. as regiões atomísticas implicam que isso não muda com as mudanças nas políticas regionais.
    ftau0 = tau[1] 
    stau0 = tau[2]     # este é o mesmo para ambos os estados
    dist10 = dist1     # mesmo para ambos os estados
    dist20 = dist2     # mesmo para ambos os estados
    
    Tss = 50
    Kgrid = np.zeros(1,Tss)
    Lgrid = np.zeros(1,Tss)
    welf1 = np.zeros(1,Tss)
    welf2 = np.zeros(1,Tss)
    tstol = .0001

    ter = 0
    tseps = 10

    thresh2 = thresh + threshdev
    ts1ub = np.matmul((stau0+.015),ones(1,Tss))
    ts1lb = np.matmul((max(stau0-.015,0)),ones(1,Tss))

    while tseps > tstol:
        ts1 = (ts1ub+ts1lb)/2
        tf = np.matmul(ftau0,np.ones(1,Tss))
        rho = np.matmul(rho0,np.ones(1,Tss))
        iter=iter+1
        tau1 = [tf,ts1]
        KPR, ELAB = dist_short(dist10,dist20,rho,tau1,thresh2,1)  # agent policy functions along transition path
        Kgrid[1,:]=KPR
        Lgrid[1,:]=ELAB

        MED, KPR, HRS, ELAB, MCARE, MCAID, YTAX, CTAX, TRANSF, WELF, BEQ, YTAXSSM, WELF2 = dist_long(dist10,dist20,1); # takes policy functions, solves all other variables

        welf1 = WELF  # this will change for each iteration
        welf2 = WELF2  # this will change for each iteration
        mcare = MCARE
        mcaid = MCAID
        med = MED
        ytax = YTAX
        ytaxssm = YTAXSSM
        ctax = CTAX
        transfer = TRANSF
        beq = BEQ
        hrs = HRS   


        s1rev = ctax[1,:] + ytaxssm[1,:]
        s1exp = Gs  + (1-FMAP)*mcaid[1,:] + .5*transfer[1,:]
        for ii1 in range(1,Tss):
            if s1rev[ii1] > s1exp[ii1]:
                ts1ub[ii1] = ts1[ii1]
            elif s1rev[ii1] <= s1exp[ii1]:
                ts1lb[ii1]=ts1[ii1]
        
        tseps = max(np.abs(ts1ub-ts1lb))
   
    return welfss,welf1,welf2