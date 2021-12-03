# Pacotes 
import numpy as np 
import scipy as sc 
import agent
import head
import tail 
from scipy.io import loadmat

def sseqm1(thresh):
    # Run main_code_params
    
    beqapprox = .0275;    # portion of bequest redistributed

    Gf1 = Gf 
    tftol=.001
    tstol=.0001
    tsub = .175
    tslb = .1
    tseps = 10

    while tseps>tstol:
        ts1 = (tsub+tslb)/2    
        tfub = .65
        tflb = .35
        tfeps = 10
        while tfeps > tftol:
            tf1 = (tfub+tflb)/2
            rho = 5.6
            rhoeps = 10
            rhotol = .001
            iter=0
            tau = [tf1 ts1]   # [federal, state]
            rhogrid = rho
            while rhoeps > rhotol:
                iter=iter+1
                [~, ~, KPR, ELAB] = distss_short(rho,tau,thresh);  % short version for iterations
                Kgrid=KPR;    % "grid" is 1x1 in steady-state, length of transition path otherwise
                Lgrid=ELAB
                rhopr = Kgrid/Lgrid
                rhogrid = [rhogrid,rhopr]
                if iter==8:
                    rhopr = np.mean(rhogrid(end-3:end))
                end
                if iter<=11:
                    rhoeps = np.abs(rho-rhopr)
                elif iter>11:
                    rhopr = np.mean(rhogrid(end-4:end))
                    rhoeps=0
                rho = .25*rho+.75*rhopr
            dist1, dist2, MED, KPR, HRS, ELAB, MCARE, MCAID, YTAX, CTAX, TRANSF, WELF, BEQ, YTAXSSM, WELF2 = distss_long(rho,tau,thresh)   %#long version for other variables
            Mdist[1] = sum(sum(dist1[:,:,1]+dist2[:,:,1]))/sum(sum(sum(dist1+dist2)))    # total Medicaid
            Mdist[2] = sum(sum(dist1[:,1:Tr-1,2]+dist2[:,1:Tr-1,2]))/sum(sum(sum(dist1+dist2)))    # total uninsured
            Mdist[3] = sum(sum(dist1[:,:,3]+dist2[:,:,3]))/sum(sum(sum(dist1+dist2)));  # total private, non-ESHI
            Mdist[4] = sum(sum(dist1[:,:,4]+dist2[:,:,4]))/sum(sum(sum(dist1+dist2)));  # total private, ESHI
            Mdist2[1] = sum(sum(dist1[:,1:Tr-1,1]+dist2[:,1:Tr-1,1]))/sum(sum(sum(dist1[:,1:Tr-1,:]+dist2[:,1:Tr-1,:])))    % working-age Medicaid
            Mdist2[2] = sum(sum(dist1[:,1:Tr-1,2]+dist2[:,1:Tr-1,2]))/sum(sum(sum(dist1[:,1:Tr-1,:]+dist2[:,1:Tr-1,:])))    % working-age uninsured
            tf1
            ts1
            welf = WELF
            welf2 = WELF2
            mcare = MCARE
            mcaid = MCAID
            med = MED
            ytax = YTAX
            ytaxssm = YTAXSSM
            ctax = CTAX
            transfer = TRANSF
            beq = BEQ
            hrs = HRS
            tfrev = ytax  +sum(mu2[Tr:T])*premmedicare + beq 
            tfexp = Gf1 + .5*transfer + FMAP*mcaid+ mcare + sum(mu2[Tr:T])*ss + beqapprox
            tfeps = np.abs(tfrev-tfexp)
            if tfrev>tfexp:
                tfub=tf1
            elif tfrev<=tfexp:
                tflb=tf1
        
    Y = (KPR^alpha)*(ELAB^(1-alpha))    # total output
    K_Y = KPR/Y    # capital-to-output ratio
    rhoeps 
    tfeps
    srev = ctax + ytaxssm;  # state revenue
    sexp = Gs  + (1-FMAP)*mcaid + .5*transfer;    # state expenditures
    if srev>sexp:
        tsub=ts1
    elif srev<=sexp:
        tslb=ts1
        
    tseps = max(abs(srev-sexp))
          
    return dist1,dist2,rho,tau,KPR,welf,welf2