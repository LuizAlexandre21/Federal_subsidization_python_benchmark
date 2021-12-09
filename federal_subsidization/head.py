import numpy as np 
import scipy as sc 
from agent  
from scipy.io import loadmat

def head(rho,tau,thresh1,agetrans):
    insdata4 = loadmat('/home/alexandre/Documents/Mestrado/Artigos de dissertação/Replication Files/Benchmark Economy/insdata4.mat')
    wage_med_data = loadmat('/home/alexandre/Documents/Mestrado/Artigos de dissertação/Replication Files/Benchmark Economy/wage_med_data1.mat')
    sspol = loadmat('/home/alexandre/Documents/Mestrado/Artigos de dissertação/Replication Files/Benchmark Economy/sspol.mat')

    beq = beqapprox 
    ngprem = 0 
    gamma = 1.11    #Health insurance load
    b = .9875

    chi3 = .3    #increase to reduce value of leisure (and increase labor supply)
    sigma4 = 4

    T = 80
    Tr = 45    # Retirement age
    alpha = .36    # capital share
    A = 1   # tfp
    phimedicare = .5
    d = .083
    nh = 2    # number of health states - good, bad
    a0 = .258
    a1 = .768    # tax function parameter 1
    ss = 0.27    # Social security payments: 4.5% of GDP.
    tau_c = .057    # Consumption tax -  Mendoza, Razin, and Tesar 1994
    premmedicare = .026    # Medicare premium
    cfloor = .08   # consumption floor - Financed by federal government
    pm = .00003625    # normalization that ensures med expenses are 18% of GDP
    Mnorm = np.matmul(pm,M)    # normalizing price of medical expenses
    tss =.124*2   # ss tax rate ( multiplied by 2 here to undo error in code re: employer + employee portion.)
    tm = .029    # medicare tax
    ybar = 3.65    # max taxable earnings for ss tax

    # Individual Capital
    nk = 50
    kgrid = np.zeros(nk,1)
    kub = 13
    klb = .001
    exppwr = 3    # ensures high concentration of gridpoints near the lower bound
    for i1 in range(1,nk):
        kgrid[i1] = (kub-klb)*((i1/nk))^exppwr + klb
    
    kgrid = [klb;kgrid]
    nk = len(kgrid)

    # Labor Grid
    nl = 25
    lub = .65
    llb = 0
    lgrid = np.zeros(nl,1)
    linc = (lub-llb)/(nl-1)
    lgrid[1] = llb
    for i1 in range(2,nl):    
        lgrid[i1] = lgrid[i1-1] + linc

    ng = 2    # group offer status
    nins = 4

    if agetrans>=Tr:
        [KGRID1, KGRID2] = ndgrid(kgrid,kgrid)    # KGRID1 is k, KGRID2 is k'
        KGRID1 = np.tile(KGRID1,[1,1,2])
        KGRID2 = np.tile(KGRID2,[1,1,2])
        ZEROGRID = np.zeros(nk,nk,2)
        ONESGRID = np.ones(nk,nk,2)
        mincons = (1/(1+tau_c))*np.matmul(cfloor,ONESGRID)

        phi2 = phimedicare    
        ssben = ss
        g1 = 1
        prem1 = premmedicare

        thresh=.125

        for t1 in range(agetrans,Tr,-1): 
            rate = 1 + A*alpha*(rho(t1))^(alpha-1) - d
            a2 = tau[:,t1]    # tax function parameter 2 ------[FED STATE]
            j2 = 1    #:nep        
            for j3 in range(1,nh):
                med = Mnorm(j3,t1)
                surv1 = surv[j2,j3,t1]
                count = j3 + (j2-1)*nh
                Pr = TrEprime[count,:,t1]
                phimedicaid = phi[j3,t1,1]
                phigrid = [phimedicaid, 1]

                EVprg1 = Pr[1]*V[:,1,1,t1+1,1,1]
                EVprb1 = Pr[2]*V[:,1,2,t1+1,1,1]
                E1 = EVprg1+EVprb1
                EVprg2 = Pr[1]*V[:,1,1,t1+1,1,2]
                EVprb2 = Pr[2]*V[:,1,2,t1+1,1,2]
                E2 = EVprg2+EVprb2
                Ehold1 = np.transpose([E1 E2])            
                Ehold2 = np.tile(Ehold1,[1,1,nk])    # replicates "E" nk times;  dims are ins',k',k
                E = np.randompermute(Ehold2,[3,2,1])     # permutes to dims k,k',ins       

                income = np.matmul(rate,KGRID1) + ssben
                tincome = np.matmul(max(rate-1,0),KGRID1) + ssben
                tincome2 = tincome[:,:,1]    # getting rid of ins dim to do it manually for medicaid/nothing below

                trfind = np.where(tincome2<=thresh)   # below means test
                ntr = len(trfind)
                [I1, I2] = ind2sub(size[tincome2],trfind)
                tr1 = sub2ind(size[tincome],I1,I2,2*np.ones[ntr,1])   # manually inputting "2" for ins dim

                trfind = find(tincome2>thresh)    # above means test
                ntr = length(trfind)
                [I1, I2] = ind2sub(size(tincome2),trfind)
                tr2 = sub2ind(size(tincome),I1,I2,1*ones(ntr,1))

                for instype in range(1,2):    # can expand to include private insurance
                    phi1 = phigrid[instype]
                    taxdeduc = max(0,phi1*(phi2*med) - .075*tincome)    # deduction
                    ytax1 = max(ZEROGRID,tincome - taxdeduc)
                    ytaxstargrid3 = np.matmul(a0,np.matmul(ytax1 - (ytax1.^(-a1) + (a2(2)),ONESGRID)).^(-1/a1))
                    ytaxstargrid2 = np,matmul(a0,((ytax1) - np.matmul(((ytax1).^(-a1) + (a2(1)),ONESGRID))).^(-1/a1))

                    # Placing the consumption floor
                    TRANSF = max(ZEROGRID, mincons - income + np.matmul((taxdeduc - beq),ONESGRID) + ytaxstargrid2 + ytaxstargrid3)
                    cons1 = (1/(1+tau_c))*(income - (prem1 + np.matmul(phi1*(phi2*med) - beq),ONESGRID) - KGRID2 - ytaxstargrid2 - ytaxstargrid3 + TRANSF)
                    tr3 = np.where(cons1<=0)
                    Vtilde = np.matmul(1/(1-sigma4)),((cons1.^chi3),(ONESGRID.^(1-chi3)))).^(1-sigma4) + (surv1*b).*E

                    Vtilde(tr1) = -10e6    # Medicaid = not uninsured                
                    Vtilde(tr2) = -10e6    # Uninsured = no medicaid
                    Vtilde(tr3) = -10e6

                    Vtilde1 = np.random.permute(Vtilde,[3,2,1])
                    V[:,j2,j3,t1,1,instype] = max(max(Vtilde1))

                    [Vhold2, Ik] = max(max(Vtilde1),[],2)

                    Vtilde1 = np.random.permute(Vtilde,[2,3,1])
                    [Vhold3, Iins] = max(max(Vtilde1),[],2)

                    insopt[:,j2,j3,t1,g1,instype] = Iins
                    kopt[:,j2,j3,t1,g1,instype] = Ik  

                    Ik = np.squeeze(Ik)
                    Ik = np.transpose(Ik)                
                    Iins = np.squeeze(Iins)               
                    Iins = np.transpose(Iins)    
                    IND1 = sub2ind(size(cons1),1:nk,Ik,Iins)

                    copt[:,j2,j3,t1,g1,instype] = cons1[IND1]
                    transf[:,j2,j3,t1,g1,instype] = TRANSF[IND1]               
                    ytax[:,j2,j3,t1,g1,instype]= ytaxstargrid2[IND1]
                    ytaxst[:,j2,j3,t1,g1,instype] = ytaxstargrid3[IND1]

        t1 = Tr-1;    

        [LGRID,KGRID1, KGRID2] = np.ndgrid(lgrid,kgrid,kgrid)
        LGRID = np.title(LGRID,[1,1,1,2])
        KGRID1 = np.title(KGRID1,[1,1,1,2])
        KGRID2 = np.title(KGRID2,[1,1,1,2])    # NOTICE: KGRID2 is now K
        ZEROGRID = np.zeros(nl,nk,nk,2)
        ONESGRID = np.ones(nl,nk,nk,2)
        mincons = (1/(1+tau_c))*np.matmul(cfloor,ONESGRID)

        thresh = thresh1
        ssben = 0    # no social security

        wage = A*(1-alpha)*(rho(t1))^(alpha)
        rate = 1 + A*alpha*(rho(t1))^(alpha-1) - d
        a2 = tau[:,t1]    # tax function parameter 2 ------[FED STATE]

        for j2 in range(1,nep):
            for j3 in range(1,nh):
                zstar = z[j2,j3,t1]
                med = Mnorm[j3,t1]
                surv1 = surv[j2,j3,t1]
                count = j3 + (j2-1)*nh
                Pr = TrE[count,:,t1]
                phimedicaid = phi[j3,t1,1]
                phipriv = phi[j3,t1,3]                                            
                phigrid = [phimedicaid, 1, phipriv, phipriv] 

                g1=1

                # Expected Value of Choosing Medicaid
                E1 = Pr[1]*V[:,1,1,t1+1,1,1]+Pr[2]*V[:,1,2,t1+1,1,1]

                # Expected Value of Choosing Uninsurance
                E2 = Pr[1]*V[:,1,1,t1+1,1,2]+Pr[2]*V[:,1,2,t1+1,1,2]

                Ehold = np.transpose([E1 E2])
                Ehold2 = np.title(Ehold,[1,1,nk,nl])    # replicates "E" (nk,nl) times            
                E = np.random.permute(Ehold2,[4,2,3,1])

                income = np.matmul(rate,KGRID2) + np.matmul((wage*zstar),LGRID)
                tincome = np.matmul(max(rate-1,0),KGRID2) + np.matmul((wage*zstar),LGRID)

                tincome3 = np.squeeze(tincome[:,:,:,1])
                trfind = np.where(tincome3<=thresh)
                ntr = len(trfind)
                [I1, I2, I3] = ind2sub(size(tincome3),trfind);
                tr1 = sub2ind(size(tincome),I1,I2,I3,2*ones(ntr,1));

                trfind = np.find(tincome3>thresh)
                ntr = len(trfind)
                [I1, I2, I3] = ind2sub(size(tincome3),trfind)
                tr2 = sub2ind(size(tincome),I1,I2,I3,1*ones(ntr,1))
                for instype = 1:4
                    phi1 = phigrid[instype]
                    taxdeduc = max(0,phi1*phi2*med - .075*tincome)  
                    ytax1 = max(ZEROGRID,tincome - taxdeduc)
                    ytaxstargrid3 = np.matmul(a0,np.matmul(ytax1 - (ytax1.^(-a1) + (a2(2)),ONESGRID)).^(-1/a1))
                    ytaxstargrid2 = np,matmul(a0,((ytax1) - np.matmul(((ytax1).^(-a1) + (a2(1)),ONESGRID))).^(-1/a1))

                    # Placing the consumption floor
                    TRANSF = np.matmul(max(ZEROGRID, mincons - income + (prem1 + phi1*phi2*med - ssben - beq),ONESGRID) + ytaxstargrid2 + ytaxstargrid3)

                    cons1 = (1/(1+tau_c))*(income - np.matmul((prem1 + phi1*phi2*med - ssben - beq),ONESGRID) - KGRID1 - ytaxstargrid2 - ytaxstargrid3 + TRANSF)

                    tr3 = find(cons1<=0)

                    Vtilde = np.matmul(1/(1-sigma4)),((cons1.^chi3),(ONESGRID.^(1-chi3)))).^(1-sigma4) + np.matmul((surv1*b),E)
                    Vtilde[tr1] = -10e6;    # Medicaid = not uninsured                
                    Vtilde[tr2] = -10e6;    # Uninsured = not medicaid eligible    
                    Vtilde[tr3] = -10e6;    # don't consume negative
                    Vtilde1 = np.random.permute(Vtilde,[1,2,4,3]);

                    V[:,j2,j3,t1,g1,instype] = max(max(max(Vtilde1)))
                    [Vhold1, Iins] = max(max(max(Vtilde1)),[],3)

                    Vtilde1 = np.random.permute(Vtilde,[1,4,2,3])
                    [Vhold2, Ik] = max(max(max(Vtilde1)),[],3)

                    Vtilde1 = np.random.permute(Vtilde,[4,2,1,3])
                    [Vhold3, Il] = max(max(max(Vtilde1)),[],3)

                    insopt[:,j2,j3,t1,g1,instype] = Iins
                    kopt[:,j2,j3,t1,g1,instype] = Ik  #(nl,nk,nk,4,T-Tr,nep,nh,ng)
                    labopt[:,j2,j3,t1,g1,instype] = lgrid[Il]

                    Il = np.squeeze(Il)             
                    Il = np.transpose(Il)               
                    Ik = np.squeeze(Ik)
                    Ik = np.transpose(Ik)              
                    Iins = np.squeeze(Iins);                
                    Iins = np.transpose(Iins)
                    IND1 = sub2ind(size(cons1),Il,Ik,1:nk,Iins)

                    copt[:,j2,j3,t1,g1,instype] = cons1[IND1]
                    transf[:,j2,j3,t1,g1,instype] = TRANSF[IND1]
                    ytax[:,j2,j3,t1,g1,instype] = ytaxstargrid2[IND1]
                    ytaxs[:,j2,j3,t1,g1,instype] = ytaxstargrid3[IND1]

                g1=2
                for instype in range(1,4):

                    phi1 = phigrid[instype]
                    taxdeduc = max(0,phi1*phi2*med - .075*tincome)    #deduction
                    ytax1 = max(ZEROGRID,tincome -taxdeduc)                                        
                    ytaxstargrid3 = np.matmul(a0,np.matmul(ytax1 - (ytax1.^(-a1) + (a2(2)),ONESGRID)).^(-1/a1))
                    ytaxstargrid2 = np,matmul(a0,((ytax1) - np.matmul(((ytax1).^(-a1) + (a2(1)),ONESGRID))).^(-1/a1))

                    # Placing the consumption floor
                    TRANSF = np.matmul(max(ZEROGRID, mincons - income + (prem1 + phi1*phi2*med - ssben - beq),ONESGRID) + ytaxstargrid2 + ytaxstargrid3)
                    cons1 = (1/(1+tau_c))*(income - np.matmul((prem1 + phi1*phi2*med - ssben - beq),ONESGRID) - KGRID1 - ytaxstargrid2 -ytaxstargrid3+ TRANSF)

                    tr3 = np.where(cons1<=0)

                    Vtilde = np.matmul(1/(1-sigma4)),((cons1.^chi3),(ONESGRID.^(1-chi3)))).^(1-sigma4) + np.matmul((surv1*b),E)
                    Vtilde(tr1) = -10e6    # Medicaid = not uninsured                
                    Vtilde(tr2) = -10e6    # Uninsured = not medicaid eligible
                    Vtilde(tr3) = -10e6

                    Vtilde1 = np.random.permute(Vtilde,[1,2,4,3])

                    V[:,j2,j3,t1,g1,instype] = max(max(max(Vtilde1)))
                    [Vhold1, Iins] = max(max(max(Vtilde1)),[],3)

                    Vtilde1 = np.random.permute[Vtilde,[1,4,2,3]]               
                    [Vhold2, Ik] = max(max(max(Vtilde1)),[],3)

                    Vtilde1 = np.random.permute(Vtilde,[4,2,1,3])
                    [Vhold3, Il] = max(max(max(Vtilde1)),[],3)

                    insopt[:,j2,j3,t1,g1,instype] = Iins
                    kopt[:,j2,j3,t1,g1,instype] = Ik  #(nl,nk,nk,4,T-Tr,nep,nh,ng)
                    labopt[:,j2,j3,t1,g1,instype] = lgrid[Il]

                    Il = np.squeeze(Il)              
                    Il = np.transpose(Il)                
                    Ik = np.squeeze(Ik)               
                    Ik = np.transpose(Ik)               
                    Iins = np.squeeze(Iins)
                    Iins = np.transpose(Iins)
                    IND1 = sub2ind(size(cons1),Il,Ik,1:nk,Iins);

                    copt[:,j2,j3,t1,g1,instype] = cons1[IND1]
                    transf[:,j2,j3,t1,g1,instype] = TRANSF[IND1]                
                    ytax[:,j2,j3,t1,g1,instype] = ytaxstargrid2[IND1]
                    ytaxst[:,j2,j3,t1,g1,instype] = ytaxstargrid3[IND1]
        
        
        ZEROGRID = np.zeros(nl,nk,nk,4)
        ONESGRID = np.ones(nl,nk,nk,4)
        mincons = (1/(1+tau_c))*np.(cfloor,ONESGRID)

        ssben = 0    # no social security


        for t1 = range(Tr-2,1-1): 
            wage = A*(1-alpha)*(rho(t1))^(alpha)
            rate = 1 + A*alpha*(rho(t1))^(alpha-1) - d
            a2 = tau[:,t1]    # tax function parameter 2 ------[FED STATE]

            for j2 in range(1,nep):
                for j3 in range(1,nh):
                    zstar = z[j2,j3,t1]
                    med = Mnorm[j3,t1]
                    surv1 = surv[j2,j3,t1]
                    count = j3 + (j2-1)*nh
                    Pr = TrE[count,:,t1]
                    phimedicaid = phi[j3,t1,1]
                    phipriv = phi[j3,t1,3]            
                    EMpriv = TrEprime[j3,1,t1]*(1-phi[1,t1+1,3])*Mnorm[1,t1+1] + TrEprime[j3,2,t1]*(1-phi[1,t1+1,3])*Mnorm[2,t1+1]            
                    prempriv = (1/rate)*gamma*EMpriv + ngprem                                    
                    phigrid = [phimedicaid, 1, phipriv, phipriv]            
                    premgrid1 = [0 0 prempriv (prempriv-ngprem)]

                    [LGRID, KGRID1, KGRID2, PREMGRID] = np.ndgrid(lgrid,kgrid,kgrid,premgrid1)    # NOTICE: KGRID2 is now K

                    g1 = 1 
                    
                    EV11 = Opr[g1,1,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,1,1]+Pr[2]*V[:,1,2,t1 - (Ttail -1] + 1,1,1]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,1,1]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,1,1] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,1,1]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,1,1]+Pr[7]*V[:,4,1,t1 - (Ttail -1) + 1,1,1]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,1,1])
                    EV21 = Opr[g1,2,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,2,1]+Pr[2]*V[:,1,2,t1 - (Ttail -1] + 1,2,1]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,2,1]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,2,1] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,2,1]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,2,1]+Pr[7]*V[:,4,1,t1 - (Ttail -1) + 1,2,1]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,2,1])
                    E1 = EV11+EV21

                    # Expected Value of Choosing Uninsurance
                    EV12 = Opr[g1,1,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,1,1]+Pr[2]*V[:,1,2,t1 - (Ttail -1) + 1,1,1]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,1,2]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,1,2] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,1,2]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,1,2)+Pr[7)*V[:,4,1,t1 - (Ttail -1) + 1,1,2]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,1,2])
                    EV22 = Opr[g1,2,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,2,2)+Pr[2]*V[:,1,2,t1 - (Ttail -1) + 1,2,2]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,2,2]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,2,2] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,2,2]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,2,2]+Pr[7]*V[:,4,1,t1 - (Ttail -1) + 1,2,2]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,2,2])
                    E2 = EV12+EV22

                    # Expected Value of Choosing Priv Insurance
                    EV13 = Opr[g1,1,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,1,3]+Pr[2]*V[:,1,2,t1 - (Ttail -1) + 1,1,3]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,1,3]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,1,3] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,1,3]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,1,3]+Pr[7]*V[:,4,1,t1 - (Ttail -1) + 1,1,3]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,1,3])
                    EV23 = Opr[g1,2,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,2,3]+Pr[2]*V[:,1,2,t1 - (Ttail -1) + 1,2,3]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,2,3]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,2,3] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,2,3]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,2,3]+Pr[7)*V[:,4,1,t1 - (Ttail -1) + 1,2,3]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,2,3])
                    E3 = EV13+EV23

                    # Expected Value of Choosing Group Insurance
                    E4 = np.matmul(-10e6,ones(nk,1))
                    Ehold = np.transpose([E1 E2 E3 E4])
                    Ehold2 = np.title(Ehold,[1,1,nk,nl])    # replicates "E" (nk,nl) times            
                    E = np.random.permute(Ehold2,[4,2,3,1])


                    income = np.matmul(rate,KGRID2) + np.matmul((wage*zstar),LGRID)
                    tincome = np.matmul(max(rate-1,0),KGRID2) + np.matmul((wage*zstar),LGRID)

                    tincome3 = np.squeeze(tincome[:,:,:,1])
                    trfind = np.where(tincome3<=thresh)
                    ntr = len(trfind)
                    [I1, I2, I3] = ind2sub(size(tincome3),trfind)
                    tr1 = sub2ind(size(tincome),I1,I2,I3,2*ones(ntr,1))

                    trfind = np.where(tincome3>thresh)
                    ntr = len(trfind)
                    [I1, I2, I3] = ind2sub(size(tincome3),trfind)
                    tr2 = sub2ind(size(tincome),I1,I2,I3,1*ones(ntr,1))


                    for instype in range(1,4):

                        phi1 = phigrid[instype]
                        taxdeduc = max(0,phi1*med - .075*tincome)    #deduction
                        ytax1 = max(ZEROGRID,tincome - taxdeduc)                                        
                        ytaxstargrid3 = np.matmul(a0,np.matmul(ytax1 - (ytax1.^(-a1) + (a2(2)),ONESGRID)).^(-1/a1))
                        ytaxstargrid2 = np,matmul(a0,((ytax1) - np.matmul(((ytax1).^(-a1) + (a2(1)),ONESGRID))).^(-1/a1))

                        # Placing the consumption floor
                        TRANSF = max(ZEROGRID, mincons - income + PREMGRID + (phi1*med - ssben - beq).*ONESGRID + ytaxstargrid2 + ytaxstargrid3);
                        cons1 = (1/(1+tau_c)).*(income - PREMGRID - (phi1*med - ssben - beq).*ONESGRID - KGRID1 - ytaxstargrid2 - ytaxstargrid3 + TRANSF); 
                        tr3 = find(cons1<=0)

                        Vtilde = np.matmul(1/(1-sigma4)),((cons1.^chi3),(ONESGRID.^(1-chi3)))).^(1-sigma4) + np.matmul((surv1*b),E)
                        Vtilde[tr1] = -10e6    # Medicaid = not uninsured                
                        Vtilde(tr2) = -10e6   # Uninsured = not medicaid eligible 
                        Vtilde(tr3) = -10e6
                        Vtilde1 = np.random.permute(Vtilde,[1,2,4,3])

                        V[:,j2,j3,t1,g1,instype] = max(max(max(Vtilde1)))
                        [Vhold1, Iins] = max(max(max(Vtilde1)),[],3)

                        Vtilde1 = np.random.permute(Vtilde,[1,4,2,3])               
                        [Vhold2, Ik] = max(max(max(Vtilde1)),[],3)

                        Vtilde1 = np.random.permute(Vtilde,[4,2,1,3])                
                        [Vhold3, Il] = max(max(max(Vtilde1)),[],3)

                        insopt[:,j2,j3,t1,g1,instype] = Iins
                        kopt[:,j2,j3,t1,g1,instype] = Ik  #(nl,nk,nk,4,T-Tr,nep,nh,ng)
                        labopt[:,j2,j3,t1,g1,instype] = lgrid[Il]

                        Il = np.squeeze(Il)
                        Il = np.transpose(Il)                
                        Ik = np.squeeze(Ik)
                        Ik = np.transpose(Ik)               
                        Iins = np.squeeze(Iins)                
                        Iins = np.transpose(Iins)
                        IND1 = sub2ind(size(cons1),Il,Ik,1:nk,Iins);

                        copt[:,j2,j3,t1,g1,instype] = cons1[IND1]
                        transf[:,j2,j3,t1,g1,instype] = TRANSF[IND1]                
                        ytax[:,j2,j3,t1,g1,instype] = ytaxstargrid2[IND1]
                        ytaxst[:,j2,j3,t1,g1,instype] = ytaxstargrid3[IND1]

    if agetrans==Tr-1:
        phi2 = phimedicare
        prem1 = premmedicare
        t1 = Tr-1    
        ssben = 0    # no social security

        [LGRID,KGRID1, KGRID2] = np.ndgrid(lgrid,kgrid,kgrid)
        LGRID = np.tile(LGRID,[1,1,1,2])
        KGRID1 = np.title(KGRID1,[1,1,1,2])
        KGRID2 = np.title(KGRID2,[1,1,1,2])    # NOTICE: KGRID2 is now K
        ZEROGRID = np.zeros(nl,nk,nk,2)
        ONESGRID = np.ones(nl,nk,nk,2)
        mincons = (1/(1+tau_c))*np.matmul(cfloor,ONESGRID)

        thresh = thresh1

        wage = A*(1-alpha)*(rho(t1))^(alpha)
        rate = 1 + A*alpha*(rho(t1))^(alpha-1) - d
        a2 = tau[:,t1]    # tax function parameter 2 ------[FED STATE]

        for j2 in range(1,nep):
            for j3 in range(1:nh):
                zstar = z[j2,j3,t1]
                med = Mnorm[j3,t1]
                surv1 = surv[j2,j3,t1]
                count = j3 + (j2-1)*nh;
                Pr = TrE[count,:,t1]
                phimedicaid = phi[j3,t1,1]
                phipriv = phi[j3,t1,3]                                               
                phigrid = [phimedicaid, 1, phipriv, phipriv]

                g1=1

                # Expected Value of Choosing Medicaid
                E1 = Pr[1]*V[:,1,1,t1+1,1,1]+Pr[2]*V[:,1,2,t1+1,1,1]

                # Expected Value of Choosing Uninsurance
                E2 = Pr[1]*V[:,1,1,t1+1,1,2]+Pr[2]*V[:,1,2,t1+1,1,2]

                Ehold = np.transpose([E1 E2])
                Ehold2 = np.tile(Ehold,[1,1,nk,nl])    # replicates "E" (nk,nl) times            
                E = np.random.permute(Ehold2,[4,2,3,1])

                income = np.matmul(rate,KGRID2) + np.matmul((wage*zstar),LGRID)
                tincome = np.matmul(max(rate-1,0),KGRID2) + np.matmul((wage*zstar),LGRID)

                tincome3 = np.squeeze(tincome[:,:,:,1])
                trfind = np.where(tincome3<=thresh)
                ntr = len(trfind);
                [I1, I2, I3] = ind2sub(size(tincome3),trfind);
                tr1 = sub2ind(size(tincome),I1,I2,I3,2*ones(ntr,1));

                trfind = np.where(tincome3>thresh);
                ntr = len(trfind);
                [I1, I2, I3] = ind2sub(size(tincome3),trfind);
                tr2 = sub2ind(size(tincome),I1,I2,I3,1*np.ones(ntr,1));


                for instype in range(1,4):
                    phi1 = phigrid[instype]
                    taxdeduc = max(0,phi1*phi2*med - .075*tincome)    # deduction
                    ytax1 = max(ZEROGRID,tincome - taxdeduc)
                    ytaxstargrid3 = np.matmul(a0,np.matmul(ytax1 - (ytax1.^(-a1) + (a2(2)),ONESGRID)).^(-1/a1))
                    ytaxstargrid2 = np,matmul(a0,((ytax1) - np.matmul(((ytax1).^(-a1) + (a2(1)),ONESGRID))).^(-1/a1))

                    # Placing the consumption floor
                    TRANSF = max(ZEROGRID, mincons - income + (prem1 + phi1*phi2*med - ssben - beq).*ONESGRID + ytaxstargrid2 + ytaxstargrid3);
                    cons1 = (1/(1+tau_c)).*(income - (prem1 + phi1*phi2*med - ssben - beq).*ONESGRID - KGRID1 - ytaxstargrid2 - ytaxstargrid3 + TRANSF);
                    tr3 = np.where(cons1<=0)

                    Vtilde = np.matmul(1/(1-sigma4)),((cons1.^chi3),(ONESGRID.^(1-chi3)))).^(1-sigma4) + np.matmul((surv1*b),E)
                    Vtilde[tr1] = -10e6   # Medicaid = not uninsured                
                    Vtilde[tr2] = -10e6   # Uninsured = not medicaid eligible    
                    Vtilde[tr3] = -10e6;    # don't consume negative
                    Vtilde1 = np.random.permute(Vtilde,[1,2,4,3])

                    V[:,j2,j3,t1,g1,instype] = max(max(max(Vtilde1)))
                    [Vhold1, Iins] = max(max(max(Vtilde1)),[],3)

                    Vtilde1 = np.random.permute(Vtilde,[1,4,2,3])
                    [Vhold2, Ik] = max(max(max(Vtilde1)),[],3)
    
                    Vtilde1 = permute(Vtilde,[4,2,1,3])
                    [Vhold3, Il] = max(max(max(Vtilde1)),[],3)

                    insopt[:,j2,j3,t1,g1,instype] = Iins
                    kopt[:,j2,j3,t1,g1,instype] = Ik  #(nl,nk,nk,4,T-Tr,nep,nh,ng)
                    labopt[:,j2,j3,t1,g1,instype] = lgrid[Il]

                    Il = np.squeeze(Il)               
                    Il = np.transpose(Il)                
                    Ik = np.squeeze(Ik)               
                    Ik = np.transpose(Ik)               
                    Iins = np.squeeze(Iins)                
                    Iins = np.transpose(Iins)
                    IND1 = sub2ind(size(cons1),Il,Ik,1:nk,Iins)

                    copt[:,j2,j3,t1,g1,instype] = cons1[IND1]                
                    transf[:,j2,j3,t1,g1,instype] = TRANSF[IND1]                
                    ytax[:,j2,j3,t1,g1,instype] = ytaxstargrid2[IND1]
                    ytaxst[:,j2,j3,t1,g1,instype] = ytaxstargrid3[IND1]
                    
        ZEROGRID = np.zeros(nl,nk,nk,4)
        ONESGRID = np.ones(nl,nk,nk,4)
        mincons = (1/(1+tau_c))*np.matmul(cfloor,ONESGRID)

        ssben = 0    # no social security

        for t1 i in range(Tr-2,1,-1):
            wage = A*(1-alpha)*(rho(t1))^(alpha)
            rate = 1 + A*alpha*(rho(t1))^(alpha-1) - d
            a2 = tau[:,t1]    # tax function parameter 2 ------[FED STATE]
            for j2 in range(1,nep):        
                for j3 in range(1,nh):
                    zstar = z[j2,j3,t1]
                    med = Mnorm[j3,t1]
                    surv1 = surv[j2,j3,t1]
                    count = j3 + (j2-1)*nh
                    Pr = TrE[count,:,t1]
                    phimedicaid = phi[j3,t1,1]
                    phipriv = phi[j3,t1,3]            
                    EMpriv = TrEprime[j3,1,t1]*(1-phi[1,t1+1,3])*Mnorm[1,t1+1] + TrEprime[j3,2,t1]*(1-phi[1,t1+1,3])*Mnorm[2,t1+1]            
                    prempriv = (1/rate)*gamma*EMpriv + ngprem
                    phigrid = [phimedicaid, 1, phipriv, phipriv]
                    premgrid1 = [0 0 prempriv (prempriv-ngprem)]

                    [LGRID, KGRID1, KGRID2, PREMGRID] = np.ndgrid(lgrid,kgrid,kgrid,premgrid1);    # NOTICE: KGRID2 is now K

                    g1=1

                    # Expected Value of Choosing Medicaid
                    EV11 = Opr[g1,1,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,1,1]+Pr[2]*V[:,1,2,t1 - (Ttail -1] + 1,1,1]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,1,1]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,1,1] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,1,1]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,1,1]+Pr[7]*V[:,4,1,t1 - (Ttail -1) + 1,1,1]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,1,1])
                    EV21 = Opr[g1,2,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,2,1]+Pr[2]*V[:,1,2,t1 - (Ttail -1] + 1,2,1]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,2,1]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,2,1] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,2,1]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,2,1]+Pr[7]*V[:,4,1,t1 - (Ttail -1) + 1,2,1]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,2,1])
                    E1 = EV11+EV21

                    # Expected Value of Choosing Uninsurance
                    EV12 = Opr[g1,1,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,1,1]+Pr[2]*V[:,1,2,t1 - (Ttail -1) + 1,1,1]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,1,2]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,1,2] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,1,2]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,1,2)+Pr[7)*V[:,4,1,t1 - (Ttail -1) + 1,1,2]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,1,2])
                    EV22 = Opr[g1,2,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,2,2)+Pr[2]*V[:,1,2,t1 - (Ttail -1) + 1,2,2]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,2,2]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,2,2] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,2,2]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,2,2]+Pr[7]*V[:,4,1,t1 - (Ttail -1) + 1,2,2]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,2,2])
                    E2 = EV12+EV22

                    # Expected Value of Choosing Priv Insurance
                    EV13 = Opr[g1,1,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,1,3]+Pr[2]*V[:,1,2,t1 - (Ttail -1) + 1,1,3]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,1,3]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,1,3] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,1,3]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,1,3]+Pr[7]*V[:,4,1,t1 - (Ttail -1) + 1,1,3]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,1,3])
                    EV23 = Opr[g1,2,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,2,3]+Pr[2]*V[:,1,2,t1 - (Ttail -1) + 1,2,3]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,2,3]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,2,3] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,2,3]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,2,3]+Pr[7)*V[:,4,1,t1 - (Ttail -1) + 1,2,3]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,2,3])
                    E3 = EV13+EV23

                    # Expected Value of Choosing Group Insurance
                    E4 = np.matmul(-10e6,np.ones(nk,1))
                    Ehold = np.transpose([E1 E2 E3 E4])
                    Ehold2 = np.tile(Ehold,[1,1,nk,nl])    # replicates "E" (nk,nl) times            
                    E = np.random.permute(Ehold2,[4,2,3,1])

                    income = np.matmul(rate,KGRID2) + np.matmul((wage*zstar),LGRID)
                    tincome = np.matmul(max(rate-1,0),KGRID2) + np.matmul((wage*zstar),LGRID)

                    tincome3 = np.squeeze(tincome[:,:,:,1])
                    trfind = np.where(tincome3<=thresh)
                    ntr = len(trfind)
                    [I1, I2, I3] = ind2sub(size(tincome3),trfind)
                    tr1 = sub2ind(size(tincome),I1,I2,I3,2*np.ones(ntr,1))

                    trfind = np.where(tincome3>thresh)
                    ntr = len(trfind)
                    [I1, I2, I3] = ind2sub(size(tincome3),trfind)
                    tr2 = sub2ind(size(tincome),I1,I2,I3,1*np.ones(ntr,1))

                    for instype in range(1,4):

                    phi1 = phigrid[instype]
                    taxdeduc = max(0,phi1*phi2*med - .075*tincome)    # deduction
                    ytax1 = max(ZEROGRID,tincome - taxdeduc)
                    ytaxstargrid3 = np.matmul(a0,np.matmul(ytax1 - (ytax1.^(-a1) + (a2(2)),ONESGRID)).^(-1/a1))
                    ytaxstargrid2 = np,matmul(a0,((ytax1) - np.matmul(((ytax1).^(-a1) + (a2(1)),ONESGRID))).^(-1/a1))

                    # Placing the consumption floor
                    TRANSF = max(ZEROGRID, mincons - income + (prem1 + phi1*phi2*med - ssben - beq).*ONESGRID + ytaxstargrid2 + ytaxstargrid3);
                    cons1 = (1/(1+tau_c)).*(income - (prem1 + phi1*phi2*med - ssben - beq).*ONESGRID - KGRID1 - ytaxstargrid2 - ytaxstargrid3 + TRANSF);
                    tr3 = np.where(cons1<=0)


                    Vtilde = np.matmul(1/(1-sigma4)),((cons1.^chi3),(ONESGRID.^(1-chi3)))).^(1-sigma4) + np.matmul((surv1*b),E)
                    Vtilde[tr1] = -10e6    # Medicaid = not uninsured                
                    Vtilde[tr2] = -10e6    # Uninsured = not medicaid eligible 
                    Vtilde[tr3] = -10e6
                    Vtilde1 = np.random.permute(Vtilde,[1,2,4,3])

                    V[:,j2,j3,t1,g1,instype] = max(max(max(Vtilde1)))
                    [Vhold1, Iins] = max(max(max(Vtilde1)),[],3);

                    Vtilde1 = np.matmul.permute(Vtilde,[1,4,2,3])
                    [Vhold2, Ik] = max(max(max(Vtilde1)),[],3)

                    Vtilde1 = np.matmul.permute(Vtilde,[4,2,1,3])                
                    [Vhold3, Il] = max(max(max(Vtilde1)),[],3)

                    insopt[:,j2,j3,t1,g1,instype] = Iins
                    kop[:,j2,j3,t1,g1,instype] = Ik  
                    labopt[:,j2,j3,t1,g1,instype] = lgrid[Il]

                    Il = np.squeeze[Il]
                    Il = np.translate(Il)                
                    Ik = np.squeeze(Ik)               
                    Ik = np.translate(Ik)               
                    Iins = np.squeeze(Iins)                
                    Iins = np.translate(Iins)
                    IND1 = sub2ind(size(cons1),Il,Ik,1:nk,Iins);

                    copt[:,j2,j3,t1,g1,instype] = cons1[IND1]                
                    transf[:,j2,j3,t1,g1,instype] = TRANSF[IND1]                
                    ytax[:,j2,j3,t1,g1,instype] = ytaxstargrid2[IND1]
                    ytaxst[:,j2,j3,t1,g1,instype] = ytaxstargrid3[IND1]                    


                    g1=2

                    # Expected Value of Choosing Medicaid
                    EV11 = Opr[g1,1,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,1,1]+Pr[2]*V[:,1,2,t1 - (Ttail -1] + 1,1,1]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,1,1]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,1,1] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,1,1]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,1,1]+Pr[7]*V[:,4,1,t1 - (Ttail -1) + 1,1,1]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,1,1])
                    EV21 = Opr[g1,2,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,2,1]+Pr[2]*V[:,1,2,t1 - (Ttail -1] + 1,2,1]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,2,1]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,2,1] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,2,1]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,2,1]+Pr[7]*V[:,4,1,t1 - (Ttail -1) + 1,2,1]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,2,1])
                    E1 = EV11+EV21

                    # Expected Value of Choosing Uninsurance
                    EV12 = Opr[g1,1,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,1,1]+Pr[2]*V[:,1,2,t1 - (Ttail -1) + 1,1,1]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,1,2]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,1,2] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,1,2]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,1,2)+Pr[7)*V[:,4,1,t1 - (Ttail -1) + 1,1,2]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,1,2])
                    EV22 = Opr[g1,2,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,2,2)+Pr[2]*V[:,1,2,t1 - (Ttail -1) + 1,2,2]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,2,2]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,2,2] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,2,2]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,2,2]+Pr[7]*V[:,4,1,t1 - (Ttail -1) + 1,2,2]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,2,2])
                    E2 = EV12+EV22

                    # Expected Value of Choosing Priv Insurance
                    EV13 = Opr[g1,1,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,1,3]+Pr[2]*V[:,1,2,t1 - (Ttail -1) + 1,1,3]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,1,3]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,1,3] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,1,3]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,1,3]+Pr[7]*V[:,4,1,t1 - (Ttail -1) + 1,1,3]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,1,3])
                    EV23 = Opr[g1,2,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,2,3]+Pr[2]*V[:,1,2,t1 - (Ttail -1) + 1,2,3]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,2,3]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,2,3] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,2,3]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,2,3]+Pr[7)*V[:,4,1,t1 - (Ttail -1) + 1,2,3]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,2,3])
                    E3 = EV13+EV23

                    # Expected Value of Choosing Group Insurance
                    EV14 = Opr[g1,1,t1]*(Pr[1]*V[]:,1,1,t1 - (Ttail -1) + 1,1,4]+Pr[2]*V[:,1,2,t1 - (Ttail -1) + 1,1,4]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,1,4]+Pr(4)*V[:,2,2,t1 - (Ttail -1) + 1,1,4] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,1,4]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,1,4]+Pr[7]*V[:,4,1,t1 - (Ttail -1) + 1,1,4]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,1,4])
                    EV21 = Opr[g1,2,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,2,1]+Pr[2]*V[:,1,2,t1 - (Ttail -1] + 1,2,4]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,2,4]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,2,4] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,2,4]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,2,4]+Pr[7]*V[:,4,1,t1 - (Ttail -1) + 1,2,4]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,2,4])
                    E4 = EV14+EV24

                    #group in last period

                    Ehold = np.transpose([E1 E2 E3 E4])            
                    Ehold2 = np.tile(Ehold,[1,1,nk,nl])    # replicates "E" (nk,nl) times            
                    E = np.random.permute(Ehold2,[4,2,3,1])   # reorders: (nl,nk',nk,nins')


                    for instype in range(1,4):

                        phi1 = phigrid[instype]
                        taxdeduc = max(0,phi1*med - .075*tincome) + max(instype-3,0)*premgrid1(4);    % deduction, including the ESHI premium
                        ytax1 = max(ZEROGRID,tincome - taxdeduc);                                        
                        ytaxstargrid3 = np.matmul(a0,np.matmul(ytax1 - (ytax1.^(-a1) + (a2(2)),ONESGRID)).^(-1/a1))
                        ytaxstargrid2 = np,matmul(a0,((ytax1) - np.matmul(((ytax1).^(-a1) + (a2(1)),ONESGRID))).^(-1/a1))

                
                        # Placing the consumption floor
                        TRANSF = np.matmul(max(ZEROGRID, mincons - income + (prem1 + phi1*phi2*med - ssben - beq),ONESGRID) + ytaxstargrid2 + ytaxstargrid3)
                        cons1 = (1/(1+tau_c))*(income - np.matmul((prem1 + phi1*phi2*med - ssben - beq),ONESGRID) - KGRID1 - ytaxstargrid2 -ytaxstargrid3+ TRANSF)
                        tr3 = np.where(cons1<=0)

                        Vtilde = np.matmul(1/(1-sigma4)),((cons1.^chi3),(ONESGRID.^(1-chi3)))).^(1-sigma4) + np.matmul((surv1*b),E)

                        Vtilde(tr1) = -10e6    # Medicaid = not uninsured                
                        Vtilde(tr2) = -10e6   # Uninsured = no medicaid
                        Vtilde(tr3) = -10e6

                        Vtilde1 = np.random.permute(Vtilde,[1,2,4,3])

                        V[:,j2,j3,t1,g1,instype] = max(max(max(Vtilde1)))
                        [Vhold1, Iins] = max(max(max(Vtilde1)),[],3)

                        Vtilde1 = np.random.permute(Vtilde,[1,4,2,3])
                        [Vhold2, Ik] = max(max(max(Vtilde1)),[],3);

                        Vtilde1 = np.random.permute(Vtilde,[4,2,1,3])                
                        [Vhold3, Il] = max(max(max(Vtilde1)),[],3)

                        insopt[:,j2,j3,t1,g1,instype] = Iins
                        kopt[:,j2,j3,t1,g1,instype] = Ik  #(nl,nk,nk,4,T-Tr,nep,nh,ng)
                        labopt[:,j2,j3,t1,g1,instype] = lgrid[Il];

                        Il = np.squeeze(Il)              
                        Il = np.transpose(Il)                
                        Ik = np.squeeze(Ik)             
                        Ik = np.transpose(Ik)               
                        Iins = np.squeeze(Iins)                
                        Iins = np.transpose(Iins)
                        IND1 = sub2ind(size(cons1),Il,Ik,1:nk,Iins);

                        copt[:,j2,j3,t1,g1,instype] = cons1[IND1]
                        transf[:,j2,j3,t1,g1,instype] = TRANSF[IND1]                
                        ytax[:,j2,j3,t1,g1,instype] = ytaxstargrid2[IND1]
                        ytaxst[:,j2,j3,t1,g1,instype] = ytaxstargrid3[IND1]                        

    if agetrans<=Tr-2:
        ZEROGRID = np.zeros(nl,nk,nk,4);
        ONESGRID = np.ones(nl,nk,nk,4);
        mincons = (1/(1+tau_c))*np.matmul(cfloor,ONESGRID)

        ssben = 0    # no social security
        thresh = thresh1


        for t1 in range(agetrans,1,-1): 
            wage = A*(1-alpha)*(rho(t1))^(alpha);
            rate = 1 + A*alpha*(rho(t1))^(alpha-1) - d;
            a2 = tau[:,t1]    # tax function parameter 2 ------[FED STATE]
            badoffer = isequal[t1,Tr-2]
            for j2 in range(1,nep):        
                for j3 in range(1:nh):
                    zstar = z[j2,j3,t1]
                    med = Mnorm[j3,t1]
                    surv1 = surv[j2,j3,t1]
                    count = j3 + (j2-1)*nh;
                    Pr = TrE[count,:,t1]
                    phimedicaid = phi[j3,t1,1]
                    phipriv = phi[j3,t1,3]           
                    EMpriv = TrEprime[j3,1,t1]*(1-phi[1,t1+1,3])*Mnorm[1,t1+1] + TrEprime[j3,2,t1]*(1-phi[1,t1+1,3])*Mnorm[2,t1+1]            
                    prempriv = (1/rate)*gamma*EMpriv + ngprem;                                    
                    phigrid = [phimedicaid, 1, phipriv, phipriv];            
                    premgrid1 = [0 0 prempriv (prempriv-ngprem)];

                    [LGRID, KGRID1, KGRID2, PREMGRID] = np.ndgrid(lgrid,kgrid,kgrid,premgrid1);    # NOTICE: KGRID2 is now K

                    g1=1

            
                    # Expected Value of Choosing Medicaid
                    EV11 = Opr[g1,1,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,1,1]+Pr[2]*V[:,1,2,t1 - (Ttail -1] + 1,1,1]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,1,1]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,1,1] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,1,1]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,1,1]+Pr[7]*V[:,4,1,t1 - (Ttail -1) + 1,1,1]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,1,1])
                    EV21 = Opr[g1,2,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,2,1]+Pr[2]*V[:,1,2,t1 - (Ttail -1] + 1,2,1]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,2,1]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,2,1] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,2,1]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,2,1]+Pr[7]*V[:,4,1,t1 - (Ttail -1) + 1,2,1]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,2,1])
                    E1 = EV11+EV21

                    # Expected Value of Choosing Uninsurance
                    EV12 = Opr[g1,1,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,1,1]+Pr[2]*V[:,1,2,t1 - (Ttail -1) + 1,1,1]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,1,2]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,1,2] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,1,2]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,1,2)+Pr[7)*V[:,4,1,t1 - (Ttail -1) + 1,1,2]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,1,2])
                    EV22 = Opr[g1,2,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,2,2)+Pr[2]*V[:,1,2,t1 - (Ttail -1) + 1,2,2]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,2,2]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,2,2] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,2,2]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,2,2]+Pr[7]*V[:,4,1,t1 - (Ttail -1) + 1,2,2]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,2,2])
                    E2 = EV12+EV22

                    # Expected Value of Choosing Priv Insurance
                    EV13 = Opr[g1,1,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,1,3]+Pr[2]*V[:,1,2,t1 - (Ttail -1) + 1,1,3]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,1,3]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,1,3] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,1,3]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,1,3]+Pr[7]*V[:,4,1,t1 - (Ttail -1) + 1,1,3]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,1,3])
                    EV23 = Opr[g1,2,t1]*(Pr[1]*V[:,1,1,t1 - (Ttail -1) + 1,2,3]+Pr[2]*V[:,1,2,t1 - (Ttail -1) + 1,2,3]+Pr[3]*V[:,2,1,t1 - (Ttail -1) + 1,2,3]+Pr[4]*V[:,2,2,t1 - (Ttail -1) + 1,2,3] +Pr[5]*V[:,3,1,t1 - (Ttail -1) + 1,2,3]+Pr[6]*V[:,3,2,t1 - (Ttail -1) + 1,2,3]+Pr[7)*V[:,4,1,t1 - (Ttail -1) + 1,2,3]+Pr[8]*V[:,4,2,t1 - (Ttail -1) + 1,2,3])
                    E3 = EV13+EV23

                    E4 = -10e6.*ones(nk,1);
                    Ehold = [E1 E2 E3 E4]';
                    Ehold2 = repmat(Ehold,[1,1,nk,nl]);    % replicates "E" (nk,nl) times            
                    E = permute(Ehold2,[4,2,3,1]);


                    income = rate.*KGRID2 + (wage*zstar).*LGRID;
                    tincome = max(rate-1,0).*KGRID2 + (wage*zstar).*LGRID;

                    tincome3 = squeeze(tincome(:,:,:,1));
                    trfind = find(tincome3<=thresh);
                    ntr = length(trfind);
                    [I1, I2, I3] = ind2sub(size(tincome3),trfind);
                    tr1 = sub2ind(size(tincome),I1,I2,I3,2*ones(ntr,1));

                    trfind = find(tincome3>thresh);
                    ntr = length(trfind);
                    [I1, I2, I3] = ind2sub(size(tincome3),trfind);
                    tr2 = sub2ind(size(tincome),I1,I2,I3,1*ones(ntr,1));


                    for instype in range(1:4):

                        phi1 = phigrid[instype];
                        taxdeduc = max(0,phi1*med - .075*tincome);    % deduction
                        ytax1 = max(ZEROGRID,tincome - taxdeduc);                                        
                        ytaxstargrid3 = np.matmul(a0,np.matmul(ytax1 - (ytax1.^(-a1) + (a2(2)),ONESGRID)).^(-1/a1))
                        ytaxstargrid2 = np,matmul(a0,((ytax1) - np.matmul(((ytax1).^(-a1) + (a2(1)),ONESGRID))).^(-1/a1))


                        # Placing the consumption floor
                        TRANSF = np.matmul(max(ZEROGRID, mincons - income + (prem1 + phi1*phi2*med - ssben - beq),ONESGRID) + ytaxstargrid2 + ytaxstargrid3)
                        cons1 = (1/(1+tau_c))*(income - np.matmul((prem1 + phi1*phi2*med - ssben - beq),ONESGRID) - KGRID1 - ytaxstargrid2 -ytaxstargrid3+ TRANSF)
                        tr3 = np.where(cons1<=0)


                        Vtilde = np.matmul(1/(1-sigma4)),((cons1.^chi3),(ONESGRID.^(1-chi3)))).^(1-sigma4) + np.matmul((surv1*b),E)
                        Vtilde[tr1] = -10e6    # Medicaid = not uninsured                
                        Vtilde[tr2] = -10e6   # Uninsured = not medicaid eligible 
                        Vtilde[tr3] = -10e6
                        Vtilde1 = np.random.permute(Vtilde,[1,2,4,3])

                        V[:,j2,j3,t1,g1,instype] = max(max(max(Vtilde1)))
                        [Vhold1, Iins] = max(max(max(Vtilde1)),[],3)

                        Vtilde1 = np.random.permute(Vtilde,[1,4,2,3])               
                        [Vhold2, Ik] = max(max(max(Vtilde1)),[],3)

                        Vtilde1 = np.random.permute(Vtilde,[4,2,1,3])                
                        [Vhold3, Il] = max(max(max(Vtilde1)),[],3);

                        insopt[:,j2,j3,t1,g1,instype] = Iins
                        kopt[:,j2,j3,t1,g1,instype] = Ik  #(nl,nk,nk,4,T-Tr,nep,nh,ng)
                        labopt[:,j2,j3,t1,g1,instype] = lgrid[Il]

                        Il = np.squeeze(Il)               
                        Il = np.transpose(Il)                
                        Ik = np.squeeze(Ik)           
                        Ik = np.transpose(Ik)               
                        Iins = np.squeeze(Iins)                
                        Iins = np.transpose(Iins)
                        IND1 = sub2ind(size(cons1),Il,Ik,1:nk,Iins);

                        copt[:,j2,j3,t1,g1,instype] = cons1[IND1]
                        transf[:,j2,j3,t1,g1,instype] = TRANSF[IND1]                
                        ytax[:,j2,j3,t1,g1,instype] = ytaxstargrid2[IND1]
                        ytaxst[:,j2,j3,t1,g1,instype] = ytaxstargrid3[IND1]

    return V,labopt, kopt, copt, ytax, transf, insopt, ytaxst