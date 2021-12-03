# Bibliotecas 
import numpy as np 
import scipy as sc 
from agent
from scipy.io import loadmat 

def distss_long(rho,tau,thresh):
    insdata4 = loadmat('/home/alexandre/Documents/Mestrado/Artigos de dissertação/Replication Files/Benchmark Economy/insdata4.mat')
    wage_med_data = loadmat('/home/alexandre/Documents/Mestrado/Artigos de dissertação/Replication Files/Benchmark Economy/wage_med_data1.mat')

    mu3 = np.matmul(np.squeeze(1-surv[1,1,:]),mu2)

    # Parametros da distribuição 
    # Com 20 anos de idades, [medicaid, uninsured, priv insured, group insured]
    insdit = [0.152, 0.449, 0.015, 0.384]
    healthdist = [0.9, 0.1]
    proddist = [0.25,0.25,0.25,0.25]
    groupdist = [0.2, 0.8]

    [V,labopt, kopt, copt, ytax, transf, insopt, ytaxSSM] = agent(np.matmul(rho,np.ones(1,T)),np.transpose(tau)*ones(1,T),thresh)
    dist1 = np.zeros(nep*nh*nk,T,4)   
    dist2 = np.zeros(nep*nh*nk,T,4)
    
    # Distribuição Inicial 
    k0 = 0 
    loc1 = np.find(kgrid>=k0,1,'first')
    loc1 = min(loc,loc1)
    if loc1 !=1:
        w1 = (kgrid[loc1]-k0)/(kgrid[loc1]-kgrid[loc1-1])
        for i1 in range(1,nep):
            for i2 in range(1,nh):
                dist1[(i1-1)*nh*nk + (i2-1)*nk  + loc1-1,1,1] = (1-w1)*healthdist[i2]*proddist[i1]*insdist[1]*groupdist[1]
                dist1[(i1-1)*nh*nk + (i2-1)*nk  + loc1-1,1,2] = (1-w1)*healthdist[i2]*proddist[i1]*insdist[2]*groupdist[1]
                dist1[(i1-1)*nh*nk + (i2-1)*nk  + loc1-1,1,3] = (1-w1)*healthdist[i2]*proddist[i1]*insdist[3]*groupdist[1]
                dist1[(i1-1)*nh*nk + (i2-1)*nk  + loc1-1,1,4] = (1-w1)*healthdist[i2]*proddist[i1]*insdist[4]*groupdist[1]
                dist1[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,1] = w1*healthdist[i2]*proddist[i1]*insdist[1]*groupdist[1]
                dist1[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,2] = w1*healthdist[i2]*proddist[i1]*insdist[2]*groupdist[1]
                dist1[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,3] = w1*healthdist[i2]*proddist[i1]*insdist[3]*groupdist[1]
                dist1[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,4] = w1*healthdist[i2]*proddist[i1]*insdist[4]*groupdist[1]
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1-1,1,1] = (1-w1)*healthdist[i2]*proddist[i1]*insdist[1]*groupdist[2]
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1-1,1,2] = (1-w1)*healthdist[i2]*proddist[i1]*insdist[2]*groupdist[2]
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1-1,1,3] = (1-w1)*healthdist[i2]*proddist[i1]*insdist[3]*groupdist[2]
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1-1,1,4] = (1-w1)*healthdist[i2]*proddist[i1]*insdist[4]*groupdist[2]
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,1] = w1*healthdist[i2]*proddist[i1]*insdist[1]*groupdist[2]
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,2] = w1*healthdist[i2]*proddist[i1]*insdist[2]*groupdist[2]
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,3] = w1*healthdist[i2]*proddist[i1]*insdist[3]*groupdist[2]
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,4] = w1*healthdist[i2]*proddist[i1]*insdist[4]*groupdist[2]

    elif  loc1==1:
        for i1 in range(1,nep):
            for i2 in range(1,nh):
                dist1[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,1] = healthdist[i2]*proddist[i1]*insdist[1]*groupdist[1]
                dist1[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,2] = healthdist[i2]*proddist[i1]*insdist[1]*groupdist[1]
                dist1[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,3] = healthdist[i2]*proddist[i1]*insdist[1]*groupdist[1]
                dist1[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,4] = healthdist[i2]*proddist[i1]*insdist[1]*groupdist[1]
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,1] = healthdist[i2]*proddist[i1]*insdist[2]*groupdist[2]
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,2] = healthdist[i2]*proddist[i1]*insdist[2]*groupdist[2]
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,3] = healthdist[i2]*proddist[i1]*insdist[2]*groupdist[2]
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,4] = healthdist[i2]*proddist[i1]*insdist[2]*groupdist[2]

    for t1 in range(1,Tr-1):
        node = 0 
        count = 0
        for j2 in range(1,nep):
            for j3 in range(1,nh):
                count = count +1 
                Pr = TrE[count,:,t1]
                for i1 in range(1,nk):
                    node = node+1
                    loc11 = kopt[i1,j2,j3,t1,1,1]
                    loc21 = kopt[i1,j2,j3,t1,1,2]
                    loc31 = kopt[i1,j2,j3,t1,1,3]
                    loc41 = kopt[i1,j2,j3,t1,1,4]
                    loc12 = kopt[i1,j2,j3,t1,2,1]
                    loc22 = kopt[i1,j2,j3,t1,2,2]
                    loc32 = kopt[i1,j2,j3,t1,2,3]
                    loc42 = kopt[i1,j2,j3,t1,2,4]
                    ins11 = insopt[i1,j2,j3,t1,1,1]
                    ins21 = insopt[i1,j2,j3,t1,1,2]
                    ins31 = insopt[i1,j2,j3,t1,1,3]
                    ins41 = insopt[i1,j2,j3,t1,1,4]
                    ins12 = insopt[i1,j2,j3,t1,2,1]
                    ins22 = insopt[i1,j2,j3,t1,2,2]
                    ins32 = insopt[i1,j2,j3,t1,2,3]
                    ins42 = insopt[i1,j2,j3,t1,2,4]
                    count2 = 0 
                    for j4 in range(1,nep):
                        for j5 in range(1,nh):
                            count2 = count2 +1                             
                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins11] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins11] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1,1]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins11] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins11] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1,1]

                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins21] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins21] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1,2]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins21] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins21] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1,2]

                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins31] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins31] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1,3]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins31] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins31] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1,3]

                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins41] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins41] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1,4]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins41] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins41] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1,4]
                            
                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins12] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins12] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1,1]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins12] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins12] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1,1]

                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins22] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins22] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1,2]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins22] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins22] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1,2]

                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins32] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins32] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1,3]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins32] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins32] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1,3]

                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins42] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins42] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1,4]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins42] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins42] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1,4]

    for t1 in range(Tr,T-1):
        node =0 
        count =0 
        j2 = 1
        for j3 in range(1,nh):
            count = count +1 
            Pr = TrE[count,:,t1]
            for i1 in range(1,nk):
                node = node +1 
                loc11 = kopt[i1,j2,j3,t1,1,1]
                loc21 = kopt[i1,j2,j3,t1,1,2]
                ins11 = insopt[i1,j2,j3,t1,1,1]
                ins21 = insopt[i1,j2,j3,t1,1,2]
                count2 =0
                j4 =1 
                for j5 in range(1,nh):
                    count2 = count2+1
                    dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins11] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins11] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1,1]
                    dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins11] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins11] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1,1]

                    dist1[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1+1,ins21] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1+1,ins21] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1,2]
                    dist2[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1+1,ins21] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1+1,ins21] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1,2]
   
    Kalive = np.zeros(1,T)
    Kdead = np.zeros(1,T)

    # Resolvendo o Capital Agregado 
    for k1 in range(1,T):
        node1 = 1 
        for k2 in range(1,nep):
            for k3 in range(1,nh):
                for k4 in range(1,nl):
                    Kalive[k1] = Kalive[k1] + mu2[k1]*kgrid[k4]*sum(np.squeeze(dist1[node1,k1,:]+dist2[node1,k1,:]))
                    Kdead[k1] = Kdead[k1] + mu2[k1]*kgrid[k4]*sum(np.squeeze(dist1[node1,k1,:]+dist2[node1,k1,:]))
                    node1 = node1+1 

    # Fixando a distribuição 
    for k1 in range(1,T):
        node1 = 1 
        for k2 in range(1,nep):
            for k3 in range(1,nh):
                for k4 in range(1,nk):
                    dist1[node1,k1,:] = mu2[k1]*dist1[node1,k1,:]
                    dist2[node1,k1,:] = mu2[k1]*dist2[node1,k1,:]
                    node1 = node1+1
    
    dist1ss = dist1
    dist2ss = dist2
    # save ssdist dist1ss dist2ss

    # Resolvendo a agregação de horas de trabalho e eficiencia 
    Lab = np.zeros(1,Tr-1)
    ELab = np.zeros(1,Tr-1)

    for k1 in range(1,Tr-1):
        node1 = 1 
        for k2 in range(1,nep):
            for k3 in range(1,nh):
                for k4 in range(1,nk):
                    for instype in range(1,4):
                        Lab[k1] = Lab[k1] + labopt[k4,k2,k3,k1,1,instype]*dist1[node1,k1,instype]+labopt[k4,k2,k3,k1,2,instype]*dist2[node1,k1,instype]
                        ELab[k1] = ELab[k1] + z[k2,k3,k1]*labopt[k4,k2,k3,k1,1,instype]*dist1[node1,k1,instype]+z[k2,k3,k1]*labopt[k4,k2,k3,k1,2,instype]*dist2[node1,k1,instype]
                    node1 = node1+1

    # Solving efficient hours of individuals that are offered group health
    # insurance
    # Solving income and consumption tax revenue, welfare, transfers
    Ytax = np.zeros[1,T]
    YtaxSSM = np.zeros[1,T]
    Ctax = np.zeros[1,T]
    welfare = np.zeros[1,T]
    transfers = np.zeros[1,T]

    for k1 in range(1,T):
        node1 - 1
        for k2 in range(1,nep):
            for k3 in range(1,nh):
                for k4 in range(1,nk):
                    for instype in range(1,4):
                        Ytax[k1] = Ytax[k1] + ytax[k4,k2,k3,k1,1,instype]*dist1[node1,k1,instype]+ ytax[k4,k2,k3,k1,2,instype]*dist2[node1,k1,instype]
                        YtaxSSM[k1] = YtaxSSM[k1] + ytaxSSM[k4,k2,k3,k1,1,instype]*dist1[node1,k1,instype]+ ytaxSSM[k4,k2,k3,k1,2,instype]*dist2[node1,k1,instype]
                        Ctax[k1] = Ctax[k1] + tau_c*copt[k4,k2,k3,k1,1,instype]*dist1[node1,k1,instype]+tau_c*copt[k4,k2,k3,k1,2,instype]*dist2[node1,k1,instype]
                        welfare[k1] = welfare[k1] + V[k4,k2,k3,k1,1,instype]*dist1[node1,k1,instype]+V[k4,k2,k3,k1,2,instype]*dist2[node1,k1,instype]
                        transfers[k1] = transfers[k1] + transf[k4,k2,k3,k1,1,instype]*dist1[node1,k1,instype]+transf[k4,k2,k3,k1,2,instype]*dist2[node1,k1,instype]
                node1 = node1 + 1

    welfare2 = 0 
    k1 =1
    node1 = 1 
    for k2 in range(1,nep):
        for k3 in range(1,nh):
            for k4 in range(1,nk):
                for g1 in range(1,ng):
                    for instype in range(1,4):
                        welfare2 = welfare2 + V[k4,k2,k3,k1,1,instype]*dist1[node1,k1,instype]+V[k4,k2,k3,k1,2,instype]*dist2[node1,k1,instype]
                node1 = node1 +1 

    # Resolvendo os gastos medicos por tipos de seguros 
    # Medicaid, non-elderly
    mcaid = np.zeros(1,T)
    for t1 in range(1,Tr-1):
        for j2 in range(1,nep):
            for j3 in range(1,nh):
                mcaid[t1] = mcaid[t1]+[1-phi[j3,t1,1]]*Mnorm[j3,t1]*sum[dist1((j2-1)*nh*nk + (j3-1)*nk + range(1,(j2-1))*nh*nk + (j3-1)*nk +nk,t1,1]+dist2[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,1)]
    
    # Medicaid, Idosos e medicare 
    for t1 in range(Tr,T):
        for j2 in range(1,nep):
            for j3 in range(1,nh):
                mcaid[t1] = mcaid[t1]+(1-phi[j3,t1,1])*(phimedicare)*Mnorm[j3,t1]*sum(dist1[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,1]+dist2[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,1)]
                mcare[t1] = mcare[t1]+phimedicare*Mnorm[j3,t1]*sum(sum(np.squeeze[dist1((j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,:]+dist2[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,:)))];
    
    # Total de gastos medicos 
    medexp=np.zeros(1,T)
    for t1 in range(1,T):
        for j2 in range(1,nep):
            for j3 in range(1,nh):
                medexp[t1] = medexp[t1]+Mnorm[j3,t1]*sum(dist1[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,1]+dist2[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,1)]
                medexp[t1] = medexp[t1]+Mnorm[j3,t1]*sum(dist1[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,2]+dist2[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,2)]
                medexp[t1] = medexp[t1]+Mnorm[j3,t1]*sum(dist1[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,3]+dist2[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,3)]
                medexp[t1] = medexp[t1]+Mnorm[j3,t1]*sum(dist1[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,4]+dist2[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,4)]
 
    MED = sum(medexp);
    KPR = sum(Kalive);
    BEQ = sum(Kdead);
    HRS = sum(Lab)/sum(mu2(1:Tr-1));
    ELAB = sum(ELab);
    MCARE = sum(mcare);
    MCAID = sum(mcaid);
    YTAX = sum(Ytax);
    YTAXSSM = sum(YtaxSSM);
    CTAX = sum(Ctax);
    TRANSF = sum(transfers);
    WELF = sum(welfare);
    WELF2=welfare2;


















    return dist1, dist2, MED, KPR, HRS, ELAB, MCARE, MCAID, YTAX, CTAX, TRANSF, WELF, BEQ, YTAXSSM, WELF2