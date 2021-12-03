import numpy as np 
import scipy as sc 
from agent  
from scipy.io import loadmat

def dist_short(rho,tau,thresh):
    # Importando dados externos 
    insdata4 = loadmat('/home/alexandre/Documents/Mestrado/Artigos de dissertação/Replication Files/Benchmark Economy/insdata4.mat')
    wage_med_data = loadmat('/home/alexandre/Documents/Mestrado/Artigos de dissertação/Replication Files/Benchmark Economy/wage_med_data1.mat')

    #Crescimento da população
    pgr =  0.012 
    
    # Corbaes's notes 
    mu2 = np.zeros(1,T)
    mu2[2] = 1 
    for i1 in range(2,T):
        mu2[i1] = (surv(1,1,i1)/(1+prg))}*mu2[i1-1]    
    mu2 = np.matmul((1/sum(mu2)),mu2)
    mu3 = np.matmul(np.squeeze[1-surv(1,1,:)],mu2)
    
    # Numeros referentes para o grupo ofertante 
    # Parametros das distribuições 
    insdist = [.152, .449, .015, .384] # Com 20 anos de [medicaid, uninsured, priv insured, group insured]
    healthdist = [.9, .1] # Distrbuição inicial da saúde tipo 
    proddist = [.25,.25,.25,.25] # Choque de produtividade 
    groupdist = [.2 .8] # Distribuição inicial dos grupos ofertantes 
    
    [V,labopt, kopt, copt, ytax, transf, insopt, ytaxSSM] = agent[rho.*ones(1,T),np.transpose(tau)*np.ones(1,T),thresh]

    dist1 = np.zeros(nep*nh*nk,T,4)
    dist2 = np.zeros(nep*nh*nk,T,4)

    # Distribuição Inicial 
    k0 = 0 
    loc1 = np.where(kgrid>=k0,1,'first')
    loc1 = min(nk,loc1)
    if loc1 !=0:
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
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1-1,1,1] = (1-w1)*healthdist[i2]*proddist[i1]*insdist[1]*groupdist[1]
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1-1,1,2] = (1-w1)*healthdist[i2]*proddist[i1]*insdist[2]*groupdist[1]
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1-1,1,3] = (1-w1)*healthdist[i2]*proddist[i1]*insdist[3]*groupdist[1]
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1-1,1,4] = (1-w1)*healthdist[i2]*proddist[i1]*insdist[4]*groupdist[1]
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,1] = w1*healthdist[i2]*proddist[i1]*insdist[1]*groupdist[1]
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,2] = w1*healthdist[i2]*proddist[i1]*insdist[2]*groupdist[1]
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,3] = w1*healthdist[i2]*proddist[i1]*insdist[3]*groupdist[1]
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,4] = w1*healthdist[i2]*proddist[i1]*insdist[4]*groupdist[1]

    elif loc1==1:
        for i1 in range(1,nep):
            for i2 in range(1,nh):
                dist1[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,1] = healthdist[i2]*proddist[i1]*insdist[1]*groupdist[1]                    
                dist1[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,2] = healthdist[i2]*proddist[i1]*insdist[1]*groupdist[2]                    
                dist1[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,3] = healthdist[i2]*proddist[i1]*insdist[1]*groupdist[3]                    
                dist1[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,4] = healthdist[i2]*proddist[i1]*insdist[1]*groupdist[4]                    
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,1] = healthdist[i2]*proddist[i1]*insdist[1]*groupdist[1]                    
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,2] = healthdist[i2]*proddist[i1]*insdist[1]*groupdist[2]                    
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,3] = healthdist[i2]*proddist[i1]*insdist[1]*groupdist[3]                    
                dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,4] = healthdist[i2]*proddist[i1]*insdist[1]*groupdist[4]                    

    for t1 in range(1,Tr-1):
        node = 0 
        count = 0
        for j2 in range(1,nep):
            for j3 in range(1,nh):
                count = count + 1
                Pr = TrE[count,:,t1]
                for i1 in range(1,nk):
                    node = node + 1
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
                            count2 = count2+1 
                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins11] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins11] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1,1]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins11] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins11] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1,1]

                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1+1,ins21] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1+1,ins21] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1,2]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1+1,ins21] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1+1,ins21] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1,2]

                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc31,t1+1,ins31] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc31,t1+1,ins31] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1,3]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc31,t1+1,ins31) = dist2((j4-1)*nh*nk + (j5-1)*nk + loc31,t1+1,ins31] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1,3]

                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc31,t1+1,ins41] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc41,t1+1,ins41] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1,4]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc31,t1+1,ins41] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc41,t1+1,ins41] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1,4]


                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc12,t1+1,ins12] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc12,t1+1,ins12] + Opr[2,1,t1]*Pr[count2]*dist2[node,t1,1]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc12,t1+1,ins12] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc12,t1+1,ins12] + Opr[2,2,t1]*Pr[count2]*dist2[node,t1,1]

                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc22,t1+1,ins22] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc22,t1+1,ins22] + Opr[2,1,t1]*Pr[count2]*dist2[node,t1,2]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc22,t1+1,ins22] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc22,t1+1,ins22] + Opr[2,2,t1]*Pr[count2]*dist2[node,t1,2]

                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc32,t1+1,ins32] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc32,t1+1,ins32] + Opr[2,1,t1]*Pr[count2]*dist2[node,t1,3]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc32,t1+1,ins32] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc32,t1+1,ins32] + Opr[2,2,t1]*Pr[count2]*dist2[node,t1,3]

                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc42,t1+1,ins42] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc42,t1+1,ins42] + Opr[2,1,t1]*Pr[count2]*dist2[node,t1,4]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc42,t1+1,ins42] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc42,t1+1,ins42] + Opr[2,2,t1]*Pr[count2]*dist2[node,t1,4]
                           
    for t1 in range(Tr,T-1):
        node = 0 
        count = 0 
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
                count2=0
                j4 = 1 
                for j5 in range(1,nh):
                    count2 = count2 +!
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
                for k4 in range(1,nk):
                    Kalive[k1] = Kalive[k1] + mu2[k1]*kgrid[k4]*sum(np.squeeze(dist1[node1,k1,:]+dist2[node1,k1,:]))
                    Kdead[k1] = Kalive[k1] + mu2[k1]*kgrid[k4]*sum(np.squeeze(dist1[node1,k1,:]+dist2[node1,k1,:]))
                    node1 = node1+1 
    
    # Fixando a distribuição 
    for k1 in range(1,T):
        node1 = 1
        for k2 in range(1,nep):
            for k3 in range(1,nk):
                dist1[node1,k1,:] = mu2[k1]*dist1[node1,k1,:]
                dist2[node1,k1,:] = mu2[k1]*dist2[node1,k1,:]
                node1 = node1+1 
    
    # Resolvendo os trabalhos agregasdos e horas de eficiências 
    ELab = np.zeros(1,Tr-1)
    for k1 in range(1,Tr-1):
        node1 = 1 
        for k2 in range(1,k2):
            for k3 in range(1,nh):
                for k4 in range(1,nk):
                    for instype in range(1,4):
                        ELab[k1] = ELab[k1] + z[k2,k3,k1]*labopt[k4,k2,k3,k1,1,instype]*dist1[node1,k1,instype]+z[k2,k3,k1]*labopt[k4,k2,k3,k1,2,instype]*dist2[node1,k1,instype]
                node1 = node1 +1

    KPR = sum(Kalive)
    ELAB = sum(ELab)

    return dist1, dist2, KPR,ELAB