# Pacotes 
import numpy as np 
import scipy as sc 
import agent
import head
import tail 
from scipy.io import loadmat

def dist_short(dist1ss,dist2ss,rho1,tau1,thresh,state):     
    insdata4 = loadmat('/home/alexandre/Documents/Mestrado/Artigos de dissertação/Replication Files/Benchmark Economy/insdata4.mat')
    wage_med_data = loadmat('/home/alexandre/Documents/Mestrado/Artigos de dissertação/Replication Files/Benchmark Economy/wage_med_data1.mat')

    # Parametro da distribuição 
    insdist = [0.152,0.449,0.015,0.384]
    healthdist = [0.9,0.1]
    proddist = [0.25,0.25,0.25,0.25]
    groupdist = [0.2,0.8]

    # Isso forma as matrizes que conterão os valores representativos. Isto  terá zeros para todas as gerações que ainda não nasceram e para aquelas que já morreram.Cada linha pertencerá a uma geração. (e uma linha vazia por acidente
    KPR = np.zeros(T+Tss,Tss)
    ELAB = np.zeros(T+Tss,Tss)

    [V,labopt, kopt, copt, ytax, transf, insopt, ytaxSSM] = agent[np.matmul(rho1(end),ones(1,T)),np.transpose([tau1(1,Tss),tau1(2,Tss)])*np.ones(1,T),thresh)

    # save sspol V labopt kopt copt ytax transf insopt ytaxSSM -double

    for tt1 in range(1,Tss):
        tt2 = len(rho1[tt1:end]);
        rho = [rho1(tt1:end),np.matmul(np.ones(1,T-tt2),rho1(end))];
        tau = [tau1[:,tt1:end], tau1[]:,Tss]*np.ones[1,T-tt2]]
    
        # Agents born into the transition
        [V,labopt, kopt, copt, ytax, transf, insopt, ytaxSSM] = head(rho,tau,thresh,min(T,Tss-(tt1-1)))
        #filename = ['agentborn' num2str(tt1) '_' num2str(state) '.mat'];
        #save(filename,'V','labopt', 'kopt', 'copt', 'ytax', 'transf', 'insopt', 'ytaxSSM');

        rho2 = [rho1,np.transpose(rho1(end),ones(1,T-length(rho1)-1))]    # this (e tau) deve ter comprimento T-1 porque iniciamos o loop nos agentes j = 2.
        tau2 = [tau1,tau1[:,Tss]*np.ones(1,T-Tss-1)]

        # Agentes vivos durante a transição
        for tt2 in range(2,T):
            rho = rho2[tt2-1:end]
            tau = tau2[:,tt2-1:end]
            [V,labopt, kopt, copt, ytax, transf, insopt, ytaxSSM] = tail[rho,tau,thresh,tt2]
#            filename = ['agenttail' num2str(tt2) '_' num2str(state) '.mat'];
#            save(filename,'V','labopt', 'kopt', 'copt', 'ytax', 'transf', 'insopt', 'ytaxSSM');

        # Vivo no início da transição
        for tt1 in range(2,T):
#          filename = ['agenttail' num2str(tt1) '_' num2str(state) '.mat'];
#           load(filename);
#           load ssresults1_1 dist1 dist2; dist1ss = dist1; dist2ss = dist2;        
            dist1 = np.zeros(nep*nh*nk,T-(tt1-1),4) #solve for the remainder of life  
            dist2 = np.zeros(nep*nh*nk,T-(tt1-1),4) 
        
            # initial distribution
            dist1[:,1,:] = dist1ss[:,tt1,:]/mu2[tt1]   # need to get distss back to "pre-fix"
            dist2[:,1,:] = dist2ss[:,tt1,:]/mu2[tt1]

            if tt1 <=Tr-1:
                for t1 in range(tt1,Tr-1):
                    node = 0 
                    count = 0 
                    for j2 in range(1,nep):
                        for j3 in range(1,nh):
                            count = count+1
                            Pr = TrE[count,:,t1]
                            for i1 in range(1,nk):
                                node = node+1 
                                loc11 = kopt[i1,j2,j3,t1 - (tt1-1),1,1]
                                loc21 = kopt[i1,j2,j3,t1 - (tt1-1),1,2]
                                loc31 = kopt[i1,j2,j3,t1 - (tt1-1),1,3]
                                loc41 = kopt[i1,j2,j3,t1 - (tt1-1),1,4]
                                loc12 = kopt[i1,j2,j3,t1 - (tt1-1),2,1]
                                loc22 = kopt[i1,j2,j3,t1 - (tt1-1),2,2]
                                loc32 = kopt[i1,j2,j3,t1 - (tt1-1),2,3]
                                loc42 = kopt[i1,j2,j3,t1 - (tt1-1),2,4]
                                ins11 = insopt[i1,j2,j3,t1 - (tt1-1),1,1]
                                ins21 = insopt[i1,j2,j3,t1 - (tt1-1),1,2]
                                ins31 = insopt[i1,j2,j3,t1 - (tt1-1),1,3]
                                ins41 = insopt[i1,j2,j3,t1 - (tt1-1),1,4]
                                ins12 = insopt[i1,j2,j3,t1 - (tt1-1),2,1]
                                ins22 = insopt[i1,j2,j3,t1 - (tt1-1),2,2]
                                ins32 = insopt[i1,j2,j3,t1 - (tt1-1),2,3]
                                ins42 = insopt[i1,j2,j3,t1 - (tt1-1),2,4]
                                count2=0
                                for j4 in range(1,nep):
                                    for j5 in range(1,nh):
                                        count2 = count2 + 1
                                        dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1 - (tt1-1) + 1,ins11] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1 - (tt1-1) + 1,ins11] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),1]
                                        dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1 - (tt1-1) + 1,ins11] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1 - (tt1-1) + 1,ins11] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),1]

                                        dist1[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1 - (tt1-1) + 1,ins21] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1 - (tt1-1) + 1,ins21] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),2]
                                        dist2[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1 - (tt1-1) + 1,ins21] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1 - (tt1-1) + 1,ins21] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),2]

                                        dist1[(j4-1)*nh*nk + (j5-1)*nk + loc31,t1 - (tt1-1) + 1,ins31] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc31,t1 - (tt1-1) + 1,ins31] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),3]
                                        dist2[(j4-1)*nh*nk + (j5-1)*nk + loc31,t1 - (tt1-1) + 1,ins31] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc31,t1 - (tt1-1) + 1,ins31] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),3]

                                        dist1[(j4-1)*nh*nk + (j5-1)*nk + loc31,t1 - (tt1-1) + 1,ins41] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc41,t1 - (tt1-1) + 1,ins41] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),4]
                                        dist2[(j4-1)*nh*nk + (j5-1)*nk + loc31,t1 - (tt1-1) + 1,ins41] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc41,t1 - (tt1-1) + 1,ins41] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),4]
                                        
                                        dist1[(j4-1)*nh*nk + (j5-1)*nk + loc12,t1 - (tt1-1) + 1,ins12] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc12,t1 - (tt1-1) + 1,ins12] + Opr[2,1,t1]*Pr[count2]*dist2[node,t1 - (tt1-1),1]
                                        dist2[(j4-1)*nh*nk + (j5-1)*nk + loc12,t1 - (tt1-1) + 1,ins12] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc12,t1 - (tt1-1) + 1,ins12] + Opr[2,2,t1]*Pr[count2]*dist2[node,t1 - (tt1-1),1]

                                        dist1[(j4-1)*nh*nk + (j5-1)*nk + loc22,t1 - (tt1-1) + 1,ins22] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc22,t1 - (tt1-1) + 1,ins22] + Opr[2,1,t1]*Pr[count2]*dist2[node,t1 - (tt1-1),2]
                                        dist2[(j4-1)*nh*nk + (j5-1)*nk + loc22,t1 - (tt1-1) + 1,ins22] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc22,t1 - (tt1-1) + 1,ins22] + Opr[2,2,t1]*Pr[count2]*dist2[node,t1 - (tt1-1),2]

                                        dist1[(j4-1)*nh*nk + (j5-1)*nk + loc32,t1 - (tt1-1) + 1,ins32] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc32,t1 - (tt1-1) + 1,ins32] + Opr[2,1,t1]*Pr[count2]*dist2[node,t1 - (tt1-1),3]
                                        dist2[(j4-1)*nh*nk + (j5-1)*nk + loc32,t1 - (tt1-1) + 1,ins32] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc32,t1 - (tt1-1) + 1,ins32] + Opr[2,2,t1]*Pr[count2]*dist2[node,t1 - (tt1-1),3]

                                        dist1[(j4-1)*nh*nk + (j5-1)*nk + loc42,t1 - (tt1-1) + 1,ins42] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc42,t1 - (tt1-1) + 1,ins42] + Opr[2,1,t1]*Pr[count2]*dist2[node,t1 - (tt1-1),4]
                                        dist2[(j4-1)*nh*nk + (j5-1)*nk + loc42,t1 - (tt1-1) + 1,ins42] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc42,t1 - (tt1-1) + 1,ins42] + Opr[2,2,t1]*Pr[count2]*dist2[node,t1 - (tt1-1),4] 
    
    for t1 in range(max(Tr,tt1),T-1):
        node = 0
        count = 0 
        j2 = 1 
        for j3 in range(1,nh):
            count = count +1 
            Pr = TrE[count,:,t1]
            for i1 in range(1,nk):
                node = node +1 
                loc11 = kopt[i1,j2,j3,t1 - (tt1-1),1,1]
                loc21 = kopt[i1,j2,j3,t1 - (tt1-1),1,2]
                ins11 = insopt[i1,j2,j3,t1 - (tt1-1),1,1]
                ins21 = insopt[i1,j2,j3,t1 - (tt1-1),1,2]
                count2 =0 
                j4 = 1 
                for j5 in range(1,nh):
                    count2 = count2+1 
                    dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1 - (tt1-1) + 1,ins11] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1 - (tt1-1) + 1,ins11] + Opr[1,1,t1]*Pr(count2]*dist1[node,t1 - (tt1-1),1]
                    dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1 - (tt1-1) + 1,ins11] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1 - (tt1-1) + 1,ins11] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),1]

                    dist1[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1 - (tt1-1) + 1,ins21] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1 - (tt1-1) + 1,ins21] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),2]
                    dist2[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1 - (tt1-1) + 1,ins21] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1 - (tt1-1) + 1,ins21] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),2]
    
    Kalive = np.zeros(1,T-(tt1-1))
    Kdead = np.zeros(1,T-(tt1-1))

    # Resolvendo o capital agregado 
    for k1 in range(tt1,T):
        node1 = 1 
        for k2 in range(1,nep):
            for k3 in range(1,nh):
                for k4 in range(1,nk):
                    Kalive[k1-(tt1-1)] = Kalive[k1-(tt1-1)] + mu2[k1]*kgrid[k4]*sum(np.squeeze(dist1[node1,k1-(tt1-1),:]+dist2[node1,k1-(tt1-1),:]))
                    Kdead[k1-(tt1-1)] = Kdead[k1-(tt1-1)] + mu3[k1]*kgrid[k4]*sum(np.squeeze(dist1[node1,k1-(tt1-1),:]+dist2[node1,k1-(tt1-1),:]))
                    node1 = node1+1
    KPR[tt1+Tss,1:min(Tss,T-(tt1-1))] = Kalive[1:min(Tss,T-(tt1-1))]

    # Fixando a distribuição 
    for k1 in range(tt1,T):
        node1 = 1 
        for k2 in range(1,nep):
            for k3 in range(1,nh):
                for k4 in range(1,nk):
                    dist1[node1,k1-(tt1-1),:] = mu2[k1]*dist1[node1,k1-(tt1-1),:]
                    dist2[node1,k1-(tt1-1),:] = mu2[k1]*dist2[node1,k1-(tt1-1),:]
                    node1 = node1+1 
    
    # Solving aggregate labor hours, efficient hours 
    if tt1<=Tr-1:
        Lab = np.zeros(1,(Tr-1)-(tt1-1))
        ELab = np.zeros(1,(Tr-1)-(tt1-1))
        for k1 in range(tt1,Tr-1):
            node1 = 1;
            for k2 in range(1,nep):
                for k3 in range(1,nh):
                    for k4 in range(1,nk):
                        for instype in range(1,4):
                            Lab[k1-(tt1-1)] = Lab[k1-(tt1-1)] + labopt[k4,k2,k3,k1-(tt1-1),1,instype]*dist1[node1,k1-(tt1-1),instype]+labopt[k4,k2,k3,k1-(tt1-1),2,instype]*dist2[node1,k1-(tt1-1),instype]
                            ELab[k1-(tt1-1)] = ELab[k1-(tt1-1)] + z[k2,k3,k1]*labopt[k4,k2,k3,k1-(tt1-1),1,instype]*dist1[node1,k1-(tt1-1),instype]+z[k2,k3,k1]*labopt[k4,k2,k3,k1-(tt1-1),2,instype]*dist2[node1,k1-(tt1-1),instype]
                            node1 = node1+1

        ELAB[tt1+Tss,1:min(Tss,Tr-1-(tt1-1))] = ELab[1:min(Tss,Tr-1-(tt1-1))]

    # Novos Nascimentos 
    for tt1 in range(1,Tss):
        dist1 = np.zeros(nep*nh*nk,T-(tt1-1),4)   
        dist2 = np.zeros(nep*nh*nk,T-(tt1-1),4)
        # initial distribution
        k0 = 0;
        loc1 = np.where(kgrid>=k0,1,'first')
        loc1 = min(nk,loc1)
        if loc1 != 1:
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
        
        elif loc1 == 1:
            for i1 in range(1,nep):
                for i2 in range(1,nh):
                    dist1[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,1] = healthdist[i2]*proddist[i1]*insdist[1]*groupdist[1]
                    dist1[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,2] = healthdist[i2]*proddist[i1]*insdist[2]*groupdist[1]            
                    dist1[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,3] = healthdist[i2]*proddist[i1]*insdist[3]*groupdist[1]
                    dist1[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,4] = healthdist[i2]*proddist[i1]*insdist[4]*groupdist[1]
                    dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,1] = healthdist[i2]*proddist[i1]*insdist[1]*groupdist[2]
                    dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,2] = healthdist[i2]*proddist[i1]*insdist[2]*groupdist[2]            
                    dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,3] = healthdist[i2]*proddist[i1]*insdist[3]*groupdist[2]
                    dist2[(i1-1)*nh*nk + (i2-1)*nk  + loc1,1,4] = healthdist[i2]*proddist[i1]*insdist[4]*groupdist[2]                    

        for t1 in range(1,min(Tr-1,Tss-(tt1-1))):
            node = 0 
            count = 0 
            for j2 in range(1,nep):
                for j3 in range(1,nh):
                    count = count +1
                    Pr = TrE[count,:,t1]
                    for i1 in range(1,nk):
                        node = node +1
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
                                dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1 - (tt1-1) + 1,ins11] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1 - (tt1-1) + 1,ins11] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),1]
                                dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1 - (tt1-1) + 1,ins11] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1 - (tt1-1) + 1,ins11] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),1]

                                dist1[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1 - (tt1-1) + 1,ins21] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1 - (tt1-1) + 1,ins21] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),2]
                                dist2[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1 - (tt1-1) + 1,ins21] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1 - (tt1-1) + 1,ins21] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),2]

                                dist1[(j4-1)*nh*nk + (j5-1)*nk + loc31,t1 - (tt1-1) + 1,ins31] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc31,t1 - (tt1-1) + 1,ins31] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),3]
                                dist2[(j4-1)*nh*nk + (j5-1)*nk + loc31,t1 - (tt1-1) + 1,ins31] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc31,t1 - (tt1-1) + 1,ins31] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),3]

                                dist1[(j4-1)*nh*nk + (j5-1)*nk + loc31,t1 - (tt1-1) + 1,ins41] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc41,t1 - (tt1-1) + 1,ins41] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),4]
                                dist2[(j4-1)*nh*nk + (j5-1)*nk + loc31,t1 - (tt1-1) + 1,ins41] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc41,t1 - (tt1-1) + 1,ins41] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),4]
                                        
                                dist1[(j4-1)*nh*nk + (j5-1)*nk + loc12,t1 - (tt1-1) + 1,ins12] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc12,t1 - (tt1-1) + 1,ins12] + Opr[2,1,t1]*Pr[count2]*dist2[node,t1 - (tt1-1),1]
                                dist2[(j4-1)*nh*nk + (j5-1)*nk + loc12,t1 - (tt1-1) + 1,ins12] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc12,t1 - (tt1-1) + 1,ins12] + Opr[2,2,t1]*Pr[count2]*dist2[node,t1 - (tt1-1),1]

                                dist1[(j4-1)*nh*nk + (j5-1)*nk + loc22,t1 - (tt1-1) + 1,ins22] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc22,t1 - (tt1-1) + 1,ins22] + Opr[2,1,t1]*Pr[count2]*dist2[node,t1 - (tt1-1),2]
                                dist2[(j4-1)*nh*nk + (j5-1)*nk + loc22,t1 - (tt1-1) + 1,ins22] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc22,t1 - (tt1-1) + 1,ins22] + Opr[2,2,t1]*Pr[count2]*dist2[node,t1 - (tt1-1),2]

                                dist1[(j4-1)*nh*nk + (j5-1)*nk + loc32,t1 - (tt1-1) + 1,ins32] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc32,t1 - (tt1-1) + 1,ins32] + Opr[2,1,t1]*Pr[count2]*dist2[node,t1 - (tt1-1),3]
                                dist2[(j4-1)*nh*nk + (j5-1)*nk + loc32,t1 - (tt1-1) + 1,ins32] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc32,t1 - (tt1-1) + 1,ins32] + Opr[2,2,t1]*Pr[count2]*dist2[node,t1 - (tt1-1),3]

                                dist1[(j4-1)*nh*nk + (j5-1)*nk + loc42,t1 - (tt1-1) + 1,ins42] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc42,t1 - (tt1-1) + 1,ins42] + Opr[2,1,t1]*Pr[count2]*dist2[node,t1 - (tt1-1),4]
                                dist2[(j4-1)*nh*nk + (j5-1)*nk + loc42,t1 - (tt1-1) + 1,ins42] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc42,t1 - (tt1-1) + 1,ins42] + Opr[2,2,t1]*Pr[count2]*dist2[node,t1 - (tt1-1),4] 
        
        if Tr<=min(Tss-(tt1-1),1):
            for t1 in range(Tr,min(T-1,Tss-(tt1-1))):
                node = 0 
                count = 0 
                j2 =1 
                for j3 in range(1,nh):
                    count = count +1
                    Pr = TrE[count,:,t1]
                    for i1 in range(1,nk):
                        node = node+1 
                        loc11 = kopt[i1,j2,j3,t1,1,1]
                        loc21 = kopt[i1,j2,j3,t1,1,2]
                        ins11 = insopt[i1,j2,j3,t1,1,1]
                        ins21 = insopt[i1,j2,j3,t1,1,2]
                        count2 = 0 
                        j4 = 1 
                        for j5 in range(1,nh):
                            count2 = count2 + 1
                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1 - (tt1-1) + 1,ins11] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1 - (tt1-1) + 1,ins11] + Opr[1,1,t1]*Pr(count2]*dist1[node,t1 - (tt1-1),1]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1 - (tt1-1) + 1,ins11] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1 - (tt1-1) + 1,ins11] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),1]

                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1 - (tt1-1) + 1,ins21] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1 - (tt1-1) + 1,ins21] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),2]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1 - (tt1-1) + 1,ins21] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1 - (tt1-1) + 1,ins21] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),2]

    Kalive = np.zeros(1,min(Tss-(tt1-1),T))
    Kdead = np.zeros(1,min(Tss=(tt1-1),T))

    # Resolvendo a Agregação do capital
    for k1 in range(1,min(Tss-(tt1-1),t)):
        node1 = 1
        for k2 in range(1,nep):
            for k3 in range(1,nh):
                for k4 in range(1,nk):
                        Kalive[k1] = Kalive[k1] + mu2[k1]*kgrid[k4]*sum(np.squeeze[dist1(node1,k1,:]+dist2[node1,k1,:]))
                        Kdead[k1] = Kdead[k1] + mu3[k1]*kgrid[k4]*sum(np.squeeze(dist1[node1,k1,:]+dist2[node1,k1,:]))
                        node1 = node1+1
    KPR[Tss-(tt1-1),tt1:tt1+min(Tss-(tt1-1),T)-1] = Kalive[1:min(Tss-(tt1-1),T)]

    # Fixando a distribuição 
    for k1 in range(1,min(Tss-(tt1-1),T)):
        node1 = 1 
        for k2 in range(1,nep):
            for k3 in range(1,nh):
                for k4 in range(1,nk):
                    dist1[node1,k1,:] = mu2[k1]*dist1[node1,k1,:]
                    dist2[node1,k1,:] = mu2[k1]*dist2[node1,k1,:]
                    node1 = node1+1

    # Solving aggregate labor hours, efficient hours 
    Lab = np.zeros(1,min(T-(tt1-1),Tr-1))
    ELab = np.zeros(1,min(T-(tt1-1),Tr-1))
    for k1 in range(1,min(Tss-(tt1-1),Tr-1)):
        node1 = 1
        for k2 in range(1,nep):
            for k3 in range(1,nh):
                for k4 in range(1,nk):
                    for instype in range(1,4):
                        Lab[k1] = Lab[k1] + labopt[k4,k2,k3,k1,1,instype]*dist1[node1,k1,instype]+labopt[k4,k2,k3,k1,2,instype]*dist2[node1,k1,instype]
                        ELab[k1] = ELab[k1] + z[k2,k3,k1]*labopt[k4,k2,k3,k1,1,instype]*dist1[node1,k1,instype]+z[k2,k3,k1]*labopt[k4,k2,k3,k1,2,instype]*dist2[node1,k1,instype]
                node1 = node1+1;
        ELAB[Tss-(tt1-1),tt1:tt1+min(Tss-(tt1-1),Tr-1)-1] = ELab[1:min(Tss-(tt1-1),Tr-1)]
         
    
    KPR = sum(KPR)
    ELAB = sum(ELAB)