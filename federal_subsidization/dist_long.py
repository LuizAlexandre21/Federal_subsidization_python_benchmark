# Pacotes 
import numpy as np 
import scipy as sc 
import agent_params
from scipy.io import loadmat

def dist_long(dist1ss,dist2ss,state):
    # Importando dados externos 
    insdata4 = loadmat('/home/alexandre/Documents/Mestrado/Artigos de dissertação/Replication Files/Benchmark Economy/insdata4.mat')
    wage_med_data = loadmat('/home/alexandre/Documents/Mestrado/Artigos de dissertação/Replication Files/Benchmark Economy/wage_med_data1.mat')

    # Parametro da distribuição 
    insdist = [0.152,0.449,0.015,0.384]
    healthdist = [0.9,0.1]
    proddist = [0.25,0.25,0.25,0.25]
    groupdist = [0.2,0.8]

    # Isso forma as matrizes que conterão os valores representativos. Isto terá zeros para todas as gerações que ainda não nasceram e para aquelas que já morreram. Cada linha pertencerá a uma geração.
    MED = np.zeros((T+Tss,Tss))
    KPR = np.zeros((T+Tss,Tss))
    BEQ = np.zeros((T+Tss,Tss))
    HRS = np.zeros((T+Tss,Tss))
    ELAB = np.zeros((T+Tss,Tss))
    MCARE = np.zeros((T+Tss,Tss))
    MCAID = np.zeros((T+Tss,Tss))
    YTAX = np.zeros((T+Tss,Tss))
    YTAXSSM = np.zeros((T+Tss,Tss))
    CTAX = np.zeros((T+Tss,Tss))
    TRANSF = np.zeros((T+Tss,Tss))
    WELF = np.zeros((T+Tss,Tss))

    val =1 
    [~] = agent_params(val)

    #  Vivo no início da transição    
    for tt1 in range(2,T):
        #filename - ['agenttail']
        #load(filename)
        
        # Resolve para a vida remanecente 
        dist1 = np.zeros(nep*nh*nk,T-(tt1-1),4)
        dist2 = np.zeros(nep*nh*nk,T-(tt1-1),4)

        # precisa voltar o distss para "pré-consertar"????
        dist1(:,1,:) = dist1ss(:,tt1,:)./mu2(tt1);   
        dist2(:,1,:) = dist2ss(:,tt1,:)./mu2(tt1);

        if tt1 <= Tr-1:
            for t1 in range(tt1,Tr-1):
                node = 0 
                count = 0 
                for j2 in range(1,nep):
                    for j3 in range(1,nh):
                        count = count +1 
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
                            count2 = 0 
                            for j4 in range(1,nep):
                                for j5 in range(1,nep):
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
    




            for t1 in range(max(Tr,tt1),T-1):
                node = 0
                count = 0 
                j2 = 1 
                for j3 in range(1,nh):
                    count = count + 1 
                    Pr = TrE[count,:,t1]
                    for i1 in range(1,nk):
                        node = node+1 
                        loc11 = kopt[i1,j2,j3,t1 - (tt1-1),1,1]
                        loc21 = kopt[i1,j2,j3,t1 - (tt1-1),1,2]
                        ins11 = insopt[i1,j2,j3,t1 - (tt1-1),1,1)]
                        ins21 = insopt[i1,j2,j3,t1 - (tt1-1),1,2]
                        count2 = 0
                        j4 = 1 
                        for j5 in range(1,nh):
                            count2 = count2+1
                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1 - (tt1-1) + 1,ins11] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1 - (tt1-1) + 1,ins11] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),1]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1 - (tt1-1) + 1,ins11] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1 - (tt1-1) + 1,ins11] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),1]

                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1 - (tt1-1) + 1,ins21] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1 - (tt1-1) + 1,ins21] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),2]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1 - (tt1-1) + 1,ins21] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1 - (tt1-1) + 1,ins21] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1 - (tt1-1),2]


            Kalive = np.zeros[1,T-(tt1-1)]
            Kdead = np.zeros[1,T-(tt1-1)]

            # Resolvendo o capital agregado 
            for k1 in range(tt1,T):
                node1 = 1
                for k2 in range(1,nep):
                    for k3 in range(1,nh):
                        Kalive[k1-(tt1-1/)] = Kalive[k1-(tt1-1)] + mu2[k1]*kgrid[k4]*np.sum(np.squeeze(dist1[node1,k1-(tt1-1),:]+dist2[node1,k1-(tt1-1),:]))
                        Kdead[k1-(tt1-1)] = Kdead[k1-(tt1-1)] + mu3[k1]*kgrid[k4]*np.sum(np.squeeze(dist1[node1,k1-(tt1-1),:]+dist2[node1,k1-(tt1-1),:]))
                        node1 = node1+1 
            
            KPR[tt1+Tss,1:min(Tss,T-(tt1-1))] = Kalive[1:min(Tss,T-(tt1-1))]
            BEQ[tt1+Tss,1:min(Tss,T-(tt1-1))] = Kalive[1:min(Tss,T-(tt1-1))]
    
            # Fixando a distribuição 
            for k1 in range(tt1,T):
                node1 = 1 
                for k2 in range(1,nep):
                    for k3 in range(1,nh):
                        for k4 in range(1,nk):
                            dist1[node1,k1-(tt1-1),:] = mu2[k1]*dist1[node1,k1-(tt1-1),:]
                            dist2[node1,k1-(tt1-1),:] = mu2[k1]*dist2[node1,k1-(tt1-1),:]
                            node1 = node1+a 
            if tt1 <= Tr-1:
                Lab = np.zeros(1,(Tr-1)-(tt1-1))
                Elab = np.zeros(1,(Tr-1)-(tt1-1))
                for k1 in range(tt1,Tr-1):
                    node1 = 1
                    for k2 in range(1,nep):
                        for k3 in range(1,nh):
                            for k4 in range(1,nk):
                                for instype in range(1,4):
                                    Lab[k1-(tt1-1)] = Lab[k1-(tt1-1)] + labopt[k4,k2,k3,k1-(tt1-1),1,instype]*dist1[node1,k1-(tt1-1),instype]+labopt[k4,k2,k3,k1-(tt1-1),2,instype]*dist2[node1,k1-(tt1-1),instype]
                                    ELab[k1-(tt1-1)] = ELab[k1-(tt1-1)] + z[k2,k3,k1]*labopt[k4,k2,k3,k1-(tt1-1),1,instype]*dist1[node1,k1-(tt1-1),instype]+z[k2,k3,k1]*labopt[k4,k2,k3,k1-(tt1-1),2,instype]*dist2[node1,k1-(tt1-1),instype]
                            node = node1+1

                HRS[tt1+Tss,1:min[Tss,Tr-1-(tt1-1)]] = Lab[1:min[Tss,Tr-1-(tt1-1)]]/sum[mu2[1:Tr-1]]
                ELAB[tt1+Tss,1:min[Tss,tr-1-(tt1-1)]] = ELab[1:min[Tss,Tr-1-(tt1-1)]]

                # Resolvendo a renda e a taxa de consumo, bem estar e transferências 
                Ytax = np.zeros(1,T-(tt1-1))
                YtaxSSM = np.zeros(1,T-(tt1-1))
                Ctax = np.zeros(1,T-(tt1-1))
                welfare = np.zeros(1,T-(tt1-1))
                transfs = np.zeros(1,T-(tt1-1))
                for k1 in range(tt1T):
                    node1 =1 
                    for k2 in range(1,nep):
                        for k3 in range(1,nh):
                            for k4 in range(1,nk):
                                for instype in range(1,4):
                                    Ytax[k1-(tt1-1)] = Ytax[k1-(tt1-1)] + ytax[k4,k2,k3,k1-(tt1-1),1,instype]*dist1[node1,k1-(tt1-1),instype]+ ytax[k4,k2,k3,k1-(tt1-1),2,instype]*dist2[node1,k1-(tt1-1),instype]
                                    YtaxSSM[k1-(tt1-1)] = YtaxSSM[k1-(tt1-1)] + ytaxSSM[k4,k2,k3,k1-(tt1-1),1,instype]*dist1[node1,k1-(tt1-1),instype]+ ytaxSSM[k4,k2,k3,k1-(tt1-1),2,instype]*dist2[node1,k1-(tt1-1),instype]
                                    Ctax[k1-(tt1-1)] = Ctax[k1-(tt1-1)] + tau_c*copt[k4,k2,k3,k1-(tt1-1),1,instype]*dist1[node1,k1-(tt1-1),instype]+tau_c*copt[k4,k2,k3,k1-(tt1-1),2,instype]*dist2[node1,k1-(tt1-1),instype]
                                    welfare[k1-(tt1-1)] = welfare[k1-(tt1-1)] + V[k4,k2,k3,k1-(tt1-1),1,instype]*dist1[node1,k1-(tt1-1),instype]+V[k4,k2,k3,k1-(tt1-1),2,instype]*dist2[node1,k1-(tt1-1),instype]
                                    transfers[k1-(tt1-1)] = transfers[k1-(tt1-1)] + transf[k4,k2,k3,k1-(tt1-1),1,instype]*dist1[node1,k1-(tt1-1),instype]+transf[k4,k2,k3,k1-(tt1-1),2,instype]*dist2[node1,k1-(tt1-1),instype]                    
                                 
                            node1 = node1+1
                        
                YTAX[tt1+Tss,1:min(Tss,T-(tt1-1))] = Ytax[1:min(Tss,T-(tt1-1))]
                YTAXSSM[tt1+Tss,1:min(Tss,T-(tt1-1))] = YtaxSSM[1:min(Tss,T-(tt1-1))]
                CTAX[tt1+Tss,1:min(Tss,T-(tt1-1))] = Ctax[1:min(Tss,T-(tt1-1))]
                TRANSF[tt1+Tss,1:min(Tss,T-(tt1-1))] = transfers[1:min(Tss,T-(tt1-1))]
                WELF[tt1+Tss,1:min(Tss,T-(tt1-1))] = welfare[1:min(Tss,T-(tt1-1))]

                # Resolvendo gastos medicos por tipos de seguros 
                mcaid = np,zeros(1,T-(tt1-1))
                if tt1<= Tr-1:
                    for t1 in range(tt1,Tr-1):
                        for j2 in range(1,nep):
                            for j3 in range(1,nh):
                                mcaid[t1-(tt1-1)] = mcaid[t1-(tt1-1) +(1-phi[j3,t1,1])*Mnorm[j3,t1]*np.sum(dist1[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1-(tt1-1),1]+dist2[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1-(tt1-1),1)]
                
                # Medicaid, idosos
                for t1 in range(max(tt1,Tr),T):
                    for j2 in range(1,nep):
                        for j3 in range(1,nh):
                            mcaid[t1-(tt1-1)] = mcaid[t1-(tt1-1)]+[1-phi(j3,t1,1)]*(phimedicare)*Mnorm[j3,t1]*sum(dist1[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1-(tt1-1),1]+dist2[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1-(tt1-1),1)]
                MCAID[tt1+Tss,1:min(Tss,T-(tt1-1))] = mcaid[1:min(Tss,T-(tt1-1))]

                # Medicaid 
                mcare = np.zeros(1,T-(tt1-1))
                for t1 in range(max(tt1,Tr),T):
                    for t2 in range(1,nep):
                        for j3 in range(1,nh):
                            mcare[t1-(tt1-1)] = mcare[t1-(tt1-1)]+phimedicare*Mnorm[j3,t1]*sum(sum(np.squeeze[dist1((j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1-(tt1-1),:]+dist2[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1-(tt1-1),:)))]
                MCARE[tt1+Tss,1:min(Tss,T-(tt1-1))] = mcare[1:min(Tss,T-(tt1-1))]

                # Total de gastos medicos 
                medexp =np.zeros(1,T-(tt1-1))
                for t1 in range(tt1,T):
                    for t2 in range(1,nh):
                        for j3 in range(1,nh):
                            medexp[t1-(tt1-1)] = medexp[t1-(tt1-1)]+Mnorm[j3,t1]*sum(dist1[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1-(tt1-1),1]+dist2[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1-(tt1-1),1)]
                            medexp[t1-(tt1-1)] = medexp[t1-(tt1-1)]+Mnorm[j3,t1]*sum(dist1[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1-(tt1-1),2]+dist2[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1-(tt1-1),2)]
                            medexp[t1-(tt1-1)] = medexp[t1-(tt1-1)]+Mnorm[j3,t1]*sum(dist1[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1-(tt1-1),3]+dist2[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1-(tt1-1),3)]
                            medexp[t1-(tt1-1)] = medexp[t1-(tt1-1)]+Mnorm[j3,t1]*sum(dist1[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1-(tt1-1),4]+dist2[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1-(tt1-1),4)]
                 MED[tt1+Tss,1:min(Tss,T-(tt1-1))] = medexp[1:min(Tss,T-(tt1-1))]


    # Nascimentos durante a transição
    for tt1 in range(1,Tss):
        #filename = ['agentborn' num2str(tt1) '_' num2str(state) '.mat'];
        #load(filename)
        dist1 = np.zeros(nep*nh*nk,T-(tt1-1),4)
        dist2 = np.zeros(nep*nh*nk,T-(tt1-1),4)

        # Distribuição inicial 
        k0 = 0 
        loc1 = np.where(kgrid>=0,1,'first')
        loc1 = min[nk,loc1]
        if loc1 !=:
            w1 = (kgrid[loc1)-k0]/(kgrid[loc1]-kgrid[loc1-1])
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
        else loc1 == 1:
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
                                count2 = count2+1 
                                dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins11] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins11] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1,1]
                                dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins11] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins11] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1,1]

                                dist1[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1+1,ins21] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1+1,ins21] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1,2]
                                dist2[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1+1,ins21] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1+1,ins21] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1,2]

                                dist1[(j4-1)*nh*nk + (j5-1)*nk + loc31,t1+1,ins31] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc31,t1+1,ins31] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1,3]
                                dist2[(j4-1)*nh*nk + (j5-1)*nk + loc31,t1+1,ins31] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc31,t1+1,ins31] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1,3]

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
                            end 
                        end 
                    end 
                end 
            end 
        end 
        if Tr<= min(Tss-tt1-1,T):
            for t1 in range(Tr,min(T-1,Tss-(tt1-1))):
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
                            count2 = count2+1
                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins11] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins11] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1,1]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins11] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc11,t1+1,ins11] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1,1]

                            dist1[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1+1,ins21] = dist1[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1+1,ins21] + Opr[1,1,t1]*Pr[count2]*dist1[node,t1,2]
                            dist2[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1+1,ins21] = dist2[(j4-1)*nh*nk + (j5-1)*nk + loc21,t1+1,ins21] + Opr[1,2,t1]*Pr[count2]*dist1[node,t1,2]
        
        Kalive = np.zeros(1,min(Tss-(tt1-1),T))
        Ldead = np.zeros(1,min(Tss-(tt1-1),T))

        # Resolvendo o capital agregado 
        for k1 in range(1,min(Tss-(tt1-1),T)):
            node1 = 1 
            for k2 in range(1,nep):
                for k3 in range(1,nh):
                    for k4 in range(1,nk):
                        Kalive[k1] = Kalive[k1] + mu2[k1]*kgrid[k4]*sum(np.squeeze[dist1[node1,k1,:])+dist2[node1,k1,:]
                        Kdead[k1] = Kdead[k1] + mu3[k1]*kgrid[k4]*sum(np.squeeze[dist1(node1,k1,:]+dist2[node1,k1,:]))
                        node1 = node1+1
        KPR[Tss-(tt1-1),tt1:tt1+min(Tss-(tt1-1),T)-1] = Kalive[1:min(Tss-(tt1-1),T)]
        BEQ[Tss-(tt1-1),tt1:tt1+min(Tss-(tt1-1),T)-1] = Kdead[1:min(Tss-(tt1-1),T)]

        # Resolvendo a distribuição 
        for k1 in range(1:min(Tss-(tt1-1),T)):
            node1 = 1 
            for k2 in range(1,nep):
                for k3 in range(1,nh):
                    for k4 in range(1,nk):
                        dist1[node1,k1,:] = mu2[k1] + dist1[node1,k1,:]
                        dist2[node1,k1,:] = mu2[k1] + dist2[node1,k1,:]
        
        # Resolvendo a hora de trabalho agregado e de eficiência 
        Lab = np.zeros(1,min(T-(tt1-1),Tr-1))
        ELab =np.zeros(1,min(T-(tt1-1),Tr-1))
        GELab = np.zeros(1,min(T-(tt1-1),Tr-1))
        
        for k1 in range(1,min(Tss-(tt1-1),Tr-1)):
            node1 =1 
            for k2 in range(1,nep):
                for k3 in range(1,nh):
                    for k4 in range(1,nk):
                        for instype in range(1,4):
                                Lab[k1] = Lab[k1] + labopt[k4,k2,k3,k1,1,instype]*dist1[node1,k1,instype]+labopt[k4,k2,k3,k1,2,instype]*dist2[node1,k1,instype]
                                ELab[k1] = ELab[k1] + z[k2,k3,k1]*labopt[k4,k2,k3,k1,1,instype]*dist1[node1,k1,instype]+z[k2,k3,k1]*labopt[k4,k2,k3,k1,2,instype]*dist2[node1,k1,instype]
                                GELab[k1] = GELab[k1] + z[k2,k3,k1]*labopt[k4,k2,k3,k1,1,instype]*dist1[node1,k1,instype]+z[k2,k3,k1]*labopt[k4,k2,k3,k1,2,instype]*dist2[node1,k1,instype]
                    node1 = node1+1 
            HRS[Tss-(tt1-1),tt1:tt1+min(Tss-(tt1-1),Tr-1)-1] = Lab[1:min(Tss-(tt1-1),Tr-1)]./sum(mu2[1:Tr-1])
            ELAB[Tss-(tt1-1),tt1:tt1+min(Tss-(tt1-1),Tr-1)-1] = ELab[1:min(Tss-(tt1-1),Tr-1)]
        
        # Resolvendo a renda e a taxa de consumo, bem estar e transferencias
        Ytax = np.zeros(1,min(Tss-(tt1-1),T))
        YtaxSSM = np.zeros(1,min(Tss-(tt1-1),T))
        Ctax = np.zeros(1,min(Tss-(tt1-1),T))
        welfare = np.zeros(1,min(Tss-(tt1-1),T))
        transfers = np.zeros(1,min(Tss-(tt1-1),T))
        for k1 in range(1,min(Tss-(tt1-1),T)):
            node1 = 1
            for k2 in range(1,nep):
                for k3 in range(1,nh):
                    for k4 in range(1,nk):
                        for instype in range(1,4):
                            Ytax[k1] = Ytax[k1] + ytax[k4,k2,k3,k1,1,instype]*dist1[node1,k1,instype]+ ytax[k4,k2,k3,k1,2,instype]*dist2[node1,k1,instype]
                            YtaxSSM[k1] = YtaxSSM[k1] + ytaxSSM[k4,k2,k3,k1,1,instype]*dist1[node1,k1,instype]+ ytaxSSM[k4,k2,k3,k1,2,instype]*dist2[node1,k1,instype]
                            Ctax[k1] = Ctax[k1] + tau_c*copt[k4,k2,k3,k1,1,instype]*dist1[node1,k1,instype]+tau_c*copt[k4,k2,k3,k1,2,instype]*dist2[node1,k1,instype]
                            welfare[k1] = welfare[k1] + V[k4,k2,k3,k1,1,instype]*dist1[node1,k1,instype]+V[k4,k2,k3,k1,2,instype]*dist2[node1,k1,instype]
                            transfers[k1] = transfers[k1] + transf[k4,k2,k3,k1,1,instype]*dist1[node1,k1,instype]+transf[k4,k2,k3,k1,2,instype]*dist2[node1,k1,instype]
                    node = node1+1 
        
        YTAX[Tss-(tt1-1),tt1:tt1+min(Tss-(tt1-1),T)-1] = Ytax[1:min(Tss-(tt1-1),T)]
        YTAXSSM[Tss-(tt1-1),tt1:tt1+min(Tss-(tt1-1),T)-1] = YtaxSSM[1:min(Tss-(tt1-1),T)]
        CTAX[Tss-(tt1-1),tt1:tt1+min(Tss-(tt1-1),T)-1] = Ctax[1:min(Tss-(tt1-1),T)]
        WELF[Tss-(tt1-1),tt1:tt1+min(Tss-(tt1-1),T)-1] = welfare[1:min(Tss-(tt1-1),T)]
        TRANSF[Tss-(tt1-1),tt1:tt1+min(Tss-(tt1-1),T)-1] = transfers[1:min(Tss-(tt1-1),T)]

        welfare = 0 

        # Solucionando os tipos de gastos medicos por tipos de seguros 
        # Medicaid, não idoso
        mcaid = np.zeros[1,min(Tss-(tt1-1),T)]
        for t1 in range(1,min(Tss-(tt1-1),Tr-1)):
            for j2 in range(1,nep):
                for j3 in range(1,nh):
                    mcaid[t1] = mcaid[t1]+(1-phi[j3,t1,1])*Mnorm[j3,t1]*sum(dist1[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,1]+dist2[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,1)]

        # Medicaid, idoso 
        if min(Tss-(tt1-1),T)>=Tr:
            for t1 in range(Tr,min(Tss-(tt1-1),T)):
                for j2 in range(1,nep):
                    for j3 in range(1,nh):
                        mcaid[t1] = mcaid[t1]+[1-phi(j3,t1,1)]*(phimedicare)*Mnorm[j3,t1]*sum(dist1[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,1]+dist2[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,1)];
        MCAID[Tss-(tt1-1),tt1:tt1+min(Tss-(tt1-1),T)-1] = mcaid[1:min(Tss-(tt1-1),T)];

        if min(Tss-(tt1-1),T)>=Tr
            mcare = zeros(1,min(Tss-(tt1-1),T));
            #Medicare
            for t1 in range(Tr:min(Tss-(tt1-1),T)):
                for j2 in range(1,nep):
                    for j3 in range(1,nh):
                        mcare[t1] = mcare[t1]+phimedicare*Mnorm[j3,t1]*sum(sum(np.squeeze(dist1[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,:]+dist2[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,:])));
                    end
                end
            end
            MCARE[Tss-(tt1-1),tt1:tt1+min(Tss-(tt1-1),T)-1] = mcare[1:min(Tss-(tt1-1),T)]
        end

        #  Total medical expenses
       medexp=np.zeros(1,min(Tss-(tt1-1),T)):
       for t1 in range(1,min(Tss-(tt1-1),T)):
           for j2 in range(1,nep):
               for j3 in range(1,nh):
                    medexp[t1] = medexp[t1]+Mnorm[j3,t1]*sum(dist1[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,1]+dist2[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,1)];
                    medexp[t1] = medexp[t1]+Mnorm[j3,t1]*sum(dist1[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,2]+dist2[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,2)];
                    medexp[t1] = medexp[t1]+Mnorm[j3,t1]*sum(dist1[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,3]+dist2[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,3)];
                    medexp[t1] = medexp[t1]+Mnorm[j3,t1]*sum(dist1[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,4]+dist2[(j2-1)*nh*nk + (j3-1)*nk +1:(j2-1)*nh*nk + (j3-1)*nk +nk,t1,4)];
        MED[Tss-(tt1-1),tt1:tt1+min(Tss-(tt1-1),T)-1] = medexp[1:min(Tss-(tt1-1),T)]

    MED = sum(MED)
    KPR = sum(KPR)
    BEQ = sum(BEQ)
    HRS = sum(HRS)
    ELAB = sum(ELAB)
    MCARE = sum(MCARE)
    MCAID = sum(MCAID)
    YTAX = sum(YTAX)
    YTAXSSM = sum(YTAXSSM)
    CTAX = sum(CTAX)
    TRANSF = sum(TRANSF)
    WELF2=sum(welfare)

    return MED, KPR, HRS, ELAB, MCARE, MCAID, YTAX, CTAX, TRANSF, WELF, BEQ, YTAXSSM, WELF2

