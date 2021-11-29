# Pacotes 
import numpy as np 
import scipy as sc 
from scipy.io import loadmat


# Criando funções auxiliares

def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)


''' Resolve o problema do agente em estado estacionário '''
# Agente no estado estacionário 
def agent(self,rho,tau2,thresh1):
    #self.beq =beqapprox 
    ngprem = 0.0 # Premio para o grupo não HI
    gamma = 1.11 # Seguro de saúde 'load'
    b = 0.9875 

    # Parametros para utilidade não separavel
    chi3 =0.3 
    sigma4 = 4 

    # Importando dados externos 
    insdata4 = loadmat('/home/alexandre/Documents/Mestrado/Artigos de dissertação/Replication Files/Benchmark Economy/insdata4.mat')
    wage_med_data = loadmat('/home/alexandre/Documents/Mestrado/Artigos de dissertação/Replication Files/Benchmark Economy/wage_med_data1.mat')

    # Parametros 
    T = 80 
    Tr = 45             # Idade de aposentadoria
    alpha = 0.36        # Compartilhamento de Capital
    A = 1               # TFP
    phimedicare = 0.5 
    d = 0.083
    nh = 2              # Numero de estados de saúde - bem e mal 
    a0 = 0.258 
    a1 = 0.768          # Parametro para função imposto 
    ss = 0.27           # Pagamento Seguridade social 4.5% do gdp
    tau_c = 0.057       # Taxa de consumo 
    premmedicare = 0.26 # Premio do Medicare 
    cfloor = 0.8        # Piso de consumo - Financiada pelo governo federal e estadual 
    pm = 0.00003625     # normalização que garante gastos médicos em 18% do PIB
    M =wage_med_data['M']
    Mnorm = pm*M        # normalizando os preços dos gastos medicos 
    tss = 0.124*2       # taxa ss 
    tm = 0.029          # imposto do medicare 
    ybar = 3.65         # ganhos tributáveis ​​máximos para imposto SS

    # Capital Individual -> Para checar
    nk = 50     
    kgrid = np.zeros((nk,1))
    kub = 13 
    klb = 0.001 
    exppwr = 3 
    for i in range(0,nk+1):
        kgrid[i] = (kub-klb)*((i/nk))**exppwr + klb
    #kgrid = list(range(klb,kgrid))
    nk = length(kgrid)
 
    # Intervalo de trabalho  -> Checado
    nl = 25 
    lub = 0.65 # Teto de oferta do trabalho 
    llb = 0 
    lgrid = zeros((nl,1))
    linc = (lub-llb)/(nl-1)
    lgrid[1] = llb 
    for i in range(1,nl):
        lgrid[i] =lgrid[i-1] +linc 

    ng = 2 
    nins = 4 
    nep = wage_med_data['nep'][0][0]
    
    # Criando arrays  -> Checado 
    V = np.zeros((nk,nep,nh,T+1,ng,nins))
    insopt = np.zeros((nk,nep,nh,T,ng,nins))
    labopt = np.zeros((nk,nep,nh,Tr-1,ng,nins))
    kopt = np.zeros((nk,nep,nh,T,ng,nins))
    copt = np.zeros((nk,nep,nh,T,ng,nins)) 
    ytax = np.zeros((nk,nep,nh,T,ng,nins))
    ytaxst = np.zeros((nk,nep,nh,T,ng,nins))
    transf = np.zeros((nk,nep,nh,T,ng,nins))

    [KGRID1, KGRID2]=np.meshgrid(kgrid,kgrid,indexing='ij') # KGRID1 é k e KGRID2 é k'
    KGRID1 = np.tile(KGRID1,[1,1,2])
    KGRID2 = np.tile(KGRID2,[1,1,2])
    ZEROGRID = np.zeros((nk,nk,2))
    ONESGRID = np.ones((nk,nk,2))
    #mincons = (1/(1+tau_c))*np.matmul(cfloor,ONESGRID)

    phi2 = phimedicare    
    ssben = ss
    g1 = 1
    prem1 = premmedicare
    thresh=.125
    for i in range(T,Tr-1,-1):
        rate = 1 + A*alpha*(rho(i))**(alpha-1) - d 
        a2 = tau2(:,t1)
        j2 =1
        for j3 in range(1,nh):
            med = Mnorm[j3,i]
            surv1 = surv[j2,j3,i]
            count = j3 + (j2-1)*nh                
            Pr = TrEprime[count,:,t1]            
            phimedicaid = phi[j3,t1,1]
            phigrid = [phimedicaid, 1]       
            
            EVprg1 = Pr[1]*V[:,1,1,i+1,1,1]
            EVprb1 = Pr[2]*V[|,1,2,i+1,1,1]
            E1 = EVprb1 +EVprv1 
            EVprg2 = Pr[1]*V[:,1,1,t1+1,1,2];
            EVprb2 = Pr[2]*V[:,1,2,t1+1,1,2];
            E2 = EVprg2+EVprb2;
            Ehold1 = np.transpose(np.array([E1,E2]));            
            Ehold2 = np.tile(Ehold1,[1,1,nk]);    # Replicando E em NK vezes 
            E = np.random.permute(Ehold2,[3,2,1]);  # Permuta para as dimensões k,k',ins        
            
            income = rate.*KGRID1 +ssben
            tincome = np.matmul(max(rate-1,0),KGRID1) + _ssben 
            tincome = tincome(:,:,1)            # Se livrando dos ins dimenções para fazer o efeito do medicaid manualmente 

            # Abaixo da media do treshold 
            trfind = np.where(trincome<=thresh) 
            ntr = length(trfind)
            I1,I2 = ind2sub(length(tincome2,trfind))
            tr2 = sub2ind(length(tincome),I1,I2,1*np.ones(ntr,1)) # Adicionando manualmente 2 para ins dim 

            # Acima da media do treshold 
            trfind = np.where(tincome2>thresh);    % above means test
            ntr = length(trfind)
            [I1, I2] = ind2sub(length(tincome2),trfind)
            tr2 = sub2ind(size(tincome),I1,I2,1*np.ones(ntr,1))

            # Podemos para expandir para incluir seguro privado 
            for instype in range(1,2):
                phi1 = phigrid(instype)
                taxdeduc = max(0,phi1*(phi2*med)- 0.75*tincome)
                ytax1 = max(ZEROGRID,tincome -taxdeduc)
                ytaxstargrid3 = np.matmul(a0,(ytax1 - (ytax1**(-a1) + np.matmul(a2(2).*ONESGRID)**(-1/a1)) + np.matmul(np.matmul(a02,max(rate-1,0)),KGRID1);
                ytaxstargrid2 = a0.*((ytax1) - ((ytax1).^(-a1) + (a2(1)).*ONESGRID).^(-1/a1));                

                # Ajustando da piso de consumo 
                TRANSF = max(ZEROGRID,  mincons - income + np.matmul((taxdeduc - beq),ONESGRID) + ytaxstargrid2 + ytaxstargrid3)
                cons1 = np.matmul((1/(1+tau_c)),(income - (prem1 + np.matmul((phi1*(phi2*med) - beq),ONESGRID) - KGRID2 - ytaxstargrid2 - ytaxstargrid3 + TRANSF)));
                tr3 = np.where(cons1<0)
                Vtilde = np.matmul((1/(1-sigma4)),(np.matmul((cons1**chi3),(ONESGRID**(1-chi3))))**(1-sigma4)) + np.matmul((surv1*b),E);
                
                # Medicaid - not uninsured                
                Vtilde[tr1] = -10e6
                # Uninsured - no medicaid
                Vtilde[tr2] = -10e6
                Vtilde[tr3] = -10e6 

                Vtilde1 = np.random.permute(Vtilde,[3,2,1])
                V(:,j2,j3,i,1,instype) = max(max(Vtilde1))

                [~,Ik] = max(max(Vtilde1),[],2)

                Vtilde1 = np.random.permute(Vtilde,[2,3,1])
                [~, Iins] = max(max(Vtilde1),[],2)
                
                insopt[:,j2,j3,t1,g1,instype] = Iins
                kopt[:,j2,j3,t1,g1,instype] = Ik          

                Ik = np.squeeze(Ik)
                Ik = np.transpose(Ik)
                Iins = np.squeeze(Iins)
                Iins = np.transpose(Iins)
                IND1 = sub2ind(size(cons1),1:nk,Ik,Iins)

                copt[:,j2,j3,i,g1,instype] = cons1[IND1]
                transf[:,j2,j3,t1,g1,instype] = TRANSF[IND1]                
                ytax[:,j2,j3,t1,g1,instype] = ytaxstargrid2[IND1]
                ytaxst[:,j2,j3,t1,g1,instype] = ytaxstargrid3[IND1]


    t1 = Tr-1 
    LGRID,KGRID1,KGRID2 = np.meshgrid(lgrid,kgrid,kgrid)
    KGRID1 = np.tile(KGRID1,[1,1,2])
    KGRID2 = np.tile(KGRID2,[1,1,2])
    ZEROGRID = np.zeros(nk,nk,2)
    ONESGRID =np.ones(nk,nk,2)

    thresh = thresh1

    wage = A*(1-alpha)*(rho(t1))^alpha 
    rate = 1 + A*alpha*(rho(t1))^(alpha-1)-d

    tau1 = tau2[:,t1]
    a2[1] = tau1[1]
    a02 = tau1[2]
    mincons = (1/(1+tau_c))*np.matmul(cfloor,ONESGRID)

    ssben = 0 
    for j2 in range(1,nep):
        for j3 in range(1,nh):

            zstar = z[j2,j3,t1]
            med = Mnorm[j3,t1]
            surv1 = surv[j2,j3,t1)]
            count = j3 +(j2-1)*nh
            Pr = TrE[count,:,t1]
            phimedicaid = phi[j3,t1,1]
            phipriv = phi[j3,t1,3]
            phigrid = phi[phimedicaid,1,phipriv,phipriv]

            g1 =1 

            # Valor esperado de escolher o medicaid 
            E1 = Pr[1]*V[:,1,1,t1+1,1,1]+Pr[2]*V[:,1,2,t1+1,1,1]

            # Valor esperado para a escolha de não seguro 
            E2 = Pr[1]*V[:,1,1,t1+1,1,2]+Pr[2]*V[:,1,2,t1+1,1,2];

            Ehold = np.transpose(np.array([E1,E2]))
            Ehold2 = np.tile(Ehold2, [1,1,nk,nl])
            E = np.random.permute(Ehold2,[4,2,3,1])

            income = rate.*KGRID2 + np.matmul((wage*zstar),LGRID)
            tincome = np.matmul(max(rate=1,0)) + np.matmul((wage*zstar),LGRID)

            tincome3 = np.squeeze(tincome(:,:,:,''))
            
            trfind - np.where(tincome3<=thresh)
            ntr = len(trfind)
            I1,I2,I3 = ind2sub(len(tincome3),trfind)
            tr1 = sub2ind(len(tincome),I1,I2,I3,2*np.ones(ntr,1))

            trfind - np.where(tincome3>thresh)
            ntr = len(trfind)
            I1,I2,I3 = ind2sub(len(tincome3),trfind)
            tr2 = sub2ind(len(tincome),I1,I2,I3,2*np.ones(ntr,1))
            
            # Dedução 
            for instype in range(1,4):
                phi1 = phigrid[instype]
                taxtdeduc = max(0,phi1*phi2*med - 0.075*tincome)
                ytax = max(ZEROGRID,tincome - taxdeduc)
                ytaxstargrid3 = np.matmul(a0,(ytax1 - (ytax1**(-a1) + np.matmul(a2[2],ONESGRID)**(-1/a1)))) + np.matmul(np.matmul(a02,max(rate-1,0)),KGRID2)
                ytaxstargrid2 = np.matmul(a0,(ytax1 - ((ytax1)**(-a1) + np.matmul(a2(1),ONESGRID)**(-1/a1))) + np.matmul((tss/2),min(ONESGRID.*ybar,ytax1)));

                # Criando o piso de consumo
                TRANSF = max(ZEROGRID, mincons - income + np.matmul((prem1 + phi1*phi2*med - ssben - beq),ONESGRID) + ytaxstargrid2 + ytaxstargrid3)
                
                cons1 = np.matmul((1/(1+tau_c)),(income - np.matmul((prem1 + phi1*phi2*med - ssben - beq),ONESGRID) - KGRID1 - ytaxstargrid2 - ytaxstargrid3 + TRANSF));

                tr3 = np.where(cons1<=0)

                Vtilde = np.matmul(np.matmul((1/(1-sigma4)),((cons1**chi3))),((ONESGRID-LGRID)**(1-chi3)))**(1-sigma4) + np.matmul(surv1*b),E)
                
                # Não assegurado - Medicaid 
                Vtilde(tr1) = -10e6     

                # Não segurando - não eligivel para o medicaid 
                Vtilde(tr2) = -10e6 

                # Não pode ter consumo negativo
                Vtilde(tr3) = -10e6

                Vtilde1 = np.random.permute(Vtilde,[1,2,4,3])

                V[:,j2,j3,t1,g1,instype] = max(max(max(Vtilde1)))
                [~, Iins] = max(max(max(Vtilde1)),[],3)

                Vtilde1 = np.random.permute(Vtilde,[1,4,2,3])               
                [~, Ik] = max(max(max(Vtilde1)),[],3)

                Vtilde1 = np.random.permute(Vtilde,[4,2,1,3])                
                [~, Il] = max(max(max(Vtilde1)),[],3)

                insopt[:,j2,j3,t1,g1,instype] = Iins
                kopt[,j2,j3,t1,g1,instype] = Ik
                labopt[:,j2,j3,t1,g1,instype] = lgrid[Il]

                Il = np.squeeze[Il]               
                Il = np.translate[Il]                
                Ik = np.squeeze[Ik]               
                Ik = np.translate[Ik]               
                Iins = np.squeeze[Iins]                
                Iins = np.translate[Iins];
                IND1 = sub2ind(size(cons1),Il,Ik,1:nk,Iins)

                copt[:,j2,j3,t1,g1,instype] = cons1[IND1]              
                transf[:,j2,j3,t1,g1,instype] = TRANSF[IND1]            
                ytax[:,j2,j3,t1,g1,instype] = ytaxstargrid2[IND1]
                ytaxst[:,j2,j3,t1,g1,instype] = ytaxstargrid3[IND1]

            g1=2 
            for instype in range(1,4):
                phi1 = phigrid[instype]
                taxtdeduc = max(0,phi1*phi2*med - 0.075*tincome)
                ytax = max(ZEROGRID,tincome - taxdeduc)
                ytaxstargrid3 = np.matmul(a0,(ytax1 - (ytax1**(-a1) + np.matmul(a2[2],ONESGRID)**(-1/a1)))) + np.matmul(np.matmul(a02,max(rate-1,0)),KGRID2)
                ytaxstargrid2 = np.matmul(a0,(ytax1 - ((ytax1)**(-a1) + np.matmul(a2(1),ONESGRID)**(-1/a1))) + np.matmul((tss/2),min(ONESGRID.*ybar,ytax1)));

                # Criando o piso de consumo
                TRANSF = max(ZEROGRID, mincons - income + np.matmul((prem1 + phi1*phi2*med - ssben - beq),ONESGRID) + ytaxstargrid2 + ytaxstargrid3)
                
                cons1 = np.matmul((1/(1+tau_c)),(income - np.matmul((prem1 + phi1*phi2*med - ssben - beq),ONESGRID) - KGRID1 - ytaxstargrid2 - ytaxstargrid3 + TRANSF));

                tr3 = np.where(cons1<=0)

                Vtilde = np.matmul(np.matmul((1/(1-sigma4)),((cons1**chi3))),((ONESGRID-LGRID)**(1-chi3)))**(1-sigma4) + np.matmul(surv1*b),E)
                
                # Não assegurado - Medicaid 
                Vtilde(tr1) = -10e6     

                # Não segurando - não eligivel para o medicaid 
                Vtilde(tr2) = -10e6 

                # Não pode ter consumo negativo
                Vtilde(tr3) = -10e6

                Vtilde1 = np.random.permute(Vtilde,[1,2,4,3])

                V[:,j2,j3,t1,g1,instype] = max(max(max(Vtilde1)))
                [~, Iins] = max(max(max(Vtilde1)),[],3)

                Vtilde1 = np.random.permute(Vtilde,[1,4,2,3])               
                [~, Ik] = max(max(max(Vtilde1)),[],3)

                Vtilde1 = np.random.permute(Vtilde,[4,2,1,3])                
                [~, Il] = max(max(max(Vtilde1)),[],3)

                insopt[:,j2,j3,t1,g1,instype] = Iins
                kopt[,j2,j3,t1,g1,instype] = Ik
                labopt[:,j2,j3,t1,g1,instype] = lgrid[Il]

                Il = np.squeeze[Il]               
                Il = np.translate[Il]                
                Ik = np.squeeze[Ik]               
                Ik = np.translate[Ik]               
                Iins = np.squeeze[Iins]                
                Iins = np.translate[Iins]
                IND1 = sub2ind(size(cons1),Il,Ik,1:nk,Iins)

                copt[:,j2,j3,t1,g1,instype] = cons1[IND1]              
                transf[:,j2,j3,t1,g1,instype] = TRANSF[IND1]            
                ytax[:,j2,j3,t1,g1,instype] = ytaxstargrid2[IND1]
                ytaxst[:,j2,j3,t1,g1,instype] = ytaxstargrid3[IND1]
    
    
    ZEROGRID = np.zeros(nl,nk,nk,4)
    ONESGRID = np.ones(nl,nk,nk,4)
    mincons = (1/(1+tau_c))*np.matmul(cfloor,ONESGRID)

    ssben = 0 

    for t1 in range(Tr-2,1,-1):
        wage = A*(1-alpha)*(rho[t1])**alpha
        rate = 1 + A*alpha**(rho(t1))^(alpha-1) - d
        a2 = tau2[:,t1]

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
                premgrid1 = [0,0,prempriv,(prempriv-ngprem)]
                [LGRID, KGRID1, KGRID2, PREMGRID] = ndgrid(lgrid,kgrid,kgrid,premgrid1)

                g1=1
            
                # Valor esperado da escolha do medicaid 
                EV11 = Opr[g1,1,t1]*(Pr[1]*V[:,1,1,t1+1,1,1]+Pr[2]*V[:,1,2,t1+1,1,1]+Pr[3]*V[:,2,1,t1+1,1,1]+Pr[4]*V[:,2,2,t1+1,1,1] +Pr[5]*V[:,3,1,t1+1,1,1]+Pr[6]*V[:,3,2,t1+1,1,1]+Pr[7]*V[:,4,1,t1+1,1,1]+Pr[8]*V[:,4,2,t1+1,1,1])
                EV21 = Opr[g1,2,t1]*(Pr[1]*V[:,1,1,t1+1,2,1]+Pr[2]*V[:,1,2,t1+1,2,1]+Pr[3]*V[:,2,1,t1+1,2,1]+Pr[4]*V[:,2,2,t1+1,2,1] +Pr[5]*V[:,3,1,t1+1,2,1]+Pr[6]*V[:,3,2,t1+1,2,1]+Pr[7]*V[:,4,1,t1+1,2,1]+Pr[8]*V[:,4,2,t1+1,2,1])
                E1 = EV11+EV21

                # Valor esperado da escolha de não assegurar
                EV12 = Opr[g1,1,t1]*(Pr[1]*V[:,1,1,t1+1,1,1]+Pr[2]*V[:,1,2,t1+1,1,1]+Pr[3]*V[:,2,1,t1+1,1,2]+Pr[4]*V[:,2,2,t1+1,1,2] +Pr[5]*V[:,3,1,t1+1,1,2]+Pr[6]*V[:,3,2,t1+1,1,2]+Pr[7]*V[:,4,1,t1+1,1,2]+Pr[8]*V[:,4,2,t1+1,1,2])
                EV22 = Opr[g1,2,t1]*(Pr[1]*V[:,1,1,t1+1,2,2]+Pr[2]*V[:,1,2,t1+1,2,2]+Pr[3]*V[:,2,1,t1+1,2,2]+Pr[4]*V[:,2,2,t1+1,2,2] +Pr[5]*V[:,3,1,t1+1,2,2]+Pr[6]*V[:,3,2,t1+1,2,2]+Pr[7]*V[:,4,1,t1+1,2,2]+Pr[8]*V[:,4,2,t1+1,2,2])
                E2 = EV12+EV22

                # Expected Value of Choosing Priv Insurance
                EV13 = Opr[g1,1,t1]*(Pr[1]*V[:,1,1,t1+1,1,3]+Pr[2]*V[:,1,2,t1+1,1,3]+Pr[3]*V[:,2,1,t1+1,1,3]+Pr[4]*V[:,2,2,t1+1,1,3] +Pr[5]*V[:,3,1,t1+1,1,3]+Pr[6]*V[:,3,2,t1+1,1,3]+Pr[7]*V[:,4,1,t1+1,1,3]+Pr[8]*V[:,4,2,t1+1,1,3])
                EV23 = Opr[g1,2,t1]*(Pr[1]*V[:,1,1,t1+1,2,3]+Pr[2]*V[:,1,2,t1+1,2,3]+Pr[3]*V[:,2,1,t1+1,2,3]+Pr[4]*V[:,2,2,t1+1,2,3] +Pr[5]*V[:,3,1,t1+1,2,3]+Pr[6]*V[:,3,2,t1+1,2,3]+Pr[7]*V[:,4,1,t1+1,2,3]+Pr[8]*V[:,4,2,t1+1,2,3])
                E3 = EV13+EV23

                # Expected Value of Choosing Group Insurance 
                E4 = np.matmul(-10e6,np.ones(nk,1))
                Ehold = [E1,E2,E3,E4]
                Ehold2 = np.tile(Ehold,[1,1,nk,nl])
                E = np.random.permute(Ehold2,[4,2,3,1])

                income = np.matmul(rate,KGRID2 + np.matmul((wage*zstar),LGRID))
                tincome = np.matmul(max(rate-1.0),KGRID2) + np.matmul((wage*zstar),LGRID)

                tincome3 = np.squeeze(tincome[:,:,:,1])
                trfind = np.where(tincome<=thresh)
                ntr = len(trfind)
                I1,I2,I3 = ind2sub(len(tincome3),trfind)
                tr1 =sub2ind(len(tincome),I1,I2,I3,2*np.ones(ntr,1))

                trfind = np.where(tincome>thresh)
                ntr = len(trfind)
                I1,I2,I3 = ind2sub(len(tincome3),trfind)
                tr2 = sub2ind(len(tincome),I1,I2,I3,1*np.where(ntr,1))

                for instype in range(1,4):
                    phi1 = phigrid[instype]
                    taxdeduc = max(0,phi1*med - .075*tincome)    # deduction
                    ytax1 = max(ZEROGRID,tincome - taxdeduc)                                        
                    ytaxstargrid3 = np.matmul(a0,(ytax1 - (ytax1**(-a1) + np.matmul(a2[2],ONESGRID)**(-1/a1))))
                    ytaxstargrid2 = np.matmul(a0,((ytax1) - ((ytax1)**(-a1) + np.matmul(a2[1],ONESGRID)**(-1/a1)))) + np.matmul((tss/2),min(np.matmul(ONESGRID,ybar),ytax1))

                    % Placing the consumption floor
                    TRANSF = max(ZEROGRID, mincons - income + PREMGRID + np.matmul((phi1*med - ssben - beq),ONESGRID) + ytaxstargrid2 + ytaxstargrid3)
                    cons1 = np.matmul((1/(1+tau_c)),(income - PREMGRID - np.matmul((phi1*med - ssben - beq),ONESGRID) - KGRID1 - ytaxstargrid2 - ytaxstargrid3 + TRANSF)) 
                    tr3 = find(cons1<=0)
                    
                    Vtilde = np.matmul(np.matmul(1/(1-sigma4),(cons1.^chi3)),ONESGRID-LGRID)**(1-chi3)))**(1-sigma4) + np.matmul((surv1*b),E)
                    Vtilde(tr1) = -10e6    # Medicaid = not uninsured                
                    Vtilde(tr2) = -10e6    # Uninsured = not medicaid eligible 
                    Vtilde(tr3) = -10e6
                    Vtilde1 = np.random.permute(Vtilde,[1,2,4,3])
                    
                    V[:,j2,j3,t1,g1,instype] = max(max(max(Vtilde1)))
                    Vhold1, Iins = max(max(max(Vtilde1)),[],3)
                    
                    Vtilde1 = np.random.permute(Vtilde,[1,4,2,3])
                    Vhold2, Ik = max(max(max(Vtilde1)),[],3)
                    
                    Vtilde1 = np.random.permute(Vtilde,[4,2,1,3])
                    Vhold3, Il = max(max(max(Vtilde1)),[],3)

                    insopt[:,j2,j3,t1,g1,instype] = Iins
                    kopt[:,j2,j3,t1,g1,instype] = Ik
                    labopt[:,j2,j3,t1,g1,instype] = lgrid(Il)
                    
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
                
                g=2

                # Expected Value of Choosing Medicaid
                EV11 = Opr[g1,1,t1]*(Pr[1]*V[:,1,1,t1+1,1,1]+Pr[2]*V[:,1,2,t1+1,1,1]+Pr[3]*V[:,2,1,t1+1,1,1]+Pr[4]*V[:,2,2,t1+1,1,1] +Pr[5]*V[:,3,1,t1+1,1,1]+Pr[6]*V[:,3,2,t1+1,1,1]+Pr[7]*V[:,4,1,t1+1,1,1]+Pr[8]*V[:,4,2,t1+1,1,1])
                EV21 = Opr[g1,2,t1]*(Pr[1]*V[:,1,1,t1+1,2,1]+Pr[2]*V[:,1,2,t1+1,2,1]+Pr[3]*V[:,2,1,t1+1,2,1]+Pr[4]*V[:,2,2,t1+1,2,1] +Pr[5]*V[:,3,1,t1+1,2,1]+Pr[6]*V[:,3,2,t1+1,2,1]+Pr[7]*V[:,4,1,t1+1,2,1]+Pr[8]*V[:,4,2,t1+1,2,1])
                E1 = EV11+EV21

                # Expected Value of Choosing Uninsurance
                EV12 = Opr[g1,1,t1]*(Pr[1]*V[:,1,1,t1+1,1,1]+Pr[2]*V[:,1,2,t1+1,1,1]+Pr[3]*V[:,2,1,t1+1,1,2]+Pr[4]*V[:,2,2,t1+1,1,2] +Pr[5]*V[:,3,1,t1+1,1,2]+Pr[6]*V[:,3,2,t1+1,1,2]+Pr[7]*V[:,4,1,t1+1,1,2]+Pr[8]*V[:,4,2,t1+1,1,2])
                EV22 = Opr[g1,2,t1]*(Pr[1]*V[:,1,1,t1+1,2,2]+Pr[2]*V[:,1,2,t1+1,2,2]+Pr[3]*V[:,2,1,t1+1,2,2]+Pr[4]*V[:,2,2,t1+1,2,2] +Pr[5]*V[:,3,1,t1+1,2,2]+Pr[6]*V[:,3,2,t1+1,2,2]+Pr[7]*V[:,4,1,t1+1,2,2]+Pr[8]*V[:,4,2,t1+1,2,2])
                E2 = EV12+EV22

                # Expected Value of Choosing Priv Insurance
                EV13 = Opr[g1,1,t1]*(Pr[1]*V[:,1,1,t1+1,1,3]+Pr[2]*V[:,1,2,t1+1,1,3]+Pr[3]*V[:,2,1,t1+1,1,3]+Pr[4]*V[:,2,2,t1+1,1,3] +Pr[5]*V[:,3,1,t1+1,1,3]+Pr[6]*V[:,3,2,t1+1,1,3]+Pr[7]*V[:,4,1,t1+1,1,3]+Pr[8]*V[:,4,2,t1+1,1,3])
                EV23 = Opr[g1,2,t1]*(Pr[1]*V[:,1,1,t1+1,2,3]+Pr[2]*V[:,1,2,t1+1,2,3]+Pr[3]*V[:,2,1,t1+1,2,3]+Pr[4]*V[:,2,2,t1+1,2,3] +Pr[5]*V[:,3,1,t1+1,2,3]+Pr[6]*V[:,3,2,t1+1,2,3]+Pr[7]*V[:,4,1,t1+1,2,3]+Pr[8]*V[:,4,2,t1+1,2,3])
                E3 = EV13+EV23
            
                #private in last period

                # Expected Value of Choosing Group Insurance
                EV14 = Opr[g1,1,t1]*(Pr[1]*V[:,1,1,t1+1,1,4]+Pr[2]*V[:,1,2,t1+1,1,4]+Pr[3]*V[:,2,1,t1+1,1,4]+Pr[4]*V[:,2,2,t1+1,1,4] +Pr[5]*V[:,3,1,t1+1,1,4]+Pr[6]*V[:,3,2,t1+1,1,4]+Pr[7]*V[:,4,1,t1+1,1,4]+Pr[8]*V[:,4,2,t1+1,1,4])
                EV24 = Opr[g1,2,t1]*(Pr[1]*V[:,1,1,t1+1,2,4]+Pr[2]*V[:,1,2,t1+1,2,4]+Pr[3]*V[:,2,1,t1+1,2,4]+Pr[4]*V[:,2,2,t1+1,2,4] +Pr[5]*V[:,3,1,t1+1,2,4]+Pr[6]*V[:,3,2,t1+1,2,4]+Pr[7]*V[:,4,1,t1+1,2,4]+Pr[8]*V[:,4,2,t1+1,2,4])
                E4 = EV14+EV24

                #group in last period

                Ehold = np.translate([E1 E2 E3 E4])            
                Ehold2 = np.tite(Ehold,[1,1,nk,nl])    # replicates "E" (nk,nl) times            
                E = np,random.permute(Ehold2,[4,2,3,1])    
                
                
                for instype in range(1,4):
                    phi1 = phigrid[instype]
                    taxdeduc = max(0,phi1*med - .075*tincome) + max(instype-3,0)*premgrid1(4);    % deduction, including the ESHI premium
                    ytax1 = max(ZEROGRID,tincome - taxdeduc);                                        
                    ytaxstargrid3 = np.matmul(a0,(ytax1 - (ytax1**(-a1) + np.matmul(a2(2),ONESGRID)**(-1/a1))))
                    ytaxstargrid2 = np.matmul(a0,((ytax1) - ((ytax1)**(-a1) + np.matmul(a2[1],ONESGRID)**(-1/a1)))) + np.matmul((tss/2),min(np.matmul(ONESGRID,ybar),ytax1))

                    % Placing the consumption floor
                    TRANSF = max(ZEROGRID, mincons - income + PREMGRID + (phi1*med - ssben - beq).*ONESGRID + ytaxstargrid2 + ytaxstargrid3);

                    cons1 = (1/(1+tau_c)).*(income - PREMGRID - (phi1*med - ssben - beq).*ONESGRID - KGRID1 - ytaxstargrid2 - ytaxstargrid3 + TRANSF); 
                    
                    tr3 = find(cons1<=0)
                    
                    Vtilde = (1/(1-sigma4)).*((cons1.^chi3).*((ONESGRID-LGRID).^(1-chi3))).^(1-sigma4) + (surv1*b).*E
                    
                    Vtilde(tr1) = -10e6;    # Medicaid = not uninsured                
                    Vtilde(tr2) = -10e6;    # Uninsured = not medicaided
                    Vtilde(tr3) = -10e6

                    Vtilde1 = permute(Vtilde,[1,2,4,3])
                    
                    V(:,j2,j3,t1,g1,instype) = max(max(max(Vtilde1)))
                    [Vhold1, Iins] = max(max(max(Vtilde1)),[],3)
                    
                    Vtilde1 = permute(Vtilde,[1,4,2,3]);               
                    [Vhold2, Ik] = max(max(max(Vtilde1)),[],3)
                    
                    Vtilde1 = permute(Vtilde,[4,2,1,3]);                
                    [Vhold3, Il] = max(max(max(Vtilde1)),[],3)

                    insopt[:,j2,j3,t1,g1,instype] = Iins
                    kopt[:,j2,j3,t1,g1,instype] = Ik
                    labopt[:,j2,j3,t1,g1,instype] = lgrid[Il]
                    
                    Il = np.squeeze(Il)               
                    Il = np.transpose(Il)                
                    Ik = np.squeeze(Ik)               
                    Ik = np.transpose(Ik)               
                    Iins = np.squeeze(Iins)                
                    Iins = np.transpose(Iins)
                    IND1 = sub2ind(len(cons1),Il,Ik,1:nk,Iins)
                    
                    copt[:,j2,j3,t1,g1,instype] = cons1[IND1]              
                    transf[:,j2,j3,t1,g1,instype] = TRANSF[IND1]                
                    ytax[:,j2,j3,t1,g1,instype] = ytaxstargrid2[IND1]
                    ytaxst[:,j2,j3,t1,g1,instype] = ytaxstargrid3[IND1]

    return V,labopt, kopt, copt, ytax, transf, insopt, ytaxst 