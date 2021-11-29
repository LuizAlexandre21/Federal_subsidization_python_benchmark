# Biblioteca 
import numpy as np 
import scipy as sc 
from scipy.io import loadmat 

''' produz parâmetros do agente para as funções de distribuição'''
# Parametros dos agentes no estado estacionário

def agent_params(val):
    # beq=beqapprox
    
    # O Premio para o não grupo HI
    ngprem = 0. 

    # Premio de Segurança 
    gamma = 1.11
    b = .9875

    # Parâmetros para utilidade não separável
    chi3 = .3
    sigma4 = 4

    # Importando dados externos 
    insdata4 = loadmat('/home/alexandre/Documents/Mestrado/Artigos de dissertação/Replication Files/Benchmark Economy/insdata4.mat')
    wage_med_data = loadmat('/home/alexandre/Documents/Mestrado/Artigos de dissertação/Replication Files/Benchmark Economy/wage_med_data1.mat')

    # Parametros 
    T = 80 
    Tr = 45 # Idade de aposentadoria
    alpha = 0.36 # Capital share 
    A=1 #tfp 
    phimedicare = 0.5
    d = 0.083
    nh = 2  # Numero de estados de saúde - bom ou mal  
    a0 = 0.258 
    a1 = 0.768 # parâmetro de função tributária 1
    ss = 0.27 # 4.5 do Pib 
    tau_c = 0.57 # Taxa de consumo 
    premmedicare = 0.026 
    cfloor = 0.08 
    pm = 0.00003625 
    Mnorm = np.matmul(pm,M)
    tss =.124*2
    tm = .029
    ybar = 3.65

    # Capital individual 
    nk = 50 
    kgrid = np.zeros(nk,1)
    kub = 13
    klb = 0.001
    exppwr = 3 
    
    for i1 in range(1,nk):
        kgrid[i1] = (kub-klb)*((i1/nk))**exppwr + klb
    kgrid =[klb,kgrid]
    nk = len(kgrid)

    # Grid de Labor 
    nl = 25 
    lub = 0.65 
    llb =0 
    lgrid =np.zeros(nl,1)
    linc = (lub-llb)/(nl-1)
    lgrid[1] = llb 
    for i1 in range(2,nl):
        lgrid[i1] = lgrid[i1-1] +linc 

    ng = 2  #status de oferta de grupo
    nins = 4


    return b 