#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 15:06:23 2022

@author: fang
"""

from math import log
import numpy as np
import anndata as ad 
from two_species import get_Y

def simulate_data(topo, tau, n, p, loga_max=4, logb_max=2, random_seed=42, loomfilepath=None):
    np.random.seed(random_seed)
    L=len(topo)
    n_states=len(set(topo.flatten()))
    t=np.linspace(tau[0],tau[-1],n)
    true_t = []
    
    theta=np.zeros((p,n_states+4))
    for j in range(n_states+2):
        theta[:,j]=np.exp(np.random.uniform(0,loga_max,size=p))-1
    theta[:,-2]=np.exp(np.random.uniform(0,logb_max,size=p))
    theta[:,-1]=np.exp(np.random.uniform(0,logb_max,size=p))
    
    Y = np.zeros((n*L,p,2))
    for l in range(L):
        theta_l = np.concatenate((theta[:,topo[l]], theta[:,-4:]), axis=1)
        Y[l*n:(l+1)*n] = get_Y(theta_l,t,tau) # m*p*2
        true_t = np.append(true_t,t)

    X = np.random.poisson(Y)
    
    if loomfilepath is not None:
        adata=ad.AnnData(np.sum(X,axis=-1))
        adata.layers["spliced"] = X[:,:,1]
        adata.layers["unspliced"] = X[:,:,0]
        adata.layers["ambiguous"]=np.zeros_like(X[:,:,0])
        adata.obs["time"]=true_t
        adata.obs["celltype"]=np.arange(n*L)//n
        adata.uns["theta"]=theta
        adata.var["true_beta"]=theta[:,-2]
        adata.var["true_gamma"]=theta[:,-1]
        adata.write_loom(loomfilepath)
    return theta, true_t, Y, X,

# Simulations

def gillvec_burst_inhom(k,tvec,tau,bvec,S,nCells,propfun,burstfun):
    n_species = S.shape[1]

    num_t_pts = len(tvec)
    X_mesh = np.zeros((nCells,num_t_pts,n_species),dtype=int) #change to float if storing floats!!!!!!! 

    t = np.zeros(nCells,dtype=float)
    tindex = np.zeros(nCells,dtype=int)

    #initialize state: integer unspliced, integer spliced 
    X = np.zeros((nCells,n_species),dtype=int)

    #initialize list of cells that are being simulated
    activecells = np.ones(nCells,dtype=bool)
    while any(activecells):
        mu = np.zeros(nCells,dtype=int)
        n_active_cells = np.sum(activecells)        
        (dt,mu_upd) = rxn_calculator(X[activecells,:],k,propfun)

        t[activecells] += dt
        mu[activecells] = mu_upd

        update = np.zeros(nCells,dtype=bool)
        update[activecells] = t[activecells] > tvec[tindex[activecells]]
        while np.any(update):
            X_mesh[update,tindex[update],:] = X[update]
            tindex += update
            ended_in_update = (tindex==num_t_pts) #less efficient
            if np.any(ended_in_update):
                activecells[ended_in_update] = False
                mu[ended_in_update] = 0
                if not np.any(activecells):
                    print('end simulation')
                    break
            update = np.zeros(nCells,dtype=bool)
            update[activecells] = t[activecells]>tvec[tindex[activecells]]
        
        burst = (mu == 1) & activecells
        bs = burstfun(tau,bvec,t[burst])
        # print(bs.shape)
        bsrand = (np.random.geometric(1/(1+bs))-1 )
        # print(bsrand.shape)
        X[burst] += bsrand[:,None]
        X[~burst] += S[mu[~burst]-1]
    return X_mesh

def rxn_calculator(X,k,propfun):
    nRxn = len(k)
    nCells = X.shape[0]

    a = np.zeros((nCells,nRxn),dtype=float)
 ################################
    a = propfun(a, k, X)
#################################
    a0 = np.sum(a,1)
    dt = np.log(1./np.random.rand(nCells)) / a0
    r2ao = a0 * np.random.rand(nCells)
    mu = np.sum(np.matlib.repmat(r2ao,nRxn+1,1).T >= np.cumsum(np.matlib.hstack((np.zeros((nCells,1)),a)),1) ,1)
    return (dt,mu)

def propfun_bursty(a,k,x):
    # a shape (nCells,nRxn)
    # x shape (nCells,nSpecies)
    k1,beta,gamma = k
    x = x.T
    a = a.T
    #######
    #fill in this part
    a[0] = k1
    a[1] = beta * x[0]
    a[2] = gamma * x[1]
    #######
    a = a.T
    return a

def Gillespie_bursty_2D_single(ts, te, x0, kvec, tau, beta, gamma, bvec, random_seed = 42):
    """
    Gillespie algorithm for the system:
        null -> X1: production rate
        X1 -> X2 : beta
        X2 -> null: degradation rate gamma
     
    Parameters
    ----------
    ts : float
        Start time 
    te : float
        End time 
    x0 : 1D array
        Initial value
    beta : float
        Splicing rate
    gamma: float
        Degradation rate 
    bs : floats
        mean burst size
    random_seed : int
        set numpy.random.seed
        
    Returns
    -------
    T : ndarray
        Time points  
    X : ndarray
        The value of x at time points in T

    """
    # np.random.seed is global
    if isinstance(random_seed, int):
        np.random.seed(random_seed)

    #np.random.seed()    
    t=ts
    x=x0.copy() #system state
    T=[]
    X=[]

    while t<te:

        T.append(t)
        X.append(x.copy()) # have to use copy!!! 
        r1,r2=np.random.uniform(0,1,2) 
        
        # calculate propensity functions
        state = np.sum(np.array(tau)<=t)-1
        k = kvec[state]
        b = bvec[state]
        a_cumsum=np.array([k, beta*x[0]+k, gamma*x[1]+beta*x[0]+k])
        wait_time=-log(r1)/a_cumsum[-1]
            
        t=t+wait_time
        
        a_normalized=a_cumsum/a_cumsum[-1]
        if r2<= a_normalized[0]:
            x[0]=x[0]+np.random.geometric(1/(b+1))-1
        elif r2<= a_normalized[1]:
            x=x+np.array([-1,1])
        else:
            x=x+np.array([0,-1])

    T.append(t)
    X.append(x)

    return np.array(T), np.array(X)

def Gillespie_bursty_2D(ncell, tvec, x0, kvec, tau, beta, gamma, bvec, random_seed = None):
    m = len(tvec)
    X = np.zeros((ncell,m,2))
    for i in range(ncell):
        T, x = Gillespie_bursty_2D_single(tvec[0], tvec[-1], x0, kvec, tau, beta, gamma, bvec, random_seed)
        k=0
        for j in range(m):
            while T[k]<=tvec[j]:
                k=k+1
            X[i,j,:]=x[k-1]
    return X