#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 15:06:23 2022

@author: fang
"""

from math import log
import numpy as np
import anndata as ad 
from importlib import import_module
import Chronocell.models.two_species_ss as two_species_ss
import Chronocell.models.two_species_ss_tau as two_species_ss_tau

def simulate_data(topo, tau, n, p, model="two_species_ss", loga_mu=2, loga_sd=1, logb_mu=1, logb_sd=0.5, rd_mu=0.25, rd_var=0, logu_mu=0, logu_sd=0, random_seed=42):    
    np.random.seed(random_seed)
    L=len(topo)
    K=len(tau)-1
    n_states=len(set(topo.flatten()))
    true_t = []
    t_samples=np.random.uniform(tau[0],tau[-1],size=100*n)
    
    if model == "two_species_ss":
        theta=np.zeros((p,n_states+2))
        for j in range(n_states):
            theta[:,j]=np.random.lognormal(loga_mu, loga_sd,size=p)
        theta[:,-2:]=np.random.lognormal(logb_mu,logb_sd,size=(p,2))
        theta[:,:n_states]/=theta[:,-2,None]
        theta[:,-1]/=np.exp(2)
        
        Y = np.zeros((n*L,p,2))
        for l in range(L):
            theta_l = np.concatenate((theta[:,topo[l]], theta[:,-2:]), axis=1)
            t = np.sort(np.random.choice(t_samples,size=n))
            Y[l*n:(l+1)*n] = two_species_ss.get_Y(theta_l,t,tau) # m*p*2
            true_t = np.append(true_t,t)
            
    elif model == "two_species_ss_tau":
        theta=np.zeros((p,n_states+2+K))
        for j in range(n_states):
            theta[:,j]=np.random.lognormal(loga_mu, loga_sd,size=p)
        theta[:,-2:]=np.random.lognormal(logb_mu,logb_sd,size=(p,2))
        theta[:,:n_states]/=theta[:,-2,None]
        theta[:,-1]/=np.exp(2)
        theta[:,n_states:-2]=np.abs(tau[None,:-1]+np.random.uniform(0,0.5,size=(p,K)))
        
        Y = np.zeros((n*L,p,2))
        for l in range(L):
            theta_l = np.concatenate((theta[:,topo[l]], theta[:,n_states:]), axis=1)
            t = np.sort(np.random.choice(t_samples,size=n))
            Y[l*n:(l+1)*n] = two_species_ss_tau.get_Y(theta_l,t,tau) # m*p*2
            true_t = np.append(true_t,t)
    else:
        raise ValueError('model not implemented')
        
    if logu_sd != 0:
        Ubias = np.random.lognormal(logu_mu,logu_sd,p)
        Y[:,:,0] *= Ubias[None,:]
    else:
        Ubias = np.ones(p)*np.exp(logu_mu)
        
    if rd_var != 0:
        a = (1-rd_mu)/rd_var - rd_mu
        b = (1/rd_mu-1)*a
        rd = np.random.beta(a=a, b=b, size=n*L)             
    else:
        rd = np.ones(n*L) * rd_mu 
        
    Y *= rd[:,None,None]          
    X = np.random.poisson(Y)  

    return theta, true_t, Y, X, Ubias, rd

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