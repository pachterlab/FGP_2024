#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 21:20:50 2022

@author: Meichen Fang
"""

"""
Outline of traj_EM:
    n cells, p genes, m time grids, L lineages
    Given Q(z) which is a n*L*m array, compute optimal theta_j for each gene j:
        compute loss which is negative log likelihood + Regularization 
        negative log likelihood is a function of theta_j and Q(z)
        
"""
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.special import softmax
import matplotlib.pyplot as plt

# global parameters: upper and lower limits for numerical stability
eps = 1e-6
omega = -1e6

def get_Y(theta, t, tau):
    # theta: p*(K+4)
    # t: len m
    # tau: len K+1
    # return m * p * 2
    p = len(theta)
    K = len(tau)-1 # number of states
    if np.shape(theta)[1]!=K+4:
        print(np.shape(theta)[1], K+4)
        raise TypeError("wrong parameters lengths")
    a = theta[:,0:K]
    beta = theta[:,-2]
    gamma = theta[:,-1]

    y1_0 = theta[:,-4]
    y2_0 = theta[:,-3]

    c = beta/(beta-gamma+eps)
    d = beta**2/((beta-gamma)*gamma+eps)
    y_0 = y2_0 + c*y1_0
    a_ = d[:,None]*a
    t = t.reshape(-1,1)
    m = len(t)
  
    I = np.ones((K+1,m),dtype=bool)
    y1 =np.zeros((m,p))
    y =np.zeros((m,p))

    # nascent
    y1=y1+y1_0[None,:]*np.exp(-beta[None,:]*t)   
    for k in range(1,K+1):
        I[k] = np.squeeze(t > tau[k])
        idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
        y1 = y1 + a[None,:,k-1] * (np.exp(- beta[None,:]*I[k,:,None] *(t-tau[k]))- np.exp(-beta[None,:]*I[k,:,None] * (t-tau[k-1])) ) \
          + a[None,:,k-1] * (1 - np.exp(- beta[None,:]*idx[:,None] *(t-tau[k-1]))) 
    
    if np.sum(np.isnan(y1)) != 0:
        raise ValueError("Nan in y1")
    # mature + c * nascent 
    y=y+y_0[None,:]*np.exp(-gamma[None,:]*t)    
    for k in range(1,K+1):
        idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
        y = y + a_[None,:,k-1] * (np.exp(-gamma[None,:]*I[k,:,None] * (t-tau[k]))- np.exp(-gamma[None,:]*I[k,:,None] * (t-tau[k-1])) ) \
          +  a_[None,:,k-1] * (1 - np.exp(-gamma[None,:]*idx[:,None]*(t-tau[k-1]))) 

    Y =np.zeros((m,p,2))
    Y[:,:,0] = y1
    Y[:,:,1] = y-c*y1
    
    Y[Y<0]=0
    
    if np.sum(np.isnan(Y)) != 0:
        raise ValueError("Nan in Y")
    return Y

def get_Y2(theta, t, tau):
    # theta: p*(K+4)
    # t: len m
    # tau: len K+1
    # return m * p * 2
    p = len(theta)
    K = len(tau)-1 # number of states
    if np.shape(theta)[1]!=K+4:
        raise TypeError("wrong parameters lengths")
    a = theta[:,0:K]
    beta = theta[:,-2]
    gamma = theta[:,-1]
    y1_0 = theta[:,-4]
    y2_0 = theta[:,-3]
    t = t.reshape(-1,1)
    m = len(t)
    I = np.ones((K+1,m),dtype=bool)
    
    # nascent
    y1=np.zeros((m,p))
    y1=y1+y1_0[None,:]*np.exp(-beta[None,:]*t)   
    for k in range(1,K+1):
        I[k] = np.squeeze(t > tau[k])
        idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
        y1[I[k]] = y1[I[k]] + a[None,:,k-1] * (np.exp(- beta[None,:]*(t[I[k]]-tau[k]))- np.exp(-beta[None,:]* (t[I[k]]-tau[k-1])) ) 
        y1[idx] = y1[idx] + a[None,:,k-1] * (1 - np.exp(- beta[None,:] *(t[idx]-tau[k-1]))) 

    if np.sum(np.isnan(y1)) != 0:
        raise ValueError("Nan in y1")
    Y =np.zeros((m,p,2))
    Y[:,:,0] = y1
    
    # mature + c * nascent 
    ## nondegenrate cases
    nondegenerate = np.abs(beta-gamma)>eps # indices of genes having similar gamma and beta
    if np.sum(nondegenerate)>0:
        y =np.zeros((m,np.sum(nondegenerate)))
        beta_, gamma_ = beta[nondegenerate], gamma[nondegenerate]
        c = beta_/(beta_-gamma_)
        d = beta_**2/((beta_-gamma_)*gamma_)
        y_0 = y2_0[nondegenerate] + c*y1_0[nondegenerate]
        a_ = d[:,None]*a[nondegenerate,:]

        y=y+y_0[None,:]*np.exp(-gamma_[None,:]*t)    
        for k in range(1,K+1):
            idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
            y[I[k]] = y[I[k]] + a_[None,:,k-1] * (np.exp(-gamma_[None,:] * (t[I[k]]-tau[k]))- np.exp(-gamma_[None,:]*(t[I[k]]-tau[k-1])) )
            y[idx] = y[idx] +  a_[None,:,k-1] * (1 - np.exp(-gamma_[None,:]*(t[idx]-tau[k-1]))) 

        if np.sum(np.isnan(y)) != 0:
            raise ValueError("Nan in y")

        Y[:,nondegenerate,1] = y-c*y1[:,nondegenerate]
    
    ## nondegenrate cases
    degenerate = ~nondegenerate
    if np.sum(degenerate)>0:
        y = np.zeros((m,np.sum(degenerate)))
        y = y + y1_0[None,degenerate]*beta[None,degenerate]*t*np.exp(-beta[None,degenerate]*t) + y2_0[None,degenerate]*np.exp(-gamma[None,degenerate]*t) 
        for k in range(1,K+1):
            idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k with a[k-1]
            y[I[k]] = y[I[k]] + a[None,degenerate,k-1]*beta[None,degenerate]*((t[I[k]]-tau[k])*(np.exp(- beta[None,degenerate]*(t[I[k]]-tau[k])) - np.exp(- beta[None,degenerate]*(t[I[k]]-tau[k-1]))))    
            y[I[k]] = y[I[k]] + a[None,degenerate,k-1] * (np.exp(- beta[None,degenerate]*(t[I[k]]-tau[k])) - np.exp(- beta[None,degenerate] *(t[I[k]]-tau[k-1])) \
                                             - beta[None,degenerate] *(tau[k]-tau[k-1])*np.exp(- beta[None,degenerate] *(t[I[k]]-tau[k-1])))
            y[idx] = y[idx] + a[None,degenerate,k-1] * (1 - np.exp(-beta[None,degenerate]*(t[idx]-tau[k-1]))\
                                             - beta[None,degenerate] *(t[idx]-tau[k-1])*np.exp(- beta[None,degenerate]*(t[idx]-tau[k-1]))) 
        Y[:,degenerate,1] = y
    
    if np.sum(np.isnan(Y)) != 0:
        raise ValueError("Nan in Y")
    return Y


def neglogL(theta,x,Q,t,tau,topo,penalty):
    # theta: length K+4
    # x: n*2
    # Q: n*L*m
    # t: len m
    # tau: len K+1
    logL = 0
    K = len(tau) - 1
    for l in range(len(topo)):
        theta_l = np.concatenate((theta[topo[l]], theta[-4:]))
        Y = get_Y(theta_l[None,:],t,tau) # m*1*2
        Y = Y[None,:,0,:] # 1*m*2
        weight_l = Q[:,l,:,None] #n*m*1
        logL += np.sum( weight_l * (x[:,None,:] * np.log(eps + Y) - Y ) )

    loss = - logL    

    ## L1 Regularization
    for l in range(len(topo)): 
        loss += penalty * np.abs(theta[topo[l][0]]-theta[-4]) 
        for k in range(1,len(topo[l])):
            loss += penalty * np.abs(theta[topo[l][k]]-theta[topo[l][k-1]]) 
    loss += penalty * np.abs(theta[-3]*theta[-1]/theta[-2]-theta[-4]) 

    return loss

  
def neglogL_bg(theta_bg,theta_a,x,Q,t,tau,topo,penalty):
    theta = np.append(theta_a, theta_bg)
    loss = neglogL(theta,x,Q,t,tau,topo,penalty)
    return loss

def neglogL_a(theta_a,theta_bg,x,Q,t,tau,topo,penalty):
    theta = np.append(theta_a, theta_bg)
    loss = neglogL(theta,x,Q,t,tau,topo,penalty)
    return loss

def minimize_wrapper(theta0, x, Q, t, tau, topo, penalty, alternative, bnd, miter = 100000, alteriter = 20):
    bounds = [[0,bnd]]*len(theta0)
    bounds[-2:] = [[1/bnd,bnd]]*2
    if alternative:
        new_theta = theta0.copy()
        for i in range(alteriter):
            ## minimize wrt a
            res1 = minimize(neglogL_a, new_theta[:-2], args=(new_theta[-2:],x,Q,t,tau,topo,penalty), bounds = bounds[:-2], options={'maxiter': int(miter/alteriter/2),'disp': False}) 
            new_theta[:-2]=res1.x
            ## minimize wrt beta and gamma
            res2 = minimize(neglogL_bg, new_theta[-2:], args=(new_theta[:-2],x,Q,t,tau,topo,penalty), bounds = bounds[-2:], options={'maxiter': int(miter/alteriter/2),'disp': False}) 
            new_theta[-2:]=res2.x
    else:
        res = minimize(neglogL, theta0, args=(x,Q,t,tau,topo,penalty), bounds=bounds, options={'maxiter': miter,'disp': False}) 
        new_theta = res.x
    return new_theta

def update_theta(X,weight,theta_G,penalty=0,alternative=False,parallel=False,n_threads=1,theta0=None, bnd=1000):
    """
    beta and gamma can not be equal
    """
    
    if type(theta_G) is dict:
        n,L,m = np.shape(weight)
        Q = weight.copy()
        tau = theta_G["tau"]
        topo = theta_G["topo"]
        
    else:       
        n,m = np.shape(weight)
        Q = weight.copy()
        Q = Q[:,None,:]
        tau = theta_G
        topo = np.array([np.arange(len(tau)-1)])
        
    n,p,s=np.shape(X)
    if s!=2:
      raise TypeError("wrong parameters lengths")
    
    t=np.linspace(0,1,m)
    n_states=len(set(topo.flatten()))

    if theta0 is None:
        theta0 = np.ones((p,n_states+4))
        theta0[:,0:-3]=np.mean(X[:,:,0],axis=0)[:,None]
        theta0[:,-3]=np.mean(X[:,:,1],axis=0)
        theta0[:,-1] = np.mean(X[:,:,0],axis=0)/(np.mean(X[:,:,1],axis=0)+eps)

    if parallel is True:
        Input_args = []
        for i in range(p):
            Input_args.append((theta0[i], X[:,i], Q, t, tau, topo, penalty, alternative, bnd))
        with Pool(n_threads) as pool:      
            theta_hat = pool.starmap(minimize_wrapper, Input_args)
        theta_hat = np.array(theta_hat)
    else:
        theta_hat = np.zeros((p,n_states+4))
        for i in range(p): 
            theta_hat[i]=minimize_wrapper(theta0[i], X[:,i], Q, t, tau, topo, penalty, alternative, bnd)
    return theta_hat

def update_weight(X,theta,theta_G,m):
    n,p,s=np.shape(X)
    t=np.linspace(0,1,m)
    if type(theta_G) is not dict:
        tau = theta_G
        Y = get_Y(theta,t,tau) # m*p*2
        logL = np.sum(X[:,None,:,:] * np.log(Y[None,:]+eps) - Y[None,:], axis=(-2,-1)) # n*m*p*2 -> n*m
        Q = softmax(logL, axis=(-1))
    else:
        tau = theta_G["tau"]
        topo = theta_G["topo"]
        Y = np.zeros((L,m,p,2))
        for l in range(L):
            theta_l = np.concatenate((theta[:,topo[l]], theta[:,-4:]), axis=1)
            Y[l] = get_Y(theta_l,t,tau) # m*p*2
        logL =  np.sum(X[:,None,None,:,:] * np.log(Y[None,:]+eps) - Y[None,:], axis=(-2,-1)) # n*L*m*p*2 -> n*L*m
        Q = softmax(logL, axis=(-2,-1))
    
    """
    Q = np.zeros((n,L,m))
    for i in range(n):
        c = np.max(logL[i])
        relative_logL = logL[i]-c        
        L = np.exp(relative_logL)
        Q[i] = L/L.sum()
    """
    
    if np.sum(np.isnan(Q)) != 0:
        raise ValueError("Nan in weight")
    return Q

def traj_EM(X, theta_G, weight0, relative_penalty=0, epoch=20, alternative=False, parallel=False, n_threads=1, bnd=1000):
    """
    X: n cells * p genes
    m grid of t=[0,1]
    """
    eps = 10**(-10)
    n,p,_=np.shape(X)
    penalty=relative_penalty*n
    
    if type(theta_G) is dict:
        n,L,m = np.shape(weight0)
        weight = weight0.copy()
        tau = theta_G["tau"]
        topo = theta_G["topo"]
        
    else:       
        n,m = np.shape(weight0)
        weight = weight0.copy()
        weight = weight[:,None,:]
        tau = theta_G
        topo = np.array([np.arange(len(tau)-1)])
        
    K=len(tau)-1
    theta_hat = np.ones((p,K+4))
    theta_hat[:,0:K+2]=np.mean(X[:,:,0],axis=0)[:,None]
    theta_hat[:,-1] = np.mean(X[:,:,0],axis=0)/(np.mean(X[:,:,1],axis=0)+eps)
    theta_hist=[] 
    weight_hist=[]
    theta_hist.append(theta_hat.copy())
    weight_hist.append(weight.copy())
    
    alternative = False
    for i in tqdm(range(epoch)):
        if i>0:
            alternative = False
        theta_hat = update_theta(X,weight,tau,topo,penalty,alternative,theta0=theta_hat,parallel=parallel,n_threads=n_threads,bnd = bnd)
        weight = update_weight(X,theta_hat,tau,m)
        theta_hist.append(theta_hat.copy())
        weight_hist.append(weight.copy())
    return theta_hist, weight_hist

