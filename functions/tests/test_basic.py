#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 11:26:08 2022

@author: fang
"""

import numpy as np
from scipy.optimize import minimize


# global parameters: upper and lower limits for numerical stability
eps = 1e-10

def get_y(theta, t, tau):
    # theta: (K+4), delta_1, ..., delta_K, u_0, eta = zeta*s_0 - u_0, beta, zeta
    # t: len m
    # tau: len K+1
    # return m * 2
    K = len(tau)-1 # number of states
    if len(theta)!=K+4:
        raise TypeError("wrong parameters lengths")
    delta = theta[0:K]
    u_0 = theta[-4]
    eta = theta[-3] # eta = s_0 - u_0
    beta = theta[-2]
    gamma = theta[-1]
    zeta = gamma / beta
    if zeta == 1:
        zeta = 1 + eps
    c = 1/(1-zeta)
    d = c-1 
    m = len(t)
  
    I = np.ones((K+1,m),dtype=bool)

    # nascent
    y1=u_0
    for k in range(K):
        I[k] = np.squeeze(t > tau[k])
        y1 = y1 + delta[k] * (1- np.exp(-(I[k]*(t-tau[k]))*beta )) 
    
    if np.sum(np.isnan(y1)) != 0:
        raise ValueError("Nan in y1")
        
    # mature * zeta
    y2 = u_0 + eta * np.exp(-t*gamma)
    for k in range(K):
        y2 = y2 + delta[k] * (1- c*np.exp(-(I[k]*(t-tau[k]))*gamma) + d*np.exp(-(I[k]*(t-tau[k]))*beta) ) 

    Y = np.zeros((m,2))
    Y[:,0] = y1
    Y[:,1] = y2/zeta
    
    Y[Y<0]=0
    
    if np.sum(np.isnan(Y)) != 0:
        raise ValueError("Nan in Y")
    return Y


def get_y_jac(theta, t, tau):
    # theta: (K+4), delta_1, ..., delta_K, u_0, eta = zeta*s_0 - u_0, beta, zeta
    # t: len m
    # tau: len K+1
    # return m * 2
    K = len(tau)-1 # number of states
    if len(theta)!=K+4:
        raise TypeError("wrong parameters lengths")
    delta = theta[0:K]
    u_0 = theta[-4]
    eta = theta[-3]
    beta = theta[-2]
    gamma = theta[-1]
    zeta = gamma / beta
    if zeta == 1:
        zeta = 1 + eps
    c = 1/(1-zeta)
    d = c-1 
    m = len(t)
  
    I = np.ones((K+1,m),dtype=bool)

    # nascent
    y1=u_0
    for k in range(K):
        I[k] = np.squeeze(t > tau[k])
        y1 = y1 + delta[k] * (1- np.exp(-(I[k]*(t-tau[k]))*beta )) 
    
    if np.sum(np.isnan(y1)) != 0:
        raise ValueError("Nan in y1")
        
    # mature * zeta
    y2 = u_0 + eta * np.exp(-t*gamma)
    for k in range(K):
        y2 = y2 + delta[k] * (1- c*np.exp(-(I[k]*(t-tau[k]))*gamma) + d*np.exp(-(I[k]*(t-tau[k]))*beta) ) 

    dY_dtheta = np.zeros((m,2,len(theta)))
    Y[:,0] = y1
    Y[:,1] = y2/zeta
    
    Y[Y<0]=0
    
    if np.sum(np.isnan(Y)) != 0:
        raise ValueError("Nan in Y")
    return dY_dtheta



def get_Y(theta, t, tau):
    # theta: p*(K+4), delta_1, ..., delta_K, u_0, eta = zeta*s_0 - u_0, beta, zeta
    # t: len m
    # tau: len K+1
    # return m * p * 2
    p = len(theta)
    K = len(tau)-1 # number of states
    if np.shape(theta)[1]!=K+4:
        print(np.shape(theta)[1], K+4)
        raise TypeError("wrong parameters lengths")
    delta = theta[:,0:K].T
    u_0 = theta[:,-4].reshape((1,-1))
    eta = theta[:,-3].reshape((1,-1))
    beta = theta[:,-2].reshape((1,-1))
    gamma = theta[:,-1].reshape((1,-1))
    zeta = gamma / beta
    
    c = 1/(1-zeta)
    d = c-1
    
    t = t.reshape(-1,1)
    m = len(t)
  
    I = np.ones((K+1,m),dtype=bool)

    # nascent
    y1=u_0
    for k in range(K):
        I[k] = np.squeeze(t > tau[k])
        y1 = y1 + delta[None,k] * (1- np.exp(-(I[k,:,None]*(t-tau[k]))@beta )) 
    
    if np.sum(np.isnan(y1)) != 0:
        raise ValueError("Nan in y1")
        
    # mature * zeta
    y2 = u_0 + eta * np.exp(-t@gamma)
    for k in range(K):
        y2 = y2 + delta[None,k] * (1- c*np.exp(-(I[k,:,None]*(t-tau[k]))@gamma) + d*np.exp(-(I[k,:,None]*(t-tau[k]))@beta) ) 

    Y = np.zeros((m,p,2))
    Y[:,:,0] = y1
    Y[:,:,1] = y2/zeta
    
    Y[Y<0]=0
    
    if np.sum(np.isnan(Y)) != 0:
        raise ValueError("Nan in Y")
    return Y


def get_Y_old(theta, t, tau):
    # theta: p*(K+4)
    # t: len m
    # tau: len K+1
    # return m * p * 2
    p = len(theta)
    K = len(tau)-1 # number of states
    if np.shape(theta)[1]!=K+4:
        print(np.shape(theta)[1], K+4)
        raise TypeError("wrong parameters lengths")
    a = theta[:,0:K].T
    beta = theta[:,-2].reshape((1,-1))
    gamma = theta[:,-1].reshape((1,-1))

    y1_0 = theta[:,-4].reshape((1,-1))
    y2_0 = theta[:,-3].reshape((1,-1))

    c = beta/(beta-gamma+eps)
    d = beta**2/((beta-gamma)*gamma+eps)
    y_0 = y2_0 + c*y1_0
    a_ = d*a
    t = t.reshape(-1,1)
    m = len(t)
  
    I = np.ones((K+1,m),dtype=bool)

    # nascent
    y1=y1_0*np.exp(-t@beta)
    for k in range(1,K+1):
        I[k] = np.squeeze(t > tau[k])
        idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
        y1 = y1 + a[None,k-1] * (np.exp(- (I[k,:,None] *(t-tau[k]))@beta)- np.exp(-(I[k,:,None]*(t-tau[k-1]))@beta )) \
          + a[None,k-1] * (1 - np.exp(- (idx[:,None] *(t-tau[k-1]))@beta ) )
    
    if np.sum(np.isnan(y1)) != 0:
        raise ValueError("Nan in y1")
    # mature + c * nascent 
    y=y_0*np.exp(-t@gamma)    
    for k in range(1,K+1):
        idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
        y = y + a_[None,k-1] * (np.exp(-(I[k,:,None] * (t-tau[k]))@gamma)- np.exp(-(I[k,:,None] * (t-tau[k-1]))@gamma )) \
          +  a_[None,k-1] * (1 - np.exp(-(idx[:,None]*(t-tau[k-1]))@gamma) )

    Y = np.zeros((m,p,2))
    Y[:,:,0] = y1
    Y[:,:,1] = y-c*y1
    
    Y[Y<0]=0
    
    if np.sum(np.isnan(Y)) != 0:
        raise ValueError("Nan in Y")
    return Y


def get_Y_old2(theta, t, tau):
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

def neglogL(theta, x_weighted, marginal_weight, t, tau, topo):
    # theta: length K+4
    # x: n*2
    # Q: n*L*m
    # t: len m
    # tau: len K+1
    logL = 0
    for l in range(len(topo)):
        theta_l = np.concatenate((theta[topo[l]], theta[-4:]))
        Y = get_y(theta_l,t,tau) # m*2
        logL += np.sum( x_weighted[l] * np.log(eps + Y) - marginal_weight[l]*Y )
    return - logL


def neglogL_jac(theta, x_weighted, marginal_weight, t, tau, topo):
    # theta: length K+4
    # x: n*2
    # Q: n*L*m
    # t: len m
    # tau: len K+1
    jac = np.zeros_like(theta)
    for l in range(len(topo)):
        theta_l = np.concatenate((theta[topo[l]], theta[-4:]))
        Y = get_y(theta_l,t,tau)[:,0,:] # m*2
        dY_dtheta = get_y_jac(theta_l,t,tau) # m*2*len(theta)
        jac += np.sum(( x_weighted[l] / (eps + Y) - marginal_weight[l] ) * dY_dtheta, axis=(0,1))
    return jac

#@profile
def update_theta_j(theta0, x, Q, t, tau, topo, bnd=1000, bnd_beta=100, miter = 1000):
    bounds = [[-bnd,bnd]]*len(theta0)
    bounds[-2:] = [[1/bnd_beta,bnd_beta]]*2
    
    n,L,m = Q.shape
    x_weighted = np.zeros((L,m,2))
    marginal_weight = np.zeros((L,m,1))
    for l in range(len(topo)):
        weight_l = Q[:,l,:] #n*m
        x_weighted[l] = weight_l.T@x # m*2
        marginal_weight[l] =  weight_l.sum(axis=0)[:,None] # m*1
    
    res = minimize(fun=neglogL, x0=theta0, args=(x_weighted,marginal_weight,t,tau,topo), method = 'Nelder-Mead' , jac = None, bounds=bounds, options={'maxiter': miter,'disp': True}) 
    return res


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    topo = np.array([[0,]])
    tau=(0,1)
    n = 10000
    p = 1
    K=len(tau)-1
    t=np.linspace(0, 1, n)
    
    theta_old=np.ones((p,K+4))
    loga_max=4
    logb_max=2
    
    theta_old[0,:K+2]=np.exp(np.random.uniform(0,loga_max,size=K+2))-1
    
    theta_old[0,-2:]=np.exp(np.random.uniform(-logb_max,logb_max,size=2))
    
    
    theta = theta_old.copy() 
    theta[0,0] -= theta[0,-4]
    theta[0,-3] = theta[0,-3]*theta[0,-1]/theta[0,-2] - theta[0,-4]  # eta = zeta * s_0 - u_0
    
    Y = get_Y(theta, t, tau)[:,0]
    y = get_y(theta[0], t, tau)
    Y_old = get_Y_old(theta_old,t,tau)[:,0]

    print(np.max(Y-Y_old))
    print(np.max(Y-y))
    
    X = np.random.poisson(Y)
        
    theta0 = np.zeros(K+4)
    theta0[-4]=np.mean(X[:,0])
    theta0[-2]=1
    theta0[-1]=np.mean(X[:,0])/np.mean(X[:,1])
    
    Q = np.diag(np.ones(n))/n
    res = update_theta_j(theta0, X, Q[:,None,:], t, tau, topo)
    
    print(res.x)
    print(theta[0]-res.x)
    
    Y_fit = get_y(res.x, t, tau)

    plt.scatter(X[:,0],X[:,1],c='gray');
    plt.scatter(Y_fit[:,0],Y_fit[:,1],c=t,cmap='Spectral');
    plt.scatter(Y[:,0],Y[:,1],c=t,cmap="viridis_r");
