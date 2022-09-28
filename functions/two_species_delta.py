#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:38:06 2022

@author: fang
"""

import numpy as np
from scipy.optimize import minimize

# global parameters: upper and lower limits for numerical stability
eps = 1e-10

def guess_theta(X,n_states):
    p = X.shape[1]
    theta = np.zeros((p,n_states+4))
    theta[:,-4]=np.mean(X[:,:,0],axis=0)
    theta[:,-2]=1
    theta[:,-1] =np.mean(X[:,:,0],axis=0)/(np.mean(X[:,:,1],axis=0)+eps)
    return theta


# delta parameterization
def get_y_delta(theta, t, tau):
    # theta: (K+4), delta_1, ..., delta_K, u_0, eta = gamma/beta * s_0 - u_0, beta, gamma
    # t: len m
    # tau: len K+1
    # return m * 2
    K = len(tau)-1 # number of states
    if len(theta)!=K+4:
        raise TypeError("wrong parameters lengths")
    delta = theta[0:K]
    u_0 = theta[-4]
    eta = theta[-3] # eta = gamma/beta * s_0 - u_0
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


def get_y_delta_jac(theta, t, tau):
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
        
    y2_gamma_coef = 1/(1-zeta)
    y2_beta_coef  = zeta/(1-zeta)
    d = beta - gamma
    
    m = len(t)
    I = np.ones((K+1,m),dtype=bool)
    dY_dtheta = np.zeros((m,2,len(theta)))
    
    # nascent
    y1=u_0
    dY_dtheta[:,0,-4] = 1
    for k in range(K):
        I[k] = np.squeeze(t > tau[k])
        y1 += delta[k] * (1- np.exp(-(I[k]*(t-tau[k]))*beta )) 
        dY_dtheta[:,0,k] = 1- np.exp(-(I[k]*(t-tau[k]))*beta )
        dY_dtheta[:,0,-2] += delta[k] * I[k] * (t-tau[k]) * np.exp(-(I[k]*(t-tau[k]))*beta)

        
    # mature * zeta
    y2 = u_0 + eta * np.exp(-t*gamma)
    dY_dtheta[:,1,-4] = 1/zeta
    dY_dtheta[:,1,-3] = np.exp(-t*gamma)/zeta
    dY_dtheta[:,1,-1] = - eta * t * np.exp(-t*gamma)
    for k in range(K):
        y2 += delta[k] * (1- y2_gamma_coef*np.exp(-(I[k]*(t-tau[k]))*gamma) + y2_beta_coef*np.exp(-(I[k]*(t-tau[k]))*beta) ) 
        dY_dtheta[:,1,k] = (1- y2_gamma_coef*np.exp(-(I[k]*(t-tau[k]))*gamma) + y2_beta_coef*np.exp(-(I[k]*(t-tau[k]))*beta)) / zeta
        dY_dtheta[:,1,-2] += delta[k]* ( np.exp(-(I[k]*(t-tau[k]))*gamma) - np.exp(-(I[k]*(t-tau[k]))*beta) - d * (t-tau[k]) * np.exp(-(I[k]*(t-tau[k]))*beta) )
        dY_dtheta[:,1,-1] += delta[k]* ( - np.exp(-(I[k]*(t-tau[k]))*gamma) + np.exp(-(I[k]*(t-tau[k]))*beta) + d * (t-tau[k]) * np.exp(-(I[k]*(t-tau[k]))*gamma) ) 
    
    Y = np.zeros((m,2))
    Y[:,0] = y1
    Y[:,1] = y2/zeta
    dY_dtheta[:,1,-2] = y2/gamma + dY_dtheta[:,1,-2]*beta/(d**2+eps)
    dY_dtheta[:,1,-1] = -Y[:,1]/gamma + dY_dtheta[:,1,-1]*beta**2/(gamma*d**2+eps)
    
    Y[Y<0]=0
    if np.sum(np.isnan(Y)) != 0:
        raise ValueError("Nan in Y")
        
    if np.sum(np.isnan(dY_dtheta)) != 0:
        raise ValueError("Nan in dY_dtheta")
        
    return Y, dY_dtheta


def get_Y_delta(theta, t, tau):
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

def neglogL_delta(theta, x_weighted, marginal_weight, t, tau, topo):
    # theta: length K+4
    # x: n*2
    # Q: n*L*m
    # t: len m
    # tau: len K+1
    logL = 0
    for l in range(len(topo)):
        theta_l = np.concatenate((theta[topo[l]], theta[-4:]))
        Y = get_y_delta(theta_l,t,tau) # m*2
        logL += np.sum( x_weighted[l] * np.log(eps + Y) - marginal_weight[l]*Y )
    return - logL


def neglogL_delta_jac(theta, x_weighted, marginal_weight, t, tau, topo):
    # theta: length K+4
    # x: n*2
    # Q: n*L*m
    # t: len m
    # tau: len K+1
    jac = np.zeros_like(theta)
    for l in range(len(topo)):
        theta_idx = np.append(topo[l],[-4,-3,-2,-1])
        theta_l = theta[theta_idx]
        Y, dY_dtheta = get_y_delta_jac(theta_l,t,tau) # m*2*len(theta)
        coef =  x_weighted[l] / (eps + Y) - marginal_weight[l]
        jac[theta_idx] += np.sum( coef [:,:,None] * dY_dtheta, axis=(0,1))
    return - jac


def update_theta_j_delta(theta0, x, Q, t, tau, topo, bnd=1000, bnd_beta=100, miter = 1000):
    bound = [[-bnd, bnd]]*len(theta0)
    bound[-2:] = [[1/bnd_beta,bnd_beta]]*2
    
    n,L,m = Q.shape
    x_weighted = np.zeros((L,m,2))
    marginal_weight = np.zeros((L,m,1))
    for l in range(len(topo)):
        weight_l = Q[:,l,:] #n*m
        x_weighted[l] = weight_l.T@x # m*2
        marginal_weight[l] =  weight_l.sum(axis=0)[:,None] # m*1
    
    res = minimize(fun=neglogL_delta, x0=theta0, args=(x_weighted,marginal_weight,t,tau,topo), method = 'L-BFGS-B' , jac = neglogL_delta_jac, bounds=bound, options={'maxiter': miter,'disp': False}) 
    return res.x
