#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 11:26:08 2022

@author: fang

This file defines functions (guess_theta, update_theta_j, update_theta_j, get_logL) for two species Poisson model. 
theta for each gene contains transcription rate of each states, u_0, s_0, beta, gamma.

"""

import numpy as np
from scipy.optimize import minimize


# global parameters: upper and lower limits for numerical stability
eps = 1e-10

def guess_theta(X,n_states):
    p = X.shape[1]
    theta = np.zeros((p,n_states+4))
    theta[:,0:-3]=np.mean(X[:,:,0],axis=0)[:,None]
    theta[:,-3]=np.mean(X[:,:,1],axis=0)
    theta[:,-2]=1
    theta[:,-1] = np.mean(X[:,:,0],axis=0)/(np.mean(X[:,:,1],axis=0)+eps)
    return theta

#%% delta parameterization
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

#%% a parameterization

def get_y_a(theta, t, tau):
    # theta: a_1, ..., a_K, u_0, s_0, beta, gamma
    # t: len m
    # tau: len K+1
    # return m * p * 2
    K = len(tau)-1 # number of states
    a = theta[0:K]
    beta = theta[-2]
    gamma = theta[-1]

    y1_0 = theta[-4]
    y2_0 = theta[-3]

    c = beta/(beta-gamma+eps)
    d = beta**2/((beta-gamma)*gamma+eps)
    y_0 = y2_0 + c*y1_0
    a_ = d*a
    t = t
    m = len(t)
  
    I = np.ones((K+1,m),dtype=bool)

    # nascent
    y1=y1_0*np.exp(-t*beta)
    y=y_0*np.exp(-t*gamma)   
    for k in range(1,K+1):
        I[k] = np.squeeze(t > tau[k])
        idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
        y1 = y1 + a[k-1] * (np.exp(- (I[k] *(t-tau[k]))*beta)- np.exp(-(I[k]*(t-tau[k-1]))*beta )) \
          + a[k-1] * (1 - np.exp(- (idx *(t-tau[k-1]))*beta ) )
        y = y + a_[k-1] * (np.exp(-(I[k] * (t-tau[k]))*gamma)- np.exp(-(I[k] * (t-tau[k-1]))*gamma )) \
          +  a_[k-1] * (1 - np.exp(-(idx*(t-tau[k-1]))*gamma) )

    Y = np.zeros((m,2))
    Y[:,0] = y1
    Y[:,1] = y-c*y1
    
    Y[Y<0]=0
    
    if np.sum(np.isnan(Y)) != 0:
        raise ValueError("Nan in Y")
    return Y

def get_y_a_jac(theta, t, tau):
    # theta: a_1, ..., a_K, u_0, s_0, beta, gamma
    # t: len m
    # tau: len K+1
    # return m * p * 2
    K = len(tau)-1 # number of states
    a = theta[0:K]
    y1_0 = theta[-4]
    y2_0 = theta[-3]
    beta = theta[-2]
    gamma = theta[-1]

    c = beta/(beta-gamma+eps) 
    dc_dbeta = - gamma/((beta-gamma)**2+eps)
    dc_dgamma = beta/((beta-gamma)**2+eps)
    d = beta**2/((beta-gamma)*gamma+eps)
    y_0 = y2_0 + c*y1_0
    a_ = d*a
    da__dbeta = a * (beta**2-2*beta*gamma)/((beta-gamma)**2*gamma+eps)
    da__dgamma = a * (-beta**3+2*beta**2*gamma)/((beta-gamma)**2*gamma**2+eps)

    
    m = len(t)
    I = np.ones((K+1,m),dtype=bool)
    dY_dtheta = np.zeros((m,2,len(theta)))
    
    # nascent
    y1=y1_0*np.exp(-t*beta)
    y=y_0*np.exp(-t*gamma)
    dY_dtheta[:,0,-4] = np.exp(-t*beta)
    dY_dtheta[:,1,-3] = np.exp(-t*gamma)   
    dY_dtheta[:,1,-4] = c * (dY_dtheta[:,1,-3] - dY_dtheta[:,0,-4])
    
    dY_dtheta[:,0,-2] = - t * y1_0 * np.exp(-t*beta)
    dY_dtheta[:,1,-2] = dc_dbeta * y1_0 * np.exp(-t*gamma)
    dY_dtheta[:,1,-1] = dc_dgamma * y1_0 * np.exp(-t*gamma)  - t * y_0 * np.exp(-t*gamma)
    
    for k in range(1,K+1):
        I[k] = np.squeeze(t > tau[k])
        idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
        
        y1_a_k_coef = np.exp(- (I[k] *(t-tau[k]))*beta) - np.exp(-(I[k] * (t-tau[k-1])) * beta ) + 1 - np.exp(- (idx * (t-tau[k-1]))*beta ) 
        y_a_k_coef = np.exp(- (I[k] * (t-tau[k]))*gamma) - np.exp(-(I[k] * (t-tau[k-1])) * gamma) + 1 - np.exp(- (idx * (t-tau[k-1]))*gamma)

        y1 += a[k-1] *  y1_a_k_coef
        y += a_[k-1] * y_a_k_coef
        
        dY_dtheta[:,0,k-1] = y1_a_k_coef
        dY_dtheta[:,1,k-1] = d * y_a_k_coef
        
        dy1_a_k_coef_dbeta = - (t-tau[k]) * I[k] * np.exp(- (I[k] * (t-tau[k])) * beta) + (t-tau[k-1]) * I[k] * np.exp(-(I[k] * (t-tau[k-1])) * beta ) +  idx * (t-tau[k-1]) * np.exp(- (idx * (t-tau[k-1])) * beta)
        dy_a_k_coef_dgamma = - (t-tau[k]) * I[k] * np.exp(- (I[k] * (t-tau[k])) * gamma) + (t-tau[k-1]) * I[k] * np.exp(-(I[k] * (t-tau[k-1])) * gamma ) + idx * (t-tau[k-1]) * np.exp(- (idx * (t-tau[k-1])) * gamma) 
        
        dY_dtheta[:,0,-2] += a[k-1] * dy1_a_k_coef_dbeta
        dY_dtheta[:,1,-2] += da__dbeta[k-1] * y_a_k_coef
        dY_dtheta[:,1,-1] += da__dgamma[k-1] * y_a_k_coef + a_[k-1] * dy_a_k_coef_dgamma
        
    Y = np.zeros((m,2))
    Y[:,0] = y1
    Y[:,1] = y-c*y1
    
    dY_dtheta[:,1,0:K] -= c * dY_dtheta[:,0,0:K]
    dY_dtheta[:,1,-2] -=  c * dY_dtheta[:,0,-2] + dc_dbeta * y1
    dY_dtheta[:,1,-1] -= dc_dgamma * y1
    
    Y[Y<0]=0
    if np.sum(np.isnan(Y)) != 0:
        raise ValueError("Nan in Y")
        
    if np.sum(np.isnan(dY_dtheta)) != 0:
        raise ValueError("Nan in dY_dtheta")
        
    return Y, dY_dtheta


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


def get_Y_a_old(theta, t, tau):
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

def neglogL_a(theta, x_weighted, marginal_weight, t, tau, topo):
    # theta: length K+4
    # x: n*2
    # Q: n*L*m
    # t: len m
    # tau: len K+1
    logL = 0
    for l in range(len(topo)):
        theta_l = np.concatenate((theta[topo[l]], theta[-4:]))
        Y = get_y_a(theta_l,t,tau) # m*2
        logL += np.sum( x_weighted[l] * np.log(eps + Y) - marginal_weight[l]*Y )
    return - logL


def neglogL_a_jac(theta, x_weighted, marginal_weight, t, tau, topo):
    # theta: length K+4
    # x: n*2
    # Q: n*L*m
    # t: len m
    # tau: len K+1
    jac = np.zeros_like(theta)
    for l in range(len(topo)):
        theta_idx = np.append(topo[l],[-4,-3,-2,-1])
        theta_l = theta[theta_idx]
        Y, dY_dtheta = get_y_a_jac(theta_l,t,tau) # m*2*len(theta)
        coef =  x_weighted[l] / (eps + Y) - marginal_weight[l]
        jac[theta_idx] += np.sum( coef [:,:,None] * dY_dtheta, axis=(0,1))
    return - jac

def get_logL(X,theta,t,tau,topo):
    L=len(topo)
    m=len(t)
    p=len(theta)
    Y = np.zeros((L,m,p,2))
    for l in range(L):
        theta_l = np.concatenate((theta[:,topo[l]], theta[:,-4:]), axis=1)
        Y[l] = get_Y(theta_l,t,tau) # m*p*2
    
    logL = np.tensordot(X, np.log(eps + Y), axes=([-2,-1],[-2,-1])) # logL:n*L*m
    logL -= np.sum(Y,axis=(-2,-1))
    return logL

def update_theta_j_a_jac(theta0, x, Q, t, tau, topo, bnd=1000, bnd_beta=100, miter = 10000):
    bound = [[-bnd,bnd]]*len(theta0)
    bound[-2:] = [[1/bnd_beta,bnd_beta]]*2
    
    n,L,m = Q.shape
    x_weighted = np.zeros((L,m,2))
    marginal_weight = np.zeros((L,m,1))
    for l in range(len(topo)):
        weight_l = Q[:,l,:] #n*m
        x_weighted[l] = weight_l.T@x # m*2 = m*n @ n*2
        marginal_weight[l] =  weight_l.sum(axis=0)[:,None] # m*1
    
    res = minimize(fun=neglogL_a, x0=theta0, args=(x_weighted,marginal_weight,t,tau,topo), method = 'L-BFGS-B' , jac = neglogL_a_jac, bounds=bound, options={'maxiter': miter,'disp': False}) 
    return res.x


def update_theta_j_a_nojac(theta0, x, Q, t, tau, topo, bnd=1000, bnd_beta=100, miter = 10000):
    bound = [[-bnd,bnd]]*len(theta0)
    bound[-2:] = [[1/bnd_beta,bnd_beta]]*2
    
    n,L,m = Q.shape
    x_weighted = np.zeros((L,m,2))
    marginal_weight = np.zeros((L,m,1))
    for l in range(len(topo)):
        weight_l = Q[:,l,:] #n*m
        x_weighted[l] = weight_l.T@x # m*2 = m*n @ n*2
        marginal_weight[l] =  weight_l.sum(axis=0)[:,None] # m*1
    
    res = minimize(fun=neglogL_a, x0=theta0, args=(x_weighted,marginal_weight,t,tau,topo), method = 'L-BFGS-B' , jac = None, bounds=bound, options={'maxiter': miter,'disp': False}) 
    return res.x


def update_theta_j(theta0, x, Q, t, tau, topo, bnd=1000, bnd_beta=100, miter=100000):
    """
    with jac

    Parameters
    ----------
    theta0 : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    Q : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    tau : TYPE
        DESCRIPTION.
    topo : TYPE
        DESCRIPTION.
    bnd : TYPE, optional
        DESCRIPTION. The default is 1000.
    bnd_beta : TYPE, optional
        DESCRIPTION. The default is 100.
    miter : TYPE, optional
        DESCRIPTION. The default is 1000.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    bound = [[-bnd,bnd]]*len(theta0)
    bound[-2:] = [[1/bnd_beta,bnd_beta]]*2
    
    n,L,m = Q.shape
    x_weighted = np.zeros((L,m,2))
    marginal_weight = np.zeros((L,m,1))
    for l in range(len(topo)):
        weight_l = Q[:,l,:] #n*m
        x_weighted[l] = weight_l.T@x # m*2 = m*n @ n*2
        marginal_weight[l] =  weight_l.sum(axis=0)[:,None] # m*1
    
    res = minimize(fun=neglogL_a, x0=theta0, args=(x_weighted,marginal_weight,t,tau,topo), method = 'L-BFGS-B' , jac = neglogL_a_jac, bounds=bound, options={'maxiter': miter,'disp': False}) 
    return res.x

def update_nested_theta_j(theta0, x, Q, t, tau, topo, restrictions, bnd=1000, bnd_beta=100, miter=10000):
    # define a new neglogL inside with fewer parameters
    redundant, blanket = restrictions # 1,0 => a[1] = a[0], -3, -4 => s_0 = u_0*beta/gamma, 0,-4 => a[0] = u_0
    if len(redundant) > len(theta0) - 4:
        theta = np.ones(len(theta0))*np.mean(x[:,0])
        theta[-3]=np.mean(x[:,1])
        theta[-2]=1
        theta[-1] = theta[-4]/theta[-3]
        
    else:
        redundant_mask = np.zeros(len(theta0), dtype=bool)
        redundant_mask[redundant] = True
        custom_theta0 = theta0[~redundant_mask]  
        bound = [[0,bnd]]*len(custom_theta0)
        bound[-2:] = [[1/bnd_beta,bnd_beta]]*2
        
        n,L,m = Q.shape
        x_weighted = np.zeros((L,m,2))
        marginal_weight = np.zeros((L,m,1))
        for l in range(len(topo)):
            weight_l = Q[:,l,:] #n*m
            x_weighted[l] = weight_l.T@x # m*2
            marginal_weight[l] =  weight_l.sum(axis=0)[:,None] # m*1
     
        def custom_neglogL(custom_theta, x_weighted, marginal_weight, t, tau, topo):
            theta = np.zeros(len(theta0))
            theta[~redundant_mask] = custom_theta
            theta[redundant] = theta[blanket]
            if -3 in redundant:
                theta[-3] = theta[-4]*theta[-2]/theta[-1]
                
            return neglogL_a(theta, x_weighted, marginal_weight, t, tau, topo)
            
       
        res = minimize(fun=custom_neglogL, x0=custom_theta0, args=(x_weighted,marginal_weight,t,tau,topo), method = 'L-BFGS-B' , jac = None, bounds=bound, options={'maxiter': miter,'disp': False}) 
        theta = np.zeros(len(theta0))
        theta[~redundant_mask] = res.x
        theta[redundant] = theta[blanket]
        if -3 in redundant:
            theta[-3] = theta[-4]*theta[-2]/theta[-1]
            
    return theta