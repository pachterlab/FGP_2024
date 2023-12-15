#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 13:43:54 2022

@author: fang

This file defines functions (guess_theta, update_theta_j, update_theta_j, get_logL) for two species 
initial steady state Poisson model. 
theta for each gene contains transcription rate of each states, beta, gamma.

"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln


# global parameters: upper and lower limits for numerical stability
eps = 1e-10

def check_params(traj):
    assert len(traj.tau) == len(traj.topo[0])
    return

def guess_theta(X,topo,tau):
    n_states=len(set(topo.flatten()))
    p = X.shape[1]
    theta = np.zeros((p,n_states+2))
    theta[:,0:-2] = np.mean(X[:,:,0],axis=0)[:,None]
    theta[:,-2] = np.mean(X[:,:,1],axis=0)/(np.mean(X[:,:,0],axis=0)+eps)
    theta[:,-1] = 1 #np.sqrt(np.mean(X[:,:,0],axis=0)/(np.mean(X[:,:,1],axis=0)+eps))
    return theta

def get_y(theta, t, tau):
    # theta: a_1, ..., a_K, u_0, s_0, beta, gamma
    # t: len m
    # tau: len K+1
    # return m * p * 2
    K = len(tau)-1 # number of states
    a = theta[1:(K+1)]
    beta = theta[-2]
    gamma = theta[-1]

    y1_0 = theta[0]

    c = beta/(beta-gamma+eps)
    d = beta**2/((beta-gamma)*gamma+eps)
    y_0 = d*y1_0
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

def get_y_jac(theta, t, tau):
    # theta: a_1, ..., a_K, u_0, s_0, beta, gamma
    # t: len m
    # tau: len K+1
    # return m * p * 2
    K = len(tau)-1 # number of states
    a = theta[1:(K+1)]
    y1_0 = theta[0]
    beta = theta[-2]
    gamma = theta[-1]

    c = beta/(beta-gamma+eps) 
    dc_dbeta = - gamma/((beta-gamma)**2+eps)
    dc_dgamma = beta/((beta-gamma)**2+eps)
    d = beta**2/((beta-gamma)*gamma+eps)
    dd_dbeta = (beta**2-2*beta*gamma)/((beta-gamma)**2*gamma+eps)
    dd_dgamma = (-beta**3+2*beta**2*gamma)/((beta-gamma)**2*gamma**2+eps)
    y_0 = d*y1_0
    a_ = d*a
    da__dbeta = a * dd_dbeta
    da__dgamma = a * dd_dgamma

    
    m = len(t)
    I = np.ones((K+1,m),dtype=bool)
    dY_dtheta = np.zeros((m,2,len(theta)))
    
    # nascent
    y1=y1_0*np.exp(-t*beta)
    y=y_0*np.exp(-t*gamma)
    dY_dtheta[:,0,0] = np.exp(-t*beta)
    dY_dtheta[:,1,0] = d * np.exp(-t*gamma) - c * dY_dtheta[:,0,0]
    
    
    dY_dtheta[:,0,-2] = - t * y1_0 * np.exp(-t*beta)
    dY_dtheta[:,1,-2] = dd_dbeta * y1_0 * np.exp(-t*gamma)
    dY_dtheta[:,1,-1] = dd_dgamma * y1_0 * np.exp(-t*gamma)  - t * y_0 * np.exp(-t*gamma)
    
    for k in range(1,K+1):
        I[k] = np.squeeze(t > tau[k])
        idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
        
        y1_a_k_coef = np.exp(- (I[k] *(t-tau[k]))*beta) - np.exp(-(I[k] * (t-tau[k-1])) * beta ) + 1 - np.exp(- (idx * (t-tau[k-1]))*beta ) 
        y_a_k_coef = np.exp(- (I[k] * (t-tau[k]))*gamma) - np.exp(-(I[k] * (t-tau[k-1])) * gamma) + 1 - np.exp(- (idx * (t-tau[k-1]))*gamma)

        y1 += a[k-1] *  y1_a_k_coef
        y += a_[k-1] * y_a_k_coef
        
        dY_dtheta[:,0,k] = y1_a_k_coef
        dY_dtheta[:,1,k] = d * y_a_k_coef
        
        dy1_a_k_coef_dbeta = - (t-tau[k]) * I[k] * np.exp(- (I[k] * (t-tau[k])) * beta) + (t-tau[k-1]) * I[k] * np.exp(-(I[k] * (t-tau[k-1])) * beta ) +  idx * (t-tau[k-1]) * np.exp(- (idx * (t-tau[k-1])) * beta)
        dy_a_k_coef_dgamma = - (t-tau[k]) * I[k] * np.exp(- (I[k] * (t-tau[k])) * gamma) + (t-tau[k-1]) * I[k] * np.exp(-(I[k] * (t-tau[k-1])) * gamma ) + idx * (t-tau[k-1]) * np.exp(- (idx * (t-tau[k-1])) * gamma) 
        
        dY_dtheta[:,0,-2] += a[k-1] * dy1_a_k_coef_dbeta
        dY_dtheta[:,1,-2] += da__dbeta[k-1] * y_a_k_coef
        dY_dtheta[:,1,-1] += da__dgamma[k-1] * y_a_k_coef + a_[k-1] * dy_a_k_coef_dgamma
        
    Y = np.zeros((m,2))
    Y[:,0] = y1
    Y[:,1] = y-c*y1
    
    dY_dtheta[:,1,1:(K+1)] -= c * dY_dtheta[:,0,1:(K+1)]
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
    assert np.shape(theta)[1]==K+3
    a = theta[:,1:(K+1)].T
    beta = theta[:,-2].reshape((1,-1))
    gamma = theta[:,-1].reshape((1,-1))

    y1_0 = theta[:,0].reshape((1,-1))

    c = beta/(beta-gamma+eps)
    d = beta**2/((beta-gamma)*gamma+eps)
    y_0 = d*y1_0
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

def neglogL(theta, x_weighted, marginal_weight, t, tau, topo, Ub=1, lambda_a=0):
    # theta: length K+4
    # x: n*2
    # Q: n*L*m
    # t: len m
    # tau: len K+1
    logL = 0
    n_states=len(set(topo.flatten()))
    parents = np.zeros(n_states,dtype=int)
    for l in range(len(topo)):
        theta_l = np.concatenate((theta[topo[l]], theta[-2:]))
        Y = get_y(theta_l,t,tau) # m*2
        Y[:,0] *= Ub
        logL += np.sum( x_weighted[l] * np.log(eps + Y) - marginal_weight[l]*Y )
        parents[topo[l][1:]] = topo[l][:-1]

    penalty_a = np.sum((theta[1:n_states]-theta[parents[1:n_states]])**2)
    return - logL + lambda_a * penalty_a

def neglogL_jac(theta, x_weighted, marginal_weight, t, tau, topo, Ub=1, lambda_a=0):
    # theta: length K+4
    # x: n*2
    # Q: n*L*m
    # t: len m
    # tau: len K+1
    n_states=len(set(topo.flatten()))
    parents = np.zeros(n_states,dtype=int)
    penalty_a_jac = np.zeros_like(theta)
    jac = np.zeros_like(theta)
    for l in range(len(topo)):
        theta_idx = np.append(topo[l],[-2,-1])
        theta_l = theta[theta_idx]
        Y, dY_dtheta = get_y_jac(theta_l,t,tau) # m*2*len(theta)
        Y[:,0] *= Ub
        dY_dtheta[:,0] *= Ub
        coef =  x_weighted[l] / (eps + Y) - marginal_weight[l]
        jac[theta_idx] += np.sum( coef[:,:,None] * dY_dtheta, axis=(0,1))
        parents[topo[l][1:]] = topo[l][:-1]
    penalty_a_jac[1:n_states] = 2*(theta[1:n_states]-theta[parents[1:n_states]])
    penalty_a_jac[parents[1:n_states]] += 2*(theta[parents[1:n_states]] - theta[1:n_states])
    return - jac + lambda_a * penalty_a_jac

def get_Y_hat(theta,t,tau,topo,params):
    L=len(topo)
    m=len(t)
    p=len(theta)
    n_states=len(set(topo.flatten()))
    Y = np.zeros((L,m,p,2))
    for l in range(L):
        theta_l = np.concatenate((theta[:,topo[l]], theta[:,n_states:]), axis=1)
        Y[l] = get_Y(theta_l,t,tau) # m*p*2
    if "Ub" in params:
        Y[:,:,:,0] *= params["Ub"][None,None,:]
    return Y
    
def get_Y_hat_jac(theta,t,tau,topo,params):
    pass

def get_gene_logL(X,theta,t,tau,topo,params):
    L=len(topo)
    m=len(t)
    p=len(theta)
    n_states=len(set(topo.flatten()))
    parents = np.zeros(n_states,dtype=int)
    Y = np.zeros((L,m,p,2))
    for l in range(L):
        theta_l = np.concatenate((theta[:,topo[l]], theta[:,n_states:]), axis=1)
        Y[l] = get_Y(theta_l,t,tau) # m*p*2
        parents[topo[l][1:]] = topo[l][:-1]
        
    if "Ub" in params:
        Y[:,:,:,0] *= params["Ub"][None,None,:]
    
    if 'r' in params:
        r = params['r'] # n
    else:
        r = np.one(len(X))
        
    logL = np.sum(X[:,None,None,:,:] * np.log(eps + Y)[None,:], axis=-1) # logL:n*L*m*p
    logL += (np.log(r)[:,None]*np.sum(X,axis=(-1)))[:,None,None,:]
    logL -= np.sum(r[:,None,None,None,None]*Y[None,:],axis=(-1))
    logL -= np.sum(gammaln(X+1),axis=(-1))[:,None,None,:]
            
    if 'lambda_a' in params:
        penalty_a = np.sum((theta[:,1:n_states]-theta[:,parents[1:n_states]])**2,axis=1)
        logL -= params['lambda_a'] * penalty_a[None,None,None,:]    
    return logL
    
def get_logL(X,theta,t,tau,topo,params):
    L=len(topo)
    m=len(t)
    p=len(theta)
    n_states=len(set(topo.flatten()))
    parents = np.zeros(n_states,dtype=int)
    Y = np.zeros((L,m,p,2))
    for l in range(L):
        theta_l = np.concatenate((theta[:,topo[l]], theta[:,n_states:]), axis=1)
        Y[l] = get_Y(theta_l,t,tau) # m*p*2
        parents[topo[l][1:]] = topo[l][:-1]    
        
    if "Ub" in params:
        Y[:,:,:,0] *= params["Ub"][None,None,:]
        
    logL = np.tensordot(X, np.log(eps + Y), axes=([-2,-1],[-2,-1])) # logL:n*L*m
    logL -= np.sum(Y,axis=(-2,-1))
    logL  -= np.sum(gammaln(X+1),axis=(-2,-1),keepdims=True)
    
    if 'r' in params:
        r = params['r'] # n
        logL += (np.log(r)*np.sum(X,axis=(-2,-1)))[:,None,None]
        logL -= (r[:,None,None]-1)*np.sum(Y,axis=(-2,-1))[None,:]
        
    if "lambda_a" in params:
        logL -= params["lambda_a"] * np.sum((theta[:,1:n_states]-theta[:,parents[1:n_states]])**2)
    
    return logL

def update_theta_j(j, theta0, x, Q, t, tau, topo, params=None, restrictions=None):
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
    n,L,m = Q.shape
    x_weighted = np.zeros((L,m,2))
    marginal_weight = np.zeros((L,m,1))
        
    if "Ub" in params:
        Ub = params["Ub"][j]
    else:
        Ub = 1
    
    if 'bnd' in params:
        bnd = params['bnd']
    else:
        bnd=1000
        
    if 'bnd_beta' in params:
        bnd_beta = params['bnd_beta']
    else:
        bnd_beta=1000
        
    if 'miter' in params:
        miter = params['miter']
    else:
        miter=100

    if 'batch_size' in params:
        x_idx = np.random.choice(n,params['batch_size'],replace=False)
    else:
        x_idx = np.arange(n)
        
            
    if 'r' in params:
        r = params['r'] # n
        for l in range(len(topo)):
            weight_l = Q[x_idx,l,:]/len(x_idx) #n*m
            x_weighted[l] = weight_l.T@x[x_idx] # m*2 = m*n @ n*2
            marginal_weight[l] =  (weight_l*r[x_idx,None]).sum(axis=0)[:,None] # m*1
    else:
        for l in range(len(topo)):
            weight_l = Q[x_idx,l,:]/len(x_idx) #n*m
            x_weighted[l] = weight_l.T@x[x_idx] # m*2 = m*n @ n*2
            marginal_weight[l] = weight_l.sum(axis=0)[:,None] # m*1

    n_states=len(set(topo.flatten()))
    theta00 = theta0.copy()
    if np.max(theta00[:n_states]) > np.maximum( np.max(x[:,0]) , np.max(x[:,1]) * theta00[-1] / theta00[-2]):
        theta00[:n_states] = np.mean(x[:,0])
    if np.min(np.abs(np.log10(theta00[-2:]))) > 1:
        theta00[-2] = np.mean(x[:,1],axis=0)/(np.mean(x[:,0],axis=0)+eps)
        theta00[-1] = 1 #np.sqrt(np.mean(x[:,0],axis=0)/(np.mean(x[:,1],axis=0)+eps))

    if "lambda_a" in params:
        lambda_a = params['lambda_a']
    else:
        lambda_a = 0
        
    if restrictions==None:
        res = update_theta_j_unrestricted(theta00, x_weighted, marginal_weight, t, tau, topo, Ub, lambda_a, bnd, bnd_beta, miter)
    else:
        res = update_theta_j_restricted(theta00, x_weighted, marginal_weight, t, tau, topo, restrictions, Ub, lambda_a, bnd, bnd_beta, miter)
    return res

def update_theta_j_unrestricted(theta0, x_weighted, marginal_weight, t, tau, topo, Ub=1, lambda_a=0, bnd=10000, bnd_beta=1000, miter=100):

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
    bound = [[0,bnd]]*len(theta0)
    bound[-2:] = [[1/bnd_beta,bnd_beta]]*2
    
    res = minimize(fun=neglogL, x0=theta0, args=(x_weighted,marginal_weight,t,tau,topo,Ub,lambda_a), method = 'L-BFGS-B' , jac = neglogL_jac, bounds=bound, options={'maxiter': miter,'disp': False}) 
    return res.x

def update_theta_j_restricted(theta0, x_weighted, marginal_weight, t, tau, topo, restrictions, Ub=1, lambda_a=0, bnd=1000, bnd_beta=1000, miter=100):

    # define a new neglogL inside with fewer parameters
    redundant, blanket = restrictions # 1,0 => a[1] = a[0], 0, -3 => a[0] = u_0,
    if len(redundant) >= len(theta0) - 3:
        theta = np.ones(len(theta0))*np.sum(x_weighted[:,0])
        theta[-2] = 1
        theta[-1] = np.sum(x_weighted[:,0])/np.sum(x_weighted[:,1])
        
    else:
        redundant_mask = np.zeros(len(theta0), dtype=bool)
        redundant_mask[redundant] = True
        custom_theta0 = theta0[~redundant_mask]  
        bound = [[0,bnd]]*len(custom_theta0)
        bound[-2:] = [[1/bnd_beta,bnd_beta]]*2
        
        def custom_neglogL(custom_theta, x_weighted, marginal_weight, t, tau, topo, Ub, lambda_a):

            theta = np.zeros(len(theta0))
            theta[~redundant_mask] = custom_theta
            theta[redundant] = theta[blanket]
               
            return neglogL(theta, x_weighted, marginal_weight, t, tau, topo, Ub, lambda_a)
            
       

        res = minimize(fun=custom_neglogL, x0=custom_theta0, args=(x_weighted,marginal_weight,t,tau,topo,Ub,lambda_a), method = 'L-BFGS-B' , jac = None, bounds=bound, options={'maxiter': miter,'disp': False}) 

        
        theta = np.zeros(len(theta0))
        theta[~redundant_mask] = res.x
        theta[redundant] = theta[blanket]
            
    return theta
