#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 11:59:09 2022

@author: fang

This file defines functions (guess_theta, update_theta_j, update_theta_j, get_logL) for one species Poisson model. 
theta for each gene contains transcription rate of each states, u_0, beta.

"""

import numpy as np
from scipy.optimize import minimize
#from scipy.special import gammaln


# global parameters: upper and lower limits for numerical stability
eps = 1e-10

def guess_theta(X,n_states):
    p = X.shape[1]
    theta = np.ones((p,n_states+2))
    theta[:,0:-1]=np.mean(X[:,:,0],axis=0)[:,None]
    return theta



def get_y(theta, t, tau):
    """
    get y for one gene

    Parameters
    ----------
    theta : 1d array
        a_1, ..., a_K, u_0, beta
    t : 1d array
        len(t)=m.
    tau : 1d array
        len(tau)=K+1.

    Raises
    ------
    ValueError
        if contains Nan.

    Returns
    -------
    Y : 2d array
        m*1.

    """
    
    K = len(tau)-1 # number of states
    a = theta[0:K]
    beta = theta[-1]

    y1_0 = theta[-2]
    m = len(t)
  
    I = np.ones((K+1,m),dtype=bool)

    # nascent
    y1=y1_0*np.exp(-t*beta)
    for k in range(1,K+1):
        I[k] = np.squeeze(t > tau[k])
        idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
        y1 = y1 + a[k-1] * (np.exp(- (I[k] *(t-tau[k]))*beta)- np.exp(-(I[k]*(t-tau[k-1]))*beta )) \
          + a[k-1] * (1 - np.exp(- (idx *(t-tau[k-1]))*beta ) )
    

    Y = np.zeros((m,1))
    Y[:,0] = y1
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
    a = theta[0:K]
    y1_0 = theta[-2]   
    beta = theta[-1]

    
    m = len(t)
    I = np.ones((K+1,m),dtype=bool)
    dY_dtheta = np.zeros((m,1,len(theta)))
    
    # nascent
    y1=y1_0*np.exp(-t*beta)
    
    dY_dtheta[:,0,-2] = np.exp(-t*beta)   
    dY_dtheta[:,0,-1] = - t * y1_0 * np.exp(-t*beta)
    
    for k in range(1,K+1):
        I[k] = np.squeeze(t > tau[k])
        idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
        
        y1_a_k_coef = np.exp(- (I[k] *(t-tau[k]))*beta) - np.exp(-(I[k] * (t-tau[k-1])) * beta ) + 1 - np.exp(- (idx * (t-tau[k-1]))*beta ) 

        y1 += a[k-1] *  y1_a_k_coef
        
        dY_dtheta[:,0,k-1] = y1_a_k_coef

        
        dy1_a_k_coef_dbeta = - (t-tau[k]) * I[k] * np.exp(- (I[k] * (t-tau[k])) * beta) + (t-tau[k-1]) * I[k] * np.exp(-(I[k] * (t-tau[k-1])) * beta ) +  idx * (t-tau[k-1]) * np.exp(- (idx * (t-tau[k-1])) * beta)
        
        dY_dtheta[:,0,-1] += a[k-1] * dy1_a_k_coef_dbeta
        
    Y = np.zeros((m,1))
    Y[:,0] = y1
    
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
    # return m * p * 1
    p = len(theta)
    K = len(tau)-1 # number of states
    a = theta[:,0:K].T
    beta = theta[:,-1].reshape((1,-1))   
    y1_0 = theta[:,-2].reshape((1,-1))
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
    
    Y = np.zeros((m,p,1))
    Y[:,:,0] = y1
    
    Y[Y<0]=0
    
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
        theta_l = np.concatenate((theta[topo[l]], theta[-2:]))
        Y = get_y(theta_l,t,tau) # m*1
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
        theta_idx = np.append(topo[l],[-2,-1])
        theta_l = theta[theta_idx]
        Y, dY_dtheta = get_y_jac(theta_l,t,tau) # m*1*len(theta)
        coef =  x_weighted[l] / (eps + Y) - marginal_weight[l]
        jac[theta_idx] += np.sum( coef[:,:,None] * dY_dtheta, axis=(0,1))
    return - jac

def get_logL(X,theta,t,tau,topo,params):
    L=len(topo)
    m=len(t)
    p=len(theta)
    Y = np.zeros((L,m,p,1))
    for l in range(L):
        theta_l = np.concatenate((theta[:,topo[l]], theta[:,-2:]), axis=1)
        Y[l] = get_Y(theta_l,t,tau) # m*p*1
        
    logL = np.tensordot(X, np.log(eps + Y), axes=([-2,-1],[-2,-1])) # logL:n*L*m
    logL -= np.sum(Y,axis=(-2,-1))

    return logL

def update_theta_j(theta0, x, Q, t, tau, topo, params, bnd=1000, bnd_beta=100, miter=1000):
    
    if restrictions==None:
        res = update_theta_j_unrestricted(theta0, x, Q, t, tau, topo, bnd, bnd_beta, miter)
    else:
        res = update_theta_j_restricted(theta0, x, Q, t, tau, topo, restrictions, bnd, bnd_beta, miter)
    return res

def update_theta_j_unrestricted(theta0, x, Q, t, tau, topo, r, bnd=10000, bnd_beta=1000, miter=10000):
    """
    with jac

    Parameters
    ----------
    theta0 : 1d array
        DESCRIPTION.
    x : 2d array
        DESCRIPTION.
    Q : 3d array
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
    bound[-1] = [1/bnd_beta,bnd_beta]
    
    n,L,m = Q.shape
    x_weighted = np.zeros((L,m,1))
    marginal_weight = np.zeros((L,m,1))
    for l in range(len(topo)):
        weight_l = Q[:,l,:] #n*m
        x_weighted[l] = weight_l.T@x # m*1 = m*n @ n*1
        marginal_weight[l] =  weight_l.sum(axis=0)[:,None] # m*1
    
    res = minimize(fun=neglogL, x0=theta0, args=(x_weighted,marginal_weight,t,tau,topo), method = 'L-BFGS-B' , jac = neglogL_jac, bounds=bound, options={'maxiter': miter,'disp': False}) 
    return res.x

def update_theta_j_restricted(theta0, x, Q, t, tau, topo, restrictions, bnd=1000, bnd_beta=100, miter=10000):
    # define a new neglogL inside with fewer parameters
    redundant, blanket = restrictions # 1,0 => a[1] = a[0], -3, -4 => s_0 = u_0*beta/gamma, 0,-4 => a[0] = u_0
    if len(redundant)+2 >= len(theta0):
        theta = np.ones(len(theta0))*np.mean(x[:,0])
        theta[-1] = 1
        
    else:
        redundant_mask = np.zeros(len(theta0), dtype=bool)
        redundant_mask[redundant] = True
        custom_theta0 = theta0[~redundant_mask]  
        bound = [[0,bnd]]*len(custom_theta0)
        bound[-1] = [1/bnd_beta,bnd_beta]
        
        n,L,m = Q.shape
        x_weighted = np.zeros((L,m,1))
        marginal_weight = np.zeros((L,m,1))
        for l in range(len(topo)):
            weight_l = Q[:,l,:] #n*m
            x_weighted[l] = weight_l.T@x # m*1
            marginal_weight[l] =  weight_l.sum(axis=0)[:,None] # m*1
     
        def custom_neglogL(custom_theta, x_weighted, marginal_weight, t, tau, topo):
            theta = np.zeros(len(theta0))
            theta[~redundant_mask] = custom_theta
            theta[redundant] = theta[blanket]
                
            return neglogL(theta, x_weighted, marginal_weight, t, tau, topo)
            
       
        res = minimize(fun=custom_neglogL, x0=custom_theta0, args=(x_weighted,marginal_weight,t,tau,topo), method = 'L-BFGS-B' , jac = None, bounds=bound, options={'maxiter': miter,'disp': False}) 
        theta = np.zeros(len(theta0))
        theta[~redundant_mask] = res.x
        theta[redundant] = theta[blanket]
        
    return theta


#%%
if __name__ == "__main__":

    np.random.seed(42)
    topo = np.array([[0,1]])
    tau=(0,0.5,1)
    n = 10000
    p = 1
    K=len(tau)-1
    t=np.linspace(0, 1, n)
    
    loga_max=4
    logb_max=2
    
    theta=np.ones((p,K+2))
    theta[0,:K+1]=np.exp(np.random.uniform(0,loga_max,size=K+1))-1
    theta[0,-1]=np.exp(np.random.uniform(-logb_max,logb_max,size=1))
    
    
    Y = get_Y(theta, t, tau)
    X = np.random.poisson(Y)
    x = X[:,0]
    Q = np.diag(np.ones(n))/n
    Q = Q[:,None,:]

    theta0 = np.zeros(K+2)
    theta0[0:-1]=np.mean(x[:,0])
    theta0[-1]=1
    theta0 += np.random.uniform(0,1,size = theta0.shape)

    res = update_theta_j(theta0, x, Q, t, tau, topo)
    print(res-theta[0])
    
    
    restrictions = np.array([[1],[-2]])
    redundant, blanket = restrictions
    
    
    theta[:,redundant]=theta[:,blanket]
    Y = get_Y(theta, t, tau)
    X = np.random.poisson(Y)
    x = X[:,0]
    res = update_nested_theta_j(theta0, x, Q, t, tau, topo, restrictions)    
    
    print(res-theta[0])
    
    import matplotlib.pyplot as plt
    y_fit = get_y(res, t, tau)
    plt.plot(y_fit,);
    plt.plot(Y[:,0]);

    
    
    
    
    
    
    
    
    
    
    
    
    