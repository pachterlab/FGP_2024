#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 14:21:36 2022

@author: fang
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 11:26:08 2022

@author: fang

This file defines functions (guess_theta, update_theta_j, update_theta_j, get_logL) for two species with Gaussian noise model.
The variance of Gaussian is set as the same as the mean, i.e. mu=var=y. 
2*logl = - ln var - (x-mu)**2/var
       = -ln y - (x-y)**2/y = - ln y - x^2/y -2x -y
theta for each gene contains transcription rate of each states, u_0, s_0, beta, gamma.

"""

import numpy as np
from scipy.optimize import minimize
#from scipy.special import gammaln


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


# a parameterization

def get_y(theta, t, tau):
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

def get_y_jac(theta, t, tau):
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


def neglogL(theta, x, Q, t, tau, topo):
    #  logl = -ln y - (x-y)**2/y = - ln y - x^2/y -2x -y
    # theta: length K+4
    # x: n*2
    # Q: n*L*m
    # t: len m
    # tau: len K+1
    logL = 0
    for l in range(len(topo)):
        theta_l = np.concatenate((theta[topo[l]], theta[-4:]))
        Y = get_y(theta_l,t,tau) # m*2
        logl = np.sum(- np.log(eps + Y[None,:]) - Y[None,:] - x[:,None]**2/(eps + Y[None,:]), axis=-1) # size (n,m)
        logL += np.sum(Q*logl)
    return - logL


def neglogL_jac(theta, x, Q, t, tau, topo):
    # theta: length K+4
    # x: n*2
    # Q: n*L*m
    # t: len m
    # tau: len K+1
    jac = np.zeros_like(theta)
    for l in range(len(topo)):
        theta_idx = np.append(topo[l],[-4,-3,-2,-1])
        theta_l = theta[theta_idx]
        Y, dY_dtheta = get_y_jac(theta_l,t,tau) # m*2*len(theta)
        coef =  np.sum( Q[:,l,:,None] * (- 1 / (eps + Y[None,:]) - 1 + x[:,None]**2/(eps + Y[None,:])**2) ,axis=0)
        jac[theta_idx] += np.sum( coef[:,:,None] * dY_dtheta, axis=(0,1))
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
    #logX = np.sum(gammaln(X+1),axis=(-2,-1),keepdims=True)
    #logL -= logX
    return logL

def update_theta_j(theta0, x, Q, t, tau, topo, bnd=10000, bnd_beta=1000, miter=1000):
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
    
    res = minimize(fun=neglogL, x0=theta0, args=(x,Q,t,tau,topo), method = 'L-BFGS-B' , jac = neglogL_jac, bounds=bound, options={'maxiter': miter,'disp': False}) 
    return res.x


def update_nested_theta_j(theta0, x, Q, t, tau, topo, restrictions, bnd=10000, bnd_beta=1000, miter=1000):
    return None



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.optimize import check_grad, approx_fprime
   
    np.random.seed(42)
    topo = np.array([[0,]])
    tau=(0,1)
    m = 100
    resol = 20
    n = m * resol
    p = 1
    K=len(tau)-1
    
    loga_max=4
    logb_max=2
    
    theta=np.ones((p,K+4))
    theta[0,:K+2]=np.exp(np.random.uniform(0,loga_max,size=K+2))-1
    theta[0,-2:]=np.exp(np.random.uniform(-logb_max,logb_max,size=2))
    
    Y = get_y(theta[0], np.repeat(np.linspace(0, 1, m), resol), tau)
    #X = np.random.normal(Y,1)
    X = np.random.poisson(Y)
    plt.scatter(X[:,0],X[:,1])
    x = X
    
#%% check jac function 
    t = np.linspace(0, 1, m)
    Q = np.zeros((n,1,m))
    for i in range(n):
        Q[i,0,i//resol]=1
        
    
    theta0 = np.zeros(K+4)
    theta0[0:-3]=np.mean(X[:,0])
    theta0[-3]=np.mean(X[:,1])
    theta0[-2]=1
    theta0[-1]=np.mean(X[:,0])/np.mean(X[:,1])
    theta0 += np.random.uniform(0,1,size = theta0.shape)
    
    check_grad(neglogL, neglogL_jac, theta[0], Y, Q, t, tau, topo)
    approx_fprime(theta[0], neglogL, 1e-10, Y, Q, t, tau, topo)
    neglogL_jac(theta[0], Y, Q, t, tau, topo)
    
    res = update_theta_j(theta0, Y, Q, t, tau, topo)    
    print(theta[0]-res.x)

    Y_fit = get_y(res.x, t, tau)

    plt.scatter(X[:,0],X[:,1],c='gray');
    plt.scatter(Y_fit[:,0],Y_fit[:,1],c=t,cmap='Spectral');
    #plt.scatter(Y[:,0],Y[:,1],c="red");


    
