#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 15:23:57 2022

@author: fang

theta = a_1,...a_K, u0, s0, beta, gamma, b
"""


# Scientific computing imports
import numpy as np
from numpy.fft import irfftn
from scipy.optimize import minimize
#from scipy.special import gammaln


# global parameters: upper and lower limits for numerical stability
eps = 1e-10

def guess_theta(X,n_states):
    p = X.shape[1]
    theta = np.zeros((p,n_states+5))
    theta[:,0:n_states+2] = np.mean(X[:,:,0],axis=0)[:,None]
    theta[:,n_states+2] = 1
    theta[:,n_states+3] = np.mean(X[:,:,0],axis=0)/(np.mean(X[:,:,1],axis=0)+eps)
    theta[:,n_states+4] = 1
    return theta


def get_logP(theta, t_array, tau, mx):
    ## theta = a_1,..., b, u0, s0, beta, gamma
    ##
    ## outline:
    ##   get u and flatten it => u 
    ##   use cumtrapz to integrate phi which has shape (len(t), mx[0]*(mx[1]//2+1) )
    ##   generating function gf=np.exp(phi)
    ##   use irfftn(gf) to get P which has shape(len(t), mx[0], mx[1])

    K = len(tau)-1
    kvec = np.array(theta[0:K])
    #bvec = np.array(theta[-1])
    u0,s0,beta,gamma,b = theta[K:]
    
    ## Get generating function argument u
    us = []
    half = mx[:]
    half[-1] = mx[-1]//2 + 1
    for i in range(len(mx)):
        l = np.arange(half[i])
        u_ = np.exp(-2j*np.pi*l/mx[i])-1
        us.append(u_)
    u = np.meshgrid(*[u_ for u_ in us], indexing='ij')
    for i in range(len(mx)):
        u[i] = u[i].flatten()

    ## Get generating function
    ## M(u)=1/(1 - bu)
    ## phi = int_0^t k*bU/(1-bU) + phi_0
    ## q = bU 
   
    c0 = (u[0]) - (beta/(beta - gamma))*(u[1])       #  coef of e^{-beta s}
    c1 = (beta/(beta - gamma))*(u[1])   # coef of e^{-gamma s}

    
    # get k(T-t) k_array[i] = k(T - t_array[i])
    t_array = np.reshape(t_array,(-1,1))
    k_array=np.ones_like(t_array)*kvec[0]

    for k in range(1,K):
      idx = tau[k]<(t_array[-1]-t_array)
      k_array[idx] = k_array[idx] + kvec[k] - kvec[k-1]
      #b_array[idx] = b_array[idx] + bvec[k] - bvec[k-1]

    c0 = c0.reshape((1,-1))
    c1 = c1.reshape((1,-1)) 
    
    phi = np.zeros((len(t_array),len(u[0])), dtype=np.complex64)    # initialize array to store ODE  
    U = c0*np.exp(-beta*t_array) + c1*np.exp(-gamma*t_array)    
    for i in range(2,len(t_array)+1):       
      #q = b_array[-i:]*U[:i,:]
      integrand = k_array[-i:]*b*U[:i,:]/(1-b*U[:i,:])
      integral = np.trapz(y=integrand, x=t_array[:i], axis=0)     # integrate ODE solution                   
      phi[i-1,:] = integral

    gf = np.exp(phi)               # get generating function
    gf = gf.reshape((len(t_array),half[0],half[1])) 
    P = irfftn(gf, s=mx)
    return np.log(P)


def neglogL(theta, x, weight, t, tau, topo):
    # theta: length K+4
    # x: n*2
    # Q: n*L*m
    # t: len m
    # tau: len K+1
    logL = 0
    mx = [x[:,0].max()+10,x[:,1].max()+10]
    for l in range(len(topo)):
        theta_l = np.concatenate((theta[topo[l]], theta[-5:]))
        logP = get_logP(theta_l,t,tau,mx) # m*mx1*mx2
        logL += np.sum( weight[l] * logP[:,x]  )
    return - logL



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

def update_theta_j(theta0, x, Q, t, tau, topo, bnd=1000, bnd_beta=100, miter=1000):
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
    
    res = minimize(fun=neglogL, x0=theta0, args=(x_weighted,marginal_weight,t,tau,topo), method = 'L-BFGS-B' , jac = neglogL_jac, bounds=bound, options={'maxiter': miter,'disp': False}) 
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
                
            return neglogL(theta, x_weighted, marginal_weight, t, tau, topo)
            
       
        res = minimize(fun=custom_neglogL, x0=custom_theta0, args=(x_weighted,marginal_weight,t,tau,topo), method = 'L-BFGS-B' , jac = None, bounds=bound, options={'maxiter': miter,'disp': False}) 
        theta = np.zeros(len(theta0))
        theta[~redundant_mask] = res.x
        theta[redundant] = theta[blanket]
        if -3 in redundant:
            theta[-3] = theta[-4]*theta[-2]/theta[-1]
            
    return theta