#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 15:23:57 2022

@author: fang

theta = a_1,...a_K, beta, gamma, b
"""


# Scientific computing imports
import numpy as np
from numpy.fft import irfftn
from scipy.optimize import minimize
#from scipy.special import gammaln


# global parameters: upper and lower limits for numerical stability
eps = 1e-100

def guess_theta(X,n_states):
    p = X.shape[1]
    theta = np.zeros((p,n_states+3))
    theta[:,0:n_states] = np.mean(X[:,:,0],axis=0)[:,None]
    theta[:,-2] = np.mean(X[:,:,0],axis=0)/(np.mean(X[:,:,1],axis=0)+eps)
    theta[:,-1] = 1
    return theta


def get_logP(theta, t_array, tau, mx):
    """
    
    ## theta = a_1,..., b, beta, gamma
    ##
    ## outline:
    ##   get u and flatten it => u 
    ##   use cumtrapz to integrate phi which has shape (len(t), mx[0]*(mx[1]//2+1) )
    ##   generating function gf=np.exp(phi)
    ##   use irfftn(gf) to get P which has shape(len(t), mx[0], mx[1])

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    t_array : TYPE
        DESCRIPTION.
    tau : TYPE
        DESCRIPTION.
    mx : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    K = len(tau)-1
    kvec = np.array(theta[0:K])
    #bvec = np.array(theta[-1])
    b,beta,gamma = theta[K:]
    
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
      integrand = k_array[-i:]*b*U[:i,:]/(1-b*U[:i,:])
      #integrand = k_array[-i:]*b*U[:i,:] #Poisson
      integral = np.trapz(y=integrand, x=t_array[:i], axis=0)     # integrate ODE solution                   
      phi[i-1,:] += integral
      
    gf = np.exp(phi)               # get generating function
    gf = gf.reshape((len(t_array),half[0],half[1])) 
    P = irfftn(gf, s=mx)
    P[P<eps]= eps
    return np.log(P)


def neglogL(theta, x, mx, weight, t, tau, topo):
    """
    

    Parameters
    ----------
    theta : 1d array of length n_states+3
        theta = a_1,..., b, beta, gamma.
    x : 2d array of shape (n,2)
        DESCRIPTION.
    mx : list of length 2
        x range.
    weight : 3d array of shape (n,L,m)
        DESCRIPTION.
    t : array of shape (m,)
        DESCRIPTION.
    tau : list of length K+1
        DESCRIPTION.
    topo : 2d array of shape (L,K)
        DESCRIPTION.

    Returns
    -------
    scalar
        negative likelihood of this gene summed over all samples.

    """
    logL = 0
    for l in range(len(topo)):
        theta_l = np.concatenate((theta[topo[l]], theta[-3:]))
        logP = get_logP(theta_l,t,tau,mx) # m*mx1*mx2
        logL += np.sum( weight[l].T * logP[:,x[:,0],x[:,1]] ) # logP[:,x]:(m,n)
    return - logL



def get_logL(X,theta,t,tau,topo):
    """
    

    Parameters
    ----------
    X : 3d array with size (n,p,2)
        DESCRIPTION.
    theta : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    tau : TYPE
        DESCRIPTION.
    topo : TYPE
        DESCRIPTION.

    Returns
    -------
    logL : (n,L,m)
        DESCRIPTION.

    """
    L = len(topo)
    m = len(t)
    n, p, _ = X.shape
    
    logL = np.zeros((n,L,m))
    
    for j in range(p):
        x = X[:,j]
        mx = [x[:,0].max().astype(int)+10,x[:,1].max().astype(int)+10]
        for l in range(len(topo)):
            theta_l = np.concatenate((theta[j,topo[l]], theta[j,-3:]))
            logP = get_logP(theta_l,t,tau,mx) # m*mx1*mx2
            logL[:,l] += logP[:,x[:,0],x[:,1]].T # logP[:,x]:(m,n)
    
    return logL

def update_theta_j(theta0, x, Q, t, tau, topo, bnd=1000, bnd_beta=100, miter=1000000):
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
    mx = [x[:,0].max().astype(int)+10,x[:,1].max().astype(int)+10]
    res = minimize(fun=neglogL, x0=theta0, args=(x,mx,Q,t,tau,topo), method = 'L-BFGS-B' , jac = None, bounds=bound, options={'maxiter': miter,'disp': True}) 
    return res

if __name__ == "__main__":  
    from simulation import Gillespie_bursty_2D
    import time 
    from scipy import stats
    import matplotlib.pyplot as plt

    nCells = 1000
    topo=np.array([[0]])
    tvec = np.linspace(0,1,101)
    tau = [0,1]
    kvec = [10.0]
    bvec = [5.0]
    beta, gamma = 5.0, 10.0
    theta = kvec + bvec + [beta, gamma]
    X = Gillespie_bursty_2D(nCells, tvec, np.array([0,0],dtype=int), kvec, tau, beta, gamma, bvec)
 
    mx = [X[:,:,0].max().astype(int)+10,X[:,:,1].max().astype(int)+10]
    
    t1 = time.time()
    logP = get_logP(theta, tvec, tau, mx)
    pmf = np.exp(logP)
    t2 = time.time()
    print(t2-t1)
                       
    bins = np.arange(mx[0])-0.5
    x = np.arange(mx[0])
    time_points=np.array(np.array([0.01,0.1,0.2,0.3,0.5,0.7,0.99])*len(tvec),
                         dtype=int)
    fig, ax = plt.subplots(1,len(time_points),figsize=(6*len(time_points),4))
    for i,j in enumerate(time_points):
        y = pmf[j].sum(axis=1)
        ax[i].plot(x,y,'r.',label='extracted pmf')
        ax[i].hist(X[:,j,0],bins=bins,density=True,facecolor='lightgray')
    y = stats.nbinom.pmf(x,kvec[-1]/beta,1/(1+bvec[-1]))
    plt.plot(x,y,'g.',label='NB')
    y = stats.poisson.pmf(x,kvec[-1]*bvec[-1]/beta)
    plt.plot(x,y,'b.',label='Poisson')
    plt.legend();
    
    bins = np.arange(mx[1])-0.5
    x = np.arange(mx[1])
    fig, ax = plt.subplots(1,len(time_points),figsize=(6*len(time_points),4))
    for i,j in enumerate(time_points):
        ax[i].hist(X[:,j,1],bins=bins,density=True,facecolor='lightgray')
        y = pmf[j].sum(axis=0)
        ax[i].plot(x,y,'r.',label='extracted pmf')
    plt.legend();
    #plt.xlim([-1,100]);
        
    u = X[:,:,0].T.flatten().astype(int)
    s = X[:,:,1].T.flatten().astype(int)
    x = np.zeros((len(u),2), dtype=int)
    x[:,0] = u
    x[:,1] = s 
    plt.plot(x[:,0],'.')
    plt.plot(x[:,1],'.')
    
    
    n = len(u)
    L = len(topo)
    K = len(tau)-1
    resol = 1000 
    m = n//resol
    t = np.linspace(0,1,m)
    Q0 = np.zeros((L*n,L,m))
    for l in range(L):
        for i in range(n):
            Q0[i+n*l,l,i//resol] = 1

    import two_species as two
    theta0 = np.ones(K+4)*np.mean(x[:,0])
    theta0[-3]=np.mean(x[:,1])
    theta0[-2]=1
    theta0[-1]=np.mean(x[:,0])/(np.mean(x[:,1])+eps)
    theta0 += np.random.uniform(0,1,size = theta0.shape)
    res = two.update_theta_j(theta0, x, Q0, tvec, tau, topo=np.array([[0]]),miter=1000)
    print(res)    
    
    
    theta0 = np.ones(K+3)*np.mean(x[:,0])
    theta0[-3]=1
    theta0[-2]=1
    theta0[-1]=2
    theta0 += np.random.uniform(0,1,size = theta0.shape)
    
    res = update_theta_j(theta0, x, Q0, t, tau, topo=np.array([[0]]))
    print(res)    

    
    plt.scatter(x[:,0],x[:,1],c=np.arange(n))

    
    
    
    
    
    
    
    
    
    
    
    
    