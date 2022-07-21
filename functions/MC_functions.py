#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 21:17:32 2022

@author: Meichen Fang
"""

#%% packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from math import exp
from scipy.stats import rankdata
import time 

#%% sim data
def eval_x_interval(k_,dt,beta,gamma,x0):
    '''
    Evaluates a single gene's average expression at a time point dt after
    the system is started. This solution results from the reaction
    rate equations, and is specifically implemented for a time interval wherein
    the parameter values are constant.
    
    Parameters
    ----------
    k_: transcription rate (constitutive)
    dt: time since system is started
    beta: splicing rate
    gamma: degradation rate
    x0: average expression at t=0
    
    Returns
    -------
    x: average expression value (vector, unspliced and spliced)
    '''
    gamma /= beta
    dt *= beta
    x = np.array([k_*(1-np.exp(-dt)) + x0[0]*np.exp(-dt),
        np.exp(-dt*gamma) * \
        (np.exp(dt*gamma) * k_ * (gamma-1) + \
        - k_*np.exp(dt*(gamma-1))*gamma \
        + k_\
        + (gamma-1)*gamma*x0[1] \
        +gamma*x0[0]*(np.exp((gamma-1)*dt)-1)) \
        /gamma/(gamma-1)])
    return x

def eval_x(k,tau,t,beta,gamma,x0=None):
    '''
    Evaluates a single gene's average expression at a time point t.
    This solution derives from the reaction rate equations. The value is
    computed by evaluating the RREs piecewise over succeeding intervals
    until time t is reached.
    
    Parameters
    ----------
    k: transcription rate (constitutive, given by a vector over "cell types")
    tau: cell type transition times
    t: time of interest
    beta: splicing rate
    gamma: degradation rate
    x0: average expression at t=0 (in this case, we assume it starts with Poisson)
    
    Returns
    -------
    x: average expression value (vector, unspliced and spliced)
    '''
    # x=x0
    if x0 is None:
        x0 = [k[0], k[0]/gamma*beta]
    x=x0
    ind_last = sum(tau<=t)
    tau = np.concatenate((tau[:(ind_last)],[t]))
    for ind in range(ind_last):
        x = eval_x_interval(k[ind],tau[ind+1]-tau[ind],beta,gamma,x)
    return x

#%% infer_theta
def negloglike(theta, x, t, tau):
    # 
    # tau = tau_0=0, tau_1,...,tau_K=1
    # a_0,..., a_K, beta = theta.flatten()
    K = len(tau)-1
    if len(theta)!=K+2:
      raise TypeError("wrong parameters lengths")
    beta = theta[-1]
    a=theta[0:(K+1)]
    x = x.reshape(-1,1)
    n = len(x)
    t = t.reshape(-1,1)
    eps = 10**(-6)
  
    I = np.ones((K+1,n),dtype=bool)
    y=np.zeros_like(x)
    y=y+a[0]*np.exp(-beta*t)
    for k in range(1,K+1):
      I[k] = np.squeeze(t > tau[k])
      idx=I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
      y= y + I[k,:,None] * a[k] * (np.exp(-beta*(t-tau[k]))- np.exp(-beta*(t-tau[k-1])) ) \
          + idx[:,None] * a[k] * (1 - np.exp(-beta*(t-tau[k-1])))
  
    logL =  np.sum(x * np.log( eps + y) - y )  
    
    return -logL

def negloglike_jac(theta, x, t, tau):
  # tau = tau_0=0, tau_1,...,tau_K=1
  # a_0,..., a_K, beta = theta.flatten()
  eps = 10**(-6)
  K = len(tau)-1
  if len(theta)!=K+2:
    raise TypeError("wrong parameters lengths")
  beta = theta[-1]
  a=theta[0:(K+1)]
  x = x.reshape(-1,1)
  n = len(x)
  t = t.reshape(-1,1)

  I = np.ones((K+1,n),dtype=bool)
  y=np.zeros_like(x)
  y_beta=np.zeros_like(x) # partial y partial beta

  y=y+a[0]*np.exp(-beta*t)
  y_beta=y_beta-t*a[0]*np.exp(-beta*t)

  y_jac = np.zeros((n,K+2))
  y_a0 =  np.exp(-beta*t)
  y_jac[:,0]=y_a0[:,0]

  for k in range(1,K+1):
    I[k] = np.squeeze(t > tau[k])
    idx=I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
    y_ak=I[k,:,None] * (np.exp(-beta*(t-tau[k]))- np.exp(-beta*(t-tau[k-1])) ) \
        + idx[:,None] * (1 - np.exp(-beta*(t-tau[k-1])))
    y_beta=y_beta- a[k]*I[k,:,None] * ((t-tau[k])*np.exp(-beta*(t-tau[k]))- (t-tau[k-1])*np.exp(-beta*(t-tau[k-1])) )  \
        + idx[:,None] * a[k] * (t-tau[k-1]) * np.exp(-beta*(t-tau[k-1]))
    y = y +  a[k] * y_ak
    y_jac[:,k]=y_ak[:,0]
  y_jac[:,K+1]=y_beta[:,0]
  jac= - np.sum(y_jac*(x/(eps + y) - 1),axis=0)
  return jac

def infer_theta(X,t,tau,theta0=None):
  # X:n*p
  # t:n*1
  # theta0:p*(K+2)
  n,p=np.shape(X)
  K=len(tau)-1
  bounds = ([[0, np.inf]]*(K+2))
  if theta0 is None:
    theta0 = np.ones((p,K+2))
    theta0[:,0:(K+1)]=np.mean(X,axis=0)[:,None]
  theta_hat = np.zeros((p,K+2))
  for i in range(p):  
    res = minimize(negloglike, theta0[i], args=(X[:,i],t,tau), method="L-BFGS-B", jac=negloglike_jac, bounds=bounds, options={'maxiter':100000,'disp': True}) 
    theta_hat[i,:] = res.x
  return theta_hat

def negloglike_jac_one(theta, x, t):
    eps = 10**(-6)
    x = x.reshape(-1,1)
    t = t.reshape(-1,1)
    a, b, beta = theta
    y = a + b * np.exp(-beta*t) + eps 
    jac_a =  np.sum(x/y - 1)
    jac_b =  np.sum((x/y - 1)*np.exp(-beta*t))
    jac_beta =  -b*np.sum( (x/y - 1)*t*np.exp(-beta*t))
    return -np.array([jac_a,jac_b,jac_beta])

    

def get_y(theta, t, tau):
    p = len(theta)
    K = len(tau)-1 # number of states
    if np.shape(theta)[1]!=K+2:
      raise TypeError("wrong parameters lengths")
    beta = theta[:,-1]
    a=theta[:,0:(K+1)]
    t = t.reshape(-1,1)
    n = len(t)
  
    I = np.ones((K+1,n),dtype=bool)
    y=np.zeros((n,p))
    y=y+a[None,:,0]*np.exp(-beta[None,:]*t)
    for k in range(1,K+1):
      I[k] = np.squeeze(t > tau[k])
      idx=I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
      y= y + I[k,:,None] * a[None,:,k] * (np.exp(-beta[None,:]*(t-tau[k]))- np.exp(-beta[None,:]*(t-tau[k-1])) ) \
          + idx[:,None] * a[None,:,k] * (1 - np.exp(-beta[None,:]*(t-tau[k-1]))) 
    return y

#%% infer t
def h(u):
  return min(u,1)
  
def V(t,x,theta,tau):
  # calculate negative log likelihood of one cell x given time t and paramters theta
  # t: scalar in [0,1]
  # x: gene expression of a cell, (p,)
  # theta: p*3
  # tau: gloabal parameters
  K=len(tau)-1
  beta = theta[:,-1]
  a=theta[:,0:(K+1)]
  y=a[:,0]*np.exp(-beta*t)
  k=1
  while(tau[k]<t):
    y=y + a[:,k] * (np.exp(-beta*(t-tau[k]))- np.exp(-beta*(t-tau[k-1])) ) 
    k=k+1
  y=y+a[:,k] * (1 - np.exp(-beta*(t-tau[k-1])) )

  eps = 10**(-6)
  logL =  np.sum(x * np.log( eps + y) - y ) 
  return -logL

def infer_t(X,theta,tau,t0=None,iters=10000):
    """
    Given gene parameter theta and global parameter tau, infer cell parameters t
    # X: n*p
    # theta: p*(K+2)
    # tau: (0,...,1)
    """
    n,p=np.shape(X)
    if t0 is None:
        t0=np.mean(np.argmin(np.abs(X[:,:,None]-theta[None,:,:-1]),axis=2),axis=1)
        pi=(rankdata(t0,method='ordinal')-1)/(n-1)
    else:
        pi=t0
    T=0.1*p/np.log(2+np.arange(iters))
    U = np.random.uniform(0,1,size=iters)
    for k in range(iters):
      i,j=np.random.choice(n,size=2,replace=False)
      dV=V(pi[i],X[i,:],theta,tau)+V(pi[j],X[j,:],theta,tau)-V(pi[j],X[i,:],theta,tau)-V(pi[i],X[j,:],theta,tau)
      if dV/T[k]>0:
        h_=1
      else:
        h_ = h(exp(dV/T[k]))
      if U[k]<h_:
        temp=pi[i]
        pi[i]=pi[j]
        pi[j]=temp
    return pi


def infer_(X,tau,Epoch=20,seed=63):
    np.random.seed(seed)
    K=len(tau)-1
    n,p=np.shape(X)
    theta = np.ones((p,K+2))
    theta[:,0:(K+1)]=np.mean(X,axis=0)[:,None]
    pi=np.random.permutation(n)/(n-1)
    for k in range(Epoch):
      theta=infer_theta(X,pi,tau,theta)
      pi=infer_t(X,theta,tau,t0=pi,iters=1000*(k+1))
    return theta, pi