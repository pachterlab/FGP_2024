#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 21:20:50 2022

@author: Meichen Fang
"""

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# global parameters: upper and lower limits for numerical stability
eps = 1e-10
omega = -1e6


def get_Y(theta, t, tau):
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



def neglogL(theta,x,weight,t,tau):
    # theta: length K+4
    # x: n*2
    # weight: n*m
    # t: len m
    # tau: len K+1
    Y = get_Y(theta[None,:],t,tau) # m*1*2
    Y = Y[None,:,0,:] # 1*m*2
    logL = np.sum( weight[:,:,None] * (x[:,None,:] * np.log( eps + Y) - Y ) )  
    return - logL
    
def minimize_wrapper(args):
    theta0, x, weight, t, tau, bounds = args
    res = minimize(neglogL, theta0, args=(x,weight,t,tau), bounds=bounds, options={'maxiter':100000,'disp': False}) 
    return res.x

def update_theta(X,weight,tau,parallel=False,n_threads=1,theta0=None):
    n,p,s=np.shape(X)
    if s!=2:
      raise TypeError("wrong parameters lengths")
    n,m=np.shape(weight)
    t=np.linspace(0,1,m)
    K=len(tau)-1
    bounds = ([[0, np.inf]]*(K+4))
    if theta0 is None:
        theta0 = np.ones((p,K+4))
        theta0[:,0:(K+2)]=np.mean(X[:,:,0],axis=0)[:,None]
        gamma_hat = np.mean(X[:,:,0],axis=0)/(np.mean(X[:,:,1],axis=0)+eps)
        theta0[:,-1] = gamma_hat

    if parallel is True:
        Input_args = []
        for i in range(p):
            Input_args.append((theta0[i], X[:,i], weight, t, tau, bounds))
        with Pool(n_threads) as pool:      
            theta_hat = pool.map(minimize_wrapper, Input_args)
        theta_hat = np.array(theta_hat)
    else:
        theta_hat = np.zeros((p,K+4))
        for i in range(p):  
          res = minimize(neglogL, theta0[i], args=(X[:,i],weight,t,tau), bounds=bounds, options={'maxiter':100000,'disp': True}) 
          theta_hat[i,:] = res.x
    return theta_hat

def update_weight(X,theta,tau,m):
    n,p,s=np.shape(X)
    t=np.linspace(0,1,m)
    Y = get_Y(theta,t,tau) # m*p*2
    Y = Y[None,:,:,:] # 1*m*p*2
    logL =  np.sum(X[:,None,:,:] * np.log(Y+eps) - Y, axis=(2,3)) # n*m*p*2 -> n*m
    w = np.zeros((n,m))
    for i in range(n):
        c = np.max(logL[i,:])
        relative_logL = logL[i,:]-c        
        relative_logL[relative_logL<omega]=omega
        L = np.exp(relative_logL)
        w[i,:] = L/L.sum()
    if np.sum(np.isnan(w)) != 0:
        raise ValueError("Nan in weight")
    return w

def traj_EM(X, tau, m=101, epoch=20, parallel=False, n_threads=1):
    """
    X: n cells * p genes
    m grid of t=[0,1]
    """
    eps = 10**(-10)
    n,p,_=np.shape(X)
    K=len(tau)-1
    theta_hat = np.ones((p,K+4))
    theta_hat[:,0:(K+1)]=np.mean(X[:,:,0],axis=0)[:,None]
    theta_hat[:,K+1]=np.mean(X[:,:,1],axis=0)
    theta_hat[:,-1] = np.mean(X[:,:,0],axis=0)/(np.mean(X[:,:,1],axis=0)+eps)
    weight=np.ones((n,m))/m

    for i in tqdm(range(epoch)):
        theta_hat = update_theta(X,weight,tau,theta0=theta_hat,parallel=parallel,n_threads=n_threads)
        weight = update_weight(X,theta_hat,tau,m)

    return theta_hat, weight


def plot_t(weight,ax=None,t=None):
    m=np.shape(weight)[1]
    h=np.linspace(0,1,m)
    t_hat=np.sum(weight*h[None,:],axis=1)
    if ax is None:
        fig, ax = plt.subplots(1,1)
    if t is not None:
        ord=np.argsort(t)
        # build a rectangle in axes coords
        left, width = .25, .5
        bottom, height = .25, .5
        right = left + width
        top = bottom + height
       
        ax.imshow(weight[ord,:],aspect="auto");
        ax.text(right, top,"cor="+str(np.around(np.corrcoef(t_hat,t)[0,1],2)), horizontalalignment='right', 
                 verticalalignment='top', transform=ax.transAxes, color="white");
    else:
        ord=np.argsort(t_hat)
        ax.imshow(weight[ord,:],aspect="auto");

def plot_theta(theta,theta_hat,K):
    fig, ax = plt.subplots(1,K+4,figsize=(6*(K+4),4))

    ax[0].plot(theta[:,-4],theta[:,-4]);
    ax[0].plot(theta[:,-4],theta_hat[:,-4],'.');
    ax[0].set_title("u0")
    ax[0].set_ylabel("fitted values")
    ax[0].set_xlabel("true values")

    ax[1].plot(theta[:,-3],theta[:,-3]);
    ax[1].plot(theta[:,-3],theta_hat[:,-3],'.');
    ax[1].set_title("s0")
    ax[1].set_ylabel("fitted values")
    ax[1].set_xlabel("true values")

    ax[2].plot(theta[:,0],theta[:,0]);
    ax[2].plot(theta[:,0],theta_hat[:,0],'.');
    ax[2].set_title("a1")
    ax[2].set_xlabel("true values")

    ax[3].plot(theta[:,1],theta[:,1]);
    ax[3].plot(theta[:,1],theta_hat[:,1],'.');
    ax[3].set_title("a2")
    ax[3].set_xlabel("true values")

    ax[-2].plot(theta[:,-2],theta[:,-2]);
    ax[-2].plot(theta[:,-2],theta_hat[:,-2],'.');
    ax[-2].set_title("beta");
    ax[-2].set_xlabel("true values");

    ax[-1].plot(theta[:,-1],theta[:,-1]);
    ax[-1].plot(theta[:,-1],theta_hat[:,-1],'.');
    ax[-1].set_title("gamma");
    ax[-1].set_xlabel("true values");


def plot_theta_hat(theta_hat,K,gene_list):
    fig, ax = plt.subplots(K+4,1,figsize=(12,4*(K+4)))

    for i in range(K+4):
        ax[i].scatter(np.array(gene_list), np.log(theta_hat[:,i]));

    ax[0].set_title("log a1");
    ax[1].set_title("log a2");
    ax[2].set_title("log u0");
    ax[3].set_title("log s0");
    ax[4].set_title("log beta");
    ax[5].set_title("log gamma");


def plot_y(theta_hat,weight,tau,gene_list):
    m=np.shape(weight)[1]
    p=np.shape(theta_hat)[0]
    h=np.linspace(0,1,m)
    t_hat=np.sum(weight*h[None,:],axis=1)
    y_hat = get_Y(theta_hat, t_hat, tau)
    fig, ax = plt.subplots(p,2,figsize=(12,4*p))
    for i in range(p):
        ax[i,0].plot(t_hat,X[:,i,0],'.');
        ax[i,0].plot(t_hat,y_hat[:,i,0],'r.');
        ax[i,0].set_title(gene_list[i]+" unspliced")

        ax[i,1].plot(t_hat,X[:,i,1],'.');
        ax[i,1].plot(t_hat,y_hat[:,i,1],'r.');
        ax[i,1].set_title(gene_list[i]+" spliced")

