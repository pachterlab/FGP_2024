#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 21:20:50 2022

@author: Meichen Fang
"""

"""
This file contains the class TI_model and its functions
    
"""
#%%
import numpy as np
from tqdm import tqdm
import time
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.special import logsumexp
from basic import *

# global parameters: upper and lower limits for numerical stability
eps = 1e-10
    np.random.seed(random_seed)
    L=len(topo)
    n_states=len(set(topo.flatten()))
    t=np.linspace(tau[0],tau[-1],n)
    theta=np.zeros((p,n_states+4))
    for k in range(n_states+2):
        theta[:,k]=np.exp(np.random.uniform(0,5,size=p))
    theta[:,-2]=np.exp(np.random.uniform(0,3,size=p))
    theta[:,-1]=np.exp(np.random.uniform(0,3,size=p))

    Y = np.zeros((n*L,p,2))
    for l in range(L):
        theta_l = np.concatenate((theta[:,topo[l]], theta[:,-4:]), axis=1)
        Y[l*n:(l+1)*n] = get_Y(theta_l,t,tau) # m*p*2

    X = np.random.poisson(Y)
    
    if loomfilepath is not None:
        adata=ad.AnnData(np.sum(X,axis=-1))
        adata.layers["spliced"] = X[:,:,1]
        adata.layers["unspliced"] = X[:,:,0]
        adata.layers["ambiguous"]=np.zeros_like(X[:,:,0])
        adata.obs["time"]=np.repeat(t,L)
        adata.obs["celltype"]=np.arange(n*L)//n
        adata.uns["theta"]=theta
        adata.var["true_beta"]=theta[:,-2]
        adata.var["true_gamma"]=theta[:,-1]
        adata.write_loom(loomfilepath)
    return theta, Y, X
    

#%%
class Trajectory:
    """
    Representation of a trajectory model probability distribution.
    
    
    Attributes
    ----------
    topo: 2D np.darray
    tau:     
    """

    def __init__(self, topo, tau):
        self.topo=topo
        self.tau=tau
        self.L=len(topo)
        self.n_states=len(set(topo.flatten()))
        return None
    
    def set_m(self,m):
        self.m=m
        self.t=np.linspace(self.tau[0],self.tau[-1],m)
    
    def _get_theta(self):
        return self.theta.copy()
    
    def _initialize_theta(self, X):
        p = X.shape[1]
        self.theta = np.ones((p,self.n_states+4))
        self.theta[:,0:-3]=np.mean(X[:,:,0],axis=0)[:,None]
        self.theta[:,-3]=np.mean(X[:,:,1],axis=0)
        self.theta[:,-1] = np.mean(X[:,:,0],axis=0)/(np.mean(X[:,:,1],axis=0)+eps)
        return 
    
    def _initialize_Q(self, n):
        Q=np.random.uniform(0,1,(n,self.L,self.m))
        Q=Q/Q.sum(axis=(-2,-1),keepdims=True)
        return Q

    def update_theta(self,X,Q,alternative=False,parallel=False,n_threads=1,theta0=None,bnd=1000,bnd_beta=100,miter=1000):
        """
        M-step
        beta and gamma can not be equal
        """
        n,L,m = np.shape(Q)
        n,p,s=np.shape(X)
        if s!=2:
            raise TypeError("wrong parameters lengths")

        if not hasattr(self, 'theta'):
            self._initialize_theta(X)

        if parallel is True:
            Input_args = []
            for j in range(p):
                Input_args.append((self.theta[j], X[:,j], Q, self.t, self.tau, self.topo, alternative, bnd, bnd_beta, miter))
            with Pool(n_threads) as pool:      
                new_theta = pool.starmap(update_theta_j, Input_args)
            new_theta = np.array(new_theta)
        else:
            new_theta = np.zeros((p,self.n_states+4))
            for j in range(p): 
                new_theta[j]=update_theta_j(self.theta[j], X[:,j], Q, self.t, self.tau, self.topo, alternative, bnd, bnd_beta,miter)
                
        self.theta = new_theta
        return

    def update_weight(self,X):
        """
        return 3D array posterior Q
        """
        if not hasattr(self, 'theta'):
            raise ValueError("self.theta not defined")
            
        n,p,s=np.shape(X)
        Y = np.zeros((self.L,self.m,p,2))
        for l in range(self.L):
            theta_l = np.concatenate((self.theta[:,self.topo[l]], self.theta[:,-4:]), axis=1)
            Y[l] = get_Y(theta_l,self.t,self.tau) # m*p*2
        #logL =  np.sum(X[:,None,None,:,:] * np.log(Y[None,:]+eps) - Y[None,:], axis=(-2,-1)) # n*L*m*p*2 -> n*L*m
        logL = np.tensordot(X, np.log(eps + Y),axes=([-2,-1],[-2,-1]))
        logL -= np.sum(Y,axis = (-2,-1))
        #logL += np.log(self.prior_) where self.prior_ = 1/L/m
        lower_bound = np.mean(logsumexp(logL, axis=(-2,-1)))
        #Q = softmax(logL, axis=(-2,-1))
        a = np.amax(logL,axis=(-2,-1),keepdims=True)
        temp = np.exp(logL-a)
        Q = temp/temp.sum(axis=(-2,-1),keepdims=True)
        return Q, lower_bound
    
    def fit(self, X, Q, relative_penalty=0, epoch=10, alternative=False, tol=0.01, parallel=False, n_threads=1):
        """
        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times
        X: n cells * p genes
        m grid of t=[0,1]
        """
        n, p, _ = np.shape(X)
        n, L, m = Q.shape
        penalty=relative_penalty*n
        
        self._initialize_theta(X)

        #self.prior_ = np.ones_like(Q)/L/m
        self.set_m(m)
        
        theta_hist=[] 
        weight_hist=[]
        lower_bounds=[]
        theta_hist.append(self._get_theta())
        weight_hist.append(Q.copy())

        alternative = False
        lower_bound = - np.inf
        self.converged = False
        time_start = time.time()
        for i in tqdm(range(epoch)):
            prev_lower_bound = lower_bound
            self.update_theta(X,Q,parallel=parallel,n_threads=n_threads)
            theta_hist.append(self._get_theta())
            Q, lower_bound = self.update_weight(X)
            weight_hist.append(Q.copy())
            lower_bounds.append(lower_bound)  
            #plot_phase(X,self.theta,Q,self.topo,self.tau)
            
            print(str(i)+" iteration: "+str(int(time.time()-time_start)),'s')
            # check converged
            change = lower_bound - prev_lower_bound
            if abs(change) < tol:
                self.converged = True
                break
                
        return theta_hist, weight_hist, lower_bounds
    
    def fit_(self, X, m, relative_penalty=0, n_init=3, epoch=10, tol=1e-6, parallel=False, n_threads=1, seed=42):
        """
        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times
        X: n cells * p genes
        m grid of t=[0,1]
        """
        n, p, _ = np.shape(X)
        penalty=relative_penalty*n
     
        self.set_m(m)
        np.random.seed(seed)
        
        elbos = []
        thetas = []
        alternative = False
        max_lower_bound = -np.inf
        for init in range(n_init):
            print("trial "+str(init+1))
            Q = self._initialize_Q(n)
            self._initialize_theta(X)
            lower_bound = -np.inf
            self.converged = False
            for i in tqdm(range(epoch)):
                prev_lower_bound = lower_bound
                self.update_theta(X,Q,parallel=parallel,n_threads=n_threads,miter=1000)
                Q, lower_bound = self.update_weight(X)
                change = lower_bound - prev_lower_bound
                if abs(change) < tol:
                    self.converged = True
                    break
            if not self.converged:
                self.update_theta(X,Q,parallel=parallel,n_threads=n_threads,miter=100000)
                Q, lower_bound = self.update_weight(X)
                
            if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                max_lower_bound = lower_bound
                best_theta = self._get_theta()
                    
            elbos.append(lower_bound)
            thetas.append(self._get_theta())
        self.theta = best_theta
        return Q, elbos, thetas
    
    def _compute_lower_bound(self,X,Q):
        n,p,s=np.shape(X)
        Y = np.zeros((self.L,self.m,p,2))
        for l in range(self.L):
            theta_l = np.concatenate((self.theta[:,self.topo[l]], self.theta[:,-4:]), axis=1)
            Y[l] = get_Y(theta_l,self.t,self.tau) # m*p*2
        #logL =  np.sum(X[:,None,None,:,:] * np.log(Y[None,:]+eps) - Y[None,:], axis=(-2,-1)) # n*L*m*p*2 -> n*L*m
        logL = np.tensordot(X, np.log(eps + Y), axes=([-2,-1],[-2,-1]))
        logL -= np.sum(Y,axis=(-2,-1))
        #logL += np.log(self.prior_) where self.prior_ = 1/L/m
        return np.mean(logsumexp(logL, axis=(-2,-1)))


#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.random.seed(2022)
    topo = np.array([[0,]])
    tau=(0,1)
    n = 2000
    p = 20
    K=len(tau)-1
    t=np.linspace(0, 1, n)
    
    theta=np.ones((p,K+4))*5
    loga_max=4
    logb_max=2
    
    for j in range(K+2):
        theta[:,j]=np.exp(np.random.uniform(0,loga_max,size=p))-1
    
    theta[:,-1]=5*(np.exp(np.random.uniform(-logb_max,logb_max,size=p)))
    theta[:,-3]=theta[:,-4]*theta[:,-2]/theta[:,-1]
    gamma = theta[:,-1]/5
    
    Y = get_Y(theta,t,tau)
    
    X = np.random.poisson(Y)
    
    c = np.random.lognormal(-0.005,0.1,n).reshape((-1,1,1))
    
    D = np.random.poisson(c*X)
    
    fig, ax = plt.subplots(1,10,figsize=(6*10,4))
    for i in range(10):
        ax[i].plot(Y[:,i,1]*gamma[i],Y[:,i,1],'--', color='gray');
        ax[i].scatter(X[:,i,0],X[:,i,1],c=t);
        ax[i].scatter(Y[:,i,0],Y[:,i,1],c='black');
        
        
    traj_D = Trajectory(topo, tau)
    Q, elbos = traj_D.fit_(D, 100, n_init=3, parallel=True, n_threads=4)
    
    plot_phase(D,traj_D.theta,Q,topo,tau)
    plot_theta(theta,traj_D.theta)
    plot_phase(D,theta,Q,topo,tau)
    
    
    traj_X = Trajectory(topo, tau)
    Q, elbos = traj_X.fit_(X, 100, n_init=3, parallel=True, n_threads=4)
    
    plot_phase(X,traj_X.theta,Q,topo,tau)
    plot_theta(theta,traj_X.theta)
    plot_phase(X,theta,Q,topo,tau)
    
    plot_theta(traj.theta,traj_X.theta)
    
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    