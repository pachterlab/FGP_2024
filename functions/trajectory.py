#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 22:22:42 2022

@author: fang
"""

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from scipy.special import logsumexp
from basic import get_Y, update_theta_j, update_nested_theta_j
    
eps = 1e-10

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
    
    def _set_m(self,m):
        self.m=m
        self.t=np.linspace(self.tau[0],self.tau[-1],m)
    
    def _get_theta(self):
        return self.theta.copy()
    
    def _initialize_theta(self, X):
        p = X.shape[1]
        self.theta = np.zeros((p,self.n_states+4))
        self.theta[:,0:-3]=np.mean(X[:,:,0],axis=0)[:,None]
        self.theta[:,-3]=np.mean(X[:,:,1],axis=0)
        self.theta[:,-2]=1
        self.theta[:,-1] = np.mean(X[:,:,0],axis=0)/(np.mean(X[:,:,1],axis=0)+eps)
        return 
    
    def _initialize_Q(self, n):
        Q=10+np.random.uniform(0,1,(n,self.L,self.m))
        Q=Q/Q.sum(axis=(-2,-1),keepdims=True)
        return Q

    def update_theta(self,X,Q,gene_idx=None,parallel=False,n_threads=1,bnd=1000,bnd_beta=100,miter=1000):
        """

        """
        n,L,m = np.shape(Q)
        n,p,_ = np.shape(X)

        if not hasattr(self, 'theta'):
            self._initialize_theta(X)
            
        if gene_idx is None:
            gene_idx = np.arange(p)
            
        if parallel is True:
            Input_args = []
            for j in gene_idx:
                Input_args.append((self.theta[j], X[:,j], Q, self.t, self.tau, self.topo, bnd, bnd_beta, miter))
            with Pool(n_threads) as pool:      
                new_theta = pool.starmap(update_theta_j, Input_args)
            new_theta = np.array(new_theta)
        else:
            new_theta = np.zeros((len(gene_idx),self.n_states+4))
            for i,j in enumerate(gene_idx): 
                new_theta[i]=update_theta_j(self.theta[j], X[:,j], Q, self.t, self.tau, self.topo, bnd, bnd_beta,miter)
                
        self.theta[gene_idx] = new_theta
        return
    
    
    def update_nested_theta(self,X,Q,nested_model,parallel=False,n_threads=1,bnd=1000,bnd_beta=100,miter=1000):
        """

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        Q : TYPE
            DESCRIPTION.
        nested_model : TYPE
            DESCRIPTION.
        parallel : TYPE, optional
            DESCRIPTION. The default is False.
        n_threads : TYPE, optional
            DESCRIPTION. The default is 1.
        theta0 : TYPE, optional
            DESCRIPTION. The default is None.
        bnd : TYPE, optional
            DESCRIPTION. The default is 1000.
        bnd_beta : TYPE, optional
            DESCRIPTION. The default is 100.
        miter : TYPE, optional
            DESCRIPTION. The default is 1000.

        Raises
        ------
        TypeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        n,L,m = np.shape(Q)
        n,p,s=np.shape(X)
        if s!=2:
            raise TypeError("wrong parameters lengths")

        if not hasattr(self, 'theta'):
            self._initialize_theta(X)

        gene_idx = np.array(list(nested_model.keys()))
        if parallel is True:
            Input_args = []           
            for j in gene_idx:
                Input_args.append((self.theta[j], X[:,j], Q, self.t, self.tau, self.topo, nested_model[j], bnd, bnd_beta, miter))                
            with Pool(n_threads) as pool:      
                new_theta = pool.starmap(update_nested_theta_j, Input_args)
            new_theta = np.array(new_theta)
            
        else:
            new_theta = np.zeros((len(gene_idx),self.n_states+4))
            for i,j in enumerate(gene_idx):
                new_theta[i]=update_nested_theta_j(self.theta[j], X[:,j], Q, self.t, self.tau, self.topo, nested_model[j], bnd, bnd_beta, miter)
                
        self.theta[gene_idx] = new_theta
        return 


    def update_weight(self,X,beta=1):
        """
        calculate q with beta

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        Q : TYPE
            DESCRIPTION.
        lower_bound : TYPE
            DESCRIPTION.

        """
        if not hasattr(self, 'theta'):
            raise AttributeError("self.theta not defined")
            
        n,p,s=np.shape(X)
        Y = np.zeros((self.L,self.m,p,2))
        for l in range(self.L):
            theta_l = np.concatenate((self.theta[:,self.topo[l]], self.theta[:,-4:]), axis=1)
            Y[l] = get_Y(theta_l,self.t,self.tau) # m*p*2
        #logL =  np.sum(X[:,None,None,:,:] * np.log(Y[None,:]+eps) - Y[None,:], axis=(-2,-1)) # n*L*m*p*2 -> n*L*m
        logl = np.tensordot(X, np.log(eps + Y),axes=([-2,-1],[-2,-1]))
        logl -= np.sum(Y,axis = (-2,-1))
        #logL += np.log(self.prior_) where self.prior_ = 1/L/m
        logL = logl/beta
        #Q = softmax(logL, axis=(-2,-1))
        a = np.amax(logL,axis=(-2,-1))
        temp = np.exp(logL-a[:,None,None])
        tempt_sum = temp.sum(axis=(-2,-1))
        Q = temp/tempt_sum[:,None,None]
        if beta == 1:
            lower_bound = np.mean( np.log(tempt_sum) + a )  - np.log(self.m) - np.log(self.L)
        else:
            lower_bound = np.mean(logsumexp(logl, axis=(-2,-1))) - np.log(self.m) - np.log(self.L)
        return Q, lower_bound
    
    def fit_warm_start(self, X, Q=None, theta=None, epoch=10, tol=1e-4, parallel=False, n_threads=1):
        """
        The method fits the model by iterating between E-step and M-step for at most `epoch` iterations.
        The warm start means that either a reasonable Q or theta is provided.
    

        Parameters
        ----------
        X : ndarray, shape (n, p, 2)
            n cells * p genes * 2 species data matrix.
        Q : TYPE
            DESCRIPTION.
        epoch : TYPE, optional
            DESCRIPTION. The default is 10.
        tol : TYPE, optional
            DESCRIPTION. The default is 0.01.
        parallel : TYPE, optional
            DESCRIPTION. The default is False.
        n_threads : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        theta_hist : TYPE
            DESCRIPTION.
        weight_hist : TYPE
            DESCRIPTION.
        lower_bounds : TYPE
            DESCRIPTION.

        """
        
        n, p, _ = np.shape(X)
        
        
        if theta is not None:
            self.theta=theta.copy()
            if Q is not None:
                n, L, m = Q.shape
            else:
                m = n//100
            self._set_m(m)
            Q, lower_bound = self.update_weight(X)
            
        else:
            if Q is not None:
                n, L, m = Q.shape
                self._set_m(m)
                lower_bound = - np.inf
            else:
                raise AssertionError("either theta or Q needs to be provided")
                
        #self.prior_ = np.ones_like(Q)/L/m

        #theta_hist=[] 
        #weight_hist=[]
        #lower_bounds=[]
        #theta_hist.append(self._get_theta())
        #weight_hist.append(Q.copy())

        self.converged = False
        for i in tqdm(range(epoch)):
            prev_lower_bound = lower_bound
            self.update_theta(X,Q,parallel=parallel,n_threads=n_threads)
            Q, lower_bound = self.update_weight(X)
            print(lower_bound)
            #theta_hist.append(self._get_theta())
            #weight_hist.append(Q.copy())
            #lower_bounds.append(lower_bound)  
            #plot_phase(X,self.theta,Q,self.topo,self.tau)
            
            ## check converged
            change = lower_bound - prev_lower_bound
            if abs(change) < tol:
                self.converged = True
                break
                
        return [Q, lower_bound]
    
    def fit_multi_init(self, X, m, n_init=3, epoch=10, tol=1e-4, parallel=False, n_threads=1, seed=42):
        """
        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times

        Parameters
        ----------
        X : ndarray, shape (n, p, 2)
            DESCRIPTION.
        m : int
            DESCRIPTION.
        n_init : int, optional
            DESCRIPTION. The default is 3.
        epoch : int, optional
            DESCRIPTION. The default is 10.
        tol : float, optional
            DESCRIPTION. The default is 1e-3.
        parallel : bool, optional
            DESCRIPTION. The default is False.
        n_threads : int, optional
            DESCRIPTION. The default is 1.
        seed : int, optional
            DESCRIPTION. The default is 42.

        Returns
        -------
        list
            DESCRIPTION.

        """
        n, p, _ = np.shape(X)
     
        self._set_m(m)
        np.random.seed(seed)
        
        elbos = []
        thetas = []

        max_lower_bound = -np.inf
        for init in range(n_init):
            print("trial "+str(init+1))
            Q = self._initialize_Q(n)
            self._initialize_theta(X)
            lower_bound = -np.inf
            self.converged = False
            for i in tqdm(range(epoch)):
                print(lower_bound)
                prev_lower_bound = lower_bound
                self.update_theta(X,Q,parallel=parallel,n_threads=n_threads,miter=10000)
                #beta=max(epoch-i,1)
                beta=1
                Q, lower_bound = self.update_weight(X,beta)
                change = lower_bound - prev_lower_bound
                if abs(change) < tol:
                    self.converged = True
                    break
                
            if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                max_lower_bound = lower_bound
                best_theta = self._get_theta()
                best_Q = Q.copy()
                    
            elbos.append(lower_bound)
            thetas.append(self._get_theta())
            
        self.theta = best_theta
        return [best_Q, elbos, thetas]
    
    def compute_lower_bound(self,X):
        """
        Compute approximation (lower bound) of log P(X|theta)

        Parameters
        ----------
        X : 3d array 
            Data

        Returns
        -------
        scalar
            lower bound value
        """
        n,p,_=np.shape(X)
        Y = np.zeros((self.L,self.m,p,2))
        for l in range(self.L):
            theta_l = np.concatenate((self.theta[:,self.topo[l]], self.theta[:,-4:]), axis=1)
            Y[l] = get_Y(theta_l,self.t,self.tau) # m*p*2
        #logL =  np.sum(X[:,None,None,:,:] * np.log(Y[None,:]+eps) - Y[None,:], axis=(-2,-1)) # n*L*m*p*2 -> n*L*m
        logL = np.tensordot(X, np.log(eps + Y), axes=([-2,-1],[-2,-1])) # logL:n*L*m
        logL -= np.sum(Y,axis=(-2,-1))
        #logL += np.log(self.prior_) where self.prior_ = 1/L/m
        return np.mean(logsumexp(a=logL, axis=(-2,-1)))-np.log(self.m)-np.log(self.L)

    def compare_model(self, X, new_model, parallel=False, n_threads=1):
        """
        

        Parameters
        ----------
        nested_model : TYPE
            nested_model is a dict of restrictions. Keys are gene indices


        Returns
        -------
        accept: Bool
            Whether accept the nested model

        """
        if not hasattr(self, 'theta'):
            raise NameError("self.theta not defined. Run fit first")
        
        n, p, _ = X.shape
        
        ##### Store the old model and compute relative AIC #####
        if not hasattr(self, 'ori_theta'):
            self.ori_theta = self.theta.copy()
            self.ori_AIC = -2 * n * self.compute_lower_bound(X)
            
        ##### update theta with restrictions #####
        Q , _ = self.update_weight(X)
            
        self.update_nested_theta(X,Q,new_model,parallel=parallel,n_threads=n_threads)
        
        ##### check AIC with old theta + new nested theta #####
        k = 0
        for j in list(new_model.keys()):
            restrictions = new_model[j]
            redundant, blanket = restrictions
            k -= len(redundant)
            if len(redundant) > self.n_states:
                k -= 1
        Q , logL = self.update_weight(X)
        self.new_AIC = -2 * n * logL + 2 * k
        
        ##### if the nested model is better, return True #####
        if self.new_AIC < self.ori_AIC:
            accept = True  
        
        ##### else, update the whole theta and check agian #####
        else:      
            ##### update all theta with new weight #####
            gene_mask = np.ones(p,dtype=bool)
            gene_idx = np.arange(p)
            gene_mask[np.array(list(new_model.keys()))] = False
            
            self.update_theta(X,Q,gene_idx[gene_mask],parallel=parallel,n_threads=n_threads)      
            self.update_nested_theta(X,Q,new_model,parallel=parallel,n_threads=n_threads)
            
            ##### compute AIC and compare #####
            Q , logL = self.update_weight(X)
            self.new_AIC = -2 * n * logL + 2 * k
            
            if self.new_AIC < self.ori_AIC:
                accept = True  
            else:
                accept = False
                
        self.new_theta = self.theta.copy()
        self.theta = self.ori_theta.copy() # return to original model
        return accept, self.new_AIC - self.ori_AIC