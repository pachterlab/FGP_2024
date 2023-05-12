#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 2023

@author: fang
"""

import numpy as np
from scipy.special import logsumexp, softmax, gammaln
from scipy.stats import poisson, nbinom
from tqdm import tqdm


    
eps = 1e-6
Omega = 1e6


class PoissonMixture:
    """
    Poisson mixture model

    Attributes
    ----------
    n_components : int, number of mixtures, default=1
    theta : ndarray of shape (n_components, p, 2). 
            Means of Poisson distribution for U and S of each gene
    weights : ndarray of shape (n_components,). Weights of each mixture as in GMM.
    """

    def __init__(self, n_components=1, verbose=0):
        self.n_components = n_components
        self.verbose = verbose    
        return
    
    def _get_parameters(self):
        return self.theta.copy(), self.weights.copy()
        
    def _initialize_Q(self, n):
        Q=np.random.uniform(0,1,size=(n, self.n_components))
        Q *= self.weights[None,:]
        Q=Q/Q.sum(axis=(-1),keepdims=True)
        return Q

    def _m_step(self,X,Q):
        self.weights = eps+np.sum(Q,axis=0)
        self.weights /= self.weights.sum()   
        self.theta = (Q[:,:,None,None]*X[:,None,:,:]).mean(axis=0)
        if self.rd is None:
            self.theta /= self.weights[:,None,None]
        else:
            self.theta /= (Q*self.rd[:,None]).mean(axis=0)[:,None,None]
        return

    def _e_step(self,X):
        if self.rd is not None:
            logL = np.sum(poisson.logpmf(k=X[:,None,:,:], mu=self.rd[:,None,None,None] * self.theta[None,:,:,:]), axis=(2,3))
        else:
            logL = np.sum(poisson.logpmf(k=X[:,None,:,:], mu=self.theta[None,:,:,:]), axis=(2,3))
        logL += np.log(self.weights)[None,:]
        Q = softmax(logL, axis=1)
        lower_bound = np.mean(logsumexp(a=logL, axis=1))
        return Q, lower_bound
        
   
    def _fit(self, X, epoch):
        silence = not bool(self.verbose)
        for i in tqdm(range(epoch), disable=silence):
            Q, lower_bound = self._e_step(X)  
            self._m_step(X, Q)
        return Q, lower_bound
        
    def fit(self, X, warm_start, Q=None, theta=None, weights=None, read_depth=None, n_init=10, epoch=100, seed=42):
        """  

        Parameters
        ----------
        X : ndarray of shape(n,p,2)
            DESCRIPTION.
        warm_start : TYPE
            DESCRIPTION.
        Q : ndarray of shape(n,n_components)
            Posteriors. The default is None.
        theta : TYPE, optional
            DESCRIPTION. The default is None.
        weights : TYPE, optional
            DESCRIPTION. The default is None.
        n_init : TYPE, optional
            DESCRIPTION. The default is 10.
        epoch : TYPE, optional
            DESCRIPTION. The default is 100.
        seed : TYPE, optional
            DESCRIPTION. The default is 42.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.warm_start = warm_start
        np.random.seed(seed)
        if read_depth is not None:
            assert np.shape(read_depth) == (len(X),) 
            if read_depth.mean() != 1 and self.verbose:
                print("read_depth is not normalized")
        self.rd = read_depth 
             
        ### Initialize weights
        if weights is None:
            self.weights = np.ones(self.n_components)/self.n_components
        elif len(weights) == self.n_components:
            self.weights = weights
        else:
            raise ValueError("check weights and n_components")
        
        if warm_start:
            if theta is None:
                assert Q is not None
                self._m_step(X, Q)
            else:
                self.theta = theta.copy()
            return self._fit(X, epoch=epoch)
            
        else:
            max_lower_bound = -np.inf
            n, p, s = X.shape
            for init in range(n_init):
                Q = self._initialize_Q(n)
                self._m_step(X, Q)
                Q, lower_bound = self._fit(X, epoch)
                
                if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                    max_lower_bound = lower_bound
                    best_params = self._get_parameters()
                    best_Q = Q
    
            self.theta, self.weights = best_params
        return best_Q, max_lower_bound

    
    def compute_lower_bound(self,X):
        if self.rd is not None:
            logL = np.sum(poisson.logpmf(k=X[:,None,:,:], mu=self.rd[:,None,None,None] * self.theta[None,:,:,:]), axis=(2,3))
        else:
            logL = np.sum(poisson.logpmf(k=X[:,None,:,:], mu=self.theta[None,:,:,:]), axis=(2,3))
        logL += np.log(self.weights)[None,:]
        return np.mean(logsumexp(a=logL, axis=1))
    
    def compute_cell_lower_bound(self,X):
        if self.rd is not None:
            logL = np.sum(poisson.logpmf(k=X[:,None,:,:], mu=self.rd[:,None,None,None] * self.theta[None,:,:,:]), axis=(2,3))
        else:
            logL = np.sum(poisson.logpmf(k=X[:,None,:,:], mu=self.theta[None,:,:,:]), axis=(2,3))
        logL += np.log(self.weights)[None,:]
        return logsumexp(a=logL, axis=1)
    
    def compute_AIC(self, X, normalized=True):
        n, p, s = np.shape(X)
        self.n_parameters = self.theta.size + self.n_components - 1      
        if normalized:
            return self.compute_lower_bound(X) - self.n_parameters/n
        else:
            return -2*n*self.compute_lower_bound(X) + 2 * self.n_parameters
            
    def compute_BIC(self, X, normalized=True):
        n, p, s = np.shape(X)
        self.n_parameters = self.theta.size + self.n_components - 1      
        if normalized:
            return self.compute_lower_bound(X) - self.n_parameters * np.log(n) / (2*n)
        else:
            return -2*n*self.compute_lower_bound(X) + self.n_parameters * np.log(n)



class GammaPoissonMixture:
    """
    Gamma-Poisson mixture model

    Attributes
    ----------
    n_components : int, number of mixtures, default=1
    theta : ndarray of shape (n_components, p, 2). 
            Means of Poisson distribution for U and S of each gene
    weights : ndarray of shape (n_components,). Weights of each mixture as in GMM.
    alpha : ndarray of shape (n_components,). alpha of the Gamma distribution.
            The lower bound is 1e-6 (the uninformative prior: alpha -> 0+).
            The upper bound is 1e6 (the Poisson limits: alpha -> +inf).
    """

    def __init__(self, n_components=1, verbose=1):
        self.n_components = n_components
        self.verbose = verbose    
        return
    
    def _get_parameters(self):
        return self.theta.copy(), self.weights.copy(), self.alpha.copy()
        
    def _initialize_Q(self, n):
        Q=np.random.uniform(0,1,size=(n, self.n_components))
        Q *= self.weights[None,:]
        Q=Q/Q.sum(axis=(-1),keepdims=True)
        return Q

    def _m_step(self,X,Q):
        """

        Parameters
        ----------
        X : ndarray of shape (n,p,s)
            Count matrix.
        Q : ndarray of shape (n,n_components)
            Posteriors/responsibilities.

        Returns
        -------
        None.

        """
        self.weights = eps+np.sum(Q,axis=0)
        self.weights /= self.weights.sum()   
        self.theta = (Q[:,:,None,None]*X[:,None,:,:]).mean(axis=0)/self.weights[:,None,None]
        total_counts = X.sum(axis=(1,2))
        for k in range(self.n_components):
            average = np.average(total_counts, weights=Q[:,k])
            variance = np.average((total_counts-average)**2, weights=Q[:,k])
            self.alpha[k] =  average**2 / (variance-average+eps)
            if self.alpha[k] < 0 or self.alpha[k] > Omega:
                self.alpha[k] = Omega
            elif self.alpha[k] < eps:
                self.alpha[k] = eps
        return

    def _e_step(self,X):
        logL = np.sum(poisson.logpmf(k=X[:,None,:,:], mu=self.theta[None,:,:,:]), axis=(2,3)) 
        # logL shape: (n,n_components)
        X_sum = X.sum(axis=(1,2)) # n
        theta_sum = self.theta.sum(axis=(1,2)) # n_components
        beta_ = self.alpha + theta_sum
        alpha_ = self.alpha[None,:] + X_sum[:,None]
        
        logL += theta_sum[None,:]
        logL += (self.alpha*np.log(self.alpha))[None,:]
        logL -= alpha_*np.log(beta_[None,:])
        logL += gammaln(alpha_)
        logL += gammaln(self.alpha)[None,:]
        
        logL += np.log(self.weights)[None,:]
        logL = np.nan_to_num(logL)
        # Q = softmax(logL, axis=1)
        a = np.amax(logL,axis=(-1))
        a = np.nan_to_num(a)
        relative_L = np.exp(logL-a[:,None])
        relative_L_sum = relative_L.sum(axis=(-1))
        Q = relative_L/relative_L_sum[:,None]
        lower_bound = np.mean( np.log(relative_L_sum) + a )
        return Q, lower_bound
        
   
    def _fit(self, X, epoch):
        silence = not bool(self.verbose)
        for i in tqdm(range(epoch), disable=silence):
            Q, lower_bound = self._e_step(X)  
            self._m_step(X, Q)
        return Q, lower_bound
        
    def fit(self, X, warm_start, Q=None, theta=None, weights=None, alpha=20, n_init=10, epoch=100, seed=42):
        """  

        Parameters
        ----------
        X : ndarray of shape(n,p,2)
            DESCRIPTION.
        warm_start : TYPE
            DESCRIPTION.
        Q : ndarray of shape(n,n_components)
            Posteriors. The default is None.
        theta : TYPE, optional
            DESCRIPTION. The default is None.
        weights : TYPE, optional
            DESCRIPTION. The default is None.
        alpha : TYPE, optional
            DESCRIPTION. The default is None.
        n_init : TYPE, optional
            DESCRIPTION. The default is 10.
        epoch : TYPE, optional
            DESCRIPTION. The default is 100.
        seed : TYPE, optional
            DESCRIPTION. The default is 42.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.warm_start = warm_start
        np.random.seed(seed)
        n,p,s = X.shape
        
        ### Initialize alphas
        self.alpha = np.zeros(self.n_components,dtype=float)
        assert np.broadcast(self.alpha,alpha).shape == (self.n_components,)
        self.alpha[:] = alpha 
             
        ### Initialize weights
        if weights is not None:
            assert len(weights) == self.n_components
            self.weights = weights/weights.sum()
        else:
            self.weights = np.ones(self.n_components)/self.n_components
            
        if warm_start:
            if theta is None:
                assert Q is not None, print("Need to provide Q or theta for warm start")
                assert Q.shape == (n,self.n_components)
                self._m_step(X, Q)
            else:
                assert theta.shape == (self.n_components,p,s)
                self.theta = theta.copy()
            return self._fit(X, epoch=epoch)
            
        else:
            max_lower_bound = -np.inf
            for init in range(n_init):
                Q = self._initialize_Q(n)
                self._m_step(X, Q)
                Q, lower_bound = self._fit(X, epoch)
                
                if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                    max_lower_bound = lower_bound
                    best_params = self._get_parameters()
                    best_Q = Q
    
            self.theta, self.weights, self.alpha = best_params
        return best_Q, max_lower_bound

    
    def compute_lower_bound(self,X):
        logL = np.sum(poisson.logpmf(k=X[:,None,:,:], mu=self.theta[None,:,:,:]), axis=(2,3)) 
        # logL shape: (n,n_components)
        X_sum = X.sum(axis=(1,2)) # n
        theta_sum = self.theta.sum(axis=(1,2)) # n_components
        beta_ = self.alpha + theta_sum
        alpha_ = self.alpha[None,:] + X_sum[:,None]
        
        logL += theta_sum[None,:]
        logL += (self.alpha*np.log(self.alpha))[None,:]
        logL -= alpha_*np.log(beta_[None,:])
        logL += gammaln(alpha_)
        logL += gammaln(self.alpha)[None,:]
        
        logL += np.log(self.weights)[None,:]
        # Q = softmax(logL, axis=1)
        a = np.amax(logL,axis=(-1))
        relative_L = np.exp(logL-a[:,None])
        relative_L_sum = relative_L.sum(axis=(-1))
        lower_bound = np.mean( np.log(relative_L_sum) + a )
        return lower_bound
    
    def compute_AIC(self, X, normalized=True):
        n, p, s = np.shape(X)
        self.n_parameters = self.theta.size + 2 * self.n_components - 1      
        if normalized:
            return self.compute_lower_bound(X) - self.n_parameters/n
        else:
            return -2*n*self.compute_lower_bound(X) + 2 * self.n_parameters
            
    
    def compute_BIC(self, X, normalized=True):
        n, p, s = np.shape(X)
        self.n_parameters = self.theta.size + 2 * self.n_components - 1      
        if normalized:
            return self.compute_lower_bound(X) - self.n_parameters * np.log(n) / (2*n)
        else:
            return -2*n*self.compute_lower_bound(X) + self.n_parameters * np.log(n)

class NBMixture:
    """
    Poisson mixture model

    Attributes
    ----------
    n_components : int, number of mixtures, default=1
    theta : ndarray of shape (n_components, p, 2). 
            Means of Poisson distribution for U and S of each gene
    weights : ndarray of shape (n_components,). Weights of each mixture as in GMM.
    """

    def __init__(self, n_components=1, verbose=0):
        self.n_components = n_components
        self.verbose = verbose    
        return
    
    def _get_parameters(self):
        return self.theta.copy(), self.weights.copy()
        
    def _initialize_Q(self, n):
        Q=np.random.uniform(0,1,size=(n, self.n_components))
        Q *= self.weights[None,:]
        Q=Q/Q.sum(axis=(-1),keepdims=True)
        return Q

    def _m_step(self,X,Q):
        self.weights = eps+np.sum(Q,axis=0)
        self.weights /= self.weights.sum()   
        self.theta = (Q[:,:,None,None]*X[:,None,:,:]).mean(axis=0)
        if self.rd is None:
            self.theta /= self.weights[:,None,None]
        else:
            self.theta /= (Q*self.rd[:,None]).mean(axis=0)[:,None,None]
        return

    def _e_step(self,X):
        if self.rd is not None:
            logL = np.sum(poisson.logpmf(k=X[:,None,:,:], mu=self.rd[:,None,None,None] * self.theta[None,:,:,:]), axis=(2,3))
        else:
            logL = np.sum(poisson.logpmf(k=X[:,None,:,:], mu=self.theta[None,:,:,:]), axis=(2,3))
        logL += np.log(self.weights)[None,:]
        Q = softmax(logL, axis=1)
        lower_bound = np.mean(logsumexp(a=logL, axis=1))
        return Q, lower_bound
        
   
    def _fit(self, X, epoch):
        silence = not bool(self.verbose)
        for i in tqdm(range(epoch), disable=silence):
            Q, lower_bound = self._e_step(X)  
            self._m_step(X, Q)
        return Q, lower_bound
        
    def fit(self, X, warm_start, Q=None, theta=None, weights=None, read_depth=None, n_init=10, epoch=100, seed=42):
        """  

        Parameters
        ----------
        X : ndarray of shape(n,p,2)
            DESCRIPTION.
        warm_start : TYPE
            DESCRIPTION.
        Q : ndarray of shape(n,n_components)
            Posteriors. The default is None.
        theta : TYPE, optional
            DESCRIPTION. The default is None.
        weights : TYPE, optional
            DESCRIPTION. The default is None.
        n_init : TYPE, optional
            DESCRIPTION. The default is 10.
        epoch : TYPE, optional
            DESCRIPTION. The default is 100.
        seed : TYPE, optional
            DESCRIPTION. The default is 42.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.warm_start = warm_start
        np.random.seed(seed)
        if read_depth is not None:
            assert np.shape(read_depth) == (len(X),) 
            if read_depth.mean() != 1:
                print("read_depth is not normalized")
        self.rd = read_depth 
             
        ### Initialize weights
        if weights is None:
            self.weights = np.ones(self.n_components)/self.n_components
        elif len(weights) == self.n_components:
            self.weights = weights
        else:
            raise ValueError("check weights and n_components")
        
        if warm_start:
            if theta is None:
                assert Q is not None
                self._m_step(X, Q)
            else:
                self.theta = theta.copy()
            return self._fit(X, epoch=epoch)
            
        else:
            max_lower_bound = -np.inf
            n, p, s = X.shape
            for init in range(n_init):
                Q = self._initialize_Q(n)
                self._m_step(X, Q)
                Q, lower_bound = self._fit(X, epoch)
                
                if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                    max_lower_bound = lower_bound
                    best_params = self._get_parameters()
                    best_Q = Q
    
            self.theta, self.weights = best_params
        return best_Q, max_lower_bound

    
    def compute_lower_bound(self,X):
        if self.rd is not None:
            logL = np.sum(poisson.logpmf(k=X[:,None,:,:], mu=self.rd[:,None,None,None] * self.theta[None,:,:,:]), axis=(2,3))
        else:
            logL = np.sum(poisson.logpmf(k=X[:,None,:,:], mu=self.theta[None,:,:,:]), axis=(2,3))
        logL += np.log(self.weights)[None,:]
        return np.mean(logsumexp(a=logL, axis=1))
    
    def compute_cell_lower_bound(self,X):
        if self.rd is not None:
            logL = np.sum(poisson.logpmf(k=X[:,None,:,:], mu=self.rd[:,None,None,None] * self.theta[None,:,:,:]), axis=(2,3))
        else:
            logL = np.sum(poisson.logpmf(k=X[:,None,:,:], mu=self.theta[None,:,:,:]), axis=(2,3))
        logL += np.log(self.weights)[None,:]
        return logsumexp(a=logL, axis=1)
    
    def compute_AIC(self, X, normalized=True):
        n, p, s = np.shape(X)
        self.n_parameters = self.theta.size + self.n_components - 1      
        if normalized:
            return self.compute_lower_bound(X) - self.n_parameters/n
        else:
            return -2*n*self.compute_lower_bound(X) + 2 * self.n_parameters
            
    def compute_BIC(self, X, normalized=True):
        n, p, s = np.shape(X)
        self.n_parameters = self.theta.size + self.n_components - 1      
        if normalized:
            return self.compute_lower_bound(X) - self.n_parameters * np.log(n) / (2*n)
        else:
            return -2*n*self.compute_lower_bound(X) + self.n_parameters * np.log(n)

    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    topo = np.array([[0,1]])
    true_tau = (0,1)
    tau = [0,1]
    n = 1000
    p = 100
    m = 2
    theta = np.random.lognormal(size=(m,p,2))
    read_depth = np.random.gamma(shape=1,scale=1,size=n)
    Y = np.random.poisson(lam=read_depth[:,None,None,None]*theta[None,:])
    X = np.concatenate([Y[:,0],Y[:,1]],axis=0)
        
    for k in range(2,3):
        GPM = GammaPoissonMixture(n_components=k)
        best_Q, max_lower_bound = GPM.fit(X, warm_start=False, theta=theta, n_init=1, epoch=3)
        print(GPM.compute_lower_bound(X),GPM.compute_AIC(X))
    
    plot_p = min(10, p)
    fig, ax = plt.subplots(1,plot_p,figsize=(6*plot_p,4))
    for i in range(plot_p):
        j = i
        ax[i].scatter(X[:,j,0],X[:,j,1],c="gray");
        ax[i].scatter(theta[:,j,0],theta[:,j,1],c='red');
        ax[i].scatter(GPM.theta[:,j,0],GPM.theta[:,j,1],c='b');
        ax[i].set_title(j)