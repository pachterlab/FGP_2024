#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import logsumexp, softmax
from scipy.stats import poisson
from tqdm import tqdm

    
eps = 1e-6


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
    Y = np.random.poisson(lam=theta,size=(n,m,p,2))
    X = np.concatenate([Y[:,0],Y[:,1]],axis=0)
    
    plot_p = min(10, p)
    fig, ax = plt.subplots(1,plot_p,figsize=(6*plot_p,4))
    for i in range(plot_p):
        j = i
        ax[i].scatter(X[:,j,0],X[:,j,1],c="gray");
        ax[i].scatter(theta[:,j,0],theta[:,j,1],c='red');
        ax[i].set_title(j)
        
    for k in range(1,5):
        PM = PoissonMixture(n_components=k)
        best_Q, max_lower_bound = PM.fit(X, warm_start=False, theta=theta, n_init=1, epoch=3)
        print(PM.compute_lower_bound(X),PM.compute_AIC(X))
