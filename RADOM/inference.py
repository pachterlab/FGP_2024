#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 22:22:42 2022

@author: fang
"""
import itertools
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from scipy.special import logsumexp
from importlib import import_module
import copy

    
eps = 1e-10


class Trajectory:
    """
    Representation of a trajectory model probability distribution.
    
    
    Attributes
    ----------
    topo: 2D np.darray
    tau:     
    """

    def __init__(self, topo, tau, model="two_species",restrictions={},verbose=0):
        self.topo=topo
        self.tau=tau
        self.L=len(topo)
        self.n_states=len(set(topo.flatten()))
        self.K=len(tau)-1
        self.model_restrictions=restrictions
        self.model=model
        self.verbose = verbose
        
        ## import model specific methods from the provided file
        tempmod = import_module("models."+model)
        #tempmod = import_module(".models."+model,"RADOM")
        self.get_Y = tempmod.get_Y
        self.guess_theta = tempmod.guess_theta
        self.get_logL =  tempmod.get_logL
        self.update_theta_j = tempmod.update_theta_j
        del tempmod 
        return None
    
    def _set_m(self,m):
        self.m=m
        self.t=np.linspace(self.tau[0],self.tau[-1],m)
    
    def _get_theta(self):
        return self.theta.copy()
    
    def _initialize_theta(self, X):
        self.theta=self.guess_theta(X,self.n_states)
        return 
    
    def _initialize_Q(self, n):
        Q=1+np.random.uniform(0,1,size=(n,self.L,self.m))
        Q=Q/Q.sum(axis=(-2,-1),keepdims=True)
        return Q

    def update_theta(self,X,Q,gene_idx=None,parallel=False,n_threads=1):
        """
        Update theta for each gene in gene_idx using update_theta_j function

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        Q : TYPE
            DESCRIPTION.
        gene_idx : list/1d array, optional
            DESCRIPTION. The default is None.
        parallel : TYPE, optional
            DESCRIPTION. The default is False.
        n_threads : TYPE, optional
            DESCRIPTION. The default is 1.
        bnd : TYPE, optional
            DESCRIPTION. The default is 1000.
        bnd_beta : TYPE, optional
            DESCRIPTION. The default is 100.
        miter : TYPE, optional
            DESCRIPTION. The default is 1000.

        Returns
        -------
        None.

        """
        p, n_theta = self.theta.shape
            
        if gene_idx is None:
            gene_idx = np.arange(p)
        restricted_gene_idx = np.array(list(self.model_restrictions.keys()))
        
        if parallel is True:
            Input_args = []
            for j in gene_idx:
                if j in restricted_gene_idx:
                    Input_args.append((self.theta[j], X[:,j], Q, self.t, self.tau, self.topo, self.model_restrictions[j]))
                else:
                    Input_args.append((self.theta[j], X[:,j], Q, self.t, self.tau, self.topo))
            with Pool(n_threads) as pool:      
                new_theta = pool.starmap(self.update_theta_j, Input_args)
            new_theta = np.array(new_theta)
        else:
            new_theta = np.zeros((len(gene_idx),n_theta))
            for i,j in enumerate(gene_idx): 
                if j in restricted_gene_idx:
                    new_theta[i]=self.update_theta_j(self.theta[j], X[:,j], Q, self.t, self.tau, self.topo, self.model_restrictions[j])
                else:
                    new_theta[i]=self.update_theta_j(self.theta[j], X[:,j], Q, self.t, self.tau, self.topo)
                    
        self.theta[gene_idx] = new_theta
        return

    def update_weight(self,X):
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
            
        #n,p,s=np.shape(X)
        
        logl = self.get_logL(X,self.theta,self.t,self.tau,self.topo) # with size (n,self.L,self.m)     
        logL = logl
        #logL += np.log(self.prior_) where self.prior_ = 1/L/m
        
        ## Q = softmax(logL, axis=(-2,-1))
        a = np.amax(logL,axis=(-2,-1))
        temp = np.exp(logL-a[:,None,None])
        temp_sum = temp.sum(axis=(-2,-1))
        Q = temp/temp_sum[:,None,None]
        lower_bound = np.mean( np.log(temp_sum) + a ) - np.log(self.m) - np.log(self.L)
        
        return Q, lower_bound
    
    def _fit(self, X, theta, epoch, tol, parallel, n_threads):
        """
        The method fits the model by iterating between E-step and M-step for at most `epoch` iterations.
        The warm start means that either Q or theta is provided.
    

        Parameters
        ----------
        X : ndarray, shape (n, p, 2)
            n cells * p genes * 2 species data matrix.
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
        lower_bound = -np.inf
        self.converged = False
        self.theta = theta.copy()
        
        silence = not bool(self.verbose)
        for i in tqdm(range(epoch), disable=silence):
            prev_lower_bound = lower_bound
            
            ## EM algorithm   
            Q, lower_bound = self.update_weight(X) ### E step   
            self.update_theta(X,Q,parallel=parallel,n_threads=n_threads) ### M step
            
            ## check converged
            change = lower_bound - prev_lower_bound
            if abs(change) < tol:
                self.converged = True
                break
                
        self.X = X
        self.Q = Q
        self.lower_bound = lower_bound
        
        return [Q, lower_bound]
    
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
        
        if theta is None:
            if Q is not None:
                n, L, m = Q.shape
                self._set_m(m)
                self._initialize_theta(X)
                self.update_theta(X,Q,parallel=parallel,n_threads=n_threads)
                theta = self.theta.copy()
            else:
                raise AssertionError("either theta or Q needs to be provided")
            
        #self.prior_ = np.ones_like(Q)/L/m

        #theta_hist=[] 
        #weight_hist=[]
        #lower_bounds=[]
        #theta_hist.append(self._get_theta())
        #weight_hist.append(Q.copy())
        
        return self._fit(X, theta, epoch, tol, parallel, n_threads)
    
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
        elbos = []
        thetas = []
        max_lower_bound = -np.inf
        
        np.random.seed(seed)
        for init in range(n_init):
            if bool(self.verbose):
                print("trial "+str(init+1))
            self._initialize_theta(X)
            Q = self._initialize_Q(n)  
            self.update_theta(X,Q,parallel=parallel,n_threads=n_threads)
            Q, lower_bound = self._fit(X, self.theta, epoch, tol, parallel, n_threads)
            theta0 = self.theta.copy()
            
            if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                max_lower_bound = lower_bound
                best_theta = self._get_theta()
                best_Q = Q.copy()
                
            elbos.append(lower_bound)
            thetas.append(self._get_theta())

            ## For each initialization, test additional configurations 
            ## generated by permutaions of states
            ## to explore the parameter spaces systematically
            
            configs = []
            theta0_mean = np.mean(theta0[:,:(self.n_states+1)],axis=0)
            for ip, perm in enumerate(itertools.permutations(range(self.n_states+1))):
                ## Check whether this configuration is equivalent to previous ones
                ## by sum over all lineages the product of (1 + theta) of all states in the lineages 
                ## If so, continue.
                ## If not, add to configs and fit.
                theta_mean = theta0_mean[np.array(perm)]
                config = 0
                for l in range(self.L):
                    config += np.exp( np.sum(np.log(np.arange(1,self.K+1)+theta_mean[self.topo[l]]))) * theta_mean[self.n_states]
                config = np.around(config,6)
                
                if config in configs:
                    continue
                else:
                    configs.append(config)
                    
                ## record the original configuration and pass
                if ip==0:
                    continue
                
                ## For new configuration
                theta = theta0.copy()
                theta[:,:(self.n_states+1)] = theta0[:,np.array(perm)]
                Q, lower_bound = self._fit(X, theta, epoch, tol, parallel, n_threads)

                if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                    max_lower_bound = lower_bound
                    best_theta = self._get_theta()
                    best_Q = Q.copy()

                elbos.append(lower_bound)
                thetas.append(self._get_theta())
                
                ## Enough additional configurations done
                if len(configs) > 10: 
                    break
            
            
        self.theta = best_theta
        self.thetas = thetas
        return [best_Q, elbos]
    
    def fit(self, X, Q=None, theta=None, model_restrictions=None, m=100, n_init=3, epoch=20, tol=1e-4, parallel=False, n_threads=1, seed=42):
        """
        

        Parameters
        ----------
        Q : TYPE, optional
            DESCRIPTION. The default is None.
        theta : TYPE, optional
            DESCRIPTION. The default is None.
        m : TYPE
            DESCRIPTION.
        n_init : TYPE, optional
            DESCRIPTION. The default is 3.
        epoch : TYPE, optional
            DESCRIPTION. The default is 10.
        tol : TYPE, optional
            DESCRIPTION. The default is 1e-4.
        parallel : TYPE, optional
            DESCRIPTION. The default is False.
        n_threads : TYPE, optional
            DESCRIPTION. The default is 1.
        seed : TYPE, optional
            DESCRIPTION. The default is 42.

        Returns
        -------
        None.

        """
        
        self._set_m(m)
        
        if Q is not None or theta is not None:
            print("run method fit_warm_start")
            res = self.fit_warm_start(X, Q=Q, theta=theta, epoch=epoch, tol=tol, parallel=parallel, n_threads=n_threads)
        else:
            print("run method fit_multi_init")
            res = self.fit_multi_init(X, m=m, n_init=n_init, epoch=epoch, tol=tol, parallel=parallel, n_threads=n_threads, seed=seed)
        
        return res
    
    def compute_lower_bound(self,X):
        """
        Compute the lower bound of marginal log likelihood log P(X|theta)

        Parameters
        ----------
        X : 3d array 
            Data

        Returns
        -------
        scalar
            lower bound value
        """
        #n,p,_=np.shape(X)
        #Y = np.zeros((self.L,self.m,p,2))
        #for l in range(self.L):
        #    theta_l = np.concatenate((self.theta[:,self.topo[l]], self.theta[:,-4:]), axis=1)
        #    Y[l] = get_Y(theta_l,self.t,self.tau) # m*p*2
        #logL =  np.sum(X[:,None,None,:,:] * np.log(Y[None,:]+eps) - Y[None,:], axis=(-2,-1)) # n*L*m*p*2 -> n*L*m
        #logL = np.tensordot(X, np.log(eps + Y), axes=([-2,-1],[-2,-1])) # logL:n*L*m
        #logL -= np.sum(Y,axis=(-2,-1))
        #logL += np.log(self.prior_) where self.prior_ = 1/L/m
        logL = self.get_logL(X,self.theta,self.t,self.tau,self.topo) # with size (n,self.L,self.m)
        return np.mean(logsumexp(a=logL, axis=(-2,-1)))-np.log(self.m)-np.log(self.L)

    
    def compute_AIC(self, X):
        """
        

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        n, p, s = np.shape(X)
        
        self.k = self.theta.size
        
        for j in list(self.model_restrictions.keys()):
            restrictions = self.model_restrictions[j]
            redundant, blanket = restrictions
            self.k -= len(redundant)
            if len(redundant) > self.n_states:
                self.k -= 1

        logL = self.compute_lower_bound(X)
        AIC = 2*self.k-2*n*logL
        AICc = AIC + 2*self.k*(self.k+1)/(X.size-self.k-1)
        return  AICc
    
    
    def compare_model(self, X, new_model, epoch=1, tol=1e-4, parallel=False, n_threads=1):
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
        
        n, p, s = X.shape
        ori_AIC = self.compute_AIC(X)
        
        ##### update theta with restrictions #####
        new = copy.deepcopy(self)
        new.model_restrictions=new_model
        new._fit(X,new.theta,epoch=1,tol=tol,parallel=parallel,n_threads=n_threads)
        new_AIC = new.compute_AIC(X)
        
        ##### if the nested model is better, return True #####
        if new_AIC < ori_AIC:
            accept = True  
        
        ##### else, update the whole theta and check agian #####
        else:      
            ##### update all theta with new weight #####
            new._fit(X,new.theta,epoch=epoch,tol=tol,parallel=parallel,n_threads=n_threads)
            new_AIC = new.compute_AIC(X)
            
            if new_AIC < ori_AIC:
                accept = True  
            else:
                accept = False

        return accept, new_AIC - ori_AIC, new
