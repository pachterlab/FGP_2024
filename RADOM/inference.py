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
from scipy.optimize import minimize_scalar
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

    def __init__(self, topo, tau, model, restrictions={}, verbose=0):
        """
        set model and global parameters
        """
        self.topo=np.array(topo,dtype=int)
        self.prior_tau=np.array(tau,dtype=float)
        self.tau=np.array(tau,dtype=float)
        self.L=len(topo)
        self.n_states=len(set(topo.flatten()))
        self.K=len(tau)-1
        self.model_restrictions=restrictions
        self.model=model
        self.verbose = verbose
        
        ## import model specific methods from the provided file
        tempmod = import_module("models."+model)
        #tempmod = import_module(".models."+model,"RADOM")
        self.guess_theta = tempmod.guess_theta
        self.check_params = tempmod.check_params
        self.get_Y_hat = tempmod.get_Y_hat
        self.get_logL =  tempmod.get_logL
        self.update_theta_j = tempmod.update_theta_j
        del tempmod 
        
        return None
    
    def _set_time_grids(self,m):
        self.m=m
        self.t=np.linspace(self.tau[0],self.tau[-1],m)
    
    def _get_theta(self):
        return self.theta.copy()
    
    def _initialize_theta(self, X):
        self.theta=self.guess_theta(X,self.topo,self.tau)
        return 
    
    def _initialize_Q(self, n):
        Q=1+np.random.uniform(0,1,size=(n,self.L,self.m))
        if self.weights is not None:
            Q *= self.weights
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
                params_j = self.params.copy()
                restrictions = None
                
                if j in restricted_gene_idx:
                    restrictions = self.model_restrictions[j]
                    
                Input_args.append((self.theta[j], X[:,j], Q, self.t, self.tau, self.topo, params_j, restrictions))
                
            with Pool(n_threads) as pool:      
                new_theta = pool.starmap(self.update_theta_j, Input_args)
            new_theta = np.array(new_theta)
        else:
            new_theta = np.zeros((len(gene_idx),n_theta))
            for i,j in enumerate(gene_idx): 
                params_j = self.params.copy()
                restrictions = None
                    
                if j in restricted_gene_idx:
                    restrictions = self.model_restrictions[j]

                new_theta[i]=self.update_theta_j(self.theta[j], X[:,j], Q, self.t, self.tau, self.topo, params_j, restrictions)
                    
        self.theta[gene_idx] = new_theta
        return

    def update_Q(self,X,beta=1):
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
        ## logl = \sum_{j=1}^p log p(x_ij|l,t_m,theta) with size (n,self.L,self.m) 
        logl = self.get_logL(X,self.theta,self.t,self.tau,self.topo,self.params)  
        logL = logl/beta
            
        ## Q is the posterior
        ## Q = softmax(logL, axis=(-2,-1))
        a = np.amax(logL,axis=(-2,-1))
        relative_L = np.exp(logL-a[:,None,None])
        if self.weights is not None:
            relative_L *= self.weights
        else:
            relative_L /= self.m*self.L
        relative_L_sum = relative_L.sum(axis=(-2,-1))
        Q = relative_L/relative_L_sum[:,None,None]
        lower_bound = np.mean( np.log(relative_L_sum) + a )
        
        return Q, lower_bound
        
    def normalize_Q(self, Q):
        n, L, m = np.shape(Q)
        marginal = Q.sum(axis=(0,1))
        idx = np.where(marginal > 0.01*n/L/m)[0]
        M = len(idx)   
        if M<m:
            if self.verbose>1:
                print("normalize_Q", M, m)
            map2idx = list(map(lambda x: idx[int(x*(M-1)/(m-1))], range(m)))

            new_Q = np.zeros_like(Q)
            new_Q += Q[:,:,map2idx]
            new_Q /= new_Q.sum(axis=(-2,-1),keepdims=True)
            return new_Q
        else:
            return Q
        
    def update_global_time_scale(self, X):
    
        def Obj(x,theta,X,t,tau,topo,params):
            new_theta = theta.copy()
            new_theta[:,-2:] *= 10**x
            logL = self.get_logL(X,new_theta,t,tau,topo,params) # with size (n,self.L,self.m)     
            if self.weights is not None:
                logL += np.log(self.weights)
            else:
                logL += - np.log(self.m) - np.log(self.L)
            
            ## Q is the posterior
            ## Q = softmax(logL, axis=(-2,-1))
            a = np.amax(logL,axis=(-2,-1))
            temp = np.exp(logL-a[:,None,None])
            temp_sum = temp.sum(axis=(-2,-1))
            lower_bound = np.mean( np.log(temp_sum) + a )
            return -lower_bound
            
        res = minimize_scalar(fun=Obj, args=(self.theta,X,self.t,self.tau,self.topo,self.params), bounds=(-1, 1), method='bounded',
                              options={'maxiter': 100,'disp': False})
        self.theta[:,-2:] *= 10**res.x
        return 
    
    def update_global_tau(self, X):
        if self.model[-3:] == "tau":
            return
        
        def Obj(dt,k,tau,theta,X,t,prior_tau,topo,params,lambda_tau):
            tau_ = tau.copy()
            tau_[k] += dt
            logL = self.get_logL(X,theta,t,tau_,topo,params) # with size (n,self.L,self.m)     
            if self.weights is not None:
                logL += np.log(self.weights)
            else:
                logL += - np.log(self.m) - np.log(self.L)

            a = np.amax(logL,axis=(-2,-1))
            temp = np.exp(logL-a[:,None,None])
            temp_sum = temp.sum(axis=(-2,-1))
            lower_bound = np.mean( np.log(temp_sum) + a )
            lower_bound -= lambda_tau * (tau_[k]-prior_tau[k])**2
            return -lower_bound
        
        if "lambda_tau" in self.params:
            lambda_tau = X.shape[1] * self.params["lambda_tau"]
        else:
            lambda_tau = X.shape[1] * 0.1
            
        new_tau = self.tau.copy()
        for k in range(1,len(self.tau)-1): 
            res = minimize_scalar(fun=Obj, args=(k,self.tau,self.theta,X,self.t,self.prior_tau,self.topo,self.params,lambda_tau), bounds=(-0.1, 0.1), method='bounded', options={'maxiter': 100,'disp': False})
            new_tau[k] += res.x
        self.tau = new_tau
        return 
    
    def _fit(self, X, epoch, beta, parallel, n_threads):
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

        """
        
        n, p, _ = np.shape(X)
        #lower_bound = -np.inf
        lower_bounds = []
        self.converged = False
        
        silence = not bool(self.verbose)
        for i in tqdm(range(epoch), disable=silence):
            #self.params['epoch']= i/epoch
            #prev_lower_bound = lower_bound
            
            ### EM algorithm  ###
            ### E step
            Q, lower_bound = self.update_Q(X,beta=beta)  
            lower_bounds.append(lower_bound)
            #self.Q_hist.append(Q)
            
            ### check converged
            #change = lower_bound - prev_lower_bound
            #if abs(change) < tol:
            #    self.converged = True
            #    break
            
            ### normalization
            #if i>3 :
            #    Q = self.normalize_Q(Q)
           
            ### M step 
            self.update_theta(X,Q,parallel=parallel,n_threads=n_threads) ### M step
            if i>1:
                if self.fit_tau:
                    self.update_global_tau(X)
                if self.fit_weights:
                    self.weights = Q.mean(axis=0,keepdims=True)
            #self.update_global_time_scale(X)
            #self.theta_hist.append(self.theta.copy())
        
        Q, lower_bound = self.update_Q(X,beta=beta)
        lower_bounds.append(lower_bound)
        return [Q, lower_bounds]

    
    def fit_multi_init(self, X, n_init=10, perm_theta=False, epoch=100, beta=1, parallel=False, n_threads=1, seed=42):
        """
        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times

        Parameters
        ----------
        X : ndarray, shape (n, p, 2)
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
     
        Qs = []
        elbos = []
        thetas = []
        max_lower_bound = -np.inf
        
        np.random.seed(seed)
        for init in range(n_init):
            if bool(self.verbose):
                print("trial "+str(init+1))
            self._initialize_theta(X)
            Q = self._initialize_Q(n)  
            self.update_theta(X=X,Q=Q,parallel=parallel,n_threads=n_threads)
            Q, lower_bound = self._fit(X=X, epoch=epoch, beta=beta, parallel=parallel, n_threads=n_threads)
            theta0 = self.theta.copy()
            

            if lower_bound[-1] > max_lower_bound or max_lower_bound == -np.inf:
                max_lower_bound = lower_bound[-1]
                best_theta = self._get_theta()
                best_Q = Q.copy()

            Qs.append(Q)
            elbos.append(lower_bound)
            thetas.append(self._get_theta())

            ## For each initialization, test additional configurations 
            ## generated by permutaions of states
            ## to explore the parameter spaces systematically
            if perm_theta==True:
                configs = []                    
                theta0_mean = np.mean(theta0[:,:self.n_states],axis=0)
                for ip, perm in enumerate(itertools.permutations(range(self.n_states))):
                    ## Check whether this configuration is equivalent to previous ones
                    ## by sum over all lineages the product of (stage + theta) of all states in the lineages 
                    ## If so, continue.
                    ## If not, add to configs and fit.
                    theta_mean = theta0_mean[np.array(perm)]
                    config = 0
                    for l in range(self.L):
                        config += np.sum(np.log(np.arange(1,self.K+2)+theta_mean[self.topo[l]]))
                    config = np.around(config,6)

                    if config in configs:
                        continue
                    else:
                        configs.append(config)

                    ## For new configuration
                    if ip!=0:
                        continue
                        
                    theta_perm = theta0.copy()
                    theta_perm[:,:(self.n_states)] = theta0[:,np.array(perm)]
                    self.theta = theta_perm.copy()
                    Q, lower_bound = self._fit(X=X, epoch=epoch, beta=beta, parallel=parallel, n_threads=n_threads)

                    if lower_bound[-1] > max_lower_bound or max_lower_bound == -np.inf:
                        max_lower_bound = lower_bound[-1]
                        best_theta = self._get_theta()
                        best_Q = Q.copy()

                    Qs.append(Q)
                    elbos.append(lower_bound)
                    thetas.append(self._get_theta())

                    ## Enough additional configurations done
                    if len(configs) > 10: 
                        break
                
        self.theta = best_theta
        self.thetas = thetas
        self.Qs = Qs        
            
        return [best_Q, elbos]
    
    def fit(self, X, warm_start=False, Q=None, theta=None, prior=None, fit_weights=False, fit_tau=False, params={}, model_restrictions=None, m=100, n_init=10, perm_theta = False, epoch=100, beta=1, parallel=False, n_threads=1, seed=42):
        """
        

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        Q : 3D array
            Posteriors/responsibilities. The default is None.
        theta : TYPE, optional
            DESCRIPTION. The default is None.
        prior : TYPE, optional
            DESCRIPTION. The default is None.
        params : TYPE, optional
            DESCRIPTION. The default is {}.
        model_restrictions : TYPE, optional
            DESCRIPTION. The default is None.
        m : TYPE, optional
            DESCRIPTION. The default is 100.
        n_init : TYPE, optional
            DESCRIPTION. The default is 10.
        perm_theta : TYPE, optional
            DESCRIPTION. The default is False.
        epoch : TYPE, optional
            DESCRIPTION. The default is 100.
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
        TYPE
            DESCRIPTION.

        """
        self.params = params
        self.weights = prior
        self.fit_weights = fit_weights
        self.fit_tau = fit_tau
        
        if "lambda_tau" not in params:
            self.params['lambda_tau'] = 0.1

        if "lambda_a" not in params:
            self.params['lambda_a'] = 0          
        
        #self.check_params(self)   
        
        if Q is not None:
            m = Q.shape[-1]  
        else:
            if prior is not None:
                m = prior.shape[-1]   
        if Q is not None:
            assert Q.shape == (len(X),self.L,m)
        if prior is not None:
            assert np.broadcast(Q, prior).shape == (len(X),self.L,m)            
        self._set_time_grids(m)
        
        if warm_start:
            if bool(self.verbose):
                print("fitting with warm start")
            if theta is None:
                assert Q is not None
                self._initialize_theta(X)
                self.update_theta(X,Q,parallel=parallel,n_threads=n_threads)  
            else:
                self.theta = theta.copy()
            res = self._fit(X=X, epoch=epoch, beta=beta, parallel=parallel, n_threads=n_threads)
        else:
            if bool(self.verbose):
                print("fitting with multiple random initializations")
            res = self.fit_multi_init(X=X, n_init=n_init, perm_theta=perm_theta, epoch=epoch, beta=beta, parallel=parallel, n_threads=n_threads, seed=seed)
        
        Q, elbos = res
        self.X = X
        self.Q = Q
        self.elbos = elbos
        return self
    
    def compute_lower_bound(self,X):
        """
        Compute the lower bound of marginal log likelihood log(P(X|theta))

        Parameters
        ----------
        X : 3d array 
            Data

        Returns
        -------
        scalar
            lower bound value
        """
        
        logL = self.get_logL(X,self.theta,self.t,self.tau,self.topo,self.params) # with size (n,self.L,self.m)
        
        if self.weights is not None:
            logL += np.log(self.weights)
        else:
            logL += - np.log(self.m) - np.log(self.L)
            
        return np.mean(logsumexp(a=logL, axis=(-2,-1)))

    
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
        AIC = logL - self.k/n
        #AICc = AIC + 2*self.k*(self.k+1)/(X.size-self.k-1)
        return  AIC
    
    
    def compare_model(self, X, new_model, epoch=9, tol=1e-4, parallel=False, n_threads=1):
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
        new._fit(X,new.theta,epoch=1,beta=beta,parallel=parallel,n_threads=n_threads)
        new_AIC = new.compute_AIC(X)
        
        ##### if the nested model is better, return True #####
        if new_AIC < ori_AIC:
            accept = True  
        
        ##### else, update the whole theta and check agian #####
        else:      
            ##### update all theta with new weight #####
            new._fit(X,new.theta,epoch=epoch,beta=beta,parallel=parallel,n_threads=n_threads)
            new_AIC = new.compute_AIC(X)
            
            if new_AIC < ori_AIC:
                accept = True  
            else:
                accept = False

        return accept, new_AIC - ori_AIC, new
