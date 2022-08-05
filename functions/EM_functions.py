#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 21:20:50 2022

@author: Meichen Fang
"""

"""
Outline of traj_EM:
    n cells, p genes, m time grids, L lineages
    Given Q(z) which is a n*L*m array, compute optimal theta_j for each gene j:
        compute loss which is negative log likelihood + Regularization 
        negative log likelihood is a function of theta_j and Q(z)
        
"""
import numpy as np
from tqdm import tqdm
import time
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.special import logsumexp
import anndata as ad

# global parameters: upper and lower limits for numerical stability
eps = 1e-20

def get_Y_ori(theta, t, tau):
    # theta: p*(K+4)
    # t: len m
    # tau: len K+1
    # return m * p * 2
    p = len(theta)
    K = len(tau)-1 # number of states
    if np.shape(theta)[1]!=K+4:
        print(np.shape(theta)[1], K+4)
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

    # nascent
    y1=y1_0[None,:]*np.exp(-beta[None,:]*t)  
    for k in range(1,K+1):
        I[k] = np.squeeze(t > tau[k])
        idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
        y1 = y1 + a[None,:,k-1] * (np.exp(- beta[None,:]*I[k,:,None] *(t-tau[k]))- np.exp(-beta[None,:]*I[k,:,None] * (t-tau[k-1])) ) \
          + a[None,:,k-1] * (1 - np.exp(- beta[None,:]*idx[:,None] *(t-tau[k-1]))) 
    
    if np.sum(np.isnan(y1)) != 0:
        raise ValueError("Nan in y1")
    # mature + c * nascent 
    y=y_0[None,:]*np.exp(-gamma[None,:]*t)    
    for k in range(1,K+1):
        idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
        y = y + a_[None,:,k-1] * (np.exp(-gamma[None,:]*I[k,:,None] * (t-tau[k]))- np.exp(-gamma[None,:]*I[k,:,None] * (t-tau[k-1])) ) \
          +  a_[None,:,k-1] * (1 - np.exp(-gamma[None,:]*idx[:,None]*(t-tau[k-1]))) 

    Y = np.zeros((m,p,2))
    Y[:,:,0] = y1
    Y[:,:,1] = y-c*y1
    
    Y[Y<0]=0
    
    if np.sum(np.isnan(Y)) != 0:
        raise ValueError("Nan in Y")
    return Y


def get_Y(theta, t, tau):
    # theta: p*(K+4)
    # t: len m
    # tau: len K+1
    # return m * p * 2
    p = len(theta)
    K = len(tau)-1 # number of states
    if np.shape(theta)[1]!=K+4:
        print(np.shape(theta)[1], K+4)
        raise TypeError("wrong parameters lengths")
    a = theta[:,0:K].T
    beta = theta[:,-2].reshape((1,-1))
    gamma = theta[:,-1].reshape((1,-1))

    y1_0 = theta[:,-4].reshape((1,-1))
    y2_0 = theta[:,-3].reshape((1,-1))

    c = beta/(beta-gamma+eps)
    d = beta**2/((beta-gamma)*gamma+eps)
    y_0 = y2_0 + c*y1_0
    a_ = d*a
    t = t.reshape(-1,1)
    m = len(t)
  
    I = np.ones((K+1,m),dtype=bool)

    # nascent
    y1=y1_0*np.exp(-t@beta)
    for k in range(1,K+1):
        I[k] = np.squeeze(t > tau[k])
        idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
        y1 = y1 + a[None,k-1] * (np.exp(- (I[k,:,None] *(t-tau[k]))@beta)- np.exp(-(I[k,:,None]*(t-tau[k-1]))@beta )) \
          + a[None,k-1] * (1 - np.exp(- (idx[:,None] *(t-tau[k-1]))@beta ) )
    
    if np.sum(np.isnan(y1)) != 0:
        raise ValueError("Nan in y1")
    # mature + c * nascent 
    y=y_0*np.exp(-t@gamma)    
    for k in range(1,K+1):
        idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
        y = y + a_[None,k-1] * (np.exp(-(I[k,:,None] * (t-tau[k]))@gamma)- np.exp(-(I[k,:,None] * (t-tau[k-1]))@gamma )) \
          +  a_[None,k-1] * (1 - np.exp(-(idx[:,None]*(t-tau[k-1]))@gamma) )

    Y = np.zeros((m,p,2))
    Y[:,:,0] = y1
    Y[:,:,1] = y-c*y1
    
    Y[Y<0]=0
    
    if np.sum(np.isnan(Y)) != 0:
        raise ValueError("Nan in Y")
    return Y


def get_Y2(theta, t, tau):
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
    t = t.reshape(-1,1)
    m = len(t)
    I = np.ones((K+1,m),dtype=bool)
    
    # nascent
    y1=np.zeros((m,p))
    y1=y1+y1_0[None,:]*np.exp(-beta[None,:]*t)   
    for k in range(1,K+1):
        I[k] = np.squeeze(t > tau[k])
        idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
        y1[I[k]] = y1[I[k]] + a[None,:,k-1] * (np.exp(- beta[None,:]*(t[I[k]]-tau[k]))- np.exp(-beta[None,:]* (t[I[k]]-tau[k-1])) ) 
        y1[idx] = y1[idx] + a[None,:,k-1] * (1 - np.exp(- beta[None,:] *(t[idx]-tau[k-1]))) 

    if np.sum(np.isnan(y1)) != 0:
        raise ValueError("Nan in y1")
    Y =np.zeros((m,p,2))
    Y[:,:,0] = y1
    
    # mature + c * nascent 
    ## nondegenrate cases
    nondegenerate = np.abs(beta-gamma)>eps # indices of genes having similar gamma and beta
    if np.sum(nondegenerate)>0:
        y =np.zeros((m,np.sum(nondegenerate)))
        beta_, gamma_ = beta[nondegenerate], gamma[nondegenerate]
        c = beta_/(beta_-gamma_)
        d = beta_**2/((beta_-gamma_)*gamma_)
        y_0 = y2_0[nondegenerate] + c*y1_0[nondegenerate]
        a_ = d[:,None]*a[nondegenerate,:]

        y=y+y_0[None,:]*np.exp(-gamma_[None,:]*t)    
        for k in range(1,K+1):
            idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
            y[I[k]] = y[I[k]] + a_[None,:,k-1] * (np.exp(-gamma_[None,:] * (t[I[k]]-tau[k]))- np.exp(-gamma_[None,:]*(t[I[k]]-tau[k-1])) )
            y[idx] = y[idx] +  a_[None,:,k-1] * (1 - np.exp(-gamma_[None,:]*(t[idx]-tau[k-1]))) 

        if np.sum(np.isnan(y)) != 0:
            raise ValueError("Nan in y")

        Y[:,nondegenerate,1] = y-c*y1[:,nondegenerate]
    
    ## nondegenrate cases
    degenerate = ~nondegenerate
    if np.sum(degenerate)>0:
        y = np.zeros((m,np.sum(degenerate)))
        y = y + y1_0[None,degenerate]*beta[None,degenerate]*t*np.exp(-beta[None,degenerate]*t) + y2_0[None,degenerate]*np.exp(-gamma[None,degenerate]*t) 
        for k in range(1,K+1):
            idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k with a[k-1]
            y[I[k]] = y[I[k]] + a[None,degenerate,k-1]*beta[None,degenerate]*((t[I[k]]-tau[k])*(np.exp(- beta[None,degenerate]*(t[I[k]]-tau[k])) - np.exp(- beta[None,degenerate]*(t[I[k]]-tau[k-1]))))    
            y[I[k]] = y[I[k]] + a[None,degenerate,k-1] * (np.exp(- beta[None,degenerate]*(t[I[k]]-tau[k])) - np.exp(- beta[None,degenerate] *(t[I[k]]-tau[k-1])) \
                                             - beta[None,degenerate] *(tau[k]-tau[k-1])*np.exp(- beta[None,degenerate] *(t[I[k]]-tau[k-1])))
            y[idx] = y[idx] + a[None,degenerate,k-1] * (1 - np.exp(-beta[None,degenerate]*(t[idx]-tau[k-1]))\
                                             - beta[None,degenerate] *(t[idx]-tau[k-1])*np.exp(- beta[None,degenerate]*(t[idx]-tau[k-1]))) 
        Y[:,degenerate,1] = y
    
    if np.sum(np.isnan(Y)) != 0:
        raise ValueError("Nan in Y")
    return Y

def neglogL(theta, x, Q, t, tau, topo, penalty):
    # theta: length K+4
    # x: n*2
    # Q: n*L*m
    # t: len m
    # tau: len K+1
    logL = 0
    for l in range(len(topo)):
        theta_l = np.concatenate((theta[topo[l]], theta[-4:]))
        Y = get_Y(theta_l[None,:],t,tau)[:,0,:] # m*2
        weight_l = Q[:,l,:] #n*m
        x_weighted = weight_l.T@x # m*2
        marginal_weight =  weight_l.sum(axis=0)[:,None] # m*1
        logL += np.sum( x_weighted * np.log(eps + Y) - marginal_weight*Y )

    loss = - logL    

    ## L1 Regularization
    if penalty>0:
        for l in range(len(topo)): 
            loss += penalty * np.abs(theta[topo[l][0]]-theta[-4]) 
            for k in range(1,len(topo[l])):
                loss += penalty * np.abs(theta[topo[l][k]]-theta[topo[l][k-1]]) 
        loss += penalty * np.abs(theta[-3]*theta[-1]/theta[-2]-theta[-4]) 

    return loss

def neglogL_old(theta, x, Q, t, tau, topo, penalty=0):
    # theta: length K+4
    # x: n
    # Q: n*L*m
    # t: len m
    # tau: len K+1
    logL = 0
    K = len(tau) - 1
    for l in range(len(topo)):
        theta_l = np.concatenate((theta[topo[l]], theta[-4:]))
        Y = get_Y(theta_l[None,:],t,tau) # m*1*2
        Y = Y[None,:,0,:] # 1*m*2
        weight_l = Q[:,l,:,None] #n*m*1
        logL += np.sum( weight_l*(x[:,None,:] * np.log(eps + Y) - Y) )

    loss = - logL    

    ## L1 Regularization
    if penalty>0:
        for l in range(len(topo)): 
            loss += penalty * np.abs(theta[topo[l][0]]-theta[-4]) 
            for k in range(1,len(topo[l])):
                loss += penalty * np.abs(theta[topo[l][k]]-theta[topo[l][k-1]]) 
        loss += penalty * np.abs(theta[-3]*theta[-1]/theta[-2]-theta[-4]) 

    return loss

  
def neglogL_bg(theta_bg,theta_a,x,Q,t,tau,topo,penalty):
    theta = np.append(theta_a, theta_bg)
    loss = neglogL(theta,x,Q,t,tau,topo,penalty)
    return loss

def neglogL_a(theta_a,theta_bg,x,Q,t,tau,topo,penalty):
    theta = np.append(theta_a, theta_bg)
    loss = neglogL(theta,x,Q,t,tau,topo,penalty)
    return loss

def update_theta_j(theta0, x, Q, t, tau, topo, penalty, alternative, bnd=1000, bnd_beta=100, miter = 1000, alteriter = 20):
    bounds = [[0,bnd]]*len(theta0)
    bounds[-2:] = [[1/bnd_beta,bnd_beta]]*2
    if alternative:
        new_theta = theta0.copy()
        for i in range(alteriter):
            ## minimize wrt a
            res1 = minimize(neglogL_a, new_theta[:-2], args=(new_theta[-2:],x,Q,t,tau,topo,penalty), bounds = bounds[:-2], options={'maxiter': int(miter/alteriter/2),'disp': False}) 
            new_theta[:-2]=res1.x
            ## minimize wrt beta and gamma
            res2 = minimize(neglogL_bg, new_theta[-2:], args=(new_theta[:-2],x,Q,t,tau,topo,penalty), bounds = bounds[-2:], options={'maxiter': int(miter/alteriter/2),'disp': False}) 
            new_theta[-2:]=res2.x
    else:
        res = minimize(neglogL, theta0, args=(x,Q,t,tau,topo,penalty), bounds=bounds, options={'maxiter': miter,'disp': False}) 
        new_theta = res.x
    return new_theta

def simulate_data(topo,tau,n,p,random_seed=2022,loomfilepath=None):
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
        adata.write_loom(loomfilepath)
    return theta, Y, X
    
class TI_model:
    """
    A object to store the data and parameters of a probabilitic trajectory inference model.
    
    
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
    
    def _initialize_Q(self, n, seed):
        Q=np.random.uniform(0,1,(n,self.L,self.m))
        Q=Q/Q.sum(axis=(-2,-1),keepdims=True)
        return Q

    def update_theta(self,X,Q,penalty=0,alternative=False,parallel=False,n_threads=1,theta0=None, bnd=1000, bnd_beta=100):
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
                Input_args.append((self.theta[j], X[:,j], Q, self.t, self.tau, self.topo, penalty, alternative, bnd, bnd_beta))
            with Pool(n_threads) as pool:      
                new_theta = pool.starmap(update_theta_j, Input_args)
            new_theta = np.array(new_theta)
        else:
            new_theta = np.zeros((p,self.n_states+4))
            for j in range(p): 
                new_theta[j]=update_theta_j(self.theta[j], X[:,j], Q, self.t, self.tau, self.topo, penalty, alternative, bnd, bnd_beta)
                
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
        for i in range(epoch):
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
    
    def fit_(self, X, m, relative_penalty=0, n_init=3, epoch=10, alternative=False, tol=0.01, parallel=False, n_threads=1):
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
        
        self._initialize_theta(X)
        self.set_m(m)
        
        alternative = False
        max_lower_bound = -np.inf
        for init in tqdm(range(n_init)):
            Q = self._initialize_Q(n, init)
            lower_bound = -np.inf
            self.converged = False
            for i in range(epoch):
                prev_lower_bound = lower_bound
                self.update_theta(X,Q,parallel=parallel,n_threads=n_threads)
                Q, lower_bound = self.update_weight(X)
                change = lower_bound - prev_lower_bound
                if abs(change) < tol:
                    self.converged = True
                    break
                
            if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                max_lower_bound = lower_bound
                best_theta = self._get_theta()
                
        self.theta = best_theta
        return
    
    def _compute_lower_bound(self,X,Q):
        n,p,s=np.shape(X)
        Y = np.zeros((self.L,self.m,p,2))
        for l in range(self.L):
            theta_l = np.concatenate((self.theta[:,self.topo[l]], self.theta[:,-4:]), axis=1)
            Y[l] = get_Y(theta_l,self.t,self.tau) # m*p*2
        #logL =  np.sum(X[:,None,None,:,:] * np.log(Y[None,:]+eps) - Y[None,:], axis=(-2,-1)) # n*L*m*p*2 -> n*L*m
        logL = np.tensordot(X, np.log(eps + Y), axes=([-2,-1],[-2,-1]))
        logL -= np.sum(Y,axis=(-2,-1))
        return np.mean(logsumexp(logL, axis=(-2,-1)))
