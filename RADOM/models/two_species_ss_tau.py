#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 21:19:11 2023


This file defines functions (guess_theta, update_theta_j, update_theta_j, get_logL) for two species 
initial steady state Poisson model. 
theta for each gene contains transcription rate of each states, u0, beta, gamma.

"""

import numpy as np
from scipy.optimize import minimize
#from scipy.special import gammaln


# global parameters: upper and lower limits for numerical stability
eps = 1e-10

def check_params(traj):
    assert len(traj.tau) == len(traj.topo[0])
    if len(traj.tau)<=2:
        print("no variable tau needed, use two_species_ss instead")
    return

def guess_theta(X,topo,tau):
    n_state=len(set(topo.flatten()))
    K = len(tau)-1
    p = X.shape[1]
    theta = np.zeros((p,n_state+K+1))
    theta[:,0:n_state] = np.mean(X[:,:,0],axis=0)[:,None]
    theta[:,n_state:-2] = tau[1:-1]
    theta[:,-2] = np.sqrt(np.mean(X[:,:,1],axis=0)/(np.mean(X[:,:,0],axis=0)+eps))
    theta[:,-1] = np.sqrt(np.mean(X[:,:,0],axis=0)/(np.mean(X[:,:,1],axis=0)+eps))
    return theta


def get_y(theta, t, global_tau):
    # theta: a_1, ..., a_K, u_0, s_0, beta, gamma
    # t: len m
    # tau: len K+1
    # return m * p * 2
    K = len(global_tau)-1 # number of states
    assert len(theta) == 2*K+2
    y1_0 = theta[0]
    a = theta[1:K+1]
    tau = np.zeros(K+1)
    tau[0]=global_tau[0]
    tau[-1]=global_tau[-1]
    tau[1:-1]=theta[K+1:-2]
    beta = theta[-2]
    gamma = theta[-1]
    if beta == gamma:
        beta += eps

    c = beta/(beta-gamma+eps)
    d = beta**2/((beta-gamma)*gamma+eps)
    y_0 = d*y1_0
    a_ = d*a
    t = t
    m = len(t)
  
    I = np.ones((K+1,m),dtype=bool)

    # nascent
    y1=y1_0*np.exp(-t*beta)
    y=y_0*np.exp(-t*gamma)   
    for k in range(1,K+1):
        I[k] = np.squeeze(t > tau[k])
        idx = I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
        y1 += a[k-1] * (np.exp(- (I[k] *(t-tau[k]))*beta)- np.exp(-(I[k]*(t-tau[k-1]))*beta )) \
          + a[k-1] * (1 - np.exp(- (idx *(t-tau[k-1]))*beta ) )
        y += a_[k-1] * (np.exp(-(I[k] * (t-tau[k]))*gamma)- np.exp(-(I[k] * (t-tau[k-1]))*gamma )) \
          +  a_[k-1] * (1 - np.exp(-(idx*(t-tau[k-1]))*gamma) )

    Y = np.zeros((m,2))
    Y[:,0] = y1
    Y[:,1] = y-c*y1
    
    Y[Y<0]=0
    
    if np.sum(np.isnan(Y)) != 0:
        raise ValueError("Nan in Y")
    return Y

def get_y_jac(theta, t, tau):
    # theta: u_0, a_1, ..., a_K, tau_1, tau_K-1, beta, gamma
    # t: len m
    # tau: len K+1
    # return m * p * 2
    K = len(tau)-1 # number of states
    assert len(theta) == 2*K+2
    y1_0 = theta[0]
    a = theta[1:K+1]
    tau_ = np.zeros(K+1)
    tau_[0]=tau[0]
    tau_[-1]=tau[-1]
    tau_[1:-1]=theta[K+1:-2]
    beta = theta[-2]
    gamma = theta[-1]

    c = beta/(beta-gamma+eps) 
    dc_dbeta = - gamma/((beta-gamma)**2+eps)
    dc_dgamma = beta/((beta-gamma)**2+eps)
    d = beta**2/((beta-gamma)*gamma+eps)
    dd_dbeta = (beta**2-2*beta*gamma)/((beta-gamma)**2*gamma+eps)
    dd_dgamma = (-beta**3+2*beta**2*gamma)/((beta-gamma)**2*gamma**2+eps)
    y_0 = d*y1_0
    a_ = d*a
    da__dbeta = a * dd_dbeta
    da__dgamma = a * dd_dgamma

    
    m = len(t)
    I = np.ones((K+1,m),dtype=bool)
    dY_dtheta = np.zeros((m,2,len(theta)))
    
    # nascent
    y1=y1_0*np.exp(-t*beta)
    y=y_0*np.exp(-t*gamma)
    dY_dtheta[:,0,0] = np.exp(-t*beta)
    dY_dtheta[:,1,0] = d * np.exp(-t*gamma) 
    
    
    dY_dtheta[:,0,-2] = - t * y1_0 * np.exp(-t*beta)
    dY_dtheta[:,1,-2] = dd_dbeta * y1_0 * np.exp(-t*gamma)
    dY_dtheta[:,1,-1] = dd_dgamma * y1_0 * np.exp(-t*gamma)  - t * y_0 * np.exp(-t*gamma)
    
    for k in range(1,K+1):
        I[k] = np.squeeze(t > tau_[k])
        idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
        
        y1_a_k_coef = np.exp(- (I[k] *(t-tau_[k]))*beta) - np.exp(-(I[k] * (t-tau_[k-1])) * beta ) + 1 - np.exp(- (idx * (t-tau_[k-1]))*beta ) 
        y_a_k_coef = np.exp(- (I[k] * (t-tau_[k]))*gamma) - np.exp(-(I[k] * (t-tau_[k-1])) * gamma) + 1 - np.exp(- (idx * (t-tau_[k-1]))*gamma)

        y1 += a[k-1] *  y1_a_k_coef
        y += a_[k-1] * y_a_k_coef
        
        dY_dtheta[:,0,k] = y1_a_k_coef
        dY_dtheta[:,1,k] = d * y_a_k_coef
        
        dy1_a_k_coef_dbeta = - (t-tau_[k]) * I[k] * np.exp(- (I[k] * (t-tau_[k])) * beta) + (t-tau_[k-1]) * I[k] * np.exp(-(I[k] * (t-tau_[k-1])) * beta ) +  idx * (t-tau_[k-1]) * np.exp(- (idx * (t-tau_[k-1])) * beta)
        dy_a_k_coef_dgamma = - (t-tau_[k]) * I[k] * np.exp(- (I[k] * (t-tau_[k])) * gamma) + (t-tau_[k-1]) * I[k] * np.exp(-(I[k] * (t-tau_[k-1])) * gamma ) + idx * (t-tau_[k-1]) * np.exp(- (idx * (t-tau_[k-1])) * gamma) 
        
        dY_dtheta[:,0,-2] += a[k-1] * dy1_a_k_coef_dbeta
        dY_dtheta[:,1,-2] += da__dbeta[k-1] * y_a_k_coef
        dY_dtheta[:,1,-1] += da__dgamma[k-1] * y_a_k_coef + a_[k-1] * dy_a_k_coef_dgamma
    
    for k in range(1,K):
        dY_dtheta[:,0,K+k] += I[k]* (a[k-1]-a[k]) * beta * np.exp(- (I[k] * (t-tau_[k])) * beta)
        dY_dtheta[:,1,K+k] += I[k]* (a_[k-1]-a_[k]) * gamma * np.exp(- (I[k] * (t-tau_[k])) * gamma)
        
    Y = np.zeros((m,2))
    Y[:,0] = y1
    Y[:,1] = y-c*y1
    
    dY_dtheta[:,1,0:-2] -= c * dY_dtheta[:,0,0:-2]
    dY_dtheta[:,1,-2] -=  c * dY_dtheta[:,0,-2] + dc_dbeta * y1
    dY_dtheta[:,1,-1] -= dc_dgamma * y1
    
    Y[Y<0]=0
    if np.sum(np.isnan(Y)) != 0:
        raise ValueError("Nan in Y")
        
    if np.sum(np.isnan(dY_dtheta)) != 0:
        raise ValueError("Nan in dY_dtheta")
        
    return Y, dY_dtheta


def get_Y(theta, t, tau):
    # theta: p*(K+4)
    # t: len m
    # tau: len K+1
    # return m * p * 2
    p = len(theta)
    K = len(tau)-1 # number of states
    assert np.shape(theta)[1] == 2*K+2, f"theta ({np.shape(theta)[1]}) and tau ({K+1}) are not compatible"
    a = theta[:,1:K+1].T
    tau_ = np.zeros((K+1,1,p))
    tau_[0,0,:] = tau[0]
    tau_[1:-1,0,:] = theta[:,K+1:-2].T
    tau_[-1,0,:] = tau[-1]
    beta = theta[:,-2].reshape((1,-1))
    gamma = theta[:,-1].reshape((1,-1))
    
    y1_0 = theta[:,0].reshape((1,-1))

    c = beta/(beta-gamma+eps)
    d = beta**2/((beta-gamma)*gamma+eps)
    y_0 = d*y1_0
    a_ = d*a
    t = t.reshape(-1,1)
    m = len(t)
  
    I = np.ones((K+1,m,p),dtype=bool)

    # nascent
    y1=y1_0*np.exp(-t@beta)
    for k in range(1,K+1):
        I[k] = t > tau_[k]
        idx = I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
        y1 += a[None,k-1] * (np.exp(- (I[k] *(t-tau_[k]))*beta)- np.exp(-(I[k]*(t-tau_[k-1]))*beta )) \
          + a[None,k-1] * (1 - np.exp(- (idx * (t-tau_[k-1]))*beta ) )
    if np.sum(np.isnan(y1)) != 0:
        raise ValueError("Nan in y1")
    # mature + c * nascent 
    y=y_0*np.exp(-t@gamma)    
    for k in range(1,K+1):
        idx =I[k-1]*(~I[k]) # tau__{k-1} < t_i <= tau__k
        y = y + a_[None,k-1] * (np.exp(-(I[k] * (t-tau_[k]))*gamma)- np.exp(-(I[k] * (t-tau_[k-1]))*gamma )) \
          +  a_[None,k-1] * (1 - np.exp(-(idx*(t-tau_[k-1]))*gamma) )

    Y = np.zeros((m,p,2))
    Y[:,:,0] = y1
    Y[:,:,1] = y-c*y1
    
    Y[Y<0]=0
    
    if np.sum(np.isnan(Y)) != 0:
        raise ValueError("Nan in Y")
    return Y

def neglogL(theta, x_weighted, marginal_weight, t, tau, topo, lam_t=0, lam_a=0):
    # theta: length K+4
    # x: n*2
    # Q: n*L*m
    # t: len m
    # tau: len K+1
    logL = 0
    n_states=len(set(topo.flatten()))
    parents = np.zeros(n_states,dtype=int)
    for l in range(len(topo)):
        theta_l = np.concatenate((theta[topo[l]], theta[n_states:]))
        Y = get_y(theta_l,t,tau) # m*2
        logL += np.sum( x_weighted[l] * np.log(eps + Y) - marginal_weight[l]*Y )
        parents[topo[l][1:]] = topo[l][:-1]
    penalty_t = np.sum((theta[n_states:-2]-tau[1:-1])**2)
    penalty_a = np.sum((theta[1:n_states]-theta[parents[1:n_states]])**2)
    return - logL + lam_t * penalty_t + lam_a * penalty_a

def neglogL_a(theta_a, theta_tau, x_weighted, marginal_weight, t, tau, topo, lam_t=0, lam_a=0):
    theta = np.insert(theta_a,-2,theta_tau)
    return neglogL(theta, x_weighted, marginal_weight, t, tau, topo, lam_t, lam_a)

def neglogL_tau(theta_tau, theta_a, x_weighted, marginal_weight, t, tau, topo, lam_t=0, lam_a=0):
    theta = np.insert(theta_a,-2,theta_tau)
    return neglogL(theta, x_weighted, marginal_weight, t, tau, topo, lam_t, lam_a)

def neglogL_jac(theta, x_weighted, marginal_weight, t, tau, topo, lam_t=0, lam_a=0):
    # theta: length K+4
    # x: n*2
    # Q: n*L*m
    # t: len m
    # tau: len K+1
    K = len(tau)-1
    n_states=len(set(topo.flatten()))
    parents = np.zeros(n_states,dtype=int)
    jac = np.zeros_like(theta)
    penalty_a_jac = np.zeros_like(theta)
    penalty_t_jac = np.zeros_like(theta)

    for l in range(len(topo)):
        theta_idx = np.append(topo[l],list(range(-K-1,0)))
        theta_l = theta[theta_idx]
        Y, dY_dtheta = get_y_jac(theta_l,t,tau) # m*2*len(theta)
        coef =  x_weighted[l] / (eps + Y) - marginal_weight[l]
        jac[theta_idx] += np.sum( coef[:,:,None] * dY_dtheta, axis=(0,1))
        parents[topo[l][1:]] = topo[l][:-1]
    penalty_a_jac[1:n_states] = 2*(theta[1:n_states]-theta[parents[1:n_states]])
    penalty_a_jac[parents[1:n_states]] += 2*(theta[parents[1:n_states]] - theta[1:n_states])
    penalty_t_jac[-(K+1):-2] = 2*(theta[-(K+1):-2]-tau[1:-1])
    return - jac + lam_t * penalty_t_jac + lam_a * penalty_a_jac

def neglogL_jac_a(theta_a, theta_tau, x_weighted, marginal_weight, t, tau, topo, lam_t=0, lam_a=0):
    theta = np.insert(theta_a,-2,theta_tau)
    a_idx = np.append(list(range(len(set(topo.flatten())))),[-2,-1])
    return neglogL_jac(theta, x_weighted, marginal_weight, t, tau, topo, lam_t, lam_a)[a_idx]

def neglogL_jac_tau(theta_tau, theta_a, x_weighted, marginal_weight, t, tau, topo, lam_t=0, lam_a=0):
    theta = np.insert(theta_a,-2,theta_tau)
    K = len(tau)-1
    return neglogL_jac(theta, x_weighted, marginal_weight, t, tau, topo, lam_t, lam_a)[-(K+1):-2]

def get_logL(X,theta,t,tau,topo,params):
    L=len(topo)
    m=len(t)
    p=len(theta)
    K = len(tau)-1
    Y = np.zeros((L,m,p,2))
    for l in range(L):
        theta_l = np.concatenate((theta[:,topo[l]], theta[:,-(K+1):]), axis=1)
        Y[l] = get_Y(theta_l,t,tau) # m*p*2
    
    if 'r' in params:
        r = params['r'] # n
        logL = np.tensordot(X, np.log(eps + Y), axes=([-2,-1],[-2,-1])) # logL:n*L*m
        logL += (np.log(r)*np.sum(X,axis=(-2,-1)))[:,None,None]
        logL -= r[:,None,None]*np.sum(Y,axis=(-2,-1))[None,:]
    else:
        logL = np.tensordot(X, np.log(eps + Y), axes=([-2,-1],[-2,-1])) # logL:n*L*m
        logL -= np.sum(Y,axis=(-2,-1))
 
    #logX = np.sum(gammaln(X+1),axis=(-2,-1),keepdims=True)
    #logL -= logX
    return logL

def get_Y_hat(theta,t,tau,topo,params):
    L=len(topo)
    m=len(t)
    p=len(theta)
    K = len(tau)-1
    Y = np.zeros((L,m,p,2))
    for l in range(L):
        theta_l = np.concatenate((theta[:,topo[l]], theta[:,-(K+1):]), axis=1)
        Y[l] = get_Y(theta_l,t,tau) # m*p*2
    return Y
    
def update_theta_j(theta0, x, Q, t, tau, topo, params, restrictions=None, bnd=1000, bnd_beta=1000, bnd_tau=0.001, miter=1000):
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
    n,L,m = Q.shape
    x_weighted = np.zeros((L,m,2))
    marginal_weight = np.zeros((L,m,1))
    
    if 'r' in params:
        r = params['r'] # n
        for l in range(len(topo)):
            weight_l = Q[:,l,:]/n #n*m
            x_weighted[l] = weight_l.T@x # m*2 = m*n @ n*2
            marginal_weight[l] =  (weight_l*r[:,None]).sum(axis=0)[:,None] # m*1
    else:
        for l in range(len(topo)):
            weight_l = Q[:,l,:]/n #n*m
            x_weighted[l] = weight_l.T@x # m*2 = m*n @ n*2
            marginal_weight[l] = weight_l.sum(axis=0)[:,None] # m*1
    
    n_states=len(set(topo.flatten()))
    theta00 = theta0.copy()
    if np.max(theta00[:n_states]) > np.maximum( np.max(x[:,0]) , np.max(x[:,1]) * theta00[-1] / theta00[-2]):
        theta00[:n_states] = np.sqrt( np.mean(x[:,0]) * np.mean(x[:,1]) * theta00[-1] / theta00[-2])
    if np.min(np.abs(np.log10(theta00[-2:]))) > 2:
        theta00[-2] = np.sqrt(np.mean(x[:,1],axis=0)/(np.mean(x[:,0],axis=0)+eps))
        theta00[-1] = np.sqrt(np.mean(x[:,0],axis=0)/(np.mean(x[:,1],axis=0)+eps))
        
    if "lambda_tau" in params:
        lambda_tau = params['lambda_tau']
    else:
        lambda_tau = 0
        
    if "lambda_a" in params:
        lambda_a = params['lambda_a']
    else:
        lambda_a = 0
        
    if restrictions==None:
        res = update_theta_j_unrestricted(theta00, x_weighted, marginal_weight, t, tau, topo, lambda_tau, lambda_a, bnd, bnd_beta, bnd_tau, miter)
    else:
        res = update_theta_j_restricted(theta00, x_weighted, marginal_weight, t, tau, topo, restrictions, lambda_tau, lambda_a, bnd, bnd_beta, bnd_tau, miter)
    return res


def update_theta_j_unrestricted_alternative(theta0, x_weighted, marginal_weight, t, tau, topo, lambda_tau, lambda_a, bnd=1000, bnd_beta=1000, bnd_tau=0.5, miter=1000):
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
    K = len(tau)-1
    n_state=len(set(topo.flatten()))
        
    if K > 1:
        bound_a = [[0,bnd]]*(n_state+2)
        bound_a[-2:] = [[1/bnd_beta,bnd_beta]]*2
        bound_tau = [[tau[0],tau[-1]]]*(K-1)
        for ii in range(1,K):
            bound_tau[ii-1] = [max(tau[0],tau[ii]-bnd_tau), min(tau[ii]+bnd_tau,tau[-1])] 
            
        a_idx = np.append(list(range(len(set(topo.flatten())))),[-2,-1])
        theta_a = theta0.copy()[a_idx]
        theta_tau = theta0.copy()[-K-1:-2]

        for iii in range(1):
            res = minimize(fun=neglogL_a, x0=theta_a, args=(theta_tau,x_weighted,marginal_weight,t,tau,topo,lambda_tau,lambda_a), method = 'L-BFGS-B' , jac = neglogL_jac_a, bounds=bound_a, options={'maxiter': miter,'disp': False}) 
            theta_a = res.x
            res = minimize(fun=neglogL_tau, x0=theta_tau, args=(theta_a,x_weighted,marginal_weight,t,tau,topo,lambda_tau,lambda_a), method = 'L-BFGS-B' , jac = neglogL_jac_tau, bounds=bound_tau, options={'maxiter': miter,'disp': False}) 
            theta_tau = res.x
        
        theta = np.zeros(len(theta0))
        theta[a_idx]=theta_a
        theta[-K-1:-2]=theta_tau  
    else:
        bound = [[0,bnd]]*len(theta0)
        bound[-2:] = [[1/bnd_beta,bnd_beta]]*2
        
        res = minimize(fun=neglogL, x0=theta0, args=(x_weighted,marginal_weight,t,tau,topo,lambda_tau,lambda_a), method = 'L-BFGS-B' , jac = neglogL_jac, bounds=bound, options={'maxiter': miter,'disp': False}) 
        theta = res.x
    return theta

def update_theta_j_unrestricted(theta0, x_weighted, marginal_weight, t, tau, topo, lambda_tau, lambda_a, bnd=1000, bnd_beta=1000, bnd_tau=0.5, miter=1000):
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
    K = len(tau)-1
    n_state=len(set(topo.flatten()))
    bound = [[0,bnd]]*len(theta0)
    bound[-2:] = [[1/bnd_beta,bnd_beta]]*2
    for ii in range(1,K):
        bound[n_state-1+ii] = [max(tau[0],tau[ii]-bnd_tau), min(tau[ii]+bnd_tau,tau[-1])] 
    
    res = minimize(fun=neglogL, x0=theta0, args=(x_weighted,marginal_weight,t,tau,topo,lambda_tau,lambda_a), method = 'L-BFGS-B' , jac = neglogL_jac, bounds=bound, options={'maxiter': miter,'disp': False}) 
    return res.x

def update_theta_j_restricted(theta0, x_weighted, marginal_weight, t, tau, topo, restrictions, lambda_tau, lambda_a, bnd=1000, bnd_beta=1000, bnd_tau=0.1, miter=1000):
    # define a new neglogL inside with fewer parameters
    redundant, blanket = restrictions # 1,0 => a[1] = a[0], 0, -3 => a[0] = u_0,
    if len(redundant) >= len(theta0) - 3:
        theta = np.ones(len(theta0))*np.sum(x_weighted[:,0])
        theta[-2] = 1
        theta[-1] = np.sum(x_weighted[:,0])/np.sum(x_weighted[:,1])
        
    else:
        redundant_mask = np.zeros(len(theta0), dtype=bool)
        redundant_mask[redundant] = True
        custom_theta0 = theta0[~redundant_mask]  
        
        K = len(tau)-1 
        n_state=len(set(topo.flatten())) 
        bound = [[0,bnd]]*len(custom_theta0)
        bound[-2:] = [[1/bnd_beta,bnd_beta]]*2
        for ii in range(1,K):
            bound[n_state-1+ii] = [max(tau[0],tau[ii]-bnd_tau), min(tau[ii]+bnd_tau,tau[-1])] 
        

        def custom_neglogL(custom_theta, x_weighted, marginal_weight, t, tau, topo, lambda_tau, lambda_a):
            theta = np.zeros(len(theta0))
            theta[~redundant_mask] = custom_theta
            theta[redundant] = theta[blanket]
               
            return neglogL(theta, x_weighted, marginal_weight, t, tau, topo, lambda_tau, lambda_a)
                 
        res = minimize(fun=custom_neglogL, x0=custom_theta0, args=(x_weighted,marginal_weight,t,tau,topo,lambda_tau,lambda_a), method = 'L-BFGS-B' , jac = None, bounds=bound, options={'maxiter': miter,'disp': False})         

        theta = np.zeros(len(theta0))
        theta[~redundant_mask] = res.x
        theta[redundant] = theta[blanket]
            
    return theta

