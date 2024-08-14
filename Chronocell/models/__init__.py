#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 11:59:09 2022

@author: fang

Each model file should define following functions:

def check_params(traj):
    
    Check parameters such topo and tau are correct for this model. 
    Raise error if not.
    
    Parameters
    ----------
    traj : Trajectory object
        DESCRIPTION.

    Returns
    -------
    None.


def guess_theta(X,topo,tau):    
    
    Initialize theta with given X, topo and tau.
    
    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    topo : TYPE
        DESCRIPTION.
    tau : TYPE
        DESCRIPTION.

    Returns
    -------
    theta : TYPE
        DESCRIPTION.

    

def get_Y_hat(theta,t,tau,topo,params):

    Get the mean values Y_hat for X at each lineage and time grids defined by topo, tau and t.

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    tau : TYPE
        DESCRIPTION.
    topo : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.

    Returns
    -------
    Y_hat : numpy array with size (L,m,p,s) 
        


def get_logL(X,theta,t,tau,topo,params):

    Get the logL for X at each lineage and time grids defined by topo, tau and t.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    theta : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    tau : TYPE
        DESCRIPTION.
    topo : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.

    Returns
    -------
    logL : TYPE
        DESCRIPTION.



def update_theta_j(theta0, x, Q, t, tau, topo, params, restrictions):

    Fit theta with initial theta0, data for one gene x, and posterior Q.

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
    params : TYPE
        DESCRIPTION.
    restrictions : TYPE
        DESCRIPTION.

    Returns
    -------
    theta : TYPE
        DESCRIPTION.


"""

