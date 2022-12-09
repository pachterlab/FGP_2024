import numpy as np
import matplotlib.pyplot as plt

from inference import Trajectory
from simulation import simulate_data
from plotting import plot_phase, plot_t, plot_theta
    
    
if __name__ == '__main__':
    #%%    
    topo = np.array([[0,1]])
    tau=(0,1,2)
    n = 1000
    p = 100
    theta, t, Y, X = simulate_data(topo, tau, n, p, model="two_species_ss")
    
    plot_p = min(10, p)
    fig, ax = plt.subplots(1,plot_p,figsize=(6*plot_p,4))
    for i in range(plot_p):
        j = i
        ax[i].scatter(X[:,j,0],X[:,j,1],c="gray");
        ax[i].scatter(Y[:,j,0],Y[:,j,1],c=t);
        ax[i].set_title(j)
       
    ##### fit with correct Q0 #####
    traj = Trajectory(topo, tau, model="two_species_ss",verbose=1)
    L = len(topo)
    resol = 100
    m = n//resol
    Q0 = np.zeros((L*n,L,m))
    for l in range(L):
        for i in range(n):
            Q0[i+n*l,l,i//resol] = 1
    Q0 = np.random.uniform(0,1,size=Q0.shape)
    Q0 /= Q0.sum(axis=(-2,-1),keepdims=True)
    traj = traj.fit(X, epoch=10)#, parallel=True, n_threads=4)
    plot_theta(theta,traj.theta)
    plot_t(traj.Q, l=0, t=t)
    #plot_phase(traj)
    
    
    #for i in range(10):
    #    #plot_theta(theta,traj.theta_hist[i])
    #    plot_t(traj.Q_hist[i*10], l=0, t=t)
    
    ##### fit #####
    """
    traj = Trajectory(topo, tau, verbose=1)
    Q, lower_bound = traj.fit(X, m=100, epoch=10, parallel=True, n_threads=4)
    plot_theta(theta,traj.theta)
    plot_t(Q,t=t)
    
        
    traj = Trajectory(topo, tau, model="two_species_ss", verbose=1)
    Q, elbos = traj.fit(X,tol=1e-10)
    plot_theta(theta,traj.theta)
    
    
 
    Q, lower_bounds = traj.fit_multi_init(X, 100, n_init=2, epoch=10, parallel=True, n_threads=4)
    plot_theta(theta,traj.theta)
    plot_t(Q, l=0, t=t)
    plot_t(Q, l=1, t=t)
    plot_phase(X, Q, traj)
    plt.plot(lower_bounds)
    print(traj.compute_AIC(X))
    #for j in range(2):
    #    theta[j,0:K] = theta[j,-4]
    #    theta[j,-3]=theta[j,-4]*theta[j,-2]/theta[j,-1]
    
    traj_ = Trajectory(topo, tau, model="two_species", restrictions={}, verbose=1)
    Q_, lower_bounds_ = traj_.fit_multi_init(X, 100, n_init=2, epoch=10, parallel=True, n_threads=4)
    plot_theta(theta,traj_.theta)
    print(-2*len(X)*traj_.compute_lower_bound(X))
    print(-2*len(X)*traj.compute_lower_bound(X))
    plot_phase(traj_)
    
    EAIC = []
    for ii in range(100):
        test_X = np.random.poisson(Y)
    #-2*len(X)*traj.compute_lower_bound(test_X)
        EAIC.append(-2*len(X)*traj.compute_lower_bound(test_X))
    print(np.mean(EAIC))
    
    EAIC_ = []
    for ii in range(100):
        test_X = np.random.poisson(Y)
    #-2*len(X)*traj.compute_lower_bound(test_X)
        EAIC_.append(-2*len(X)*traj_.compute_lower_bound(test_X))
    print(np.mean(EAIC_))

    #accept, delta, new = traj_.compare_model(X, new_model={1:[[1],[0]],5:[[1],[0]]})
    #print(accept, delta)
    

    error = np.abs(theta-traj.theta)/theta**(1/4)
    plt.plot(theta[:,0],error[:,0],'.',alpha=0.1)
    plt.yscale("log")
    plt.title(np.around(np.corrcoef(theta[:,0],error[:,0],)[0,1],3))
    
    error_r = np.abs(theta-traj.theta)/theta
    plt.plot(theta[:,0],error_r[:,0],'.',alpha=0.1)
    plt.yscale("log")
    plt.title(np.around(np.corrcoef(theta[:,0],error_r[:,0],)[0,1],3))
    
    error_a = np.abs(theta-traj.theta)/theta**(1/2)
    plt.plot(theta[:,-1],error_a[:,-1],'.',alpha=0.1)
    plt.yscale("log")
    plt.title(np.around(np.corrcoef(theta[:,-1],error_a[:,-1],)[0,1],3))
    
    #plt.plot(traj.theta[:,-1],'.')
    #plt.axhline(y=1,color='r')
    #plt.yscale('log')
    
    #plot_t(Q, l=0, t=t)
    #plot_phase(X,traj.theta[:plot_p],Q,topo,tau)
    
    plot_t(Q, l=1, t=t)
    
    traj = Trajectory(topo, tau)
    Q, lower_bounds, thetas = traj.fit_multi_init(X, 10, n_init=1, epoch=10, parallel=True, n_threads=4)
    
    plot_phase(X,traj.theta[:plot_p],Q,topo,tau)
    plt.plot(lower_bounds)
    plot_theta(theta,traj.theta)
    plot_t(Q, l=0, t=t)
    plot_t(Q, l=1, t=t)
    
    Q, lower_bound = traj.fit_warm_start(X, Q=Q, theta=traj.theta, epoch=10, parallel=True, n_threads=4)
    
    
    
    accepts = []
    diffs = []
    for j in range(p):
        nested_model = {j:[[1],[0]]}
        accept, diff = traj.compare_model(X, nested_model)
        accepts.append(accept)
        diffs.append(diff)
    
    # no jac 2862.8
    """
    
    #%% test AIC
    
    """
    old_AIC = traj.compute_AIC(X)
    test_AIC = -2* traj.compute_lower_bound(test_X)
    
    nested_model = {9:[[1,-3],[-4,-4]]}
    
    traj_nested = Trajectory(topo, tau, nested_model)
    Q, logL = traj_nested.fit_restrictions(X, Q, traj.theta)      
    
    
    new_AIC = traj_nested.compute_AIC(X)
    new_test_AIC = -2 * traj_nested.compute_lower_bound(test_X)
    new_AIC - old_AIC
    
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
