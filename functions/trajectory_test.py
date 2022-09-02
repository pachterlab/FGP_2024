import numpy as np
import matplotlib.pyplot as plt

from trajectory import Trajectory
from utils import simulate_data
from plotting import plot_phase, plot_t, plot_theta

if __name__ == "__main__":  
    
    topo = np.array([[0],[1]])
    tau=(0,1)
    n = 2000
    p = 20
    theta, t, Y, X = simulate_data(topo, tau, n, p)
    
    #for j in range(2):
    #    theta[j,0:K] = theta[j,-4]
    #    theta[j,-3]=theta[j,-4]*theta[j,-2]/theta[j,-1]
    
    #Y = get_Y(theta,t,tau)
    
    #X = np.random.poisson(Y)
    
    plot_p = min(10, p)
    fig, ax = plt.subplots(1,plot_p,figsize=(6*plot_p,4))
    for i in range(plot_p):
        j = i + 10
        ax[i].scatter(X[:,j,0],X[:,j,1],c="gray");
        ax[i].scatter(Y[:,j,0],Y[:,j,1],c=t);
       
    ##### fit with correct theta0 #####
    traj = Trajectory(topo, tau)
    Q, lower_bound = traj.fit_warm_start(X, theta=theta+np.random.normal(0,0.1,size=theta.shape), epoch=1, parallel=True, n_threads=4)
    plot_theta(theta,traj.theta)
    plot_t(Q, l=0, t=t)
    plot_t(Q, l=1, t=t)

    ##### fit with correct Q0 #####
    traj = Trajectory(topo, tau)
    L = len(topo)
    resol = 50
    m = n//resol
    Q0 = np.zeros((L*n,L,m))
    for l in range(L):
        for i in range(n):
            Q0[i+n*l,l,i//resol] = 1
    
    Q, lower_bound = traj.fit_warm_start(X, Q=Q0, epoch=10, parallel=True, n_threads=4)
    plot_theta(theta,traj.theta)
    plot_t(Q, l=0, t=t)
    plot_t(Q, l=1, t=t)
    
    traj = Trajectory(topo, tau)
    Q, lower_bounds, thetas = traj.fit_multi_init(X, 10, n_init=6, epoch=10, parallel=True, n_threads=4)
    
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
    
    
    #%% test AIC
    """
    old_AIC = - 2 * n * traj_X.compute_lower_bound(X)
    old_theta = traj_X.theta.copy()
    nested_model = {0:[[1,-3],[-4,-4]], 1:[[1,-3],[-4,-4]]}
    traj_X.theta = old_theta.copy()
    for j in nested_model.keys():
        traj_X.theta[j,0:-3]=np.mean(X[:,j,0])
        traj_X.theta[j,-3]=np.mean(X[:,j,1])
        traj_X.theta[j,-2]=1
        traj_X.theta[j,-1]=np.mean(X[:,j,0])/np.mean(X[:,j,1])
        
    k = - ( K + 2 ) * 2
    new_AIC = - 2 * n * traj_X.compute_lower_bound(X) + 2 * k
    new_AIC - old_AIC
    """

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    