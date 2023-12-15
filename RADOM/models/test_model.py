if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.optimize import check_grad, approx_fprime
    import numpy as np    

#%% check jac function
    np.random.seed(2024)
    topo = np.array([[0,1]])
    tau=(0,1)
    n = 1000
    p = 1
    K=len(tau)-1
    L=len(topo)
    n_states=len(set(topo.flatten()))
    t=np.linspace(tau[0], tau[-1], n)
    
    loga_mu=2
    loga_sd=1
    logb_mu=1
    logb_sd=0.5
    
    ### change this when changing models ###
    from two_species_ss_tau import check_params, guess_theta, get_y, get_Y_hat, get_logL, update_theta_j, neglogL, neglogL_jac, neglogL_a, neglogL_jac_a
    theta=np.ones((p,n_states+K+2))
    theta[:,:n_states]=np.random.lognormal(loga_mu, loga_sd,size=(p,n_states))
    theta[:,-2:]=np.random.lognormal(logb_mu,logb_sd,size=(p,2))
    theta[:,:n_states]/=theta[:,-2,None]
    theta[:,-1]/=np.exp(2)
    
    theta[:,n_states:-2]=tau[:-1]#+np.random.uniform(0,1,size=p)
    #theta[:,-3]=np.exp(np.random.uniform(0,loga_max,size=p))-1
    y = get_y(theta[0], t, tau)
    ### change this when changing models ###    
    Y_ = get_Y_hat(theta, t, tau, topo, {})[0]
    
    X = np.random.poisson(Y_)
    
    x = X[:,0]
    
    m=100
    t_grids = np.linspace(tau[0], tau[-1], m)
    Q = np.zeros((n,L,m))
    for l in range(L):
        for i in range(n):
            Q[i+n*l,l,int(t[i]/tau[-1]*(m-1))] = 1
    
    Q=Q/Q.sum(axis=(1,2),keepdims=True)
    x_weighted = np.zeros((L,m,2))
    marginal_weight = np.zeros((L,m,1))
    
    for l in range(len(topo)):
        weight_l = Q[:,l,:]/n #n*m
        x_weighted[l] = weight_l.T@x # m*2 = m*n @ n*2
        marginal_weight[l] = weight_l.sum(axis=0)[:,None] # m*1
    
    theta0 = guess_theta(X,topo,tau)
    print('True theta',theta[0])
    theta_hat = update_theta_j(0,theta0[0],x,Q,t_grids,tau,topo,params={"lambda_a":0},restrictions=None)
    print('Estimated theta',theta_hat)
    a_idx = np.append(list(range(n_states)),[-2,-1])
    theta_a = theta_hat[a_idx]
    theta_tau = theta_hat[n_states:-2]
    #print(update_theta_j(theta0[0],x,Q,t_grids,tau,topo,params={"lambda_a":1e-4,"lambda_tau":1},restrictions=[[3],[2]]))
    print(check_grad(neglogL_a, neglogL_jac_a, theta_a, theta_tau, x_weighted, marginal_weight, t_grids, tau, topo, 1))
    approx_fprime(theta_hat, neglogL, 1e-6, x_weighted, marginal_weight, t_grids, tau, topo, 1)
    neglogL_a(theta_a, theta_tau,  x_weighted, marginal_weight, t_grids, tau, topo, 1)
    neglogL(theta_hat, x_weighted, marginal_weight, t_grids, tau, topo, 1)
    y_hat = get_y(theta_hat, t, tau)
    
    from two_species_ss import check_params, guess_theta, get_y, get_Y_hat, get_logL, update_theta_j, neglogL, neglogL_jac
    print(check_grad(neglogL, neglogL_jac, theta_a,x_weighted, marginal_weight, t_grids, tau, topo, 1))
    neglogL(theta_a, x_weighted, marginal_weight, t_grids, tau, topo, 1)
    
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    ax[0].plot(t,x[:,0],'k.')
    ax[0].plot(t,y[:,0],'r')
    ax[0].plot(t,y_hat[:,0],'b')
    
    ax[1].plot(t,x[:,1],'k.')
    ax[1].plot(t,y[:,1],'r')
    ax[1].plot(t,y_hat[:,1],'b')

    
#%%
   
    """
    #%% jac function debug
    def get_y_a_debug(theta, t, tau):
        # theta: a_1, ..., a_K, u_0, s_0, beta, gamma
        # t: len m
        # tau: len K+1
        # return m * p * 2
        K = len(tau)-1 # number of states
        a = theta[0:K]
        beta = theta[-2]
        gamma = theta[-1]
    
        y1_0 = theta[-4]
        y2_0 = theta[-3]
    
        c = beta/(beta-gamma+eps)
        d = beta**2/((beta-gamma)*gamma+eps)
        y_0 = y2_0 + c*y1_0
        a_ = d*a
        t = t
        m = len(t)
      
        I = np.ones((K+1,m),dtype=bool)
    
        # nascent
        y1=y1_0*np.exp(-t*beta)
        y=y_0*np.exp(-t*gamma)   
        for k in range(1,K+1):
            I[k] = np.squeeze(t > tau[k])
            idx =I[k-1]*(~I[k]) # tau_{k-1} < t_i <= tau_k
            y1 = y1 + a[k-1] * (np.exp(- (I[k] *(t-tau[k]))*beta)- np.exp(-(I[k]*(t-tau[k-1]))*beta )) \
              + a[k-1] * (1 - np.exp(- (idx *(t-tau[k-1]))*beta ) )
            y = y + a_[k-1] * (np.exp(-(I[k] * (t-tau[k]))*gamma)- np.exp(-(I[k] * (t-tau[k-1]))*gamma )) \
              +  a_[k-1] * (1 - np.exp(-(idx*(t-tau[k-1]))*gamma) )
    
        Y = np.zeros((m,2))
        Y[:,0] = y1
        Y[:,1] = y-c*y1
        
        Y[Y<0]=0
        
        if np.sum(np.isnan(Y)) != 0:
            raise ValueError("Nan in Y")
        return Y[:,1]
     
    #theta0_a[-2]=1
    #theta0_a[-1]=0.9
    t=np.array([0.01])
    print(approx_fprime(theta0_a, get_y_a_debug, 1e-1000,  t, tau))
    y, jac = get_y_a_jac(theta0_a, t, tau)
    print(jac[:,1])
    
    #Y, dY_dtheta = get_y_a_jac(theta_a[0],t,tau)
    
    #%% update_theta_j
    np.random.seed(42)
    topo = np.array([[0,]])
    tau=(0,1)
    n = 10000
    p = 1
    K=len(tau)-1
    t=np.linspace(0, 1, n)
    
    loga_max=4
    logb_max=2
    
    theta_a=np.ones((p,K+2))
    theta_a[0,:K+1]=np.exp(np.random.uniform(0,loga_max,size=K+1))-1
    theta_a[0,-1:]=np.exp(np.random.uniform(-logb_max,logb_max,size=1))
    
    theta_d = theta_a.copy() 
    theta_d[0,0] -= theta_d[0,-4]
    theta_d[0,1:K] -= theta_a[0,0:(K-1)]
    theta_d[0,-3] = theta_d[0,-3]*theta_d[0,-1]/theta_d[0,-2] - theta_d[0,-4]  # eta = zeta * s_0 - u_0
    
    Y = get_y_a(theta_a[0], t, tau)
    X = np.random.poisson(Y)
    x = X
    Q = np.diag(np.ones(n))/n
    Q = Q[:,None,:]

    theta0_a = np.zeros(K+4)
    theta0_a[0:-3]=np.mean(X[:,0])
    theta0_a[-3]=np.mean(X[:,1])
    theta0_a[-2]=1
    theta0_a[-1]=np.mean(X[:,0])/np.mean(X[:,1])
    theta0_a += np.random.uniform(0,1,size = theta0_a.shape)


    theta0_d = np.zeros(K+4)
    theta0_d[-4]=np.mean(X[:,0])
    theta0_d[-2]=1
    theta0_d[-1]=np.mean(X[:,0])/np.mean(X[:,1])
    
    res_d = update_theta_j_delta(theta0_d, X, Q, t, tau, topo)
    print(theta_d[0]-res_d)
    
    #%% check update_nested_theta_j_a(theta0, x, Q, t, tau, topo, restrictions)
    theta_a=np.zeros((p,K+4))
    theta_a[0,-4]=np.exp(np.random.uniform(0,loga_max,size=1))
    theta_a[0,0:K]=theta_a[0,-4]
    theta_a[0,-2:]=np.exp(np.random.uniform(-logb_max,logb_max,size=2))
    theta_a[0,-3] = theta_a[0,-4] * theta_a[0,-2]/theta_a[0,-1]
    Y = get_y_a(theta_a[0], t, tau)
    X = np.random.poisson(Y)
    res_a = update_theta_j(theta0_a, X, Q, t, tau, topo)   
    print(theta_a[0]-res_a)

    
    Y_fit_a = get_y_a(res_a, t, tau)

    plt.scatter(X[:,0],X[:,1],c=t);
    plt.scatter(Y_fit_a[:,0],Y_fit_a[:,1],c=t,cmap='Spectral');
    plt.scatter(Y[:,0],Y[:,1],c="red");
       
    restrictions = np.array([[0,-3],[-4,-4]])
    theta = update_nested_theta_j(theta0_a, X, Q, t, tau, topo, restrictions)    
    
    print(theta-theta_a[0])
        
    Y_fit = get_y_a(theta, t, tau)

    plt.scatter(X[:,0],X[:,1],c=t);
    plt.scatter(Y_fit[:,0],Y_fit[:,1],c=t,cmap='Spectral');
    plt.scatter(Y[:,0],Y[:,1],c="red");
    """
