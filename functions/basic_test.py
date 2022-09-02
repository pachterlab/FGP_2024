

#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.optimize import check_grad, approx_fprime
    """
    np.random.seed(42)
    topo = np.array([[0,]])
    tau=(0,1)
    n = 10000
    p = 1
    K=len(tau)-1
    t=np.linspace(0, 1, n)
    
    loga_max=4
    logb_max=2
    
    theta_a=np.ones((p,K+4))
    theta_a[0,:K+2]=np.exp(np.random.uniform(0,loga_max,size=K+2))-1
    theta_a[0,-2:]=np.exp(np.random.uniform(-logb_max,logb_max,size=2))
    
    theta_d = theta_a.copy() 
    theta_d[0,0] -= theta_d[0,-4]
    theta_d[0,1:K] -= theta_a[0,0:(K-1)]
    theta_d[0,-3] = theta_d[0,-3]*theta_d[0,-1]/theta_d[0,-2] - theta_d[0,-4]  # eta = zeta * s_0 - u_0
    
    Y_a = get_y_a(theta_a[0], t, tau)
    Y_d = get_y_delta(theta_d[0], t, tau)


    print(np.max(Y_a-Y_d))
    #print(np.max(Y-y))

#%% check jac function
    np.random.seed(42)
    topo = np.array([[0,]])
    tau=(0,1)
    n = 3
    p = 1
    K=len(tau)-1
    t=np.linspace(0, 1, n)
    
    loga_max=4
    logb_max=2
    
    theta_a=np.ones((p,K+4))
    theta_a[0,:K+2]=np.exp(np.random.uniform(0,loga_max,size=K+2))-1
    theta_a[0,-2:]=np.exp(np.random.uniform(-logb_max,logb_max,size=2))
    
    Y_a = get_y_a(theta_a[0], t, tau)

    Y = Y_a
    X = np.random.poisson(Y)
    x = X
    Q = np.diag(np.ones(n))/n
    Q = Q[:,None,:]
    n,L,m = Q.shape
    x_weighted = np.zeros((L,m,2))
    marginal_weight = np.zeros((L,m,1))
    for l in range(len(topo)):
        weight_l = Q[:,l,:] #n*m
        x_weighted[l] = weight_l.T@x # m*2
        marginal_weight[l] =  weight_l.sum(axis=0)[:,None] # m*1
        
    
    theta0_a = np.zeros(K+4)
    theta0_a[0:-3]=np.mean(X[:,0])
    theta0_a[-3]=np.mean(X[:,1])
    theta0_a[-2]=1
    theta0_a[-1]=np.mean(X[:,0])/np.mean(X[:,1])
    theta0_a += np.random.uniform(0,1,size = theta0_a.shape)
    
    #check_grad(neglogL_a, neglogL_a_jac, theta0_a, x_weighted, marginal_weight, t, tau, topo)
    #approx_fprime(theta_a[0], neglogL_a, 1e-100, x_weighted, marginal_weight, t, tau, topo)
    #neglogL_a_jac(theta_a[0], x_weighted, marginal_weight, t, tau, topo)
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
    """
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
    
    theta_a=np.ones((p,K+4))
    theta_a[0,:K+2]=np.exp(np.random.uniform(0,loga_max,size=K+2))-1
    theta_a[0,-2:]=np.exp(np.random.uniform(-logb_max,logb_max,size=2))
    
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
    """
    Y_fit_delta = get_y_delta(res_d.x, t, tau)
    
    plt.scatter(X[:,0],X[:,1],c='gray');
    plt.scatter(Y_fit_delta[:,0],Y_fit_delta[:,1],c=t,cmap='Spectral');
    plt.scatter(Y[:,0],Y[:,1],c="red");
    """
    res_a = update_theta_j(theta0_a, X, Q, t, tau, topo)   
    res_a_nojac = update_theta_j_a_nojac(theta0_a, X, Q, t, tau, topo)   
    print(theta_a[0]-res_a)
    print(theta_a[0]-res_a_nojac)
    

    
    Y_fit_a = get_y_a(res_a.x, t, tau)

    plt.scatter(X[:,0],X[:,1],c='gray');
    plt.scatter(Y_fit_a[:,0],Y_fit_a[:,1],c=t,cmap='Spectral');
    plt.scatter(Y[:,0],Y[:,1],c="red");

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
