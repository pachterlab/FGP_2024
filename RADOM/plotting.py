import numpy as np
import matplotlib.pyplot as plt

cmps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
colors20 = np.vstack((plt.cm.tab20b(np.linspace(0., 1, 20))[::2], plt.cm.tab20c(np.linspace(0, 1, 20))[1::2]))    

def plot_t(weight,l=0,ax=None,t=None):
    m=np.shape(weight)[-1]
    h=np.linspace(0,1,m)
    t_hat=np.sum(weight*h[None,None,:],axis=(-2,-1))
    if ax is None:
        fig, ax = plt.subplots(1,1)
    if t is not None:
        # ord=np.argsort(t)
        # build a rectangle in axes coords
        left, width = .25, .5
        bottom, height = .25, .5
        right = left + width
        top = bottom + height
       
        ax.imshow(weight[:,l,:],aspect="auto");
        ax.text(right, top,"cor="+str(np.around(np.corrcoef(t_hat,t)[0,1],2)), horizontalalignment='right', 
                 verticalalignment='top', transform=ax.transAxes, color="white");
    else:
        ord=np.argsort(t_hat)
        ax.imshow(weight[ord,l,:],aspect="auto");

        
def plot_theta(theta,theta_hat):
    K=np.shape(theta)[1]-4
    fig, ax = plt.subplots(1,K+4,figsize=(6*(K+4),4))

    for i in range(K):
        ax[i].plot(1+theta[:,i],1+theta[:,i]);
        ax[i].plot(1+theta[:,i],1+theta_hat[:,i],'.');
        ax[i].set_title("a"+str(i+1))
        ax[i].set_xlabel("true values + 1")
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')

    ax[-4].plot(1+theta[:,-4],1+theta[:,-4]);
    ax[-4].plot(1+theta[:,-4],1+theta_hat[:,-4],'.');
    ax[-4].set_title("u0")
    ax[-4].set_ylabel("fitted values + 1")
    ax[-4].set_xlabel("true values + 1")
    ax[-4].set_xscale('log')
    ax[-4].set_yscale('log')

    ax[-3].plot(1+theta[:,-3],1+theta[:,-3]);
    ax[-3].plot(1+theta[:,-3],1+theta_hat[:,-3],'.');
    ax[-3].set_title("s0")
    ax[-3].set_ylabel("fitted values + 1")
    ax[-3].set_xlabel("true values + 1")
    ax[-3].set_xscale('log')
    ax[-3].set_yscale('log')
    
    ax[-2].plot(theta[:,-2],theta[:,-2]);
    ax[-2].plot(theta[:,-2],theta_hat[:,-2],'.');
    ax[-2].set_title("log beta");
    ax[-2].set_xlabel("true values");
    ax[-2].set_xscale('log')
    ax[-2].set_yscale('log')
    

    ax[-1].plot(theta[:,-1],theta[:,-1]);
    ax[-1].plot(theta[:,-1],theta_hat[:,-1],'.');
    ax[-1].set_title("log gamma");
    ax[-1].set_xlabel("true values");
    ax[-1].set_xscale('log')
    ax[-1].set_yscale('log')
    


def plot_theta_hat(theta_hat,K,gene_list=None):
    if gene_list is None:
        gene_list = np.arange(len(theta_hat))
    fig, ax = plt.subplots(K+4,1,figsize=(12,4*(K+4)))

    for i in range(K+4):
        ax[i].scatter(np.array(gene_list), np.log(theta_hat[:,i]));

    ax[0].set_title("log a1");
    ax[1].set_title("log a2");
    ax[2].set_title("log u0");
    ax[3].set_title("log s0");
    ax[4].set_title("log beta");
    ax[5].set_title("log gamma");



def plot_theta_diff(theta_hat,K,gene_list=None):
    if gene_list is None:
        gene_list = np.arange(len(theta_hat))
    fig, ax = plt.subplots(K+4,1,figsize=(12,4*(K+4)))
    ax[0].scatter(np.array(gene_list), theta_hat[:,0]);
    ax[-3].scatter(np.array(gene_list), theta_hat[:,-3]);
    for i in range(1,K+1):
        ax[i].scatter(np.array(gene_list), theta_hat[:,i]-theta_hat[:,0]);
        
    for i in range(-2,0):
        ax[i].scatter(np.array(gene_list), theta_hat[:,i]);

    ax[0].set_title("a1");
    ax[1].set_title("a2 - a1");
    ax[2].set_title("u0 - a1");
    ax[3].set_title("s0");
    ax[4].set_title("beta");
    ax[5].set_title("gamma");

          
def plot_y(X,weight,traj,idx,gene_name=None,cell_colors=None):
    theta_hat = traj.theta.copy()
    tau=traj.tau
    n,L,m=np.shape(weight)
    p=np.shape(traj.theta)[0]
    h=np.linspace(0,tau[-1],m)
    t_hat=np.sum(weight[:,:,:]*h[None,None,:],axis=(1,2))
    if gene_name is None:
        gene_name = np.arange(p)
    if cell_colors is None:
        cell_colors = t_hat
    fig, ax = plt.subplots(2,len(idx),figsize=(6*len(idx),4*2))
    if len(idx)==1:
        i=idx[0]
        ax[0].scatter(t_hat,X[:,i,0],c=cell_colors);
        ax[0].set_title(gene_name[0]+" unspliced")

        ax[1].scatter(t_hat,X[:,i,1],c=cell_colors);
        ax[1].set_title(gene_name[0]+" spliced")

    else:
        for i,j in enumerate(idx):
            ax[0,i].scatter(t_hat,X[:,j,0],c=cell_colors);
            ax[0,i].set_title(gene_name[j])
            ax[1,i].scatter(t_hat,X[:,j,1],c=cell_colors);
            ax[1,i].set_title(gene_name[j])
    
    Y_hat = np.zeros((L,n,p,2))
    for l in range(L):
        t_hat=np.sum(weight[:,l,:]*h[None,:],axis=1)
        theta_l_hat = np.concatenate((theta_hat[:,traj.topo[l]], theta_hat[:,-4:]), axis=1)
        Y_hat[l] = traj.get_Y(theta_l_hat,t_hat,tau) # m*p*2
        y_hat = Y_hat[l]
        if len(idx)==1:
            i=idx[0]
            ax.scatter(y_hat[:,i,0],y_hat[:,i,1],c=t_hat,cmap=cmps[l]);
        else:
            for i,j in enumerate(idx):
                ax[0,i].plot(t_hat,y_hat[:,j,0],'.');
                ax[0,i].set_title(gene_list[j]+" unspliced")

                ax[1,i].plot(t_hat,y_hat[:,j,1],'.');
                ax[1,i].set_title(gene_list[j]+" spliced")
                          
            
def plot_phase(X,weight,traj,idx,gene_name=None,cell_colors=None):
    theta_hat = traj.theta.copy()
    tau=traj.tau
    n,L,m=np.shape(weight)
    p=np.shape(traj.theta)[0]
    h=np.linspace(0,tau[-1],m)
    t_hat=np.sum(weight[:,:,:]*h[None,None,:],axis=(1,2))
    if gene_name is None:
        gene_name = np.arange(p)
    if cell_colors is None:
        cell_colors = t_hat
      
    fig, ax = plt.subplots(1,len(idx),figsize=(6*len(idx),4))
    for l in range(L):
        t_hat=np.sum(weight[:,l,:]*h[None,:],axis=1)
        theta_l_hat = np.concatenate((theta_hat[:,traj.topo[l]], theta_hat[:,-4:]), axis=1)
        y_hat = traj.get_Y(theta_l_hat,t_hat,tau) # m*p*2
        if len(idx)==1:
            i=idx[0]
            ax.scatter(y_hat[:,i,0],y_hat[:,i,1],c=t_hat,cmap=cmps[l]);
            ax.scatter(X[:,i,0]+np.random.normal(0,0.1,n),X[:,i,1]+np.random.normal(0,0.1,n),c=cell_colors,s=0.5);
            ax.set_title(gene_name[0])
        else:
            for i,j in enumerate(idx):
                ax[i].scatter(y_hat[:,j,0],y_hat[:,j,1],c=t_hat,cmap=cmps[l]);
                ax[i].scatter(X[:,j,0]+np.random.normal(0,0.1,n),X[:,j,1]+np.random.normal(0,0.1,n),c=cell_colors,s=0.5);
                ax[i].set_title(gene_name[j])
        
                         