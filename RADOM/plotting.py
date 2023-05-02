import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.neighbors import NearestNeighbors
#import cmasher as cmr

cmap_Q = 'Greys'
cmap_t = 'Blues'
cmap_ts = ['Blues', 'Reds', 'Purples', 'Greens']

#cmap_Q = cmr.get_sub_cmap('Greys', 0.2, 1)
#cmap_t = cmr.get_sub_cmap('Blues', 0.2, 1)
#cmap_ts = [cmr.get_sub_cmap('Blues', 0.2, 1), cmr.get_sub_cmap('Reds', 0.2, 1),cmr.get_sub_cmap('Purples', 0.2, 1), cmr.get_sub_cmap('Greens', 0.2, 1)]
label_font =24
      
def CCC(y_pred, y_true):
    # Pearson product-moment correlation coefficients
    cor = np.corrcoef(y_true, y_pred)[0][1]
    
    # Mean
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    # Variance
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    
    # Standard deviation
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    
    return numerator / denominator

def getNeighbors(embed, n_neigh=100, p=1):
    """Get indices of nearest neighbors in embedding (shape n_samples,n_features)"""
    nbrs = NearestNeighbors(n_neighbors=n_neigh, p=p).fit(embed.reshape((-1,1)))
    distances, indices = nbrs.kneighbors(embed)
    return indices

def getJaccard(x1, x2, n_neigh=100):
    '''
    Get jaccard distance between embeddings
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    embed1_neighbor = getNeighbors(x1.reshape(-1, 1),n_neigh)
    embed2_neighbor = getNeighbors(x2.reshape(-1, 1),n_neigh)
    frac = [0]*embed1_neighbor.shape[0]
    for i in range(embed1_neighbor.shape[0]):
        inter = set(embed1_neighbor[i,:]).intersection(embed2_neighbor[i,:])
        union = set(embed1_neighbor[i,:]).union(embed2_neighbor[i,:])
        frac[i] = 1-len(inter)/len(union)
    return frac

def plot_t(traj,Q=None,l=0,ax=None,t=None,order_cells=False):
    if Q is None:
        Q = traj.Q
    t_hat=Q[:,l]@traj.t
    if ax is None:
        fig, ax = plt.subplots(1,1)
    if t is not None:
        if order_cells:
            order=np.argsort(t)
        else:
            order=np.arange(len(t))
        im = ax.imshow(Q[order,l,:],aspect="auto",cmap=cmap_Q);
        ax.text(0.9, 0.9, "CCC="+str(np.around(CCC(t_hat,t),3)), horizontalalignment='right', 
                 verticalalignment='top', transform=ax.transAxes, color="black",fontsize=24);
    else:
        if order_cells:
            order=np.argsort(t_hat)
        else:
            order=np.arange(len(t_hat))
        im = ax.imshow(Q[order,l,:],aspect="auto",cmap=cmap_Q);
        
    #plt.colorbar(im) # adding the colobar on the right
    return ax

def plot_theta(theta,theta_hat,dot_color='grey'):
    n_theta = theta.shape[1]
    assert theta_hat.shape[1] == n_theta
    fig, ax = plt.subplots(1,n_theta,figsize=(6*n_theta,5))

    for i in range(n_theta):
        ax[i].plot(theta[:,i],theta[:,i],color='black');
        ax[i].plot(theta[:,i],theta_hat[:,i],'.',color='tab:red');
        ax[i].text(0.9, 0.2, "CCC="+str(np.around(CCC(theta_hat[:,i],theta[:,i]),3)), horizontalalignment='right', 
                 verticalalignment='top', transform=ax[i].transAxes, color="black",fontsize=24);
        ax[i].set_title("a"+str(i+1))
        ax[i].set_xlabel("true values")
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')

    
    ax[-2].set_title("beta");
    ax[-1].set_title("gamma");
    
    plt.tight_layout()
    
    #fig.supxlabel("true values", fontsize = label_font);
    #fig.supylabel("estimates", fontsize = label_font)
    plt.tight_layout()
    
    return fig,ax
    
def plot_theta_1(theta,theta_hat):
    n_theta = theta.shape[1]
    assert theta_hat.shape[1] == n_theta
    fig, ax = plt.subplots(1,n_theta,figsize=(6*n_theta,5))

    for i in range(n_theta):
        ax[i].plot(1+theta[:,i],1+theta[:,i],color='black');
        ax[i].plot(1+theta[:,i],1+theta_hat[:,i],'.',color='tab:red');
        ax[i].text(0.9, 0.2, "CCC="+str(np.around(CCC(theta_hat[:,i],theta[:,i]),3)), horizontalalignment='right', 
                 verticalalignment='top', transform=ax[i].transAxes, color="black",fontsize=24);
        ax[i].set_title("a"+str(i+1))
        ax[i].set_xlabel("true values + 1")
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')

    
    ax[-2].set_title("beta");
    ax[-1].set_title("gamma");
    
    plt.tight_layout()
    

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

          
def plot_y(traj,X=None,idx=np.arange(10),gene_name=None,cell_colors=None):
    if X is None:
        X = traj.X
    weight = traj.Q
        
    n,p,c=X.shape
    theta_hat = traj.theta.copy()
    tau=traj.tau
    n,L,m=np.shape(weight)
    t_hat=np.sum(weight[:,:,:]*traj.t[None,None,:],axis=(1,2))
    t = np.linspace(tau[0], tau[-1],1001)
    
    if gene_name is None:
        gene_name = np.arange(p)
    if cell_colors is None:
        cell_colors = 'grey'
        
    fig, ax = plt.subplots(c,len(idx),figsize=(6*len(idx),4*c))
    
    ## plot X and set title
    if len(idx)==1:
        if c==1:
            i=idx[0]
            ax.scatter(t_hat,X[:,i,0]+np.random.normal(0,0.1,n),c=cell_colors,s=10);
            ax.set_title(gene_name[i])
        else:
            i=idx[0]
            ax[0].scatter(t_hat,X[:,i,0]+np.random.normal(0,0.1,n),c=cell_colors,s=10);
            ax[0].set_title(gene_name[i]+" unspliced")
            for ic in range(1,c):
                ax[ic].scatter(t_hat,X[:,i,1]+np.random.normal(0,0.1,n),c=cell_colors,s=10);
                ax[ic].set_title(gene_name[i]+" spliced")

    else:
        for i,j in enumerate(idx):
            if c==1:
                ax[i].scatter(t_hat,X[:,j,0]+np.random.normal(0,0.1,n),c=cell_colors,s=10);
                ax[i].set_title(gene_name[j])
            else:
                ax[0,i].scatter(t_hat,X[:,j,0]+np.random.normal(0,0.1,n),c=cell_colors,s=10);
                ax[0,i].set_title(gene_name[j])
                for ic in range(1,c):
                    ax[ic,i].scatter(t_hat,X[:,j,ic]+np.random.normal(0,0.1,n),c=cell_colors,s=10);
    
    ## plot Y
    Y_hat = traj.get_Y_hat(theta_hat,t,tau,traj.topo,traj.params) #(L,len(t),p,2)                                                                 
       
    for l in range(L):                                                                                                        
        y_hat = Y_hat[l]
        if len(idx)==1:
            i=idx[0]
            if c==1:
                ax.scatter(t,y_hat[:,i,0],c=t,cmap=cmap_ts[l],alpha=0.5)
            else:
                for ic in range(c):
                    ax[ic].scatter(t,y_hat[:,i,ic],c=t,cmap=cmap_ts[l],alpha=0.5);
        else:
            for i,j in enumerate(idx):
                if c==1:
                    ax[i].scatter(t,y_hat[:,j,0],c=t,cmap=cmap_ts[l],alpha=0.5);

                else:
                    for ic in range(c):
                        ax[ic,i].scatter(t,y_hat[:,j,ic],c=t,cmap=cmap_ts[l],alpha=0.5);

def plot_phase(traj,X=None,idx=np.arange(10),gene_name=None,cell_colors=None):
    if X is None:
        X = traj.X
    weight = traj.Q
    theta_hat = traj.theta.copy()
    tau=traj.tau
    n,L,m=np.shape(weight)
    p=np.shape(traj.theta)[0]
    #t_hat=np.sum(weight[:,:,:]*traj.t[None,None,:],axis=(1,2))
    t = np.linspace(tau[0], tau[-1],1001)
    if gene_name is None:
        gene_name = np.arange(p)
    if cell_colors is None:
        cell_colors = 'grey'
      
    fig, ax = plt.subplots(1,len(idx),figsize=(6*len(idx),4))
    if len(idx)==1:
        i=idx[0]
        ax.scatter(X[:,i,0]+np.random.normal(0,0.1,n),X[:,i,1]+np.random.normal(0,0.1,n),c=cell_colors,s=10);
        ax.set_title(gene_name[i])
    else:
        for i,j in enumerate(idx):
            ax[i].scatter(X[:,j,0]+np.random.normal(0,0.1,n),X[:,j,1]+np.random.normal(0,0.1,n),c=cell_colors,s=10);
            ax[i].set_title(gene_name[j])
        
    Y_hat = traj.get_Y_hat(theta_hat,t,tau,traj.topo,traj.params) #(L,len(t),p,2)                                                                 
    for l in range(L):                                                               
        y_hat = Y_hat[l]
        if len(idx)==1:
            i=idx[0]
            ax.scatter(y_hat[:,i,0],y_hat[:,i,1],c=t,cmap=cmap_ts[l]);
            
        else:
            for i,j in enumerate(idx):
                ax[i].scatter(y_hat[:,j,0],y_hat[:,j,1],c=t,cmap=cmap_ts[l]);
    return fig, ax