import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score, precision_recall_curve, mean_squared_error
import cmasher as cmr
#from .utils import RE, CCC, AE

plt.rcParams['font.size'] = '24'
plt.rcParams['lines.linewidth'] = 12
plt.rcParams['lines.markersize'] = 12
plt.rcParams['figure.constrained_layout.use'] = True
label_font = '30'
legend_font = '24'
title_font = '36'

cmap_Q = cmr.get_sub_cmap('Greys', 0, 1)
cmap_t = cmr.get_sub_cmap('Greys', 0, 1)
cmap_ts = [cmr.get_sub_cmap('Blues', 0.2, 1), cmr.get_sub_cmap('Reds', 0.2, 1),cmr.get_sub_cmap('Purples', 0.2, 1), cmr.get_sub_cmap('Greens', 0.2, 1)]

def AE(y_pred, y_true):
    return np.abs(y_pred - y_true).mean()
    
def RE(y_pred, y_true, scale=0.5):
    return (np.abs(y_pred - y_true)/np.power(y_true,scale)).mean()

def R2(y_pred, y_true):
    # R squared
    SS_Res = np.sum((y_pred - y_true)**2)
    SS_Total = np.sum((y_pred - y_pred.mean())**2)
    
    return 1 - SS_Res / SS_Total



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


def plot_t(traj,Q=None,l=None,ax=None,t=None,order_cells=False,measure="AE"):
    if Q is None:
        Q = traj.Q
    if l is None:
        Q_ = Q.sum(1)
    else:
        Q_ = Q[:,l]
        
    t_hat=Q_@traj.t
    if ax is None:
        fig, ax = plt.subplots(1,1)
    if t is not None:
        if order_cells:
            order=np.argsort(t)
        else:
            order=np.arange(len(t))
        im = ax.imshow(Q_[order],aspect="auto",cmap=cmap_Q);
        ax.set_xticks(traj.tau/traj.tau[-1]*traj.m,labels=np.around(traj.tau,1))
        ax.text(0.9, 0.9, "RMSE="+str(np.around(np.sqrt(mean_squared_error(y_pred=t_hat,y_true=t)),3)), horizontalalignment='right', 
                 verticalalignment='top', transform=ax.transAxes, color="black",fontsize=24);
    else:
        if order_cells:
            order=np.argsort(t_hat)
        else:
            order=np.arange(len(t_hat))
        im = ax.imshow(Q_[order],aspect="auto",cmap=cmap_Q);
        ax.set_xticks(traj.tau/traj.tau[-1]*traj.m,labels=np.around(traj.tau,1))
        
    #plt.colorbar(im) # adding the colobar on the right
    return ax

def plot_theta(theta,theta_hat,dot_color='lightgray'):
    n_theta = theta.shape[1]
    assert theta_hat.shape[1] == n_theta
    fig, ax = plt.subplots(1,n_theta,figsize=(6*n_theta,5),constrained_layout=True)

    for i in range(n_theta):
        ax[i].plot(theta[:,i],theta[:,i],color='black');
        ax[i].plot(theta[:,i],theta_hat[:,i],'.',color='tab:red');
        ax[i].text(0.9, 0.2, "Error="+str(np.around(RE(theta_hat[:,i],theta[:,i]),3)), horizontalalignment='right', 
                 verticalalignment='top', transform=ax[i].transAxes, color="black",fontsize=24);
        ax[i].set_title("a"+str(i))
        ax[i].set_xlabel("true values")
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')

    
    ax[-2].set_title("beta");
    ax[-1].set_title("gamma");
    
    #fig.supxlabel("true values", fontsize = label_font);
    #fig.supylabel("estimates", fontsize = label_font)
    
    return fig,ax

          
def plot_y(traj,X=None,idx=np.arange(10),l=0,gene_name=None,cell_colors=None):
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
        cell_colors = 'lightgray'
        
    fig, ax = plt.subplots(c,len(idx),figsize=(6.4*len(idx),4.8*c),constrained_layout=True)
    
    ## plot X and set title
    if len(idx)==1:
        if c==1:
            i=idx[0]
            ax.scatter(t_hat,X[:,i,0]+np.random.normal(0,0.1,n),c=cell_colors,s=10);
            ax.set_title(gene_name[i],fontsize=title_font)
        else:
            i=idx[0]
            ax[0].scatter(t_hat,X[:,i,0]+np.random.normal(0,0.1,n),c=cell_colors,s=10);
            ax[0].set_title(gene_name[i]+" unspliced",fontsize=title_font)
            for ic in range(1,c):
                ax[ic].scatter(t_hat,X[:,i,1]+np.random.normal(0,0.1,n),c=cell_colors,s=10);
                ax[ic].set_title(gene_name[i]+" spliced",fontsize=title_font)

    else:
        for i,j in enumerate(idx):
            if c==1:
                ax[i].scatter(t_hat,X[:,j,0]+np.random.normal(0,0.1,n),c=cell_colors,s=10);
                ax[i].set_title(gene_name[j],fontsize=title_font)
            else:
                ax[0,i].scatter(t_hat,X[:,j,0]+np.random.normal(0,0.1,n),c=cell_colors,s=10);
                ax[0,i].set_title(gene_name[j],fontsize=title_font)
                for ic in range(1,c):
                    ax[ic,i].scatter(t_hat,X[:,j,ic]+np.random.normal(0,0.1,n),c=cell_colors,s=10);
    
    ## plot Y
    Y_hat = traj.get_Y_hat(theta_hat,t,tau,traj.topo,traj.params) #(L,len(t),p,2)                                                                                  
    if 'r' in traj.params:
        y_hat = Y_hat[l]*traj.params['r'].mean()
    else:
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

def plot_phase(traj,X=None,idx=np.arange(10),l=0,gene_name=None,cell_colors=None):
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
        cell_colors = 'lightgrey'
      
    fig, ax = plt.subplots(1,len(idx),figsize=(6.4*len(idx),4.8),constrained_layout=True)
    if len(idx)==1:
        i=idx[0]
        ax.scatter(X[:,i,0]+np.random.normal(0,0.1,n),X[:,i,1]+np.random.normal(0,0.1,n),c=cell_colors,s=10);
        ax.set_title(gene_name[i],fontsize=title_font)
    else:
        for i,j in enumerate(idx):
            ax[i].scatter(X[:,j,0]+np.random.normal(0,0.1,n),X[:,j,1]+np.random.normal(0,0.1,n),c=cell_colors,s=10);
            ax[i].set_title(gene_name[j],fontsize=title_font)
        
    Y_hat = traj.get_Y_hat(theta_hat,t,tau,traj.topo,traj.params) #(L,len(t),p,2)                                                                                                                            
    if 'r' in traj.params:
        y_hat = Y_hat[l]*traj.params['r'].mean()
    else:
        y_hat = Y_hat[l]
        
    if len(idx)==1:
        i=idx[0]
        ax.scatter(y_hat[:,i,0],y_hat[:,i,1],c=t,cmap=cmap_ts[l]);
        
    else:
        for i,j in enumerate(idx):
            ax[i].scatter(y_hat[:,j,0],y_hat[:,j,1],c=t,cmap=cmap_ts[l]);
    return fig, ax