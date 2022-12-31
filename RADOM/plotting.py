import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
import matplotlib.colors as pltc

cmap_Q = cmr.get_sub_cmap('Greys', 0.1, 1)
#cmap_t = cmr.get_sub_cmap('YlGnBu', 0.2, 1)
cmap_t = cmr.get_sub_cmap('Blues', 0.2, 0.8)
cmap_ts = [cmr.get_sub_cmap('Blues', 0.2, 0.8),cmr.get_sub_cmap('Reds', 0.2, 0.8),cmr.get_sub_cmap('Purples', 0.2, 0.8),cmr.get_sub_cmap('Greens', 0.2, 0.8)]
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

def plot_t(traj,l=0,ax=None,t=None,order_cells=False):
    t_hat=traj.Q[:,l]@traj.t
    if ax is None:
        fig, ax = plt.subplots(1,1)
    if t is not None:
        if order_cells:
            order=np.argsort(t)
        else:
            order=np.arange(len(t))
        im = ax.imshow(traj.Q[order,l,:],aspect="auto",cmap=cmap_Q);
        ax.text(0.9, 0.9, "CCC="+str(np.around(CCC(t_hat,t))), horizontalalignment='right', 
                 verticalalignment='top', transform=ax.transAxes, color="black",fontsize=24);
    else:
        if order_cells:
            order=np.argsort(t_hat)
        else:
            order=np.arange(len(t_hat))
        im = ax.imshow(traj.Q[order,l,:],aspect="auto",cmap=cmap_Q);
        
    plt.colorbar(im) # adding the colobar on the right
    return ax

def plot_theta_ss(theta,theta_hat,dot_color='grey'):
    K=np.shape(theta)[1]-3
    fig, ax = plt.subplots(1,K+3,figsize=(4*(K+3),3.5))

    for i in range(K):
        ax[i+1].plot(theta[:,i],theta[:,i],color='black');
        ax[i+1].plot(theta[:,i],theta_hat[:,i],'.',color=dot_color);
        ax[i+1].text(0.9, 0.2, "CCC="+str(np.around(CCC(theta_hat[:,i],theta[:,i]))), horizontalalignment='right', 
                 verticalalignment='top', transform=ax.transAxes, color="black",fontsize=24);
        ax[i+1].set_title("α"+str(i+1), fontsize = label_font)
        ax[i+1].set_xscale('log')
        ax[i+1].set_yscale('log')

    ax[0].plot(theta[:,-3],theta[:,-3],color='black');
    ax[0].plot(theta[:,-3],theta_hat[:,-3],'.',color=dot_color);
    ax[0].text(0.9, 0.2, "CCC="+str(np.around(CCC(theta_hat[:,-3],theta[:,-3]))), horizontalalignment='right', 
                 verticalalignment='top', transform=ax.transAxes, color="black",fontsize=24);
    ax[0].set_title("u0", fontsize = label_font)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    
    ax[-2].plot(theta[:,-2],theta[:,-2],color='black');
    ax[-2].plot(theta[:,-2],theta_hat[:,-2],'.',color=dot_color);
    ax[-2].text(0.9, 0.2, "CCC="+str(np.around(CCC(theta_hat[:,-2],theta[:,-2]))), horizontalalignment='right', 
                 verticalalignment='top', transform=ax.transAxes, color="black",fontsize=24);
    ax[-2].set_title("β", fontsize = label_font);
    ax[-2].set_xscale('log')
    ax[-2].set_yscale('log')
    

    ax[-1].plot(theta[:,-1],theta[:,-1],color='black');
    ax[-1].plot(theta[:,-1],theta_hat[:,-1],'.',color=dot_color);
    ax[-1].text(0.9, 0.2, "CCC="+str(np.around(CCC(theta_hat[:,-1],theta[:,-1]))), horizontalalignment='right', 
                 verticalalignment='top', transform=ax.transAxes, color="black",fontsize=24);
    ax[-1].set_title("γ",fontsize = label_font);
    ax[-1].set_xscale('log')
    ax[-1].set_yscale('log')
    
    #fig.supxlabel("true values", fontsize = label_font);
    #fig.supylabel("estimates", fontsize = label_font)
    plt.tight_layout()
    
    return fig
    
def plot_theta(theta,theta_hat):
    K=np.shape(theta)[1]-4
    fig, ax = plt.subplots(1,K+4,figsize=(6*(K+4),4))

    for i in range(K):
        ax[i+2].plot(1+theta[:,i],1+theta[:,i],color='black');
        ax[i+2].plot(1+theta[:,i],1+theta_hat[:,i],'.',color='tab:red');
        ax[i+2].text(0.9, 0.2, "CCC="+str(np.around(CCC(theta_hat[:,i],theta[:,i]))), horizontalalignment='right', 
                 verticalalignment='top', transform=ax.transAxes, color="black",fontsize=24);
        ax[i+2].set_title("a"+str(i+1))
        ax[i+2].set_xlabel("true values + 1")
        ax[i+2].set_xscale('log')
        ax[i+2].set_yscale('log')

    ax[0].plot(1+theta[:,-4],1+theta[:,-4],color='black');
    ax[0].plot(1+theta[:,-4],1+theta_hat[:,-4],'.',color='tab:red');
    ax[0].text(0.9, 0.2, "CCC="+str(np.around(CCC(theta_hat[:,-4],theta[:,-4]))), horizontalalignment='right', 
                 verticalalignment='top', transform=ax.transAxes, color="black",fontsize=24);
    ax[0].set_title("u0")
    ax[0].set_ylabel("fitted values + 1")
    ax[0].set_xlabel("true values + 1")
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')

    ax[1].plot(1+theta[:,-3],1+theta[:,-3],color='black');
    ax[1].plot(1+theta[:,-3],1+theta_hat[:,-3],'.',color='tab:red');
    ax[1].text(0.9, 0.2, "CCC="+str(np.around(CCC(theta_hat[:,-3],theta[:,-3]))), horizontalalignment='right', 
                 verticalalignment='top', transform=ax.transAxes, color="black",fontsize=24);
    ax[1].set_title("s0")
    ax[1].set_ylabel("fitted values + 1")
    ax[1].set_xlabel("true values + 1")
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    
    ax[-2].plot(theta[:,-2],theta[:,-2],color='black');
    ax[-2].plot(theta[:,-2],theta_hat[:,-2],'.',color='tab:red');
    ax[-2].text(0.9, 0.2, "CCC="+str(np.around(CCC(theta_hat[:,-2],theta[:,-2]))), horizontalalignment='right', 
                 verticalalignment='top', transform=ax.transAxes, color="black",fontsize=24);
    ax[-2].set_title("log beta");
    ax[-2].set_xlabel("true values");
    ax[-2].set_xscale('log')
    ax[-2].set_yscale('log')
    

    ax[-1].plot(theta[:,-1],theta[:,-1],color='black');
    ax[-1].plot(theta[:,-1],theta_hat[:,-1],'.',color='tab:red');
    ax[-1].text(0.9, 0.2, "CCC="+str(np.around(CCC(theta_hat[:,-1],theta[:,-1]))), horizontalalignment='right', 
                 verticalalignment='top', transform=ax.transAxes, color="black",fontsize=24);
    ax[-1].set_title("log gamma");
    ax[-1].set_xlabel("true values");
    ax[-1].set_xscale('log')
    ax[-1].set_yscale('log')
  
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

          
def plot_y(traj,idx=np.arange(10),gene_name=None,cell_colors=None):
    X,weight = traj.X, traj.Q
    n,p,c=X.shape
    theta_hat = traj.theta.copy()
    tau=traj.tau
    n,L,m=np.shape(weight)
    h=np.linspace(tau[0],tau[-1],m)
    t_hat=np.sum(weight[:,:,:]*h[None,None,:],axis=(1,2))
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
    Y_hat = np.zeros((L,10000,p,2))
    for l in range(L):
        t=np.linspace(tau[0],tau[-1],10000)
        if traj.model == "two_species":
            theta_l_hat = np.concatenate((theta_hat[:,traj.topo[l]], theta_hat[:,-4:]), axis=1)
        elif traj.model == "two_species_ss":
            theta_l_hat = np.concatenate((theta_hat[:,traj.topo[l]], theta_hat[:,-3:]), axis=1)
        else:
            print("update plot_phase to include", traj.model)
        Y_hat[l] = traj.get_Y(theta_l_hat,t,tau) # m*p*2
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

def plot_phase(traj,idx=np.arange(10),gene_name=None,cell_colors=None):
    X,weight = traj.X, traj.Q
    theta_hat = traj.theta.copy()
    tau=traj.tau
    n,L,m=np.shape(weight)
    p=np.shape(traj.theta)[0]
    h=traj.t
    t_hat=np.sum(weight[:,:,:]*h[None,None,:],axis=(1,2))
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
        
                         
    for l in range(L):
        t=np.linspace(traj.tau[0],traj.tau[-1],100*traj.m)
        if traj.model == "two_species":
            theta_l_hat = np.concatenate((theta_hat[:,traj.topo[l]], theta_hat[:,-4:]), axis=1)
        elif traj.model == "two_species_ss":
            theta_l_hat = np.concatenate((theta_hat[:,traj.topo[l]], theta_hat[:,-3:]), axis=1)
        else:
            print("update plot_phase to include", traj.model)
        y_hat = traj.get_Y(theta_l_hat,t,tau) # m*p*2
        if len(idx)==1:
            i=idx[0]
            ax.scatter(y_hat[:,i,0],y_hat[:,i,1],c=t,cmap=cmap_ts[l]);
            
        else:
            for i,j in enumerate(idx):
                ax[i].scatter(y_hat[:,j,0],y_hat[:,j,1],c=t,cmap=cmap_ts[l]);
                         