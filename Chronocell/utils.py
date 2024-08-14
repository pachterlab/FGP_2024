import numpy as np
from scipy.special import gammaln
from scipy.stats import chi2, poisson, nbinom, t
import pandas as pd
import matplotlib.pyplot as plt
import anndata

eps = 1e-6

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

def linear_regression(Theta):
    k,p,s = Theta.shape
    gene_idx = (Theta[:,:,0].var(0)>0) & (Theta[:,:,1].var(0)>0)
    U = Theta[:,gene_idx,0]
    S = Theta[:,gene_idx,1]
    
    U_mean = U.mean(axis=0)
    S_mean = S.mean(axis=0)
    U_var = U.var(axis=0)*k
    S_var = S.var(axis=0)*k
    beta = np.sum((U-U_mean[None,:])*(S-S_mean[None,:]),axis=0)/np.sum((U-U_mean[None,:])**2,axis=0)
    alpha = S_mean - beta * U_mean
    MSE = np.sum((S - beta[None,:]*U - alpha[None,:])**2,axis=0)/(k-2)
    MSE += 1e-10 # to avoid MSE == 0
    corr = beta*np.sqrt(np.sum((U-U_mean[None,:])**2,axis=0))/np.sqrt(np.sum((S-S_mean[None,:])**2,axis=0))
    s_alpha = np.sqrt(MSE*(1/k+S_mean**2/S_var))
    pval = 2*(1-t.cdf(np.abs(alpha/s_alpha),df=k-2))
    return beta, alpha, corr, pval

def calculate_entropy(X):
    """
    Shannon entropy of the empirical distributions for each gene
    """
    n,p,s = X.shape
    entropy = np.zeros(p)
    DF = np.zeros(p)
    for j in range(p):
        x=X[:,j]
        value, counts = np.unique(x, return_counts=True)
        norm_counts = counts / counts.sum()
        entropy[j] -= (norm_counts * np.log(norm_counts)).sum()
        DF[j] = len(value)
    return entropy, DF


def select_genes_by_G(adata,model=None,n_genes=30,params={},verbose=True):
    """
    Perform G test of Poisson fits and select n_genes with highest G-test statistic
    """
    n = adata.n_obs
    U_mean = adata.layers["unspliced"].toarray().mean(0)
    S_mean = adata.layers["spliced"].toarray().mean(0)
    idx = (U_mean > 0) & (U_mean > 0) # & ( np.abs(np.log10(U_mean/S_mean)) < 2 )
    X=np.zeros((n,idx.sum(),2))
    X[:,:,0]=adata.layers["unspliced"][:,idx].toarray()
    X[:,:,1]=adata.layers["spliced"][:,idx].toarray()
    Y = np.mean(X,axis=0,keepdims=True)
    
    if model == "NB":
        r = adata.layers["unspliced"].toarray().sum(1) + adata.layers["spliced"].toarray().sum(1)
        s = (r.var()-r.mean())/r.mean()**2
        if verbose:
            print("phi="+str(s))
        sigma2 = Y + s * Y**2
        p = Y / sigma2
        n = 1 / s
        logL = np.sum(nbinom.logpmf(k=X,n=n,p=p),axis=(0,2))
        
    elif model == "adjusted Poisson":
        r = adata.layers["unspliced"].toarray().sum(1) + adata.layers["spliced"].toarray().sum(1)
        if "read_depth" in params:
            if np.shape(params["read_depth"])==(n,):
                r = params["read_depth"].copy()
        r /= r.mean()
        logL = np.sum(poisson.logpmf(k=X,mu=r[:,None,None]*Y),axis=(0,2))
        
    elif model == "Poisson":
        logL = np.sum(poisson.logpmf(k=X,mu=Y),axis=(0,2))
        
    else:
        print(model + " model not implemented")
        return
        
    #logL = np.sum(X*np.log(r[:,None,None]*Y)-r[:,None,None]*Y-gammaln(X+1),axis=(0,2))
    entropy, DF = calculate_entropy(X)
    negG = logL + n*entropy
    p_value = 1-chi2.cdf(-2*negG,df=DF-2)
    
    ## return data X and gene_names
    selected_genes = adata.var_names[idx][np.argsort(negG)[:n_genes]]
    selected_idx = [np.where(adata.var_names == gene)[0][0] for  gene in selected_genes]
    X=np.zeros((n,len(selected_idx),2))
    X[:,:,0]=adata.layers["unspliced"][:,selected_idx].toarray()
    X[:,:,1]=adata.layers["spliced"][:,selected_idx].toarray()
    
    adata.var['e logL'] = np.nan
    adata.var['Poisson logL'] = np.nan
    adata.var['p value'] = np.nan
    adata.var['e logL'][idx] = -n*entropy
    adata.var['Poisson logL'][idx] = logL
    adata.var['p value'][idx] = p_value
    
    return X, selected_genes
