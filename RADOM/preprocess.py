import numpy as np
from scipy.special import gammaln
from scipy.stats import chi2, poisson
import pandas as pd
import matplotlib.pyplot as plt
import anndata
pd.options.mode.chained_assignment = None  # default='warn'

def entropy(X):
    """
    Shannon entropy of the empirical distributions for each gene
    """
    n,p,s = X.shape
    entropy = np.zeros(p)
    for j in range(p):
        for k in range(s):
            x=X[:,j,k]
            value,counts = np.unique(x, return_counts=True)
            norm_counts = counts / counts.sum()
            entropy[j] -= (norm_counts * np.log(norm_counts)).sum()
    return entropy


def select_genes(adata,r=None,n_genes=30):
    """
    Perform G test of Poisson fits and select n_genes with highest G-test statistic
    """
    n = adata.n_obs
    idx = (adata.layers["unspliced"].toarray().mean(0) > 0) & (adata.layers["spliced"].toarray().mean(0) > 0)
    X=np.zeros((n,idx.sum(),2))
    X[:,:,0]=adata.layers["unspliced"][:,idx].toarray()
    X[:,:,1]=adata.layers["spliced"][:,idx].toarray()
    Y = np.mean(X,axis=0,keepdims=True)
    if r is None:
        logL = np.sum(poisson.logpmf(k=X,mu=Y),axis=(0,2))
    elif np.shape(r)==(n,):
        r /= r.mean()
        logL = np.sum(poisson.logpmf(k=X,mu=r[:,None,None]*Y),axis=(0,2))
    else:
        r = adata.layers["unspliced"].toarray().sum(1) + adata.layers["spliced"].toarray().sum(1)
        r /= r.mean()
        logL = np.sum(poisson.logpmf(k=X,mu=r[:,None,None]*Y),axis=(0,2))
    #logL = np.sum(X*np.log(r[:,None,None]*Y)-r[:,None,None]*Y-gammaln(X+1),axis=(0,2))
    
    negG = logL + n*entropy(X)
    p_value = 1-chi2.cdf(-2*negG,df=n-1)
    selected_genes = adata.var_names[idx][np.argsort(negG)[:n_genes]]
    adata.var['e logL'] = np.nan
    adata.var['Poisson logL'] = np.nan
    adata.var['p value'] = np.nan
    adata.var['e logL'][idx] = -n*entropy(X)
    adata.var['Poisson logL'][idx] = logL
    adata.var['p value'][idx] = p_value
    
    ## return data X and gene_names
    idx = [np.where(adata.var_names == gene)[0][0] for  gene in selected_genes]
    X=np.zeros((n,len(idx),2))
    X[:,:,0]=adata.layers["unspliced"][:,idx].toarray()
    X[:,:,1]=adata.layers["spliced"][:,idx].toarray()

    return X, selected_genes

