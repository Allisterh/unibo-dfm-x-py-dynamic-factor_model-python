import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('figure', figsize=(16,6))
plt.rc('lines', linewidth=1.5)
sns.set_style('darkgrid')

def ABC_crit(data: pd.DataFrame, kmax: int , nbck: int | None = None, cmax: int | None = 3,
             seed: int | None = 1776, standardize: bool | None = True, ax: plt.Axes | None = None):
    """
    The ABC criterion for determining the number of common factors.
    
    This function implements the information criterion described in Alessi,
    Barigozzi and Capasso (2010).
    
    
    
    Reference: Alessi, L., M. Barigozzi, and M. Capasso (2010). 
    Improved penalization for determining the number of factors 
    in approximate static factor models. 
    Statistics and Probability Letters 80, 1806?1813.
    
    Parameters
    ----------
    data : pd.DataFrame
        Stationary input data matrix of size T x n.
    kmax : int
        Maximum number of factors.
    nbck : int | None, optional
        Number of sub-blocks to use. The default is floor(n/10).
    cmax : int | None, optional
        Maximum value for the penalty constant. The default is 3.
    seed : int | None, optional
        Seed for random permutation generator. The default is 1776.
    standardize : bool | None, optional
        Whether to standardize the input data matrix. The default is True.
    ax : plt.Axes | None, optional
        Canvas on which to plot the graphs as in the paper. The default is None.

    Raises
    ------
    ValueError
        If input data is not a pandas dataframe.
    ValueError
        If data set includes NaN values.
    ValueError
        If kmax is not provided.

    Returns
    -------
    out : list
        A list containing the estimated number of factors and the plt.Axes object with the plots.

    Authors
    -------
    - Original MATLAB Code: Matteo Barigozzi
    - Python implementation: Shahriyar (Donn) Majidi    

    """
        
    if type(data) != pd.core.frame.DataFrame:
        raise ValueError(f'Input data must be a pandas DataFrame, {type(data)} provided instead.')
    
    if data.isna().any().any():
        raise ValueError('Dataset includes NaN values.')
        
    if kmax is None:
        raise ValueError('Must provide a value for kmax.')
    
    kmax = int(kmax)
    npace = 1
    step = 500
    T, n  = data.shape
    
    if nbck is None:
        nbck = int(np.floor(n/10))
    
    if standardize:
        data = (data - data.mean())/data.std()
        
    ## initialize the Criterion
    
    s = 0
    IC1 = np.zeros(kmax+1)
    abc = np.zeros(((nbck + 1) // npace, int(cmax * step)), dtype=int)
    rng = np.random.default_rng(seed=seed)
    
    for N in range(n-nbck,n+1,npace):
        perm = rng.permutation(n)
        
        xs = data.iloc[:,perm[:N]]
        _eigvals = np.flip(np.linalg.eigvalsh(xs.cov()))
        
        for k in range(kmax+1):
            IC1[k] = _eigvals[k:N].sum()
        
        p = (N+T)/(N*T) * np.log( (N*T)/(N+T) ) ## Bai and Ng penalty
        T0 = np.arange(kmax + 1) * p
        
        for c in range(1,int(cmax*step)+1):
            cc = c/step
            IC = IC1/N + T0*cc
            rr = np.argmin(IC)
            
            abc[s,c-1] = rr
            
        s += 1
            
    cr = np.arange(1,int(cmax*step)+1)/step
    
    ABC = np.zeros((cr.shape[0]+1, 4))
    ABC[0,0] = kmax
    
    sabc = np.std(abc, axis=0, ddof=1)
    
    c1 = 1
    for i in range(len(cr)):
        if sabc[i] == 0:
            if abc[-1,i] == ABC[c1-1,0]:
                ABC[c1-1,2] = cr[i]
            else:
                ABC[c1,0] = abc[-1,i]
                ABC[c1,1] = cr[i]
                ABC[c1,2] = cr[i]
                c1 += 1
    
    ABC[:,3] = ABC[:,2] - ABC[:,1]
    ABC = ABC[:c1]
    
    idx_5 = np.where(ABC[1:, 3] > 0.05)[0]
    idx_1 = np.where(ABC[1:, 3] > 0.01)[0]
    
    rhat1 = int(ABC[idx_5 + 1, 0][0]) if idx_5.size > 0 else kmax
    rhat2 = int(ABC[idx_1 + 1, 0][0]) if idx_1.size > 0 else kmax
    
    out = [rhat1, rhat2]

    if ax is not None:
        ax.plot(cr,abc[-1,:], 'r-', label=r'$r^{*T}_{c;N}$')
        ax.plot(cr,5*sabc, 'b--', label=r'$S_c$')
        ax.set(xlim=[0,1])
        ax.set(xlabel='Tuning parameter - C')
        ax.set(ylabel='Number of factors - r')
        ax.legend()
        ax.set_title('ABC estimated number of factors')
        out.append(ax)
        
    return out
