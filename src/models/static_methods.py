import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import subspace_angles
from statsmodels.tools.validation import (string_like,
                                          array_like,
                                          bool_like,
                                          float_like,
                                          int_like,
                                          )

## Global attributes
plt.rc('figure', figsize=(16,6))
plt.rc('lines', linewidth=1.5)
sns.set_style('darkgrid')

## Useful functions
def _norm(x):
    return np.sqrt(np.sum(x*x))

class SFM_StateSpace:
    
    def __init__(self, data: pd.DataFrame, nfactors: int | None = None,
                 demean: bool | None = False, standardize: bool | None = True,
                 method: str | None = 'svd', missing: str | None = 'fill-em',
                 normalize: bool | None = True, loadings_mat: object | None = None,
                 tol_em: float | None = 5e-8, max_em_iter: int | None = 1000):
        
        self._nobs, self._nvar = data.shape
        self._nfactors = int_like(nfactors, 'nfactors', optional=True)
        self._demean = bool_like(demean, 'demean', optional=True)
        self._standardize = bool_like(standardize, 'standardize', optional=True)
        self._method = string_like(method, 'method')
        self._missing = string_like(missing, 'missing')
        self._normalize = bool_like(normalize, 'normalize', optional=True)
        self._loadings_mat = array_like(loadings_mat, 'loaginds_mat', ndim=2, optional=True)
        self._tol_em = float_like(tol_em, 'tol_em')
        self._max_em_iter = int_like(max_em_iter, 'max_em_iter')
        
        self._index = None
        self._columns = []
        if isinstance(data, pd.DataFrame):
            self._index = data.index
            self._columns = data.columns
            
        ## If the number of common factors is not provided set to the upper bound suggested by Bai & Ng
        if self._nfactors is None:            
            kf = int((min(self._nvar,self._nobs)/100)**0.25)
            kmax = 8 * kf if kf > 0 else 8
            self._nfactors = kmax
            
        self.data = array_like(data, 'data', ndim=2)
        self._adjusted_data = self.data
        self.rows = np.arange(self._nobs)
        self.cols = np.arange(self._nvar)
        ## Handle missing values
        self._impute_missing()
        
        ## Update recorded size of the data after imputation
        self._nobs, self._nvar = self._adjusted_data.shape
        
        ## Decomposition method validation
        if self._method not in ('svd', 'eig'):
            raise ValueError(f'Unrecognized decomposition method {self._method}. '
                             'Must be chosen from "eig" or "svd".')
            
        ## Declare class parameters and attributes
        self.principal_components = None
        self.eigenvals = None
        self.eigenvecs = None
        self.factors = None
        self.loadings = None
        self.transformed_data = None
        self.projection = None
        self.residuals = None
        self.standardized_residuals = None
        
        ## Initialize the data
        self.transformed_data = self._initialize_data()
        
        ## Perform Principal Components Analysis
        self._pca()
        
        ## Prepare the output
        if self._index is not None:
            self._to_pandas()
        
    def __str__(self):
        string = 'Factor Analysis via Static Methods ('
        string += 'nobs: ' + str(self._nobs) + ', '
        string += 'nvar: ' + str(self._nvar) + ', '
        if self._standardize:
            kind = 'Standardize (Correlation)'
        elif self._demean:
            kind = 'Demean (Covariance)'
        else:
            kind = 'Original Unscaled Data'
            
        string += 'transformation: ' + kind + ', '
            
        if self._normalize:
            model_id = 'Factors with unit variance with the loadings on the first factor all positive'
        else:
            model_id = 'Inner product of the matrix of loadings equal to the identity with the observations on the first factor all positive'
            
        string += 'model identification (Global): ' + model_id + ', '
        string += 'number of common factors: ' + str(self._nfactors) + ', '
        string += 'decomposition method: ' 
        string += 'Eigen Decomposition' if self._method == 'eig' else 'SVD'
        string += ')'
        return string
        
    def __repr__(self):
        string = self.__str__()
        string = string[:-1]
        string += ', id: ' + hex(id(self)) + ')'
        return string
        
    def _initialize_data(self):
        adj_data = self._adjusted_data
        
        if np.all(np.isnan(adj_data)):
            return np.empty(adj_data.shape[1]).fill(np.nan)
            
        self._mu = np.nanmean(adj_data, axis=0)
        self._sigma = np.sqrt(np.nanmean( (adj_data - self._mu) ** 2.0, axis=0))
            
        if self._demean:
            data = adj_data - self._mu
        elif self._standardize:
            data = (adj_data - self._mu)/self._sigma
        else:
            data = adj_data
                
        return data
        
    def _impute_missing(self):
        ## Impute missing values according to the chosen method
        if self._missing == 'mean':
            self._adjuted_data = self._fill_missing_mean()
        elif self._missing == 'fill-em':
            self._adjuted_data = self._fill_missing_em()
        elif self._missing == None:
            if not np.isfinite(self._adjuted_data).all():
                raise ValueError('''Data contains missing values.\
                                 These values should either be dropped or imputed via one of the methods "mean" or "fill-em".''')
        else:
            raise ValueError(f'Unkown imputation method {self._missing}. Must be chosen from "mean" or "fill-em".')
        
        if self._index is not None:
            self._index = self._index[self.rows]
            self._columns = self._columns[self.cols]
                
        ## Check adjusted data size and raise ValueError if all data is lost due to adjustment
        if self._adjusted_data.size == 0:
            raise ValueError('The requested missing value imputation method has eliminated all the data.')

    def _fill_missing_mean(self):
        non_missing = np.logical_not(np.isnan(self.data))
        if np.all(non_missing):
            return self.data
            
        data = self.data
        ## Compute mean discarding missing values
        mu = np.nanmean(data, axis=0)
            
        ## Get mask
        mask = np.isnan(data)
            
        ## Replace missing values with mean
        projection = np.ones((self._nobs, 1)) * mu
        projection_masked = projection[mask]
        data[mask] = projection_masked
            
        return data
        
    def _fill_missing_em(self):
        ## EM Algorithm to impute missing values
        
        non_missing = np.logical_not(np.isnan(self.data))
        if np.all(non_missing):
            return self.data
        
        nfactors = self._nfactors
        
        ## Controll the feasibility of the EM algorithm
        col_non_missing = np.sum(non_missing, axis=1)
        row_non_missing = np.sum(non_missing, axis=0)
        if np.any(col_non_missing < nfactors) or np.any(row_non_missing < nfactors):
            raise ValueError('Implementation of the EM algorithm for imputing the missing values '
                             'is impossible since at least one row or one column has fewer than '
                             f'nfactors: {nfactors} non-missing values which is the necessary threshold for '
                             'the implementation of the EM algorithm.')
        
        ## First standardize the data
        data = self.transformed_data = np.asarray(self._initialize_data())
        
        ## Get mask
        mask = np.isnan(data)
                
        ## Replace missing values with zeros
        projection = np.zeros((self._nobs, self._nvar))
        projection_masked = projection[mask]
        data[mask] = projection_masked
            
        ## Fit the factor model with the maximum number of factors and replace missing values with the estimated common component
        diff = 1.0
        _iter = 0
        while diff > self._tol_em and _iter < self._max_em_iter:
            last_projection_masked = projection_masked
            ## Update transformed data
            self.transformed_data = data
            ## Call the decomposition function
            self._obtain_eig()
            ## Now estimate the factor model via PCA on the eigen or singular values
            self._compute_pca()
            projection = np.asarray(self.project(transform=False))
            projection_masked = projection[mask]
            data[mask] = projection_masked
            ## Compute threshold
            delta = last_projection_masked - projection_masked
            diff = _norm(delta) / _norm(projection_masked)
            _iter += 1
            
        ## Copy data to avoid overwriting original values
        data = self._adjusted_data + 0.0
        projection = np.asarray(self.project())
        data[mask] = projection[mask]
        
        return data
        
    def _obtain_eig(self):
        if self._method == 'eig':
            return self._decomp_eig()
        else:
            return self._decomp_svd()
            
    def _decomp_eig(self):
        ## Obtain eigenvalues and the corresponding eigenvectors from the eigendecomposition of the design matrix.
        X = self.transformed_data
        evals, evecs = np.linalg.eigh(X.T.dot(X))
        '''
        What follows is a simple procedure to ensure that the eigenvalues and the corresponding
        eigenvectors are always sorted from larges to smallest.
        '''
        indices = np.argsort(evals)
        indices = indices[::-1]
        evals = evals[indices]
        evecs = evecs[:,indices]
        self.eigenvals, self.eigenvecs = evals, evecs
            
    def _decomp_svd(self):
        ## Obtain singular values and the corresponding singular vectors from the singula value decomposition of the data matrix.
        X = self.transformed_data
        U, s, Vh = np.linalg.svd(X, full_matrices=False)
        self.eigenvals = s**2
        self.eigenvecs = Vh.T
            
    def _compute_pca(self):
        ## Compute the common component via PCA from eigenvalues and eigenvectors
        evals, evecs = self.eigenvals, self.eigenvecs
            
        '''
        For the remainder of the code, we only select the relevant number of components
        according to the number of factors provided.
        '''
        evals = evals[:self._nfactors]
        evecs = evecs[:, :self._nfactors]
            
        #self.eigenvals, self.eigenvecs = evals, evecs
            
        ## Assign the values of the PCA estimates of the factors and the loadings based on the identification assumptions.
        if self._normalize:
            self.loadings = evecs @ np.diag(np.sqrt(evals)) / np.sqrt(self._nobs)
            H = np.diag(np.sign(self.loadings[0,:]))
            self.loadings = self.loadings @ H
            P_coeff = self.loadings @ np.linalg.inv(H.T @ np.diag(evals)/self._nobs @ H)
            self.factors = self.transformed_data @ P_coeff
                
        else:
            self.factors = self.transformed_data @ evecs / np.sqrt(self._nvar)
            H = np.diag(np.sign(self.factors[0,:]))
            self.factors = self.factors @ H
            P_coeff = self.factors @ np.linalg.inv(self.factors.T @ self.factors)
            self.loadings = self.transformed_data.T @ P_coeff
                
    def project(self, nfactors: int | None = None, transform: bool | None = True):
        nfactors = self._nfactors if nfactors is None else nfactors
        factors = np.asarray(self.factors)
        loadings = np.asarray(self.loadings)
                
        projection = factors @ loadings.T
                
        ## Undo data transformation for transformed data
        if transform:
            if self._standardize:
                projection *= self._sigma
            if self._standardize or self._demean:
                projection += self._mu
                        
        if self._index is not None:
            projection = pd.DataFrame(projection,
                                      index=self._index,
                                      columns=self._columns)
        return projection
            
    def _pca(self):
        '''
        Main PCA Routine
        '''
        self._obtain_eig()
        self._compute_pca()
        self.projection = self.project()
        self._residual_maker()
                
    def _residual_maker(self):
        self.residuals = self._adjusted_data - self.projection
        self.standardized_residuals = self.transformed_data - self.factors @ self.loadings.T
            
    def _to_pandas(self):
        '''
        Return all parameters as a pandas DataFrame if the input data were a pandas DataFrame.

        '''
        index = self._index
                
        # Factors
        num_zeros = np.ceil(np.log10(self._nfactors))
        comp_str = 'factor_{0:0' + str(int(num_zeros)) + 'd}'
        cols = [comp_str.format(i) for i in range(self._nfactors)]
        df = pd.DataFrame(self.factors, index=index,
                          columns=cols)
        self.factors = df
                
        # Loadings
        df = pd.DataFrame(self.loadings, index=self._columns,
                          columns=cols)
        self.loadings = df
                
        # Projections
        df = pd.DataFrame(self.projection, index=index,
                          columns=self._columns)
        self.projection = df
                
        # Transformed Data
        df = pd.DataFrame(self.transformed_data, index=index,
                          columns=self._columns)
        self.transformed_data = df
                
        # Residuals (also standardized)
        ## 1. Raw Residuals
        df = pd.DataFrame(self.residuals, index=index,
                          columns=self._columns)
        self.residuals = df
        ## 2. Standardized Residuals
        df = pd.DataFrame(self.standardized_residuals, index=index,
                          columns=self._columns)
        self.standardized_residuals = df
                
        # Eigenvalues
        self.eigenvals = pd.Series(self.eigenvals)
        self.eigenvals.name = 'eigenvalues'
                
        # Eigenvectors
        eigenvec_str = comp_str.replace('factor', 'eigenvector')
        cols = [eigenvec_str.format(i) for i in range(self.eigenvecs.shape[1])]
        self.eigenvecs = pd.DataFrame(self.eigenvecs, columns=cols)
                
    def plot_scree(self, ncomp: int | None = None):
        #fig, ax = plt.subplots()
        if ncomp is None:
            ncomp = len(self.eigenvals)
        
        data = self.eigenvals[:ncomp]
        
        ax = sns.scatterplot(data = data/np.sqrt(self._nobs))
        if self._method == 'eig':
            ylabel = '$Eigen Value (Scaled by \sqrt{T})$'
        else:
            ylabel = '$Singular Value (Scaled by \sqrt{T})$'
        
        ax.set(xlabel='Component Index',
               ylabel=ylabel,
               title='Scree Plot')
        return ax
    
    def loadings_subspace_angles(self):
        '''
        Compare the span of the estimated matrix of loadings with the original matrix.
        Raises
        ------
        ValueError
            If the original matrix of loadings is not provided.

        Returns
        -------
        df : TYPE
            Dataframe containing the subspace angles in descending order.

        '''
        loagings_mat = self._loadings_mat
        if loagings_mat is None:
            raise ValueError('The matrix of loadings is not provided. Nothing to compare against.')
         
        loadings = self.loadings
        ang_sep = subspace_angles(loagings_mat, loadings).reshape(1,self._nfactors)
        
        num_zeros = np.ceil(np.log10(self._nfactors))
        ang_str = 'angle_{0:0' + str(int(num_zeros)) + 'd}'
        cols = [ang_str.format(i) for i in range(self._nfactors)]
        df = pd.DataFrame(ang_sep, columns=cols)
        return df
    
def StaticFM(data, nfactors: int | None = None, demean: bool | None = False,
                    standardize: bool | None = True, method: str | None = 'svd',
                    missing: str | None = 'fill-em', normalize: bool | None = True,
                    loadings_mat: object | None = None):
    '''

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    nfactors : int | None, optional
        DESCRIPTION. The default is None.
    demean : bool | None, optional
        DESCRIPTION. The default is False.
    standardize : bool | None, optional
        DESCRIPTION. The default is True.
    method : str | None, optional
        DESCRIPTION. The default is 'svd'.
    missing : str | None, optional
        DESCRIPTION. The default is 'fill-em'.
    normalize : bool | None, optional
        DESCRIPTION. The default is True.
    loadings_mat : object | None, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    Notes
    -----
    Wrapper function around the SFM_StateSpace class.

    '''
    ssm = SFM_StateSpace(data, nfactors=nfactors, standardize=standardize,
                                 method = method, missing = missing, normalize = normalize,
                                 loadings_mat = loadings_mat)
    
    return(ssm.factors, ssm.loadings, ssm.projection, ssm.residuals, ssm.eigenvals, ssm.eigenvecs)
            
            
            
            
            
                
        
        