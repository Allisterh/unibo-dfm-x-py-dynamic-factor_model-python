import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sys
from dataclasses import dataclass
from statsmodels.tools.validation import (string_like,
                                          array_like,
                                          bool_like,
                                          float_like,
                                          int_like,
                                          )

sys.path.insert(-1, '../modules/')
from ABC_crit import ABC_crit

class SFM_StateSpace:
    
    def __init__(self, data: pd.DataFrame, max_factors: int | None = None,
                 demean: bool | None = None, standardize: bool | None = True,
                 method: str | None = 'svd', missing: str | None = None,
                 loadings_mat: object | None = None):
        
        self._nobs, self._nvar = data.shape
        self._standardize = standardize
        self._max_factors = max_factors
        self._method = method
        self._missing = missing
        
        ## Input validation
        if self._method not in ('svd', 'eig'):
            raise ValueError(f'Unkown method: {self._method}. Must be chosen from "svd" or "eig".')
        
    def _impute_missing(self):
        
        if self._missing == 'mean':
            pass
        elif self._missing == 'fill-em':
            pass
        elif self._missing == None:
            raise ValueError('''Data contains missing values.\
                             These values should either be dropped or imputed via one of the methods "mean" or "fill-em".''')
        else:
            raise ValueError(f'Unkown imputation method {self._missing}. Must be chosen from "mean" or "fill-em".')
    
    def _initialize_data(self):
        pass
    
    def _compute_subspace_angles(self):
        pass
    
    def _compute_pca_eig(self):
        pass
    
    def _compute_pca_svd(self):
        pass
    
    def _to_pandas(self):
        pass
    
    def _spectral_periodogram(self):
        pass
    
    def plot_periodogram(self, component: int | None = None):
        pass
    
    def plot_scree(self):
        pass
    
    def plot_ABC(self):
        pass
    
    def plot_rsquare(self):
        pass
    