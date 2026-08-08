import numpy as np
import pandas as pd
from scipy.signal import periodogram
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('figure', figsize=(16,10))
plt.rc('lines', linewidth=1.5)
sns.set_style('darkgrid')

__PGRAM_DEFAULT_PARAMS__ = {'fs': None, 'scaling': 'spectrum', 'window': 'bartlett',
                            'nfft': None}
__PGRAM_FREQS__ = {'QD', 'MD'}

@dataclass
class DFMPeriodogram:
    """Results container for the spectral density periodogram estimates."""
    fxx: np.ndarray     # Array of sample frequencies.
    Pxx: np.ndarray     # Estimated power spectral density or power spectrum of the input series.
    fs: float           # Frequency scaling factor.
    figure: object      # matplotlib Figure for the plot of the estimated power spectral density or the powe spectrum.
    scaling: str        # Nature of the estimated periodogram, density for the power spectral density in units of V^2/Hz and spectrum for the power spectrum in units of V^2.    
    window: str         # The kernel window used in estimating the periodogram. Defaults to 'Bartlett'

def dfm_periodogram(data: pd.Series, freq: str, ax: plt.Axes | None = None, **kwargs) -> DFMPeriodogram:
    
    ## Input validation
    if type(data) != pd.Series:
        raise ValueError(f'Invalid input type {type(data)}. Input data should be a pandas Series.')
        
    if freq == None:
        raise ValueError('Please specify the frequency of the input data. Either "QD" for quartely data or "MD" for monthly data.')
        
    unsupported_freq = {freq} - __PGRAM_FREQS__
    
    if unsupported_freq:
        raise ValueError(f'Unsupported frequency {freq}. Choose either "QD" for quarterlyl data or "MD" for monthly data.')

    params = {k: kwargs.get(k, v) for k,v in  __PGRAM_DEFAULT_PARAMS__.items()}    
    fs, scaling, window, nfft = params.values()
    
    if fs == None:
        fs = 4.0 if freq == 'QD' else 12.0
        plot_xlabel = 'Frequency Cycles - 1.00 = 4 Quarters' if freq == 'QD' else 'Frequency Cycles - 1.00 = 12 Months'
    
    if fs != None:
        plot_xlabel = f'Frequency Cycles - 1.00 = {fs} Periods'
    
    if scaling == None:
        scaling = 'spectrum'
        
    if scaling == 'spectrum':
        plot_ylabel = 'Spectral Power - In units of $x^2$'
    else:
        plot_ylabel = 'Spectral Density - In units of $x^2/Hz$'
        
    if window == None:
        window = 'bartlett'
        
    if nfft == None:
        nfft = len(data)
        
    fxx, Pxx = periodogram(data, fs=fs, window=window, scaling=scaling, nfft=nfft)
    
    if ax is not None:
        ax.plot(fxx, Pxx, label=data.name)
        ax.set(xlabel=plot_xlabel, ylabel = plot_ylabel)
        ax.legend()
        ax.set_title(f'Sample Periodogram with {window.capitalize()} Kernel')
        
    return DFMPeriodogram(fxx, Pxx, fs, ax, scaling, window)