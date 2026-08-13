# unibo-dfm-x-py
A Python framework for macroeconomic forecasting and nowcasting with dynamic factor models (DFMs).

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Modules](#modules)
- [Workflow](#workflow)
- [Usage Example](#usage-example)
- [Requirements](#requirements)
- [References](#references)
- [License](#license)

## Overview

## Installation

## Modules

### `Fred_transform`
This function transforms each of the series in the FRED QD and MD datasets according to their transformation codes and returns a clean pandas dataframe containing the transformed series indexed by time.
- The function takes as input the raw dataset downloaded from the FRED database and loaded via pandas.
- Additionally, the function takes a two character string input ('QD' or 'MD') to determine the frequency of the input data.

```python
import pandas as pd
from src.modules.Fred_transform import Fred_transform

## Quarterly data
fred_qd = pd.read_csv('./data/2026-04-QD.csv')
fred_qd_tr = Fred_transform(fred_qd, 'QD')

## Monthly data
fred_md = pd.read_csv('./data/2026-04-MD.csv')
fred_md_tr = Fred_transform(fred_md, 'MD')
```

### `ABC_crit`
This function implements the information criterion of Alessi, Barigozzi and Capasso (2010).
- The input data has to be a pandas DataFrame object. It should not contain missing values or else the function throws out an error.
- The upper bound for the number of factors `kmax` is a required input and must provided.
- If `ax` is provided, the function also plots the IC on the same canvas on `ax`.

```python
import matplotlib.pyplot as plt
from src.modules.ABC_crit import ABC_crit

## load the data
T, n = data.shape
## Bai and Ng (2002) upper bound for kmax
kf = int((min(n,T)/100)**0.25)
kmax = 8 * kf if kf > 0 else 8

fig, ax = plt.subplots()
rhat1, rhat2, ax = ABC_crit(data, kmax=kmax, ax=ax, demean=True)
plt.show()
```

### `DFM_periodogram`
This function provides the means for a simple spectral analysis of univariate series.
- The input data has to be a pandas Series object and it cannot contain missing values or else the function throws out an error.
- The frequency of the input data is a required argument. This argument takes an optional two character string input ('QD' or 'MD') to determine the frequency of the input data.
- By default the scaling of the output of the function is such that it corresponds to the power spectrum in units of V^2. Can be set to 'density' if the other scaling is desired.
- The default kernel window is the Bartlett kernel. Can be set to any of the valid kernels recognized by scipy.
- The function returns a data class that contains the array of sample frequencies, the estimated periodogram, and the plot of the periodogram against the sample frequencies.

```python
import matplotlib.pyplot as plt
from src.modules.DFM_periodogram import DFM_periodogram

## load the data
univariate_series = data.iloc[:,0] # the first series for instance

fig, ax = plt.subplots()
dfm_p = DFM_periodogram(univariate_series, freq='QD', ax=ax)
plt.show()
```

## Workflow
The full estimation pipeline proceeds in five stages:
 
```
FRED-QD or FRED-MD raw data
      │
      ▼
1. Fred_transform           — apply stationarity transformations per tcode
      │
      ▼
2. ABC_crit                 — estimate number of common factors via the ABC criterion
      │
      ▼
3. SVD / PCA                — initialize loadings Λ and factors F via SVD of standardized data
      │
      ▼
4. VAR lag selection        — AIC on VAR(F), confirmed via likelihood ratio test
      │
      ▼
5. DynamicFactorMQ          — state space estimation via Kalman filter + EM algorithm
```
 
---
## Usage Example

## Requirements

## References

## License
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
