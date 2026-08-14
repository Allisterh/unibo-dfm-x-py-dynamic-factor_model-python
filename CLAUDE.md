# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A Python framework for macroeconomic forecasting and nowcasting with dynamic factor models (DFMs), built on top of `statsmodels`, `scipy`, `numpy`, and `pandas`. There is no package manager config (no `requirements.txt`/`pyproject.toml`), no test suite, and no build/lint tooling configured — this is a research codebase driven by `playground.py`, not an installable package.

## Running the code

There is no CLI entrypoint (`main.py` is currently empty). Work happens by running `playground.py` (or an equivalent script/notebook) from the repo root, which adds `./src/` to `sys.path` and drives the pipeline end to end:

```bash
python3 playground.py
```

Key third-party dependencies observed in imports (install as needed; no lockfile exists): `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`.

## Architecture

The estimation pipeline (documented in more detail in README.md) proceeds in five stages, spanning `src/modules/` (standalone, reusable analysis functions) and `src/models/` (stateful model classes):

```
FRED-QD / FRED-MD raw CSV (data/*.csv)
      │
      ▼
1. Fred_transform (src/modules/Fred_transform.py)   — apply per-series stationarity transforms via FRED tcodes (1-7)
      │
      ▼
2. ABC_crit (src/modules/ABC_crit.py)                — Alessi-Barigozzi-Capasso IC to pick the number of common factors
      │
      ▼
3. SVD / PCA (src/models/static_methods.py)          — SFM_StateSpace / StaticFM: initialize loadings Λ and factors F
      │
      ▼
4. VAR lag selection (statsmodels sm.tsa.VAR)         — AIC, confirmed via likelihood ratio test
      │
      ▼
5. DynamicFactorMQ (statsmodels) / src/models/dynamic_methods.py — state-space estimation via Kalman filter + EM
```

### Module responsibilities

- **`src/modules/Fred_transform.py`** — `Fred_transform(data, freq)` applies the FRED-QD/MD stationarity transform codes (1=none, 2=diff, 3=2nd diff, 4=log, 5=log-diff, 6=2nd log-diff, 7=pct change) to a raw FRED CSV loaded via pandas. `freq` must be `'QD'` or `'MD'` — this determines which row/column offsets hold the tcodes vs. data. `Fred_qd_transform` is a deprecated QD-only predecessor kept for backward compatibility; prefer `Fred_transform`.
- **`src/modules/ABC_crit.py`** — `ABC_crit(data, kmax, ...)` implements the Alessi-Barigozzi-Capasso (2010) information criterion for selecting the number of common factors via random sub-sampling of columns across a grid of penalty constants. Input must be a NaN-free `pd.DataFrame`. Returns `[rhat1, rhat2]` (5%/1% threshold estimates), plus the `Axes` if `ax` is passed.
- **`src/modules/DFM_periodogram.py`** — `DFM_periodogram(series, freq, ax=None, **kwargs)` wraps `scipy.signal.periodogram` for spectral analysis of a single series, returning a `DFMPeriodogram` dataclass. `freq` (`'QD'`/`'MD'`) only affects the sampling-frequency label/scale (`fs=4` or `12`).
- **`src/models/static_methods.py`** — `SFM_StateSpace` is the core static factor model class: handles missing-data imputation (`'mean'` or `'fill-em'`, an internal iterative PCA-based EM), standardization/demeaning, PCA via SVD or eigendecomposition, and factor/loadings identification (normalizing either the factor covariance to `I` or fixing loading signs on the first row). `StaticFM(...)` is a thin functional wrapper returning `(factors, loadings, projection, residuals, eigenvals, eigenvecs)`. This class follows the `statsmodels`-style validation convention (`string_like`, `int_like`, etc. from `statsmodels.tools.validation`) for constructor args.
- **`src/models/dynamic_methods.py`** — `DFM_StateSpace`, the dynamic (state-space) counterpart to `SFM_StateSpace`; currently a stub (`__init__` unimplemented).
- **`src/models/VAR_simulate.py`** — `transition_mat(dimension, order, stable=False, alpha=None, seed=1776)` generates (optionally stability-rescaled) companion-form transition matrices for simulating VAR processes, used for testing/validating the dynamic factor model machinery.
- **`src/utils/impute.py`** — `impute_missing(series, max_iter=50)` fills NaNs in a univariate series using a local-level (random walk + noise) model fit by MLE, with Kalman-smoothed state means substituted at missing positions.
- **`src/diagnostics.py`** — statistical test wrappers and unit-root diagnostics: `adf_wrapper`/`kpss_wrapper` return `adfuller`/`kpss` results as labeled `pd.Series`; `CT_unitroot` implements the Cavaliere & Taylor (2008) wild bootstrap ADF test (Rademacher weights), robust to nonstationary volatility, via internal helpers `_get_adf_residuals` and `_wild_bootstrap_adf`.

### Conventions to preserve

- Model/estimator classes (`SFM_StateSpace`, and the `DFM_StateSpace` stub) validate constructor arguments using `statsmodels.tools.validation` helpers (`string_like`, `array_like`, `bool_like`, `float_like`, `int_like`) rather than manual `isinstance` checks — follow this pattern for new model classes.
- Functions operating on FRED data consistently distinguish `'QD'` (quarterly) vs `'MD'` (monthly) frequency via a two-character string argument, affecting both index offsets and (for periodogram/plotting code) the sampling-frequency scale.
- Global matplotlib/seaborn styling (`plt.rc('figure', figsize=(16,6))`, `sns.set_style('darkgrid')`) is set at import time in most `src/modules/*.py` and `src/models/*.py` files — keep new plotting modules consistent with this.
- Factor/loadings identification schemes are derived analytically in code comments before implementation (see `static_methods.py` and `playground.py`) — when modifying identification logic, preserve or update these derivations since they document *why* the linear algebra is structured that way, not just what it does.
- `playground.py` is the living reference for how the pipeline stages are meant to be wired together end-to-end (including judgment calls like which nonstationary FRED-QD series to drop/detrend); consult it when integrating new modules into the pipeline rather than inferring flow from module signatures alone.
