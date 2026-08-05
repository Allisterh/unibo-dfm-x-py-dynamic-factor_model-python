import numpy as np
import pandas as pd
import warnings

__transform_codes__ = [1,2,3,4,5,6,7]
__frequencies__ = ['QD', 'MD']

def _fred_series_transform (data: pd.Series, transform_code: int):
    """
    This function transforms the input data to achieve stationarity.
    Transformations are based on integer codes which determin the
    type of transformation.
    
    transform_code  == 1 -> No transformation.
                    == 2 -> First difference.
                    == 3 -> Second order difference.
                    == 4 -> Log transform.
                    == 5 -> Log difference.
                    == 6 -> Second order log difference.
                    == 7 -> Relative difference.
                    
    Reference: FRED-QD: A Quarterly Database for Macroeconomic Research McCracken and Ng, 2020.

    Parameters
    ----------
    data : pd.Series
        Univariate input time-series.
    transform_code : int
        Transformation code.

    Returns
    -------
    pd.Series.
    
     
    Raises
    ------
    ValueError
        If transformation codes are invalid.

    """
    
    if transform_code not in __transform_codes__:
        raise ValueError(f"Transformation code {transform_code} is invalid. It has to be chosen from {__transform_codes__}")
        
    match transform_code:
        case 1:
            data_tr = data
            
        case 2:
            data_tr = data.diff()
            
        case 3:
            data_tr = data.diff().diff()
            
        case 4:
            data_tr = np.log(data)
            
        case 5:
            data_tr = np.log(data).diff()
            
        case 6:
            data_tr = np.log(data).diff().diff()
            
        case 7:
            data_tr = data/data.shift(1) - 1
            
    return data_tr
            
def fred_transform(data: pd.DataFrame, freq: str):
    """
    This function transforms the input data to achieve stationarity.
    Transformations are based on integer codes which determine the
    type of transformation.
    
    transform_code  == 1 -> No transformation.
                    == 2 -> First difference.
                    == 3 -> Second order difference.
                    == 4 -> Log transform.
                    == 5 -> Log difference.
                    == 6 -> Second order log difference.
                    == 7 -> Relative difference.
                    
    Reference: FRED-QD: A Quarterly Database for Macroeconomic Research McCracken and Ng, 2020.

    Parameters
    ----------
    data : pd.DataFrame
        Raw FRED-QD or FRED-MD data frame.
        
    freq : str
        String indicating the frequency of the input data. It has to be either "QD" for quarterly data or "MD" for monthly data.

    Returns
    -------
    Transformed series in a pd.DataFrame object.
    
    Raises
    ------
    ValueError
        If frequency is not recognized.

    """
    
    if freq not in __frequencies__:
        raise ValueError(f"Unrecognized frequency {freq}. Choose 'QD' for quarterly data or 'MD' for monthly data.")
    
    transformed = []
    
    if freq == 'QD':
        init_idx = 2
        trcode_idx = 1
        
    elif freq == 'MD':
        init_idx = 1
        trcode_idx = 0
    
    for i in range(1,len(data.columns)):
        
        input_series = data.iloc[init_idx:,i]
        transform_code = data.iloc[trcode_idx,i]
        input_dates = data.iloc[init_idx:,0]
        
        _input_series_tr = _fred_series_transform(input_series, transform_code)
        _input_series_tr.name = data.columns[i]
        _input_series_tr.index = pd.to_datetime(input_dates, format="%m/%d/%Y", errors='coerce')

        transformed.append(_input_series_tr)
        
    transformed_df = pd.concat(transformed, axis=1, join='outer')
    transformed_df.index.freq = transformed_df.index.inferred_freq
    
    return transformed_df       

def fred_qd_transform(data: pd.DataFrame):
    warnings.warn('This function is deprecated and will probably be removed in a future release. Use fred_transform instead.',
                  DeprecationWarning,
                  stacklevel=2)
    """
    This function transforms the input data to achieve stationarity.
    Transformations are based on integer codes which determine the
    type of transformation.
    
    transform_code  == 1 -> No transformation.
                    == 2 -> First difference.
                    == 3 -> Second order difference.
                    == 4 -> Log transform.
                    == 5 -> Log difference.
                    == 6 -> Second order log difference.
                    == 7 -> Relative difference.
                    
    Reference: FRED-QD: A Quarterly Database for Macroeconomic Research McCracken and Ng, 2020.

    Parameters
    ----------
    data : pd.DataFrame
        Raw FRED-QD data input.
        
    Returns
    -------
    Transformed series in a pd.DataFrame object.
    
    """
    
    transformed = []
    
    for i in range(1,len(data.columns)):
        input_series = data.iloc[2:,i]
        transform_code = data.iloc[1,i]
        input_dates = data.iloc[2:,0]
        
        match transform_code:
            case 1:
                _input_series_tr = input_series
                
            case 2:
                _input_series_tr = input_series.diff()
                
            case 3:
                _input_series_tr = input_series.diff().diff()
                
            case 4:
                _input_series_tr = np.log(input_series)
                
            case 5:
                _input_series_tr = np.log(input_series).diff()
                
            case 6:
                _input_series_tr = np.log(input_series).diff().diff()
                
            case 7:
                _input_series_tr = input_series/input_series.shift(1) - 1
            
            
        _input_series_tr.name = data.columns[i]
        _input_series_tr.index = pd.to_datetime(input_dates, format="%m/%d/%Y", errors='coerce')
        
        transformed.append(_input_series_tr)
        
    transformed_df = pd.concat(transformed, axis=1, join='outer')
    
    return transformed_df
