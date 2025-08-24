cimport cython
import numpy as np
cimport numpy as np

# cython: np_pythran=True


def sma(np.ndarray[np.float32_t, ndim=1] input_array = None, int timeperiod = 2, bint normalize=True, int n_values = 999999999):
    """
    Calculate the Simple Moving Average (SMA) over a specified time period.

    The SMA is the unweighted mean of the previous `timeperiod` data points. It smooths 
    the data to identify trends over a specified window.

    Parameters:
    ----------
    input_array : np.ndarray[np.float32_t, ndim=1]
        Array of input values (e.g., prices or other time series data) for the calculation.

    timeperiod : int
        The lookback period for calculating the SMA.

    normalize : bint, optional
        If True, the SMA values are normalized relative to the input values:
            Normalized SMA = (SMA - Value) / (Value if Value != 0 else 1e-9)
        If False, the raw SMA values are returned. Defaults to True.

    n_values : int, optional
        Number of values from the end of the input array to calculate. Defaults to a very large number, effectively using all elements.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the Simple Moving Average (SMA) values (normalized or raw). 
        Values outside the calculated range remain NaN.

    Notes:
    -----
    - The SMA is calculated using a sliding window:
        SMA[i] = Σ(Value[i - timeperiod + 1] ... Value[i]) / timeperiod
      where only non-NaN values are considered in the sum and count.
    - If `normalize` is True, the SMA is adjusted relative to the current value at each index.
    - A bitmask (`non_nan_mask`) is used to efficiently handle NaN values in the input array.
    - The sliding window approach optimizes computation by reusing sums from the previous window.
    - Initial values, corresponding to the `timeperiod`, remain NaN due to insufficient data.
    """    """
    Calculate the Simple Moving Average (SMA) over a specified time period.

    The SMA is the unweighted mean of the previous `timeperiod` data points. It smooths 
    the data to identify trends over a specified window.

    Parameters:
    ----------
    input_array : np.ndarray[np.float32_t, ndim=1]
        Array of input values (e.g., prices or other time series data) for the calculation.

    timeperiod : int
        The lookback period for calculating the SMA.

    normalize : bint, optional
        If True, the SMA values are normalized relative to the input values:
            Normalized SMA = (SMA - Value) / (Value if Value != 0 else 1e-9)
        If False, the raw SMA values are returned. Defaults to True.

    n_values : int, optional
        Number of values from the end of the input array to calculate. Defaults to a very large number, effectively using all elements.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the Simple Moving Average (SMA) values (normalized or raw). 
        Values outside the calculated range remain NaN.

    Notes:
    -----
    - The SMA is calculated using a sliding window:
        SMA[i] = Σ(Value[i - timeperiod + 1] ... Value[i]) / timeperiod
      where only non-NaN values are considered in the sum and count.
    - If `normalize` is True, the SMA is adjusted relative to the current value at each index.
    - A bitmask (`non_nan_mask`) is used to efficiently handle NaN values in the input array.
    - The sliding window approach optimizes computation by reusing sums from the previous window.
    - Initial values, corresponding to the `timeperiod`, remain NaN due to insufficient data.
    """
    cdef int n = len(input_array), i, non_nan_count = 0
    cdef np.ndarray[np.float32_t, ndim=1] sma = np.full(n, np.nan, dtype=np.float32)
    cdef np.ndarray[np.uint8_t, ndim = 1, cast=True] non_nan_mask = ~np.isnan(input_array)
    cdef float sum = 0.0
    k = max(timeperiod - 1, n - n_values)
    for i in range(k - timeperiod, k):
        if non_nan_mask[i]:
            sum += input_array[i]
            non_nan_count += 1
    if normalize:
        for i in range(k, n):
            new = input_array[i]
            if non_nan_mask[i]:
                sum += new
                non_nan_count += 1
            if non_nan_mask[i - timeperiod]:
                sum -= input_array[i - timeperiod]
                non_nan_count -= 1
            if non_nan_count > 0:
                sma[i] = ((sum / non_nan_count) - new) / (new if new!=0 else 1e-9)     
    else:
        for i in range(k, n):
            new = input_array[i]
            if non_nan_mask[i]:
                sum += new
                non_nan_count += 1
            if non_nan_mask[i - timeperiod]:
                sum -= input_array[i - timeperiod]
                non_nan_count -= 1
            if non_nan_count > 0:
                sma[i] = sum / non_nan_count
    del input_array
    return sma


def linreg_slope(np.ndarray[np.float32_t, ndim=1] input_array = None, int timeperiod = 1, int n_values = 999999999):
    """
    Calculate the slope of a linear regression line over a specified time period.

    This function computes the slope of the line of best fit for a sliding window of 
    data points in the input array. The slope is calculated using least squares regression.

    Parameters:
    ----------
    input_array : np.ndarray[np.float32_t, ndim=1]
        Array of input values (e.g., prices or other time series data) for the calculation.

    timeperiod : int
        The lookback period for calculating the linear regression slope.

    n_values : int, optional
        Number of values from the end of the input array to calculate. Defaults to a very large number, effectively using all elements.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the linear regression slope values for each time step. 
        Values outside the calculated range remain NaN.

    Notes:
    -----
    - The slope of the linear regression line is calculated using the formula:
        Slope = (N * Σ(xy) - Σx * Σy) / (N * Σ(x²) - (Σx)²)
      where:
        - N is the number of points (timeperiod),
        - Σxy is the sum of x * y (x is the index, y is the value),
        - Σx is the sum of x,
        - Σy is the sum of y,
        - Σx² is the sum of x².
    - The function skips NaN values until the first non-NaN value is encountered.
    - If there is insufficient data to calculate the slope for a given index, the result is NaN.
    - A sliding window approach is used to optimize performance, reusing sums for efficient computation.
    - Denominator checks are in place to avoid division by zero.

    """    
    cdef int n = len(input_array), i, j, first_non_nan_index = -1, start_index
    cdef np.ndarray[np.float32_t, ndim=1] lin = np.full(n, np.nan, dtype=np.float32)
    cdef float sum_y = 0.0, sum_xy = 0.0, sum_x = 0.0, sum_x2 = 0.0, denom, deduction = 0.0
    for i in range(n - 2*timeperiod - min(n_values, n - 2*timeperiod), n - timeperiod):
        if not np.isnan(input_array[i]):
            first_non_nan_index = i
            break
    if first_non_nan_index == -1:
        return lin
    start_index = max(first_non_nan_index, n - n_values - timeperiod)
    for j in range(start_index, start_index + timeperiod):
        if j > start_index:
            deduction += input_array[j]
        sum_x += (j - start_index) + 1
        sum_x2 += ((j - start_index) + 1) * ((j - start_index) + 1)
        sum_y += input_array[j]
        sum_xy += input_array[j] * ((j - start_index) + 1)
    denom = timeperiod * sum_x2 - sum_x * sum_x
    if (denom != 0) and (timeperiod + start_index - 1 >= n - n_values):
        lin[(timeperiod + start_index) - 1] = (timeperiod * sum_xy - sum_x * sum_y) / denom
    for i in range(1, n - (timeperiod + start_index) + 1):
        sum_y += input_array[i + (timeperiod + start_index) - 1] - input_array[i - 1 + start_index]
        sum_xy += input_array[i + (timeperiod + start_index) - 1] * timeperiod - input_array[i - 1 + start_index] - deduction
        deduction += input_array[i + (timeperiod + start_index) - 1] - input_array[i + start_index]
        if denom != 0:
            lin[i + (timeperiod + start_index) - 1] = (timeperiod * sum_xy - sum_x * sum_y) / denom  
    del input_array
    return lin



def midpoint(np.ndarray[np.float32_t, ndim=1] high = None, np.ndarray[np.float32_t, ndim=1] low = None, np.ndarray[np.float32_t, ndim=1] close = None, int timeperiod = 1, bint normalize=True, int n_values = 999999999):
    """
    Calculate the midpoint of the high and low prices over a specified time period.

    The midpoint is a simple average of the highest high and lowest low within the lookback period. 
    It can be normalized relative to the closing price to measure its relative deviation.

    Parameters:
    ----------
    high : np.ndarray[np.float32_t, ndim=1]
        Array of high prices for each time step.

    low : np.ndarray[np.float32_t, ndim=1]
        Array of low prices for each time step.

    close : np.ndarray[np.float32_t, ndim=1]
        Array of closing prices for each time step.

    timeperiod : int
        The lookback period for calculating the highest high and lowest low.

    normalize : bint, optional
        If True, the midpoint is normalized relative to the closing price:
            Normalized Midpoint = (Midpoint - Close) / Close
        If False, the raw midpoint values are returned. Defaults to True.

    n_values : int, optional
        Number of values from the end of the input arrays to calculate. Defaults to a very large number, effectively using all elements.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the midpoint values (normalized or raw). 
        Values outside the calculated range remain NaN.

    Notes:
    -----
    - The midpoint is calculated as:
        Midpoint = Low + (High - Low) / 2
    - If `normalize` is True, the midpoint is adjusted relative to the closing price.
    - A sliding window approach is used to efficiently track the highest high and lowest low 
      over the lookback period.
    - Initial values, corresponding to the `timeperiod`, remain NaN due to insufficient data.
    """
    cdef int n = len(close), i, high_start = 0, high_end = 0, low_start = 0, low_end = 0
    cdef int k = max(timeperiod - 1, n - n_values)
    cdef np.ndarray[np.float32_t, ndim=1] mid = np.full(n, np.nan, dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] high_indices = np.zeros(n, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] low_indices = np.zeros(n, dtype=np.int32)
    for i in range(k - timeperiod + 1, k):
        while high_end > high_start and high[high_indices[high_end - 1]] <= high[i]:
            high_end -= 1
        high_indices[high_end] = i
        high_end += 1
        while low_end > low_start and low[low_indices[low_end - 1]] >= low[i]:
            low_end -= 1
        low_indices[low_end] = i
        low_end += 1
    if normalize:
        for i in range(k, n):
            new_close = close[i]
            if high_end > high_start and high_indices[high_start] <= i - timeperiod:
                high_start += 1
            if low_end > low_start and low_indices[low_start] <= i - timeperiod:
                low_start += 1
            while high_end > high_start and high[high_indices[high_end - 1]] <= high[i]:
                high_end -= 1
            high_indices[high_end] = i
            high_end += 1
            while low_end > low_start and low[low_indices[low_end - 1]] >= low[i]:
                low_end -= 1
            low_indices[low_end] = i
            low_end += 1
            mid[i] = ((low[low_indices[low_start]] + (high[high_indices[high_start]] - low[low_indices[low_start]]) / 2) - new_close) / (new_close if new_close!=0 else 1e-9) 
    else:
        for i in range(k, n):
            if high_end > high_start and high_indices[high_start] <= i - timeperiod:
                high_start += 1
            if low_end > low_start and low_indices[low_start] <= i - timeperiod:
                low_start += 1
            while high_end > high_start and high[high_indices[high_end - 1]] <= high[i]:
                high_end -= 1
            high_indices[high_end] = i
            high_end += 1
            while low_end > low_start and low[low_indices[low_end - 1]] >= low[i]:
                low_end -= 1
            low_indices[low_end] = i
            low_end += 1
            mid[i] = low[low_indices[low_start]] + (high[high_indices[high_start]] - low[low_indices[low_start]]) / 2
    del high, low, close, high_indices, low_indices
    return mid

 

def ema(np.ndarray[np.float32_t, ndim=1] input_array = None, int timeperiod = 1, bint normalize=True, int n_values = 999999999):
    """
    Calculate the Exponential Moving Average (EMA) over a specified time period.

    The EMA is a type of moving average that places a greater weight and significance on 
    recent data points. This makes it more responsive to recent price changes compared 
    to a simple moving average.

    Parameters:
    ----------
    input_array : np.ndarray[np.float32_t, ndim=1]
        Array of input values (e.g., closing prices) for the calculation.

    timeperiod : int
        The lookback period for calculating the EMA.

    normalize : bint, optional
        If True, the EMA values are normalized relative to the input values:
            Normalized EMA = (EMA - Value) / Value
        If False, the raw EMA values are returned. Defaults to True.

    n_values : int, optional
        Number of values from the end of the input array to calculate. Defaults to a very large number, effectively using all elements.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the Exponential Moving Average (EMA) values. 
        Values outside the calculated range remain NaN.

    Notes:
    -----
    - The EMA is calculated using the formula:
        EMA[i] = α * Value[i] + (1 - α) * EMA[i-1]
      where α (alpha) = 2 / (timeperiod + 1).
    - The function skips NaN values in the input array until the first non-NaN value is encountered.
    - If `normalize` is True, the EMA is adjusted relative to the input values to measure deviation.
    - The initial values before the first non-NaN and before `n_values` are NaN as they cannot be computed.
    - The function is optimized to handle both normalized and unnormalized calculations in a single pass.
    """
    cdef int n = len(input_array), i, first_non_nan = -1
    cdef float alpha = 2.0 / (timeperiod + 1),  ema_value
    cdef np.ndarray[np.float32_t, ndim=1] ema = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        if np.isnan(input_array[i]):
            continue
        else:
            ema_value = input_array[i]
            if i >= n - n_values:
                ema[i] = ema_value
            first_non_nan = i
            break
    if normalize:
        if first_non_nan < n - n_values:
            for i in range(first_non_nan + 1, n - n_values):
                ema_value = alpha * input_array[i] + (1 - alpha) * ema_value
            for i in range(n - n_values, n):
                new = input_array[i]
                ema_value = alpha * new + (1 - alpha) * ema_value
                ema[i] = (ema_value - new) / (new if new!=0 else 1e-9) 
        else:
            for i in range(first_non_nan, n):
                new = input_array[i]
                ema_value = alpha * new + (1 - alpha) * ema_value
                ema[i] = (ema_value - new) / (new if new!=0 else 1e-9) 
    else:
        if first_non_nan < n - n_values:
            for i in range(first_non_nan + 1, n - n_values):
                ema_value = alpha * input_array[i] + (1 - alpha) * ema_value
            for i in range(n - n_values, n):
                ema_value = alpha * input_array[i] + (1 - alpha) * ema_value
                ema[i] = ema_value
        else:
            for i in range(first_non_nan, n):
                ema_value = alpha * input_array[i] + (1 - alpha) * ema_value
                ema[i] = ema_value
    del input_array
    return ema



def ema2(np.ndarray[np.float32_t, ndim=1] input_array = None, int timeperiod = 1, bint input_is_ema1=False, int n_values=999999999):
    """
    Calculate the second-level Exponential Moving Average (EMA2) over a specified time period.

    EMA2 is the exponential moving average (EMA) of the EMA of the input array, providing a smoother representation 
    of the data compared to a single EMA. It is often used in multi-level smoothing techniques.

    Parameters:
    ----------
    input_array : np.ndarray[np.float32_t, ndim=1]
        Input array representing the raw data or the first-level EMA, depending on the `input_is_ema1` flag.

    timeperiod : int
        The lookback period for calculating each level of the EMA.

    input_is_ema1 : bint, optional
        If True, `input_array` is treated as the first-level EMA (ema1), and only the second EMA is calculated. Defaults to False.

    n_values : int, optional
        Number of values from the end of the input array to calculate. Defaults to a very large number, effectively using all elements.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the second-level Exponential Moving Average (EMA2) values. 
        Values outside the calculated range remain NaN.

    Notes:
    -----
    - EMA2 is calculated recursively:
        - If `input_is_ema1` is True, the second EMA is computed directly from the provided input array.
        - If `input_is_ema1` is False, the first EMA is calculated from the input array, and then the second EMA is computed.
    - The exponential moving average applies smoothing, making the data progressively less sensitive to recent changes.
    - Initial values required for each EMA result in NaN values at the start of the array.
    """    
    cdef np.ndarray[np.float32_t, ndim=1] ema1, ema2
    if input_is_ema1:
        ema2 = ema(input_array, timeperiod, normalize=False, n_values=n_values)
    else:
        ema1 = ema(input_array, timeperiod, normalize=True)
        ema2 = ema(ema1, timeperiod, normalize=False, n_values=n_values)
        del ema1
    del input_array
    return ema2



def ema3(np.ndarray[np.float32_t, ndim=1] input_array = None, int timeperiod = 1, bint input_is_ema2=False, input_is_ema1=False, int n_values=999999999):
    """
    Calculate the third-level Exponential Moving Average (EMA3) over a specified time period.

    EMA3 is the exponential moving average (EMA) of the EMA of the EMA of the input array. 
    This multi-level smoothing reduces noise and highlights long-term trends.

    Parameters:
    ----------
    input_array : np.ndarray[np.float32_t, ndim=1]
        Input array representing the raw data or a precomputed lower-level EMA, depending on the flags.

    timeperiod : int
        The lookback period for calculating each level of the EMA.

    input_is_ema2 : bint, optional
        If True, `input_array` is treated as the second-level EMA (ema2). Defaults to False.

    input_is_ema1 : bint, optional
        If True, `input_array` is treated as the first-level EMA (ema1). Defaults to False.

    n_values : int, optional
        Number of values from the end of the input array to calculate. Defaults to a very large number, effectively using all elements.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the third-level Exponential Moving Average (EMA3) values. 
        Values outside the calculated range remain NaN.

    Notes:
    -----
    - The function supports precomputed lower-level EMAs to avoid redundant calculations:
        - If `input_is_ema2` is True, no additional EMAs are calculated, and the third EMA is computed directly.
        - If `input_is_ema1` is True, the second EMA is computed from `input_array`, followed by the third EMA.
        - If neither flag is set, all three levels of EMAs are calculated starting from the raw input array.
    - The exponential moving average is calculated recursively with smoothing applied at each level.
    - Initial values required for each EMA result in NaN values at the start of the array.
    """
    cdef np.ndarray[np.float32_t, ndim=1] ema1, ema2, ema3
    if input_is_ema2:
        ema3 = ema(input_array, timeperiod, normalize=False, n_values=n_values)
    elif input_is_ema1:
        ema2 = ema(input_array, timeperiod, normalize=False)
        ema3 = ema(ema2, timeperiod, normalize=False, n_values=n_values)
        del ema2
    else:
        ema1 = ema(input_array, timeperiod, normalize=True)
        ema2 = ema(ema1, timeperiod, normalize=False)
        del ema1
        ema3 = ema(ema2, timeperiod, normalize=False, n_values=n_values)
        del ema2
    del input_array
    return ema3



def dema(np.ndarray[np.float32_t, ndim=1] input1 = None, np.ndarray[np.float32_t, ndim=1] input2=None, int timeperiod=1, bint input1_is_ema1=False, bint input2_is_ema2=False, int n_values = 999999999):
    """
    Calculate the Double Exponential Moving Average (DEMA) over a specified time period.

    The DEMA is a smoother and more responsive trend-following indicator compared to a 
    simple EMA. It uses a combination of a single EMA and its EMA to reduce lag.

    Parameters:
    ----------
    input1 : np.ndarray[np.float32_t, ndim=1]
        Input array representing the raw data or the first EMA, depending on the flags.

    input2 : np.ndarray[np.float32_t, ndim=1], optional
        Input array representing the second EMA, if `input2_is_ema2` is True. Defaults to None.

    timeperiod : int, optional
        The lookback period for calculating the exponential moving averages (EMAs). Defaults to 1.

    input1_is_ema1 : bint, optional
        If True, `input1` is treated as the first EMA (ema1). Defaults to False.

    input2_is_ema2 : bint, optional
        If True, `input2` is treated as the second EMA (ema2). Defaults to False.

    n_values : int, optional
        Number of values from the end of the input arrays to calculate. Defaults to a very large number, effectively using all elements.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the Double Exponential Moving Average (DEMA) values. 
        Values outside the calculated range remain NaN.

    Notes:
    -----
    - DEMA is calculated as:
        DEMA = 2 * EMA1 - EMA2
      where EMA1 is the single exponential moving average, and EMA2 is the EMA of EMA1.
    - The function provides flexibility:
        - If `input1_is_ema1` and `input2_is_ema2` are True, no additional EMA calculations are performed.
        - If only `input1_is_ema1` is True, the second EMA is calculated using `ema`.
        - If neither flag is set, both EMAs are calculated starting from `input1`.
    - Initial values required for each EMA calculation result in NaN values at the start of the array.
    """    
    cdef int n = len(input1), i
    cdef int k = max(0, n - n_values)
    cdef np.ndarray[np.float32_t, ndim=1] dema = np.full(n, np.nan, dtype=np.float32), ema1, ema2
    if input1_is_ema1 and input2_is_ema2:
        ema1 = input1
        ema2 = input2
    elif input1_is_ema1 and not input2_is_ema2:
        ema1 = input1
        ema2 = ema(ema1, timeperiod, normalize=False, n_values=n_values)
    elif not input1_is_ema1 and input2_is_ema2:
        ema1 = ema(input1, timeperiod, normalize=True, n_values=n_values)
        ema2 = input2
    else:
        ema1 = ema(input1, timeperiod, normalize=True)
        ema2 = ema(ema1, timeperiod, normalize=False, n_values=n_values)
    for i in range(k, n):
        dema[i] = 2 * ema1[i] - ema2[i]
    del input1, input2, ema1, ema2
    return dema


def tema(np.ndarray[np.float32_t, ndim=1] input1 = None, np.ndarray[np.float32_t, ndim=1] input2=None, np.ndarray[np.float32_t, ndim=1] input3=None, int timeperiod = 1, bint input3_is_ema3=False, bint input2_is_ema2=False, bint input1_is_ema1=False, int n_values=999999999):
    """
    Calculate the Triple Exponential Moving Average (TEMA) over a specified time period.

    The TEMA is a trend-following indicator that reduces lag by combining a single EMA, 
    double EMA, and triple EMA. It helps smooth price movements while being more responsive 
    to recent price changes.

    Parameters:
    ----------
    input1 : np.ndarray[np.float32_t, ndim=1]
        Input array representing the raw data or the first EMA, depending on the flags.

    input2 : np.ndarray[np.float32_t, ndim=1], optional
        Input array representing the second EMA, if `input2_is_ema2` is True. Defaults to None.

    input3 : np.ndarray[np.float32_t, ndim=1], optional
        Input array representing the third EMA, if `input3_is_ema3` is True. Defaults to None.

    timeperiod : int, optional
        The lookback period for calculating the exponential moving averages (EMAs). Defaults to 1.

    input3_is_ema3 : bint, optional
        If True, `input3` is treated as the third EMA (ema3). Defaults to False.

    input2_is_ema2 : bint, optional
        If True, `input2` is treated as the second EMA (ema2). Defaults to False.

    input1_is_ema1 : bint, optional
        If True, `input1` is treated as the first EMA (ema1). Defaults to False.

    n_values : int, optional
        Number of values from the end of the input arrays to calculate. Defaults to a very large number, effectively using all elements.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the Triple Exponential Moving Average (TEMA) values. 
        Values outside the calculated range remain NaN.

    Notes:
    -----
    - TEMA is calculated as:
        TEMA = 3 * EMA1 - 3 * EMA2 + EMA3
      where EMA1 is the single exponential moving average, EMA2 is the EMA of EMA1, 
      and EMA3 is the EMA of EMA2.
    - The function allows for precomputed EMAs to be passed as inputs via the flags `input1_is_ema1`, 
      `input2_is_ema2`, and `input3_is_ema3`. Depending on the flags, unnecessary EMA calculations 
      are skipped to save computational resources.
    - If no flags are set, the function calculates all three EMAs starting from `input1`.
    - Initial values required for each EMA calculation result in NaN values at the start of the array.
    """    
    cdef int n = len(input1), i
    cdef int k = max(0, n - n_values)
    cdef np.ndarray[np.float32_t, ndim=1] tema = np.full(n, np.nan, dtype=np.float32), ema1, ema2, ema3
    if input1_is_ema1 and input2_is_ema2 and input3_is_ema3:
        ema1 = input1
        ema2 = input2
        ema3 = input3
    elif input1_is_ema1 and input2_is_ema2 and not input3_is_ema3:
        ema1 = input1
        ema2 = input2
        ema3 = ema(ema2, timeperiod, normalize=False, n_values=n_values)
    elif input1_is_ema1 and not input2_is_ema2 and input3_is_ema3:
        ema1 = input1
        ema2 = ema(ema1, timeperiod, normalize=False, n_values=n_values)
        ema3 = input3
    elif not input1_is_ema1 and input2_is_ema2 and input3_is_ema3:
        ema1 = ema(input1, timeperiod, normalize=True, n_values=n_values)
        ema2 = input2
        ema3 = input3
    elif input1_is_ema1 and not input2_is_ema2 and not input3_is_ema3:
        ema1 = input1
        ema2 = ema(ema1, timeperiod, normalize=False)
        ema3 = ema(ema2, timeperiod, normalize=False, n_values=n_values)
    elif not input1_is_ema1 and input2_is_ema2 and not input3_is_ema3:
        ema1 = ema(input1, timeperiod, normalize=True, n_values=n_values)
        ema2 = input2
        ema3 = ema(ema2, timeperiod, normalize=False, n_values=n_values)
    elif not input1_is_ema1 and not input2_is_ema2 and input3_is_ema3:
        ema1 = ema(input1, timeperiod, normalize=True)
        ema2 = ema(ema1, timeperiod, normalize=False, n_values=n_values)
        ema3 = input3
    else:
        ema1 = ema(input1, timeperiod, normalize=True)
        ema2 = ema(ema1, timeperiod, normalize=False)
        ema3 = ema(ema2, timeperiod, normalize=False, n_values=n_values)
    for i in range(k, n):
        tema[i] = 3 * ema1[i] - 3 * ema2[i] + ema3[i]
    del ema1, ema2, ema3, input1, input2, input3
    return tema


  
def trix(np.ndarray[np.float32_t, ndim=1] arr = None, int timeperiod = 1, bint input_is_ema3 = False, bint input_is_ema2=False, bint input_is_ema1=False, int n_values = 999999999):
    """
    Calculate the Triple Exponential Moving Average Oscillator (TRIX) over a specified time period.

    The TRIX is a momentum oscillator that measures the rate of change of the third exponential 
    moving average (EMA) of a time series. It is used to identify trends and possible reversals.

    Parameters:
    ----------
    arr : np.ndarray[np.float32_t, ndim=1]
        Array of input values (e.g., closing prices) for the calculation.

    timeperiod : int, optional
        The lookback period for calculating the exponential moving averages (EMAs). Defaults to 1.

    input_is_ema3 : bint, optional
        If True, the input `arr` is treated as the third EMA (ema3). Defaults to False.

    input_is_ema2 : bint, optional
        If True, the input `arr` is treated as the second EMA (ema2). Defaults to False.

    input_is_ema1 : bint, optional
        If True, the input `arr` is treated as the first EMA (ema1). Defaults to False.

    n_values : int, optional
        Number of values from the end of the input array to calculate. Defaults to a very large number, effectively using all elements.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the TRIX values. Values outside the calculated range remain NaN.

    Notes:
    -----
    - The TRIX is calculated as:
        TRIX[i] = (EMA3[i] - EMA3[i-1]) / EMA3[i-1]
      where EMA3 is the third exponential moving average (of the EMA of the EMA) of the input values.
    - If `EMA3[i-1]` equals zero, a small epsilon (1e-9) is added to the denominator to prevent division by zero.
    - The function allows for flexibility:
        - If `input_is_ema3` is True, no additional EMA calculations are performed, and `arr` is used directly as `ema3`.
        - If `input_is_ema2` is True, only one additional EMA is calculated to derive `ema3`.
        - If `input_is_ema1` is True, two additional EMAs are calculated to derive `ema3`.
        - If none of the flags are set, three levels of EMA are calculated starting from the input array.
    - The initial `timeperiod` values and the starting values required for each EMA remain NaN due to insufficient data for calculation.
    """    
    cdef int n = arr.shape[0]
    cdef int k = max(1, n - n_values)
    cdef np.ndarray[np.float32_t, ndim=1] trix = np.full(n, np.nan, dtype=np.float32), ema1, ema2, ema3
    if input_is_ema3:
        ema3 = arr
    elif input_is_ema2:
        ema3 = ema(arr, timeperiod, normalize=False, n_values=n_values+1)
    elif input_is_ema1:
        ema2 = ema(arr, timeperiod, normalize=False)
        ema3 = ema(ema2, timeperiod, normalize=False, n_values=n_values+1)
        del ema2
    else:
        ema1 = ema(arr, timeperiod, normalize=True)
        ema2 = ema(ema1, timeperiod, normalize=False)
        del ema1
        ema3 = ema(ema2, timeperiod, normalize=False, n_values=n_values+1)
        del ema2
    for i in range(k, n):
        trix[i] = (ema3[i] - ema3[i - 1]) / (ema3[i - 1] + (0 if ema3[i-1]!=0 else 1e-9))
    del ema3, arr
    return trix


def deriv(np.ndarray[np.float32_t, ndim=1] input_array,
          int window_size=7,
          int polyorder=2,
          int deriv_order=1):

    cdef int n = len(input_array)
    cdef np.ndarray[np.float32_t, ndim=1] result = np.full(n, np.nan, dtype=np.float32)
    cdef int i, j, k

    # Ensure constraints
    if window_size % 2 == 0 or polyorder < deriv_order or window_size <= polyorder:
        raise ValueError("Invalid window_size or polyorder")

    # Build the Vandermonde matrix X for each window [0, 1, ..., window_size - 1]
    cdef int W = window_size
    cdef np.ndarray[np.float64_t, ndim=2] X = np.zeros((W, polyorder + 1), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] XtX_inv = np.zeros((polyorder + 1, polyorder + 1), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] Xt = np.zeros((polyorder + 1, W), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] coeffs = np.zeros(polyorder + 1, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] window = np.zeros(W, dtype=np.float64)

    for i in range(W):
        for j in range(polyorder + 1):
            X[i, j] = pow(i, j)

    # Precompute (XᵀX)⁻¹ Xᵀ
    Xt = X.T.copy()
    XtX_inv = np.linalg.pinv(X.T @ X) @ Xt  # shape: (polyorder+1, window_size)

    # Compute factorial for scaling derivative
    cdef double factorial = 1.0
    for i in range(1, deriv_order + 1):
        factorial *= i

    # Final convolution kernel: derivative_coeffs[i] = coeff of i-th input in the window
    cdef np.ndarray[np.float64_t, ndim=1] deriv_kernel = np.zeros(W, dtype=np.float64)
    for k in range(W):
        deriv_kernel[k] = XtX_inv[deriv_order, k] * factorial

    # Apply filter over input_array
    for i in range(W - 1, n):
        for j in range(W):
            window[j] = input_array[i - W + 1 + j]
        if np.isnan(window).any():
            continue
        result[i] = np.dot(deriv_kernel, window)

    return result.astype(np.float32)



def roc(np.ndarray[np.float32_t, ndim=1] input_array, int timeperiod=1, int n_values=999999999):
    cdef Py_ssize_t n = input_array.shape[0]
    cdef Py_ssize_t k = max(n - n_values, timeperiod)
    cdef Py_ssize_t i

    cdef np.ndarray[np.float32_t, ndim=1] out = np.full(n, np.nan, dtype=np.float32)

    for i in range(k, n):
        if input_array[i - timeperiod] != 0:
            out[i] = (input_array[i] - input_array[i - timeperiod]) / input_array[i - timeperiod]
        # else: stays NaN

    return out

 

def rsi(np.ndarray[np.float32_t, ndim=1] close = None, int timeperiod = 1, int n_values = 999999999):
    """
    Calculate the Relative Strength Index (RSI) over a specified time period.

    The RSI is a momentum oscillator that measures the speed and change of price movements. 
    It oscillates between 0 and 100 and is used to identify overbought or oversold conditions.

    Parameters:
    ----------
    close : np.ndarray[np.float32_t, ndim=1]
        Array of closing prices for each time step.

    timeperiod : int
        The lookback period for calculating the RSI.

    n_values : int, optional
        Number of values from the end of the input array to calculate. Defaults to a very large number, effectively using all elements.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the Relative Strength Index (RSI) values. 
        Values outside the calculated range remain NaN.

    Notes:
    -----
    - RSI is calculated using the following formula:
        RSI = 100 - (100 / (1 + RS))
      where RS (Relative Strength) is the average gain divided by the average loss.
    - Average gains and losses are calculated using an exponential moving average 
      to ensure smoothness:
        Avg_Gain = (Previous_Avg_Gain * (timeperiod - 1) + Current_Gain) / timeperiod
        Avg_Loss = (Previous_Avg_Loss * (timeperiod - 1) + Current_Loss) / timeperiod
    - RSI is undefined (set to 100) if the average loss is zero to avoid division errors.
    - The initial `timeperiod` values remain NaN since there is insufficient data to compute RSI.
    """
    cdef int n = len(close), i
    cdef int k = max(timeperiod + 1, n - n_values)
    cdef np.ndarray[np.float32_t, ndim=1] rsi = np.full(n, np.nan, dtype=np.float32)
    cdef float total_gain = 0.0, total_loss = 0.0, diff, gain, loss
    for i in range(1, timeperiod + 1):
        diff = close[i] - close[i - 1]
        total_gain += diff if diff > 0 else 0
        total_loss += -diff if diff < 0 else 0
    for i in range(timeperiod + 1, k):
        diff = close[i] - close[i - 1]
        gain = diff if diff > 0 else 0.0
        loss = -diff if diff < 0 else 0.0
        total_gain += -(total_gain / timeperiod) + gain
        total_loss += -(total_loss / timeperiod) + loss        
    for i in range(k, n):
        diff = close[i] - close[i - 1]
        gain = diff if diff > 0 else 0.0
        loss = -diff if diff < 0 else 0.0
        total_gain += -(total_gain / timeperiod) + gain
        total_loss += -(total_loss / timeperiod) + loss
        rsi[i] = 100 if total_loss == 0 else 100 - (100 / (1 + (total_gain / total_loss)))
    del close
    return rsi



def willr(np.ndarray[np.float32_t, ndim=1] high_arr = None, np.ndarray[np.float32_t, ndim=1] low_arr = None, np.ndarray[np.float32_t, ndim=1] close_arr = None, int timeperiod = 1, int n_values = 999999999):
    """
    Calculate the Williams %R (WILLR) over a specified time period.

    The Williams %R is a momentum indicator that measures overbought or oversold levels 
    by comparing the closing price to the highest high and lowest low over the lookback period.

    Parameters:
    ----------
    high_arr : np.ndarray[np.float32_t, ndim=1]
        Array of high prices for each time step.

    low_arr : np.ndarray[np.float32_t, ndim=1]
        Array of low prices for each time step.

    close_arr : np.ndarray[np.float32_t, ndim=1]
        Array of closing prices for each time step.

    timeperiod : int
        The lookback period for calculating the Williams %R.

    n_values : int, optional
        Number of values from the end of the input arrays to calculate. Defaults to a very large number, effectively using all elements.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the Williams %R values. Values outside the calculated range remain NaN.

    Notes:
    -----
    - Williams %R is calculated as:
        %R = ((Highest High - Close) / (Highest High - Lowest Low))
      where `Highest High` and `Lowest Low` are the maximum and minimum values 
      of the high and low prices over the `timeperiod`.
    - The function uses a deque-like structure for efficiently tracking the highest high 
      and lowest low over the sliding window.
    - If the `Highest High` is equal to the `Lowest Low`, the %R value is not defined 
      and remains NaN.
    - Initial values, corresponding to the lookback period, remain NaN since 
      sufficient data is not available for calculation.
    """
    cdef int n = len(close_arr), i, high_start = 0, high_end = 0, low_start = 0, low_end = 0
    cdef int k = max(timeperiod - 1, n - n_values)
    cdef np.ndarray[np.float32_t, ndim=1] willr = np.full(n, np.nan, dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] high_indices = np.zeros(n, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] low_indices = np.zeros(n, dtype=np.int32)
    cdef float highest_high, lowest_low
    for i in range(k - timeperiod + 1, k): 
        while high_end > high_start and high_arr[high_indices[high_end - 1]] <= high_arr[i]:
            high_end -= 1
        high_indices[high_end] = i
        high_end += 1
        while low_end > low_start and low_arr[low_indices[low_end - 1]] >= low_arr[i]:
            low_end -= 1
        low_indices[low_end] = i
        low_end += 1
    for i in range(k, n):
        if high_end > high_start and high_indices[high_start] <= i - timeperiod:
            high_start += 1
        if low_end > low_start and low_indices[low_start] <= i - timeperiod:
            low_start += 1
        while high_end > high_start and high_arr[high_indices[high_end - 1]] <= high_arr[i]:
            high_end -= 1
        high_indices[high_end] = i
        high_end += 1
        while low_end > low_start and low_arr[low_indices[low_end - 1]] >= low_arr[i]:
            low_end -= 1
        low_indices[low_end] = i
        low_end += 1
        high_start = max(0, min(high_start, high_end - 1))
        low_start = max(0, min(low_start, low_end - 1))
        highest_high = high_arr[high_indices[high_start]]
        lowest_low = low_arr[low_indices[low_start]]
        if highest_high != lowest_low:
            willr[i] = ((highest_high - close_arr[i]) / (highest_high - lowest_low))
    del high_arr, low_arr, close_arr, high_indices, low_indices
    return willr



def cci(np.ndarray[np.float32_t, ndim=1] high_arr = None, np.ndarray[np.float32_t, ndim=1] low_arr = None, np.ndarray[np.float32_t, ndim=1] close_arr = None, int timeperiod = 1, int n_values = 999999999):
    """
    Calculate the Commodity Channel Index (CCI) using a mean squared deviation for faster execution.

    The CCI measures the variation of a security's price from its average price over a specified 
    time period. It identifies overbought or oversold conditions relative to historical average prices.

    This implementation uses mean squared deviation instead of mean absolute deviation to improve 
    computational efficiency.

    Parameters:
    ----------
    high_arr : np.ndarray[np.float32_t, ndim=1]
        Array of high prices for each time step.

    low_arr : np.ndarray[np.float32_t, ndim=1]
        Array of low prices for each time step.

    close_arr : np.ndarray[np.float32_t, ndim=1]
        Array of closing prices for each time step.

    timeperiod : int
        The lookback period for calculating the CCI.

    n_values : int, optional
        Number of values from the end of the input arrays to calculate. Defaults to a very large number, effectively using all elements.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the Commodity Channel Index (CCI) values. 
        Values outside the calculated range remain NaN.

    Notes:
    -----
    - Typical Price (TP) is calculated as the average of the high, low, and close prices:
        TP = (High + Low + Close) / 3
    - The CCI formula is:
        CCI = (TP - SMA(TP)) / (0.015 * Mean Squared Deviation)
      where SMA(TP) is the simple moving average of the Typical Price, 
      and Mean Squared Deviation is used instead of Mean Absolute Deviation for optimization.
    - A `timeperiod` offset is required for valid CCI calculations, so initial values will remain NaN.
    - If the Mean Squared Deviation is zero, the CCI is set to 0 to avoid division errors.
    """
    cdef int n = len(close_arr), i, j
    cdef int k = max(timeperiod, n - n_values)
    cdef np.ndarray[np.float32_t, ndim=1] cci = np.full(n, np.nan, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] typical_price = (high_arr + low_arr + close_arr) / 3
    cdef float sum_tp = 0.0, sum_tp_squared = 0.0, old_avg_tp, new_avg_tp, old_tp, new_tp, mean_squared_deviation
    for j in range(k - timeperiod, k):
        sum_tp += typical_price[j]
        sum_tp_squared += typical_price[j] ** 2
    if k - 1 >= n - n_values:
        old_avg_tp = sum_tp / timeperiod
        mean_squared_deviation = (sum_tp_squared / timeperiod) - (old_avg_tp ** 2)
        cci[k - 1] = (typical_price[timeperiod - 1] - old_avg_tp) / (0.015 * np.sqrt(mean_squared_deviation)) if mean_squared_deviation != 0 else 0
    for i in range(k, n):
        old_tp = typical_price[i - timeperiod]
        new_tp = typical_price[i]
        sum_tp += new_tp - old_tp
        sum_tp_squared += new_tp ** 2 - old_tp ** 2
        new_avg_tp = sum_tp / timeperiod
        mean_squared_deviation = (sum_tp_squared / timeperiod) - (new_avg_tp ** 2)
        cci[i] = (new_tp - new_avg_tp) / (0.015 * np.sqrt(mean_squared_deviation)) if mean_squared_deviation != 0 else 0
        old_avg_tp = new_avg_tp
    del typical_price, high_arr, low_arr, close_arr
    return cci


def dx(np.ndarray[np.float32_t, ndim=1] high = None, np.ndarray[np.float32_t, ndim=1] low = None, np.ndarray[np.float32_t, ndim=1] close = None, int timeperiod = 1, int n_values = 999999999):
    """
    Calculate the Directional Movement Index (DX) over a specified time period.

    The DX is a component of the Average Directional Index (ADX) and measures the 
    difference between positive and negative directional movement as a proportion 
    of the True Range (TR). It indicates the strength of a directional trend.

    Parameters:
    ----------
    high : np.ndarray[np.float32_t, ndim=1]
        Array of high prices for each time step.

    low : np.ndarray[np.float32_t, ndim=1]
        Array of low prices for each time step.

    close : np.ndarray[np.float32_t, ndim=1]
        Array of closing prices for each time step.

    timeperiod : int
        The lookback period for calculating the DX.

    n_values : int, optional
        Number of values from the end of the input arrays to calculate. Defaults to a very large number, effectively using all elements.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the Directional Movement Index (DX) values. 
        Values outside the calculated range remain NaN.

    Notes:
    -----
    - The True Range (TR) for each period is the maximum of:
        1. The difference between the current high and low.
        2. The absolute difference between the current high and the previous close.
        3. The absolute difference between the current low and the previous close.
    - Positive Directional Movement (+DM) is calculated when the current high 
      minus the previous high is greater than the previous low minus the current low.
    - Negative Directional Movement (-DM) is calculated when the previous low 
      minus the current low is greater than the current high minus the previous high.
    - Both +DM and -DM are smoothed using an exponential moving average over the `timeperiod`.
    - DX is computed as:
        DX = abs((+DM / TR) - (-DM / TR)) / ((+DM / TR) + (-DM / TR))
    - A `timeperiod` offset is required for valid DX calculations, so initial values will remain NaN.
    """
    cdef int n = len(high), i
    cdef np.ndarray[np.float32_t, ndim=1] dx = np.full(n, np.nan, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] plus_dm = np.zeros(n, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] minus_dm = np.zeros(n, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] atr = np.zeros(n, dtype=np.float32)
    cdef float tr, plus_dm_sum = 0.0, minus_dm_sum = 0.0, atr_sum = 0.0
    for i in range(1, timeperiod + 1):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr[i] = tr
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
        plus_dm_sum += plus_dm[i]
        minus_dm_sum += minus_dm[i]
        atr_sum += atr[i]
    plus_dm_sum /= timeperiod
    minus_dm_sum /= timeperiod
    atr_sum /= timeperiod
    for i in range(timeperiod + 1, max(timeperiod + 1, n - n_values)):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr[i] = tr
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
        atr_sum = (atr_sum * (timeperiod - 1) + atr[i]) / timeperiod
        plus_dm_sum = (plus_dm_sum * (timeperiod - 1) + plus_dm[i]) / timeperiod
        minus_dm_sum = (minus_dm_sum * (timeperiod - 1) + minus_dm[i]) / timeperiod
    for i in range(max(timeperiod + 1, n - n_values), n):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr[i] = tr
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
        atr_sum = (atr_sum * (timeperiod - 1) + atr[i]) / timeperiod
        plus_dm_sum = (plus_dm_sum * (timeperiod - 1) + plus_dm[i]) / timeperiod
        minus_dm_sum = (minus_dm_sum * (timeperiod - 1) + minus_dm[i]) / timeperiod
        dx[i] = abs((plus_dm_sum / atr_sum) - (minus_dm_sum / atr_sum)) / ((plus_dm_sum / atr_sum) + (minus_dm_sum / atr_sum))
    del plus_dm, minus_dm, atr, high, low, close
    return dx

 
def adx(np.ndarray[np.float32_t, ndim=1] high_arr = None, np.ndarray[np.float32_t, ndim=1] low_arr = None, np.ndarray[np.float32_t, ndim=1] close_arr = None, int timeperiod = 1, int n_values = 999999999, bint close_arr_is_dx=False):
    """
    Calculate the Average Directional Index (ADX) over a specified time period.

    The ADX is a trend strength indicator that measures the degree of price movement, 
    regardless of direction. It is calculated as a smoothed average of the Directional Movement Index (DX).

    Parameters:
    ----------
    high_arr : np.ndarray[np.float32_t, ndim=1], optional
        Array of high prices for each time step. Required unless `close_arr_is_dx` is True.

    low_arr : np.ndarray[np.float32_t, ndim=1], optional
        Array of low prices for each time step. Required unless `close_arr_is_dx` is True.

    close_arr : np.ndarray[np.float32_t, ndim=1], optional
        Array of closing prices for each time step, or precomputed DX values if `close_arr_is_dx` is True.

    timeperiod : int, optional
        The lookback period for calculating the ADX. Defaults to 1.

    n_values : int, optional
        Number of values from the end of the input arrays to calculate. Defaults to a very large number, effectively using all elements.

    close_arr_is_dx : bint, optional
        If True, the `close_arr` input is treated as precomputed DX values. 
        If False, the DX will be calculated internally using `dx`.
        Defaults to False.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the Average Directional Index (ADX) values. 
        Values outside the calculated range remain NaN.

    Notes:
    -----
    - The DX (Directional Movement Index) measures the percentage difference between 
      the positive and negative directional movements.
    - The ADX is a smoothed average of DX values, calculated using an exponential moving average.
    - If `close_arr_is_dx` is False, the DX values are calculated internally using the `dx` function.
    - The initial `2 * timeperiod` values are required for the calculation, so values before this offset remain NaN.
    """
    cdef int n = len(close_arr), i
    k = max(2 * timeperiod, n - n_values)
    cdef np.ndarray[np.float32_t, ndim=1] adx = np.full(n, np.nan, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] dx_arr = close_arr if close_arr_is_dx else dx(high_arr, low_arr, close_arr, timeperiod)
    cdef float adx_value    


    ### this line is causing an error
    adx_value = np.nanmean(dx_arr[timeperiod:2 * timeperiod])



    for i in range(2 * timeperiod, k):
        adx_value = ((adx_value * (timeperiod - 1)) + dx_arr[i]) / timeperiod
    for i in range(k, n):
        adx_value = ((adx_value * (timeperiod - 1)) + dx_arr[i]) / timeperiod
        adx[i] = adx_value
    del dx_arr, high_arr, low_arr, close_arr
    return adx

 
def adxr(np.ndarray[np.float32_t, ndim=1] high_arr = None, np.ndarray[np.float32_t, ndim=1] low_arr = None, np.ndarray[np.float32_t, ndim=1] close_arr = None, int timeperiod = 1, int n_values = 999999999, bint close_arr_is_adx=False):
    """
    Calculate the Average Directional Movement Rating (ADXR) over a specified time period.

    ADXR is a smoothed version of the Average Directional Index (ADX). It is calculated 
    as the simple average of the ADX value at the current time step and the ADX value 
    `timeperiod` steps earlier. This smoothing reduces volatility in the ADX values.

    Parameters:
    ----------
    high_arr : np.ndarray[np.float32_t, ndim=1]
        Array of high prices for each time step.

    low_arr : np.ndarray[np.float32_t, ndim=1]
        Array of low prices for each time step.

    close_arr : np.ndarray[np.float32_t, ndim=1]
        Array of closing prices for each time step, or precomputed ADX values if `close_arr_is_adx` is True.

    timeperiod : int
        The lookback period for calculating the ADXR.

    n_values : int, optional
        Number of values from the end of the input arrays to calculate. Defaults to a very large number, effectively using all elements.

    close_arr_is_adx : bint, optional
        If True, the `close_arr` input is treated as precomputed ADX values. 
        If False, the ADX will be computed internally using `adx`.
        Defaults to False.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the Average Directional Movement Rating (ADXR) values. 
        Values outside the calculated range remain NaN.

    Notes:
    -----
    - ADXR is calculated as:
        ADXR[i] = (ADX[i] + ADX[i - timeperiod]) / 2
      where ADX is the Average Directional Index.
    - If `close_arr_is_adx` is False, the ADX values are calculated internally using the `adx` function.
    - A `timeperiod` offset is required for valid ADXR calculations, so initial values will remain NaN.
    """
    cdef int n = len(close_arr), i
    cdef int k = max(timeperiod, n - n_values)
    cdef np.ndarray[np.float32_t, ndim=1] adx_array = close_arr if close_arr_is_adx else adx(high_arr, low_arr, close_arr, timeperiod)
    cdef np.ndarray[np.float32_t, ndim=1] adxr = np.full(n, np.nan, dtype=np.float32)
    for i in range(k - timeperiod, n - timeperiod):
        adxr[i + timeperiod] = (adx_array[i] + adx_array[i + timeperiod]) / 2
    del adx_array, high_arr, low_arr, close_arr
    return adxr


def cmo(np.ndarray[np.float32_t, ndim=1] close_arr = None, int timeperiod = 1, int n_values = 999999999):
    """
    Calculate the Chande Momentum Oscillator (CMO) over a specified time period.

    The Chande Momentum Oscillator measures momentum by comparing the sum of recent 
    gains to the sum of recent losses over a lookback period. It oscillates between -1 and 1.

    The formula is:
        CMO = (Sum of Gains - Sum of Losses) / (Sum of Gains + Sum of Losses)

    Parameters:
    ----------
    close_arr : np.ndarray[np.float32_t, ndim=1]
        Array of closing prices for each time step.

    timeperiod : int
        The lookback period for calculating the CMO.

    n_values : int, optional
        Number of values from the end of the input array to calculate. Defaults to a very large number, effectively using all elements.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the Chande Momentum Oscillator (CMO) values. 
        Values outside the calculated range remain NaN.

    Notes:
    -----
    - Gains are defined as positive changes in closing prices compared to the previous period.
    - Losses are defined as negative changes in closing prices (converted to positive values).
    - The CMO value is zero when the sum of gains and losses is zero to avoid division errors.
    - A sliding window approach ensures efficient calculation by updating the gains and losses incrementally.
    """
    cdef int n = len(close_arr), i
    cdef int k = max(n - n_values, timeperiod)
    cdef np.ndarray[np.float32_t, ndim=1] cmo = np.full(n, np.nan, dtype=np.float32)
    cdef float sum_gains = 0.0, sum_losses = 0.0, change
    for i in range(k + 1 - timeperiod, k + 1):
        change = close_arr[i] - close_arr[i - 1]
        if change >= 0:
            sum_gains += change
        elif change < 0:
            sum_losses -= change
    cmo[k] = 0 if (sum_gains + sum_losses == 0) else (sum_gains - sum_losses) / (sum_gains + sum_losses)
    for i in range(k + 1, n):
        new_change = close_arr[i] - close_arr[i - 1]
        old_change = close_arr[i - timeperiod] - close_arr[i - timeperiod - 1]
        if new_change >= 0:
            sum_gains += new_change
        elif new_change < 0:
            sum_losses -= new_change
        if old_change >= 0:
            sum_gains -= old_change
        elif old_change < 0:
            sum_losses += old_change
        cmo[i] = 0 if (sum_gains + sum_losses == 0) else (sum_gains - sum_losses) / (sum_gains + sum_losses)
    del close_arr
    return cmo


def natr(np.ndarray[np.float32_t, ndim=1] high_arr = None, np.ndarray[np.float32_t, ndim=1] low_arr = None, np.ndarray[np.float32_t, ndim=1] close_arr = None, int timeperiod = 1, int n_values = 999999999):
    """
    Calculate the Normalized Average True Range (NATR) over a specified time period.

    NATR is a volatility indicator that normalizes the Average True Range (ATR) 
    by dividing it by the closing price. It provides a percentage-based measure 
    of volatility to compare across assets with different price ranges.

    Parameters:
    ----------
    high_arr : np.ndarray[np.float32_t, ndim=1]
        Array of high prices for each time step.

    low_arr : np.ndarray[np.float32_t, ndim=1]
        Array of low prices for each time step.

    close_arr : np.ndarray[np.float32_t, ndim=1]
        Array of closing prices for each time step.

    timeperiod : int
        The lookback period for calculating the Average True Range (ATR).

    n_values : int, optional
        Number of values from the end of the input arrays to calculate. Defaults to a very large number, effectively using all elements.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the Normalized Average True Range (NATR) values. 
        Values outside the calculated range remain NaN.

    Notes:
    -----
    - The True Range (TR) for each period is the maximum of:
        1. The difference between the current high and low.
        2. The absolute difference between the current high and the previous close.
        3. The absolute difference between the current low and the previous close.
    - The ATR is calculated using a rolling exponential average of the True Range values.
    - The NATR is derived by normalizing the ATR using the current closing price.
    """
    cdef int n = len(close_arr), i
    cdef int k = max(timeperiod, n - n_values)
    cdef np.ndarray[np.float32_t, ndim=1] tr = np.zeros(n, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] atr = np.full(n, np.nan, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] natr = np.full(n, np.nan, dtype=np.float32)
    cdef float atr_value = 0.0
    for i in range(1, timeperiod):
        tr[i] = max(high_arr[i] - low_arr[i], abs(high_arr[i] - close_arr[i - 1]), abs(low_arr[i] - close_arr[i - 1]))
    atr_value = np.nanmean(tr[:timeperiod])
    atr[timeperiod - 1] = atr_value
    for i in range(timeperiod, k):
        tr[i] = max(high_arr[i] - low_arr[i], abs(high_arr[i] - close_arr[i - 1]), abs(low_arr[i] - close_arr[i - 1]))
        atr_value = (atr_value * (timeperiod - 1) + tr[i]) / timeperiod
    for i in range(k, n):
        tr[i] = max(high_arr[i] - low_arr[i], abs(high_arr[i] - close_arr[i - 1]), abs(low_arr[i] - close_arr[i - 1]))
        atr_value = (atr_value * (timeperiod - 1) + tr[i]) / timeperiod
        atr[i] = atr_value
        natr[i] = atr[i] / close_arr[i]
    del tr, atr, high_arr, low_arr, close_arr
    return natr

 
def ult(np.ndarray[np.float32_t, ndim=1] high_arr = None, np.ndarray[np.float32_t, ndim=1] low_arr = None, np.ndarray[np.float32_t, ndim=1] close_arr = None, int short_period = 1, int medium_period = 2, int long_period = 4, int n_values = 999999999):
    """
    Calculate the Ultimate Oscillator (UO) over three specified time periods.

    The Ultimate Oscillator is a momentum indicator that combines short-term, medium-term, 
    and long-term price movements to reduce the effect of false signals. It measures the 
    average of buying pressure (BP) relative to the true range (TR) across multiple periods.

    Parameters:
    ----------
    high_arr : np.ndarray[np.float32_t, ndim=1]
        Array of high prices for each time step.

    low_arr : np.ndarray[np.float32_t, ndim=1]
        Array of low prices for each time step.

    close_arr : np.ndarray[np.float32_t, ndim=1]
        Array of closing prices for each time step.

    short_period : int
        The short-term lookback period for calculating the Ultimate Oscillator.

    medium_period : int
        The medium-term lookback period for calculating the Ultimate Oscillator.

    long_period : int
        The long-term lookback period for calculating the Ultimate Oscillator.

    n_values : int, optional
        Number of values from the end of the input arrays to calculate. Defaults to a very large number, effectively using all elements.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the Ultimate Oscillator values. Values outside the calculated range remain NaN.

    Notes:
    -----
    - Buying Pressure (BP) is defined as the difference between the current close price and 
      the minimum of the current low and the previous close.
    - True Range (TR) is defined as the difference between the maximum of the current high 
      and the previous close, and the minimum of the current low and the previous close.
    - The Ultimate Oscillator combines the weighted average of BP/TR ratios for the short, 
      medium, and long time periods.
    - A small epsilon (1e-9) is added to the denominator if the True Range (TR) is zero to prevent division errors.
    """
    cdef int n = len(close_arr), i
    cdef int k = max(long_period, n - n_values)
    cdef np.ndarray[np.float32_t, ndim=1] uo = np.full(n, np.nan, dtype=np.float32)
    cdef float sum_bp_short = 0.0, sum_tr_short = 0.0, sum_bp_medium = 0.0, sum_tr_medium = 0.0, sum_bp_long = 0.0, sum_tr_long = 0.0, eps = 1e-9
    cdef float bp, tr
    for i in range(k - long_period, k):
        bp = close_arr[i] - min(low_arr[i], close_arr[i - 1])
        tr = max(high_arr[i], close_arr[i - 1]) - min(low_arr[i], close_arr[i - 1])
        if k - short_period <= i:
            sum_bp_short += bp
            sum_tr_short += tr
        if k - medium_period <= i:
            sum_bp_medium += bp
            sum_tr_medium += tr
        sum_bp_long += bp
        sum_tr_long += tr
    for i in range(k, n):
        bp = close_arr[i] - min(low_arr[i], close_arr[i - 1])
        tr = max(high_arr[i], close_arr[i - 1]) - min(low_arr[i], close_arr[i - 1])
        sum_bp_short += bp - (close_arr[i - short_period] - min(low_arr[i - short_period], close_arr[i - short_period - 1]))
        sum_tr_short += tr - (max(high_arr[i - short_period], close_arr[i - short_period - 1]) - min(low_arr[i - short_period], close_arr[i - short_period - 1]))
        sum_bp_medium += bp - (close_arr[i - medium_period] - min(low_arr[i - medium_period], close_arr[i - medium_period - 1]))
        sum_tr_medium += tr - (max(high_arr[i - medium_period], close_arr[i - medium_period - 1]) - min(low_arr[i - medium_period], close_arr[i - medium_period - 1]))
        sum_bp_long += bp - (close_arr[i - long_period] - min(low_arr[i - long_period], close_arr[i - long_period - 1]))
        sum_tr_long += tr - (max(high_arr[i - long_period], close_arr[i - long_period - 1]) - min(low_arr[i - long_period], close_arr[i - long_period - 1]))
        sum_tr_short += eps if sum_tr_short == 0 else 0
        sum_tr_medium += eps if sum_tr_medium == 0 else 0
        sum_tr_long += eps if sum_tr_long == 0 else 0
        uo[i] = ((long_period * sum_bp_short / sum_tr_short) + (medium_period * sum_bp_medium / sum_tr_medium) + (short_period * sum_bp_long / sum_tr_long)) / (short_period + medium_period + long_period)
    del high_arr, low_arr, close_arr
    return uo


def stochf(np.ndarray[np.float32_t, ndim=1] high_arr = None, np.ndarray[np.float32_t, ndim=1] low_arr = None, np.ndarray[np.float32_t, ndim=1] close_arr = None, int timeperiod = 1, int n_values = 999999999):
    """
    Calculate the Stochastic Fast %K indicator over a specified time period.

    The Stochastic Fast %K is a momentum indicator that compares the closing price to the 
    range of high and low prices over a specified lookback period. It measures the position 
    of the close relative to the range.

    Parameters:
    ----------
    high_arr : np.ndarray[np.float32_t, ndim=1]
        Array of high prices for each time step.

    low_arr : np.ndarray[np.float32_t, ndim=1]
        Array of low prices for each time step.

    close_arr : np.ndarray[np.float32_t, ndim=1]
        Array of closing prices for each time step.

    timeperiod : int
        The lookback period for calculating the Stochastic Fast %K.

    n_values : int, optional
        Number of values from the end of the input arrays to calculate. Defaults to a very large number, effectively using all elements.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the Stochastic Fast %K values. 
        Values outside the calculated range remain NaN.

    Notes:
    -----
    - The Stochastic Fast %K is calculated as:
        %K = (Close - Lowest Low) / (Highest High - Lowest Low)
      where `Highest High` and `Lowest Low` are the maximum and minimum values of the high and low prices over the `timeperiod`.
    - A deque-like sliding window approach is used to efficiently track the highest high and lowest low over the lookback period.
    - Initial values, corresponding to the `timeperiod`, remain NaN since there is insufficient data to compute %K.
    - If the `Highest High` equals the `Lowest Low`, the %K value is not defined and remains NaN.
    """
    cdef int n = len(close_arr)
    cdef int k = max(timeperiod, n - n_values)
    cdef np.ndarray[np.float32_t, ndim=1] stochf_k = np.full(n, np.nan, dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] high_indices = np.zeros(n, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] low_indices = np.zeros(n, dtype=np.int32)
    cdef int high_start = 0, high_end = 0
    cdef int low_start = 0, low_end = 0
    cdef int i, highest_idx, lowest_idx
    cdef float highest_high, lowest_low
    
    for i in range(k - timeperiod, k - 1):
        while high_end > high_start and high_arr[high_indices[high_end - 1]] <= high_arr[i]:
            high_end -= 1
        high_indices[high_end] = i
        high_end += 1
        while low_end > low_start and low_arr[low_indices[low_end - 1]] >= low_arr[i]:
            low_end -= 1
        low_indices[low_end] = i
        low_end += 1
    for i in range(k - 1, n):
        if high_end > high_start and high_indices[high_start] <= i - timeperiod:
            high_start += 1
        if low_end > low_start and low_indices[low_start] <= i - timeperiod:
            low_start += 1
        while high_end > high_start and high_arr[high_indices[high_end - 1]] <= high_arr[i]:
            high_end -= 1
        high_indices[high_end] = i
        high_end += 1
        while low_end > low_start and low_arr[low_indices[low_end - 1]] >= low_arr[i]:
            low_end -= 1
        low_indices[low_end] = i
        low_end += 1
        high_start = max(0, min(high_start, high_end - 1))
        low_start = max(0, min(low_start, low_end - 1))
        highest_idx = high_indices[high_start]
        lowest_idx = low_indices[low_start]
        highest_high = high_arr[highest_idx]
        lowest_low = low_arr[lowest_idx]
        if highest_high != lowest_low:
            stochf_k[i] = (close_arr[i] - lowest_low) / (highest_high - lowest_low)
    del high_arr, low_arr, close_arr, high_indices, low_indices
    return stochf_k


def net_wick(np.ndarray[np.float32_t, ndim=1] high_arr = None, np.ndarray[np.float32_t, ndim=1] low_arr = None, np.ndarray[np.float32_t, ndim=1] close_arr = None, int timeperiod = 1, int n_values = 999999999):
    """
    Calculate the net wick of candlesticks over a given time period.

    The net wick is defined as the top wick of a candlestick minus the bottom wick, normalized by the close price.
    The top wick is the difference between the highest price and the relevant close price.
    The bottom wick is the difference between the relevant close price and the lowest price.

    Parameters:
    ----------
    high_arr : np.ndarray[np.float32_t, ndim=1]
        Array of high prices for each time step.

    low_arr : np.ndarray[np.float32_t, ndim=1]
        Array of low prices for each time step.

    close_arr : np.ndarray[np.float32_t, ndim=1]
        Array of closing prices for each time step.

    timeperiod : int
        The lookback period for calculating the net wick.

    n_values : int, optional
        Number of values from the end of the input arrays to calculate. Defaults to a very large number, effectively using all elements.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the net wick values for each time step. Values outside the calculated range will remain NaN.

    Notes:
    -----
    - The calculation considers the relative position of the current close price compared to the close price `timeperiod` steps ago.
    - If the current close price is higher than the previous close, the top wick is the difference between the highest price and the current close, while the bottom wick is the difference between the previous close and the lowest price.
    - If the current close price is lower, the top wick is the difference between the highest price and the previous close, while the bottom wick is the difference between the current close and the lowest price.
    - The result is normalized by dividing the net wick (top wick - bottom wick) by the current close price.
    """
    cdef int n = len(close_arr), high_start = 0, high_end = 0, low_start = 0, low_end = 0, i, highest_idx, lowest_idx
    cdef int k = max(timeperiod - 1, n - n_values)
    cdef np.ndarray[np.float32_t, ndim=1] wick_arr = np.full(n, np.nan, dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] high_indices = np.zeros(n, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] low_indices = np.zeros(n, dtype=np.int32)
    cdef float top_wick, bottom_wick, max_high, min_low
    for i in range(k - timeperiod + 1, k):
        while high_end > high_start and high_arr[high_indices[high_end - 1]] <= high_arr[i]:
            high_end -= 1
        high_indices[high_end] = i
        high_end += 1
        while low_end > low_start and low_arr[low_indices[low_end - 1]] >= low_arr[i]:
            low_end -= 1
        low_indices[low_end] = i
        low_end += 1
    for i in range(k, n):
        if high_end > high_start and high_indices[high_start] <= i - timeperiod:
            high_start += 1
        if low_end > low_start and low_indices[low_start] <= i - timeperiod:
            low_start += 1
        while high_end > high_start and high_arr[high_indices[high_end - 1]] <= high_arr[i]:
            high_end -= 1
        high_indices[high_end] = i
        high_end += 1
        while low_end > low_start and low_arr[low_indices[low_end - 1]] >= low_arr[i]:
            low_end -= 1
        low_indices[low_end] = i
        low_end += 1
        high_start = max(0, min(high_start, high_end - 1))
        low_start = max(0, min(low_start, low_end - 1))
        highest_idx = high_indices[high_start]
        lowest_idx = low_indices[low_start]
        max_high = high_arr[highest_idx]
        min_low = low_arr[lowest_idx]
        if i - timeperiod >= 0:
            previous_close = close_arr[i - timeperiod]
            if close_arr[i] > previous_close:
                top_wick = max_high - close_arr[i]
                bottom_wick = previous_close - min_low
            else:
                top_wick = max_high - previous_close
                bottom_wick = close_arr[i] - min_low
            wick_arr[i] = (top_wick - bottom_wick) / close_arr[i]
        else:
            wick_arr[i] = np.nan
    del high_arr, low_arr, close_arr, high_indices, low_indices
    return wick_arr


def macd(np.ndarray[np.float32_t, ndim=1] input0=None, np.ndarray[np.float32_t, ndim=1] input1=None, np.ndarray[np.float32_t, ndim=1] input2=None, int short_period=1, int long_period=1, bint input1_is_fast_ema =False, bint input2_is_slow_ema=False, int n_values = 999999999):
    """
    Calculate the Moving Average Convergence Divergence (MACD) indicator.

    MACD is calculated as the difference between two exponential moving averages (EMAs):
        - The "fast" EMA (short period)
        - The "slow" EMA (long period)

    Parameters:
    ----------
    input0 : np.ndarray[np.float32_t, ndim=1], optional
        The primary input array of prices or values used to compute EMAs.
        Required if `input1` and `input2` are not precomputed EMAs.

    input1 : np.ndarray[np.float32_t, ndim=1], optional
        Precomputed "fast" EMA values. Used directly if `input1_is_fast_ema` is True.

    input2 : np.ndarray[np.float32_t, ndim=1], optional
        Precomputed "slow" EMA values. Used directly if `input2_is_slow_ema` is True.

    short_period : int, optional
        The lookback period for the "fast" EMA. Defaults to 1.

    long_period : int, optional
        The lookback period for the "slow" EMA. Defaults to 1.

    input1_is_fast_ema : bint, optional
        If True, `input1` is treated as precomputed "fast" EMA values.
        Defaults to False.

    input2_is_slow_ema : bint, optional
        If True, `input2` is treated as precomputed "slow" EMA values.
        Defaults to False.

    n_values : int, optional
        Number of values from the end of the input to calculate. Defaults to a very large number, effectively using all elements.

    Returns:
    -------
    np.ndarray[np.float32_t, ndim=1]
        An array containing the MACD values. The values outside the calculated range remain NaN.

    Notes:
    -----
    - If both `input1` and `input2` are provided as precomputed EMAs (`input1_is_fast_ema=True`, `input2_is_slow_ema=True`), 
      no additional EMA calculation is performed.
    - If only one of `input1` or `input2` is provided as a precomputed EMA, the other is calculated using `input0`.
    - If neither `input1` nor `input2` are provided as precomputed EMAs, both are calculated using `input0`.
    """
    cdef int n = len(input0), i
    cdef int k = max(long_period, n - n_values)
    cdef np.ndarray[np.float32_t, ndim=1] macd = np.full(n, np.nan, dtype=np.float32), fast_ema, slow_ema
    if input1_is_fast_ema and input2_is_slow_ema:
        for i in range(k, n):
            macd[i] = input1[i] - input2[i]
    elif input1_is_fast_ema and not input2_is_slow_ema:
        slow_ema = ema(input0, timeperiod=long_period, n_values=n_values)
        for i in range(k, n):
            macd[i] = input1[i] - slow_ema[i]
        del slow_ema
    elif not input1_is_fast_ema and input2_is_slow_ema:
        fast_ema = ema(input0, timeperiod=short_period, n_values=n_values)
        for i in range(k, n):
            macd[i] = fast_ema[i] - input2[i]   
        del fast_ema     
    else:
        fast_ema = ema(input0, timeperiod=short_period, n_values=n_values)
        slow_ema = ema(input0, timeperiod=long_period, n_values=n_values)
        for i in range(k, n):
            macd[i] = fast_ema[i] - slow_ema[i]
        del fast_ema, slow_ema
    del input0, input1, input2
    return macd

##################################### volume indicators ##

def obv(np.ndarray[np.float32_t, ndim=1] close, np.ndarray[np.float32_t, ndim=1] volume):
    cdef int n = len(close), i
    cdef np.ndarray[np.float32_t, ndim=1] obv_vals = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        if close[i] > close[i-1]:
            obv_vals[i] = obv_vals[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv_vals[i] = obv_vals[i-1] - volume[i]
        else:
            obv_vals[i] = obv_vals[i-1]
    return obv_vals

def vwap(np.ndarray[np.float32_t, ndim=1] close, np.ndarray[np.float32_t, ndim=1] volume):
    cdef int n = len(close), i
    cdef np.ndarray[np.float32_t, ndim=1] vwap_vals = np.zeros(n, dtype=np.float32)
    cdef float cum_vol = 0.0, cum_vol_price = 0.0
    for i in range(n):
        cum_vol += volume[i]
        cum_vol_price += close[i] * volume[i]
        if cum_vol > 0:
            vwap_vals[i] = cum_vol_price / cum_vol
    return vwap_vals

def mfi(np.ndarray[np.float32_t, ndim=1] high, np.ndarray[np.float32_t, ndim=1] low,
        np.ndarray[np.float32_t, ndim=1] close, np.ndarray[np.float32_t, ndim=1] volume, int period=14):
    cdef int n = len(close), i
    cdef np.ndarray[np.float32_t, ndim=1] money_flow = np.zeros(n, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] mfi_vals = np.full(n, np.nan, dtype=np.float32)
    cdef float typical_price, pos_flow, neg_flow
    
    for i in range(n):
        typical_price = (high[i] + low[i] + close[i]) / 3
        money_flow[i] = typical_price * volume[i]
    
    for i in range(period, n):
        pos_flow = np.sum(money_flow[i-period:i][close[i-period:i] > close[i-period-1:i-1]])
        neg_flow = np.sum(money_flow[i-period:i][close[i-period:i] < close[i-period-1:i-1]])
        if neg_flow == 0:
            mfi_vals[i] = 100
        else:
            mfi_vals[i] = 100 - (100 / (1 + (pos_flow / neg_flow)))
    return mfi_vals

def ad_line(np.ndarray[np.float32_t, ndim=1] high, np.ndarray[np.float32_t, ndim=1] low,
            np.ndarray[np.float32_t, ndim=1] close, np.ndarray[np.float32_t, ndim=1] volume):
    cdef int n = len(close), i
    cdef np.ndarray[np.float32_t, ndim=1] ad_vals = np.zeros(n, dtype=np.float32)
    cdef float money_flow_multiplier
    
    for i in range(n):
        if high[i] != low[i]:
            money_flow_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            ad_vals[i] = ad_vals[i-1] + money_flow_multiplier * volume[i]
    return ad_vals

def cmf(np.ndarray[np.float32_t, ndim=1] high, np.ndarray[np.float32_t, ndim=1] low,
        np.ndarray[np.float32_t, ndim=1] close, np.ndarray[np.float32_t, ndim=1] volume, int period=20):
    cdef int n = len(close), i
    cdef np.ndarray[np.float32_t, ndim=1] cmf_vals = np.full(n, np.nan, dtype=np.float32)
    cdef float money_flow_multiplier, money_flow_volume, sum_mfv, sum_volume
    
    for i in range(period, n):
        sum_mfv = 0.0
        sum_volume = 0.0
        for j in range(i - period, i):
            if high[j] != low[j]:
                money_flow_multiplier = ((close[j] - low[j]) - (high[j] - close[j])) / (high[j] - low[j])
                money_flow_volume = money_flow_multiplier * volume[j]
                sum_mfv += money_flow_volume
                sum_volume += volume[j]
        if sum_volume != 0:
            cmf_vals[i] = sum_mfv / sum_volume
    return cmf_vals

def pvt(np.ndarray[np.float32_t, ndim=1] close, np.ndarray[np.float32_t, ndim=1] volume):
    cdef int n = len(close), i
    cdef np.ndarray[np.float32_t, ndim=1] pvt_vals = np.zeros(n, dtype=np.float32)
    
    for i in range(1, n):
        if close[i-1] != 0:
            pvt_vals[i] = pvt_vals[i-1] + (volume[i] * (close[i] - close[i-1]) / close[i-1])
    return pvt_vals

def volume_oscillator(np.ndarray[np.float32_t, ndim=1] volume, int short_period=5, int long_period=10):
    cdef int n = len(volume), i
    cdef np.ndarray[np.float32_t, ndim=1] vo_vals = np.full(n, np.nan, dtype=np.float32)
    cdef float short_ma, long_ma
    
    for i in range(long_period, n):
        short_ma = np.mean(volume[i-short_period:i])
        long_ma = np.mean(volume[i-long_period:i])
        if long_ma != 0:
            vo_vals[i] = ((short_ma - long_ma) / long_ma) * 100
    return vo_vals