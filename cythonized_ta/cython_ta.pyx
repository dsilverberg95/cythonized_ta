

import numpy as np
import pandas as pd
cimport cython
cimport numpy as np


def cython_sma(np.ndarray[np.float32_t, ndim=1] input_array, int timeperiod, bint normalize=True, int n_values = 999999999):
    cdef int n = len(input_array), i, non_nan_count = 0
    cdef np.ndarray[np.float32_t, ndim=1] sma = np.full(n, np.nan, dtype=np.float32)
    cdef float sum = 0.0
    k = max(timeperiod - 1, n - n_values)
    for i in range(k - timeperiod, k):
        if not np.isnan(input_array[i]):
            sum += input_array[i]
            non_nan_count += 1
    if normalize:
        for i in range(k, n):
            if not np.isnan(input_array[i]):
                sum += input_array[i]
                non_nan_count += 1
            if not np.isnan(input_array[i - timeperiod]):
                sum -= input_array[i - timeperiod]
                non_nan_count -= 1
            if non_nan_count > 0:
                sma[i] = ((sum / non_nan_count) - input_array[i]) / input_array[i]                    
    else:
        for i in range(k, n):
            if not np.isnan(input_array[i]):
                sum += input_array[i]
                non_nan_count += 1
            if not np.isnan(input_array[i - timeperiod]):
                sum -= input_array[i - timeperiod]
                non_nan_count -= 1
            if non_nan_count > 0:
                sma[i] = sum / non_nan_count
    del input_array
    return sma
        


def cython_lin(np.ndarray[np.float32_t, ndim=1] input_array, int timeperiod, int n_values = 999999999):
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



def cython_mid(np.ndarray[np.float32_t, ndim=1] high, np.ndarray[np.float32_t, ndim=1] low, np.ndarray[np.float32_t, ndim=1] close, int timeperiod, bint normalize=True, int n_values = 999999999):
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
            mid[i] = ((low[low_indices[low_start]] + (high[high_indices[high_start]] - low[low_indices[low_start]]) / 2) - close[i]) / close[i]
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

 

def cython_ema(np.ndarray[np.float32_t, ndim=1] input_array, int timeperiod, bint normalize=True, int n_values = 999999999):
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
                ema_value = alpha * input_array[i] + (1 - alpha) * ema_value
                ema[i] = (ema_value - input_array[i]) / input_array[i]
        else:
            for i in range(first_non_nan, n):
                ema_value = alpha * input_array[i] + (1 - alpha) * ema_value
                ema[i] = (ema_value - input_array[i]) / input_array[i]
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



def cython_ema2(np.ndarray[np.float32_t, ndim=1] input_array, int timeperiod, bint input_is_ema1=False, int n_values=999999999):
    cdef np.ndarray[np.float32_t, ndim=1] ema1, ema2
    if input_is_ema1:
        ema2 = cython_ema(input_array, timeperiod, normalize=False, n_values=n_values)
    else:
        ema1 = cython_ema(input_array, timeperiod, normalize=True)
        ema2 = cython_ema(ema1, timeperiod, normalize=False, n_values=n_values)
        del ema1
    del input_array
    return ema2



def cython_ema3(np.ndarray[np.float32_t, ndim=1] input_array, int timeperiod, bint input_is_ema2=False, input_is_ema1=False, int n_values=999999999):
    cdef np.ndarray[np.float32_t, ndim=1] ema1, ema2, ema3
    if input_is_ema2:
        ema3 = cython_ema(input_array, timeperiod, normalize=False, n_values=n_values)
    elif input_is_ema1:
        ema2 = cython_ema(input_array, timeperiod, normalize=False)
        ema3 = cython_ema(ema2, timeperiod, normalize=False, n_values=n_values)
        del ema2
    else:
        ema1 = cython_ema(input_array, timeperiod, normalize=True)
        ema2 = cython_ema(ema1, timeperiod, normalize=False)
        del ema1
        ema3 = cython_ema(ema2, timeperiod, normalize=False, n_values=n_values)
        del ema2
    del input_array
    return ema3



def cython_dema(np.ndarray[np.float32_t, ndim=1] input1, np.ndarray[np.float32_t, ndim=1] input2=None, int timeperiod=1, bint input1_is_ema1=False, bint input2_is_ema2=False, int n_values = 999999999):
    cdef int n = len(input1), i
    cdef int k = max(0, n - n_values)
    cdef np.ndarray[np.float32_t, ndim=1] dema = np.full(n, np.nan, dtype=np.float32), ema1, ema2
    if input1_is_ema1 and input2_is_ema2:
        ema1 = input1
        ema2 = input2
    elif input1_is_ema1 and not input2_is_ema2:
        ema1 = input1
        ema2 = cython_ema(ema1, timeperiod, normalize=False, n_values=n_values)
    elif not input1_is_ema1 and input2_is_ema2:
        ema1 = cython_ema(input1, timeperiod, normalize=True, n_values=n_values)
        ema2 = input2
    else:
        ema1 = cython_ema(input1, timeperiod, normalize=True)
        ema2 = cython_ema(ema1, timeperiod, normalize=False, n_values=n_values)
    for i in range(k, n):
        dema[i] = 2 * ema1[i] - ema2[i]
    del input1, input2, ema1, ema2
    return dema


def cython_tema(np.ndarray[np.float32_t, ndim=1] input1, np.ndarray[np.float32_t, ndim=1] input2=None, np.ndarray[np.float32_t, ndim=1] input3=None, int timeperiod = 1, bint input3_is_ema3=False, bint input2_is_ema2=False, bint input1_is_ema1=False, int n_values=999999999):
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
        ema3 = cython_ema(ema2, timeperiod, normalize=False, n_values=n_values)
    elif input1_is_ema1 and not input2_is_ema2 and input3_is_ema3:
        ema1 = input1
        ema2 = cython_ema(ema1, timeperiod, normalize=False, n_values=n_values)
        ema3 = input3
    elif not input1_is_ema1 and input2_is_ema2 and input3_is_ema3:
        ema1 = cython_ema(input1, timeperiod, normalize=True, n_values=n_values)
        ema2 = input2
        ema3 = input3
    elif input1_is_ema1 and not input2_is_ema2 and not input3_is_ema3:
        ema1 = input1
        ema2 = cython_ema(ema1, timeperiod, normalize=False)
        ema3 = cython_ema(ema2, timeperiod, normalize=False, n_values=n_values)
    elif not input1_is_ema1 and input2_is_ema2 and not input3_is_ema3:
        ema1 = cython_ema(input1, timeperiod, normalize=True, n_values=n_values)
        ema2 = input2
        ema3 = cython_ema(ema2, timeperiod, normalize=False, n_values=n_values)
    elif not input1_is_ema1 and not input2_is_ema2 and input3_is_ema3:
        ema1 = cython_ema(input1, timeperiod, normalize=True)
        ema2 = cython_ema(ema1, timeperiod, normalize=False, n_values=n_values)
        ema3 = input3
    else:
        ema1 = cython_ema(input1, timeperiod, normalize=True)
        ema2 = cython_ema(ema1, timeperiod, normalize=False)
        ema3 = cython_ema(ema2, timeperiod, normalize=False, n_values=n_values)
    for i in range(k, n):
        tema[i] = 3 * ema1[i] - 3 * ema2[i] + ema3[i]
    del ema1, ema2, ema3, input1, input2, input3
    return tema


  
def cython_trix(np.ndarray[np.float32_t, ndim=1] arr, int timeperiod = 1, bint input_is_ema3 = False, bint input_is_ema2=False, bint input_is_ema1=False, int n_values = 999999999):
    cdef int n = arr.shape[0]
    cdef int k = max(1, n - n_values)
    cdef np.ndarray[np.float32_t, ndim=1] trix = np.full(n, np.nan, dtype=np.float32), ema1, ema2, ema3
    if input_is_ema3:
        ema3 = arr
    elif input_is_ema2:
        ema3 = cython_ema(arr, timeperiod, normalize=False, n_values=n_values+1)
    elif input_is_ema1:
        ema2 = cython_ema(arr, timeperiod, normalize=False)
        ema3 = cython_ema(ema2, timeperiod, normalize=False, n_values=n_values+1)
        del ema2
    else:
        ema1 = cython_ema(arr, timeperiod, normalize=True)
        ema2 = cython_ema(ema1, timeperiod, normalize=False)
        del ema1
        ema3 = cython_ema(ema2, timeperiod, normalize=False, n_values=n_values+1)
        del ema2
    for i in range(k, n):
        trix[i] = (ema3[i] - ema3[i - 1]) / (ema3[i - 1] + (0 if ema3[i-1]!=0 else 1e-9))
    del ema3, arr
    return trix



def cython_roc(np.ndarray[np.float32_t, ndim=1] input_array, int timeperiod, int n_values = 999999999):
    cdef int n = len(input_array)
    cdef np.ndarray[np.float32_t, ndim=1] roc = np.full(n, np.nan, dtype=np.float32)
    k = max(n - n_values, timeperiod)
    for i in range(k, n):
        roc[i] = (input_array[i] - input_array[i - timeperiod]) / (input_array[i - timeperiod] + (0 if input_array[i - timeperiod]!=0 else 1e-9))
    del input_array
    return roc

 

def cython_rsi(np.ndarray[np.float32_t, ndim=1] close, int timeperiod, int n_values = 999999999):
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



def cython_willr(np.ndarray[np.float32_t, ndim=1] high_arr, np.ndarray[np.float32_t, ndim=1] low_arr, np.ndarray[np.float32_t, ndim=1] close_arr, int timeperiod, int n_values = 999999999):
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



def cython_cci(np.ndarray[np.float32_t, ndim=1] high_arr, np.ndarray[np.float32_t, ndim=1] low_arr, np.ndarray[np.float32_t, ndim=1] close_arr, int timeperiod, int n_values = 999999999):
    '''
    Note: this function uses mean squared deviation instead of mean absolute deviation to achieve faster execution.
    '''
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


def cython_dx(np.ndarray[np.float32_t, ndim=1] high, np.ndarray[np.float32_t, ndim=1] low, np.ndarray[np.float32_t, ndim=1] close, int timeperiod, int n_values = 999999999):
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

 
def cython_adx(np.ndarray[np.float32_t, ndim=1] high_arr = None, np.ndarray[np.float32_t, ndim=1] low_arr = None, np.ndarray[np.float32_t, ndim=1] close_arr = None, int timeperiod = 1, int n_values = 999999999, bint close_arr_is_dx=False):
    cdef int n = len(close_arr), i
    k = max(2 * timeperiod, n - n_values)
    cdef np.ndarray[np.float32_t, ndim=1] adx = np.full(n, np.nan, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] dx = close_arr if close_arr_is_dx else cython_dx(high_arr, low_arr, close_arr, timeperiod)
    cdef float adx_value    
    adx_value = np.nanmean(dx[timeperiod:2 * timeperiod])
    for i in range(2 * timeperiod, k):
        adx_value = ((adx_value * (timeperiod - 1)) + dx[i]) / timeperiod
    for i in range(k, n):
        adx_value = ((adx_value * (timeperiod - 1)) + dx[i]) / timeperiod
        adx[i] = adx_value
    del dx, high_arr, low_arr, close_arr
    return adx

 
def cython_adxr(np.ndarray[np.float32_t, ndim=1] high_arr, np.ndarray[np.float32_t, ndim=1] low_arr, np.ndarray[np.float32_t, ndim=1] close_arr, int timeperiod, int n_values = 999999999, bint close_arr_is_adx=False):
    cdef int n = len(close_arr), i
    cdef int k = max(timeperiod, n - n_values)
    cdef np.ndarray[np.float32_t, ndim=1] adx = close_arr if close_arr_is_adx else cython_adx(high_arr, low_arr, close_arr, timeperiod)
    cdef np.ndarray[np.float32_t, ndim=1] adxr = np.full(n, np.nan, dtype=np.float32)
    for i in range(k - timeperiod, n - timeperiod):
        adxr[i + timeperiod] = (adx[i] + adx[i + timeperiod]) / 2
    del adx, high_arr, low_arr, close_arr
    return adxr


def cython_cmo(np.ndarray[np.float32_t, ndim=1] close_arr, int timeperiod, int n_values = 999999999):
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


def cython_natr(np.ndarray[np.float32_t, ndim=1] high_arr, np.ndarray[np.float32_t, ndim=1] low_arr, np.ndarray[np.float32_t, ndim=1] close_arr, int timeperiod, int n_values = 999999999):
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

 
def cython_ult(np.ndarray[np.float32_t, ndim=1] high_arr, np.ndarray[np.float32_t, ndim=1] low_arr, np.ndarray[np.float32_t, ndim=1] close_arr, int short_period, int medium_period, int long_period, int n_values = 999999999):
    cdef int n = len(close_arr), i
    cdef int k = max(long_period, n - n_values)
    cdef np.ndarray[np.float32_t, ndim=1] uo = np.full(n, np.nan, dtype=np.float32)
    cdef float sum_bp_short = 0.0, sum_tr_short = 0.0, sum_bp_medium = 0.0, sum_tr_medium = 0.0, sum_bp_long = 0.0, sum_tr_long = 0.0, eps = 1e-10
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

 
def cython_stochf(np.ndarray[np.float32_t, ndim=1] high_arr, np.ndarray[np.float32_t, ndim=1] low_arr, np.ndarray[np.float32_t, ndim=1] close_arr, int timeperiod, bint stream=False, bint last_value_only=False):
    cdef int n = len(close_arr)
    cdef np.ndarray[np.float32_t, ndim=1] stochf_k = np.full(n, np.nan, dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] high_indices = np.zeros(n, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] low_indices = np.zeros(n, dtype=np.int32)
    cdef int high_start = 0, high_end = 0
    cdef int low_start = 0, low_end = 0
    cdef int i, highest_idx, lowest_idx
    cdef float highest_high, lowest_low
    for i in range(timeperiod - 1):
        while high_end > high_start and high_arr[high_indices[high_end - 1]] <= high_arr[i]:
            high_end -= 1
        high_indices[high_end] = i
        high_end += 1
        while low_end > low_start and low_arr[low_indices[low_end - 1]] >= low_arr[i]:
            low_end -= 1
        low_indices[low_end] = i
        low_end += 1
    for i in range(timeperiod - 1, n):
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


def cython_wick(np.ndarray[np.float32_t, ndim=1] high_arr, np.ndarray[np.float32_t, ndim=1] low_arr, np.ndarray[np.float32_t, ndim=1] close_arr, int timeperiod, int n_values = 999999999):
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


def cython_macd(np.ndarray[np.float32_t, ndim=1] input0=None, np.ndarray[np.float32_t, ndim=1] input1=None, np.ndarray[np.float32_t, ndim=1] input2=None, int short_period=1, int long_period=1, bint input1_is_fast_ema =False, bint input2_is_slow_ema=False, int n_values = 999999999):
    cdef int n = len(input0), i
    cdef int k = max(long_period, n - n_values)
    cdef np.ndarray[np.float32_t, ndim=1] macd = np.full(n, np.nan, dtype=np.float32), fast_ema, slow_ema
    if input1_is_fast_ema and input2_is_slow_ema:
        for i in range(k, n):
            macd[i] = input1[i] - input2[i]
    elif input1_is_fast_ema and not input2_is_slow_ema:
        slow_ema = cython_ema(input0, timeperiod=long_period, n_values=n_values)
        for i in range(k, n):
            macd[i] = input1[i] - slow_ema[i]
        del slow_ema
    elif not input1_is_fast_ema and input2_is_slow_ema:
        fast_ema = cython_ema(input0, timeperiod=short_period, n_values=n_values)
        for i in range(k, n):
            macd[i] = fast_ema[i] - input2[i]   
        del fast_ema     
    else:
        fast_ema = cython_ema(input0, timeperiod=short_period, n_values=n_values)
        slow_ema = cython_ema(input0, timeperiod=long_period, n_values=n_values)
        for i in range(k, n):
            macd[i] = fast_ema[i] - slow_ema[i]
        del fast_ema, slow_ema
    del input0, input1, input2
    return macd


def cython_adif(np.ndarray[np.float32_t, ndim=1] input_array_1, np.ndarray[np.float32_t, ndim=1] input_array_2, int n_values=999999999):
    cdef int n = len(input_array_1), i
    cdef np.ndarray[np.float32_t, ndim=1] dif = np.full(n, np.nan, dtype=np.float32) 
    n_values = min(n_values, n)
    for i in range(n - n_values, n):
        dif[i] = input_array_1[i] - input_array_2[i]
    del input_array_1, input_array_2
    return dif


def cython_rdif(np.ndarray[np.float32_t, ndim=1] input_array_1, np.ndarray[np.float32_t, ndim=1] input_array_2, int n_values = 999999999):
    cdef int n = len(input_array_1), i
    cdef np.ndarray[np.float32_t, ndim=1] dif = np.full(n, np.nan, dtype=np.float32)
    n_values = min(n_values, n)
    for i in range(n - n_values, n):
        if input_array_2[i] != 0:
            dif[i] = (input_array_1[i] - input_array_2[i]) / input_array_2[i]
        else:
            dif[i] = (input_array_1[i] - input_array_2[i]) / (input_array_2[i] + 1e-9)
    del input_array_1, input_array_2
    return dif