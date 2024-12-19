# cythonized-ta

`cythonized-ta` is a high-performance Python library for financial technical analysis, designed for speed, flexibility, and efficiency. It is implemented entirely in Cython and NumPy, making it ideal for large datasets and real-time analysis.

---

## Features

### 🚀 **High Performance**
- Fully implemented in **Cython** and **NumPy** for optimized computation.
- Efficient algorithms minimize memory usage and maximize speed.

### ⚙️ **Flexibility**
- **Customizable Computation Scope**: Specify the exact number of values to compute using the `n_values` parameter.
- **Precomputed Values Support**: Use intermediate results to speed up complex calculations without redundant computations.

### 📊 **Comprehensive Technical Indicators**
Includes a wide range of indicators for trend, momentum, volatility, and more:
- **Trend Indicators**: SMA, EMA, DEMA, TEMA
- **Momentum Indicators**: RSI, MACD, TRIX, CMO
- **Volatility Indicators**: ATR, NATR, Bollinger Bands
- **Volume Indicators**: On-Balance Volume (OBV), Volume Oscillator
- **Specialized Indicators**: Williams %R, Ultimate Oscillator, Stochastic Fast %K

---

## Installation

Install the library from PyPI:

```bash
pip install cythonized-ta
```

Or clone the repository for the latest version:
```bash
git clone https://github.com/yourusername/cythonized-ta.git
cd cythonized-ta
pip install .
```

## Quick Start

### Example: Simple Moving Average (SMA)

```python
import numpy as np
import cythonized_ta as ta
```

```python
# Sample data
data = np.array([100, 102, 104, 103, 105, 107, 109], dtype=np.float32)

# Compute SMA with a 3-period window
sma = ta.sma(data, timeperiod=3, n_values=5)
print(sma)
```

### Example: Relative Strengh Index (RSI)


```python
# Sample data
data = np.array([100, 102, 104, 103, 105, 107, 109], dtype=np.float32)

# Compute SMA with a 3-period window
sma = ta.sma(data, timeperiod=3) # if n_values isn't specified, computes all available values
print(sma)
```
---

## Documentation

Detailed documentation, including the full API reference, is available at [Documentation Link](https://your-documentation-url).

---

## Advanced Usage

### Precomputed Values for Efficiency
Avoid redundant calculations by providing intermediate values:

```python
# Precompute the first EMA
ema1 = ta.ema(data, timeperiod=10)

# Use the precomputed EMA to calculate a DEMA
dema = ta.dema(data, input1=ema1, input1_is_ema1=True, timeperiod=10)
```

---

## Performance Benchmarks

cythonized-ta outperforms similar libraries in terms of speed and efficiency, especially for large datasets. Benchmarks can be found in the Performance Documentation.