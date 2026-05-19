# LASSO Regression via the Homotopy Method

A from-scratch implementation of LASSO (L1-regularized) regression using an iterative homotopy approach — built with NumPy only, no scikit-learn models used.

## What it does
LASSO adds an L1 penalty to linear regression, driving irrelevant feature coefficients to exactly zero — performing automatic feature selection. This implementation gradually decreases regularization strength (λ) from high to low, tracking how coefficients change at each step (the homotopy path).

**Use it when:**
- You have high-dimensional data with many features
- You suspect only a few features truly matter (sparse solution)
- You want an interpretable model with automatic feature selection

## Installation

```bash
git clone https://github.com/krishna-reddy-ds/lasso-homotopy.git
cd lasso-homotopy
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```python
from LassoHomotopy import LassoHomotopyModel
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([3, 7, 11])

model = LassoHomotopyModel(lambda_max=1.0, lambda_min=1e-4, step_size=0.9)
results = model.fit(X, y)

predictions = results.predict(X)
print("Predictions:", predictions)
print("Coefficients:", results.coefficients)
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `lambda_max` | Auto | Starting regularization strength |
| `lambda_min` | 1e-4 | Stopping threshold for λ |
| `step_size` | 0.9 | λ decay factor per iteration |
| `max_iter` | 1000 | Maximum iterations |
| `fit_intercept` | True | Whether to fit an intercept term |

## Running Tests

```bash
pytest test_LassoHomotopy.py -v
```

Tests cover:
- Basic prediction accuracy
- Collinear feature suppression
- Irrelevant/noisy feature zeroing
- All-zero feature handling
- Extreme collinearity scenarios
- CSV-based data loading

## Limitations
- Very high noise or perfect collinearity can challenge numerical stability
- Large datasets may require more efficient sparse updates
- Feature scaling recommended before fitting

## Tech Stack
Python · NumPy · Pytest

## Course
IIT Chicago — Applied Machine Learning (2024)
