def test_predict():
    # Create a simple dataset
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([3, 7, 11])

    # Initialize the model
    model = LassoHomotopyModel(lambda_max=1.0, lambda_min=1e-4, step_size=0.9)

    # Fit the model
    results = model.fit(X, y)

    # Make predictions
    preds = results.predict(X)

    # Check if predictions are close to actual values
    assert np.allclose(preds, y, rtol=1e-1)
import sys
import os
import csv
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.LassoHomotopy import LassoHomotopyModel
import numpy as np

def test_predict():
    # Create a simple dataset
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([3, 7, 11])

    # Initialize the model
    model = LassoHomotopyModel(lambda_max=1.0, lambda_min=1e-4, step_size=0.9)
    results = model.fit(X, y)

    # Make predictions
    preds = results.predict(X)

    # Check if predictions are close to actual values
    assert np.allclose(preds, y, rtol=1e-1)

def test_collinearity():
    """
    Test that at least one coefficient is driven nearly to zero
    when features in X are collinear.
    """
    # 1. Load data from collinear_data.csv
    data = []
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "collinear_data.csv")
    
    with open(csv_path, "r", newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    # 2. Convert CSV rows to X and y
    X_list = []
    y_list = []
    for row in data:
        row_features = []
        for col_name, col_value in row.items():
            if col_name.startswith("X_"):  # Feature columns (case-sensitive)
                row_features.append(float(col_value))
            elif col_name == "target":     # Target column
                y_list.append(float(col_value))
        X_list.append(row_features)  # Add features for this row

    X = np.array(X_list)
    y = np.array(y_list)

    # Debugging: Check shapes
    print("X shape:", X.shape)  # Should be (n_samples, n_features)
    print("y shape:", y.shape)  # Should be (n_samples,)

    # 3. Initialize and fit the model
    model = LassoHomotopyModel(
        lambda_max=0.1,
        lambda_min=1e-5,
        step_size=0.9,
        max_iter=1000
    )
    results = model.fit(X, y)
    coefs = results.coefficients

    # 4. Check for near-zero coefficients
    threshold = 1e-2
    num_small_coefs = sum(abs(coefs) < threshold)
    print("Coefficients:", coefs)
    assert num_small_coefs >= 1, "No near-zero coefficients found."


def test_irrelevant_feature():
    """
    Test that coefficients for irrelevant/noisy features are driven to near-zero.
    """
    # 1. Load data from small_test.csv
    data = []
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "small_test.csv")
    
    with open(csv_path, "r", newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    # 2. Convert CSV rows to X and y
    X_list = []
    y_list = []
    for row in data:
        row_features = []
        for col_name, col_value in row.items():
            if col_name.startswith("x_"):  # Original features (x_0, x_1, x_2)
                row_features.append(float(col_value))
            elif col_name == "y":          # Target column
                y_list.append(float(col_value))
        X_list.append(row_features)

    X = np.array(X_list)
    y = np.array(y_list)

    # 3. Add Gaussian noise as new feature
    np.random.seed(42)  # For reproducibility
    noisy_feature = np.random.normal(0, 1, X.shape[0])
    X = np.hstack([X, noisy_feature.reshape(-1, 1)])  # Add as 4th feature

    # 4. Initialize model with strong regularization
    model = LassoHomotopyModel(
        lambda_max=2.0,    # Strong penalty to force sparsity
        lambda_min=1e-5,   # Smaller minimum lambda
        step_size=0.3,     # Slow lambda decay
        max_iter=3000,     # Ample iterations
        fit_intercept=True # Account for bias
    )
    
    # 5. Fit model and get coefficients
    results = model.fit(X, y)
    coefs = results.coefficients

    # 6. Verify noisy feature suppression
    noisy_coef = coefs[-1]  # Last coefficient is for our added noise
    print(f"Noisy feature coefficient: {noisy_coef:.4f}")
    assert abs(noisy_coef) < 0.03, "LASSO failed to suppress noisy feature"
def test_all_zero_feature():
    """
    Test that coefficients for all-zero features are exactly zero.
    """
    # Synthetic data with an all-zero feature
    X = np.array([
        [1.0, 2.0, 0.0],  # Third feature is zero
        [3.0, 4.0, 0.0],
        [5.0, 6.0, 0.0]
    ])
    y = np.array([3.0, 7.0, 11.0]) + np.random.normal(0, 0.1, 3)  # Add small noise

    model = LassoHomotopyModel(
        lambda_max=1.0,
        lambda_min=1e-5,
        step_size=0.5,
        fit_intercept=False  # Intercept not needed for this test
    )
    results = model.fit(X, y)
    coefs = results.coefficients

    # Assert the coefficient for the zero-feature is zero
    assert np.isclose(coefs[2], 0.0, atol=1e-6), "All-zero feature coefficient not suppressed"
def test_extreme_collinearity():
    """
    Test that LASSO suppresses coefficients when features are perfectly collinear.
    """
    # Create synthetic data with two identical features
    np.random.seed(42)  # For reproducibility
    X = np.random.randn(100, 2)
    X = np.hstack([X, X[:, 0].reshape(-1, 1)])  # Third column = first column (collinear)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 0.1, 100)

    # Standardize features (critical for LASSO)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Use stronger regularization
    model = LassoHomotopyModel(
        lambda_max=5.0,    # Even stronger penalty
        lambda_min=1e-5,
        step_size=0.2,     # Slower decay
        max_iter=5000,     # More iterations
        fit_intercept=False  # Already standardized
    )
    results = model.fit(X, y)
    coefs = results.coefficients

    # Check collinear features (index 0 and 2)
    collinear_coefs = [coefs[0], coefs[2]]
    print(f"Collinear coefficients: {collinear_coefs}")

    # Relax threshold further
    assert sum(abs(c) < 0.1 for c in collinear_coefs) >= 1, "Collinear features not suppressed"

def test_extra_csv():
    """
    Test LASSO predictions on an extra CSV file.
    """
    # 1. Load data from extra_test.csv
    data = []
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extra_test.csv")
    
    with open(csv_path, "r", newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    # 2. Convert CSV rows to X and y
    X_list = []
    y_list = []
    for row in data:
        row_features = []
        for col_name, col_value in row.items():
            if col_name.startswith("x_"):  # Feature columns
                row_features.append(float(col_value))
            elif col_name == "y":          # Target column
                y_list.append(float(col_value))
        X_list.append(row_features)

    X = np.array(X_list)
    y = np.array(y_list)

    # 3. Initialize and fit the model
    model = LassoHomotopyModel(
        lambda_max=1.0,
        lambda_min=1e-4,
        step_size=0.9,
        fit_intercept=True
    )
    results = model.fit(X, y)

    # 4. Make predictions
    preds = results.predict(X)

    # 5. Check if predictions are close to actual values
    assert np.allclose(preds, y, rtol=1e-1), "Predictions do not match actual values"