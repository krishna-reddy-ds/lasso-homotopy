from LassoHomotopy import LassoHomotopyModel
import numpy as np

# Initialize the model
model = LassoHomotopyModel(lambda_max=1.0, lambda_min=1e-4, step_size=0.9)

# Create sample data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([3, 7, 11])

# Fit the model
results = model.fit(X, y)

# Make predictions
preds = results.predict(X)
print("Predictions:", preds)