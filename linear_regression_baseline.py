import numpy as np

def ridge_closed_form(X, y, l2_penalty=1.0):
    # Add a column of ones to X for the bias term (intercept)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    n_features = X_b.shape[1]
    
    # Create the Identity matrix
    I = np.eye(n_features)
    # We do not regularize the bias term, so set the top-left element to 0
    I[0, 0] = 0 
    
    # Solve: (X^T * X + lambda * I) * beta = X^T * y
    # np.linalg.solve is numerically more stable than using np.linalg.inv
    theta = np.linalg.solve((X_b.T @ X_b) + (l2_penalty * I), (X_b.T) @ y)
    
    return theta # The first value is the bias, the rest are the feature weights