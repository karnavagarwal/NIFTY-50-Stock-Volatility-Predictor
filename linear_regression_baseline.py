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

class Ridge:
    def __init__(self, l2_penalty=1.0):
        self.l2_penalty = l2_penalty
        self.theta_ = None

    def fit(self, X, y):
        # Convert pandas DataFrames to numpy arrays
        X_array = np.array(X, dtype=float)
        y_array = np.array(y, dtype=float)
        
        # Calculate weights
        self.theta_ = ridge_closed_form(X_array, y_array, self.l2_penalty)
        return self

    def predict(self, X):
        if self.theta_ is None:
            raise ValueError("The model has not been fitted yet!")
            
        X_array = np.array(X, dtype=float)
        # Add the column of ones to the new data for intercept
        X_b = np.c_[np.ones((X_array.shape[0], 1)), X_array]
        
        # y = X * beta
        return X_b @ self.theta_