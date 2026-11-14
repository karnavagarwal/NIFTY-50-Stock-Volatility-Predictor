import numpy as np
import pandas as pd
#using the object oriented approach
#treenode class for each node of the xgb tree
class TreeNode:
    def __init__(self, feature_id = None, threshold = None, left = None, right = None, value = None):
        self.feature_id = feature_id
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
#XGBRegressionTree class for a single tree
#We will be using trees here, not stumps for xg boost, as stumps wont be able to figure our the relation among features.
class XGBRegressionTree:
    def __init__(self, max_depth=3, lambda_=1.0, gamma=0.0, min_samples_split=2):
        self.max_depth = max_depth
        self.lambda_ = lambda_
        self.gamma = gamma
        self.min_samples_split = min_samples_split
        self.root = None
    def calculate_similarity(self, g, h):
        return (np.sum(g) ** 2)/(np.sum(h) + self.lambda_) #xgb formula. did not use 1 as different types of loss other than mse will be tried.
    def calculate_weight(self, g, h):
        return -np.sum(g)/(np.sum(h) + self.lambda_)
    def find_split(self, X, g, h):
        best_gain = 0.0
        best_split = None
        root_similarity = self.calculate_similarity(g, h)
        n_features = X.shape[1]
        for feature_id in range(n_features):
            feature_values = X[:, feature_id]
            #testing thresholds on all rows will be unneccessarily bulky, so it will only be done on unique percentiles
            thresholds = np.unique(np.percentile(feature_values, np.arange(0, 100, 10)))
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold
                #If all the data is on one side
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                g_left, h_left = g[left_mask], h[left_mask]
                g_right, h_right = g[right_mask], h[right_mask]
                sim_left = self.calculate_similarity(g_left, h_left)
                sim_right = self.calculate_similarity(g_right, h_right)
                gain = sim_left + sim_right - root_similarity
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_id': feature_id,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask,
                        'gain': gain
                    }
        return best_split
    #recursively building the tree
    def build_tree(self, X, g, h, depth):
        #2 conditions where we terminate the tree
        if depth>=self.max_depth or X.shape[0]<self.min_samples_split:
            return TreeNode(value = self.calculate_weight(g, h))
        best_split = self.find_split(X, g, h)
        if best_split is None or best_split['gain'] < self.gamma:
            return TreeNode(value = self.calculate_weight(g, h))
        left_subtree = self.build_tree(
            X[best_split['left_mask']], g[best_split['left_mask']], h[best_split['left_mask']], depth + 1
        )
        right_subtree = self.build_tree(
            X[best_split['right_mask']], g[best_split['right_mask']], h[best_split['right_mask']], depth + 1
        )
        return TreeNode(
            feature_id=best_split['feature_id'],
            threshold = best_split['threshold'],
            left = left_subtree,
            right = right_subtree
        )
    def fit(self, X, g, h):
        self.root = self.build_tree(X, g, h, depth = 0)
    def predict_single(self, x, node):
        #predicting a single point
        if node.value is not None:
            return node.value
        if x[node.feature_id] <= node.threshold:
            return self.predict_single(x, node.left)
        else:
            return self.predict_single(x, node.right)
    def predict(self, X):
        #predicting all the points
        return np.array([self.predict_single(x, self.root) for x in X])

class XGBoost:
        def __init__(self, n_estimators=20, learning_rate=0.1, max_depth=3, lambda_=1.0, gamma=0.0, delta = 1.0):
            self.n_estimators = n_estimators
            self.learning_rate = learning_rate
            self.max_depth = max_depth
            self.lambda_ = lambda_
            self.gamma = gamma
            self.delta = delta
            self.trees = []
            self.base_prediction = None
        def fit(self, X, y):
            X_array = np.array(X, dtype = float)
            y_array = np.array(y, dtype = float)
            #F0 = mean of y
            self.base_prediction = np.mean(y_array)
            current_predictions = np.full(y_array.shape, self.base_prediction)
            for i in range(self.n_estimators):
                #starting with mse loss here, only need to change these 2 lines of code to switch to huber loss
                a = current_predictions - y_array
                g = a/np.sqrt(1 + (a/self.delta)**2)
                h = 1.0/(1 + (a/self.delta)**2)**1.5
                #g = current_predictions - y_array
                #h = np.ones_like(y_array)
                tree = XGBRegressionTree(
                    max_depth=self.max_depth,
                    lambda_=self.lambda_,
                    gamma = self.gamma
                )
                tree.fit(X_array, g, h)
                self.trees.append(tree)
                current_predictions += self.learning_rate * tree.predict(X_array)
                loss = np.mean(self.delta**2 * (np.sqrt(1 + (a/self.delta)**2)-1))
                print(f"Tree {i+1} of {self.n_estimators} built! Training loss: {loss:.6f}")
            return self
        def predict(self, X):
            X_array = np.array(X, dtype = float)
            predictions = np.full(X_array.shape[0], self.base_prediction)
            for tree in self.trees:
                predictions += self.learning_rate * tree.predict(X_array)
            return predictions
        