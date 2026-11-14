import numpy as np

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Index of feature split on
        self.threshold = threshold      # Value to split at
        self.left = left                # Left child node
        self.right = right              # Right child node
        self.value = value              # Target value (leaf prediction)

class DecisionTree:
    def __init__(self, max_depth=5, min_sample_split=2, max_features=None, rng=None):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.max_features = max_features  # Number of random features per split
        self.rng = rng if rng is not None else np.random.RandomState()
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        # Base case: stopping criteria
        if depth >= self.max_depth or n_samples < self.min_sample_split or len(np.unique(y)) == 1:
            return Node(value=np.mean(y))

        # Find the best split across a random subset of features
        best_feat, best_thresh = self._best_split(X, y, n_samples, n_features)

        # If no split improves variance, create a leaf
        if best_feat is None:
            return Node(value=np.mean(y))

        # Split data on best threshold
        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = X[:, best_feat] > best_thresh

        # Guard against zero-division / no-op splits
        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            return Node(value=np.mean(y))

        # Recursively construct subtrees
        left_child = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(feature_idx=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def _best_split(self, X, y, n_samples, n_features):
        best_split_var = np.var(y)  # Must strictly improve upon current node variance
        best_feat, best_thresh = None, None

        # RANDOM FOREST FEATURE SUBSAMPLING:
        # Determine number of features to consider
        if self.max_features is None:
            n_sub_features = n_features
        elif isinstance(self.max_features, int):
            n_sub_features = min(self.max_features, n_features)
        elif self.max_features == 'sqrt':
            n_sub_features = max(1, int(np.sqrt(n_features)))
        else:
            n_sub_features = n_features

        # Randomly choose subset of feature indices at this split
        feat_idxs = self.rng.choice(n_features, n_sub_features, replace=False)

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            unique_sorted_values = np.sort(np.unique(X_column))
            if len(unique_sorted_values) <= 1:
                continue

            thresholds = (unique_sorted_values[:-1] + unique_sorted_values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = X_column <= threshold
                right_mask = ~left_mask

                left_y = y[left_mask]
                right_y = y[right_mask]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                # Weighted variance
                n_l, n_r = len(left_y), len(right_y)
                var_l = np.var(left_y)
                var_r = np.var(right_y)
                weighted_var = (n_l * var_l + n_r * var_r) / (n_l + n_r)

                if weighted_var < best_split_var:
                    best_split_var = weighted_var
                    best_feat = feat_idx
                    best_thresh = threshold

        return best_feat, best_thresh

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForest:
    def __init__(self, n_estimators=10, max_depth=5, min_sample_split=2, 
                 max_features='sqrt', random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.trees = []
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]

        for _ in range(self.n_estimators):
            # Bootstrap sample (sampling rows with replacement)
            idxs = rng.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[idxs]
            y_bootstrap = y[idxs]

            # Fit individual tree with random feature selection
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_sample_split=self.min_sample_split,
                max_features=self.max_features,
                rng=rng
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Collect predictions from each tree
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        # Average the predictions across all trees
        return np.mean(tree_preds, axis=0)