import numpy as np
import pandas as pd

class DecisionTreeRegression:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if self.max_depth is not None and depth >= self.max_depth:
            return self._terminal_node(y)
        if len(y) <= self.min_samples_split:
            return self._terminal_node(y)
        best_feature, best_split_value, best_left_X, best_left_y, best_right_X, best_right_y = self._find_best_split(X, y)
        left_node = self._build_tree(best_left_X, best_left_y, depth + 1)
        right_node = self._build_tree(best_right_X, best_right_y, depth + 1)
        return Node(best_feature, best_split_value, left_node, right_node)

    def _find_best_split(self, X, y):
        best_feature = None
        best_split_value = None
        best_gain = -float('inf')
        for feature_idx in range(X.shape[1]):
            sorted_indices = np.argsort(X[:, feature_idx])
            sorted_X = X[sorted_indices]
            sorted_y = y[sorted_indices]
            for i in range(1, len(sorted_X)):
                split_value = (sorted_X[i-1, feature_idx] + sorted_X[i, feature_idx]) / 2
                left_X, right_X = sorted_X[:i], sorted_X[i:]
                left_y, right_y = sorted_y[:i], sorted_y[i:]
                gain = self._impurity_gain(y, left_y, right_y)
                if gain > best_gain:
                    best_feature = feature_idx
                    best_split_value = split_value
                    best_gain = gain
                    best_left_X, best_left_y = left_X, left_y
                    best_right_X, best_right_y = right_X, right_y
        return best_feature, best_split_value, best_left_X, best_left_y, best_right_X, best_right_y

    def _impurity_gain(self, y, left_y, right_y):
        gini_parent = self._gini(y)
        gini_left = self._gini(left_y)
        gini_right = self._gini(right_y)
        weighted_gini = (len(left_y) * gini_left + len(right_y) * gini_right) / len(y)
        gain = gini_parent - weighted_gini
        return gain

    def _gini(self, y):
        classes = np.unique(y)
        gini = 0
        for c in classes:
            proportion = np.sum(y == c) / len(y)
            gini += proportion * (1 - proportion)
        return gini

    def _terminal_node(self, y):
        return PredictionNode(np.mean(y))

    def predict(self, X):
        return np.array([self._predict_node(x, self.root) for x in X])

    def _predict_node(self, x, node):
        if isinstance(node, PredictionNode):
            return node.prediction
        if x[node.feature] <= node.split_value:
            return self._predict_node(x, node.left)
        else:
            return self._predict_node(x, node.right)

class Node:
    def __init__(self, feature, split_value, left, right):
        self.feature = feature
        self.split_value = split_value
        self.left = left
        self.right = right

class PredictionNode:
    def __init__(self, prediction):
        self.prediction = prediction

cancer_data = pd.read_csv('../Assignment 2/Data/cancer.csv')
data = cancer_data.values

X = data[:, 1:]
y = data[:, 0]

num_train_samples = int(0.8 * len(data))
X_train, y_train = X[:num_train_samples], y[:num_train_samples]
X_test, y_test = X[num_train_samples:], y[num_train_samples:]

regressor = DecisionTreeRegression(max_depth=10 )
regressor.fit(X_train, y_train)

print("PREDICTED VALUE FOR THE TEST DATA PRESENT IN THE CSV FILE (1->Malignant, 0->Benign)")

for i, x in enumerate(X_test):
    prediction = regressor.predict([x])[0]
    print(f"[{x[::]}] ==> {prediction}")
    print()
    
print("----------------------------------------------------------------------------------------------------------------------------")
print("PREDICTED VALUE FOR THE USER INPUT DATA (1->Malignant, 0->Benign)")  
    
    
    
feature_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
                 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 
                 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 
                 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 
                 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 
                 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 
                 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

user_data = []
for feature in feature_names:
    value = float(input(f"Enter value for {feature}: "))
    user_data.append(value)

user_prediction = regressor.predict([user_data])[0]
print(f"Predicted outcome: {user_prediction}")
    

