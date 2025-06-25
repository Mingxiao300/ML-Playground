"""
KNN Core concepts
- Euclidean distance
- No training (fit) process, learn from data at prediction time
- Follow sklearn conventions: fit, predict, score

Modules:
- numpy euclidean_distance function: np.linalg.norm(x-y)
- python collections Counter for most common label: Counter(k_nearest_labels).most_common(1) --> returning list of tuples [(number, count)]

"""

import numpy as np
from collections import Counter

# np.linalg.norm()
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(x1-x2)**2)

class KNN:
    
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        # store training data
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        # list comprehension to predict labels for each instance in X
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self, x):
        # calculate distances from x to all training samples
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k] #np.argsort returns indices
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # get most common label among k nearest samples
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]