import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(X_train.shape)
# print(X_train[0])
# print(y_train.shape)
# print(y_train[0])

# plt.figure()
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, edgecolor='k', s=20)
# plt.title('Training Data')
# plt.show()

# from collections import Counter
# a = [1,1,2,3,3,3,3]
# most_common = Counter(a).most_common(1)
# print(most_common)

from knn import KNN
clf = KNN(k=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
acc = np.sum(y_pred == y_test) / len(y_test)
print(acc)