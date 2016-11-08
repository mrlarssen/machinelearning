from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import random
from knn import KNN


iris = load_iris()

features = pd.DataFrame(iris.feature_names)
target = pd.DataFrame(iris.target_names)
data = pd.DataFrame(iris.data[0])
print len(iris.data)

test_ids = [0,50,100]

#training data
train_data = np.delete(iris.data, test_ids, axis=0)
train_target = np.delete(iris.target, test_ids)

print(train_data)
print(train_target)

#testing data
test_data = iris.data[test_ids]
test_target = iris.target[test_ids]

print(test_data)
print(test_target)

from sklearn.model_selection import train_test_split
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

from sklearn import tree
dt = tree.DecisionTreeClassifier()

dt.fit(X_train, y_train)
predictions_dt = dt.predict(X_test)

#from sklearn.neighbors import KNeighborsClassifier
#kn = KNeighborsClassifier()
knn = KNN()

knn.fit(X_train, y_train)
predictions_kn = knn.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions_dt)
print accuracy_score(y_test, predictions_kn)
