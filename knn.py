import random
from scipy.spatial import distance

"""    
Pros:
    - Relatively simple

Cons:
    - Computationally intensive
    - Hard to represent relationships between features




"""

    

class KNN():
    
    def euc(self, a,b):
        return distance.euclidean(a,b)
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
        
    def closest(self, row):
        best_dist = self.euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = self.euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]
                
                
                
                
