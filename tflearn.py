from sklearn import metrics, model_selection
import tensorflow as tflearn
from tensorflow.contrib import learn

#Load dataset.
iris = learn.datasets.load_dataset('iris')
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)
    
#Build 3 layer DNN with 10,20,10 units respectively.
classifier = learn.DNNClassifier(hidden_units=[100,200,200,100], n_classes=3)

#Fit and predict.
classifier.fit(x_train, y_train, steps=1000)
score = metrics.accuracy_score(y_test, classifier.predict(x_test))

print('Accuracy: {0:f}'.format(score))