import tensorflow as tf
import pandas as pd
import numpy as np

one_hot_dict = {
    0: [0,0,1],
    1: [0,1,0],
    2: [1,0,0]
}

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
y_ = []

for i,v in enumerate(y):
    y_.append(one_hot_dict[v])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y_, test_size = .2)

#Number of input nodes - 4 features
n_nodes_input = 4

#Number of nodes in each hidden layer
n_nodes_hl1 = 100
n_nodes_hl2 = 200
n_nodes_hl3 = 200
n_nodes_hl4 = 100

#Number of classes classes - 
n_classes = 3

#Input tensor placeholder
x = tf.placeholder('float', [None, n_nodes_input])

#Output tensor placeholder
y = tf.placeholder('float', [None, n_classes])

#Defining our neural network model
def neural_network_model(data):

    #Initialize weights and biases in the hidden layers 
    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([n_nodes_input, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))
                    }
    
    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))
                     }
    
    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))
                     }
                     
    hidden_layer_4 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                  'biases': tf.Variable(tf.random_normal([n_nodes_hl4]))
                 }
    
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl4, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))
                     }
    
    #input * weights + bias for each layer 
    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)
    
    l4 = tf.add(tf.matmul(l3, hidden_layer_4['weights']), hidden_layer_4['biases'])
    l4 = tf.nn.relu(l4)
    
    output = tf.add(tf.matmul(l4, output_layer['weights']), output_layer['biases'])
    
    return output

#Training our neural network model
def train_neural_network(x):
    prediction = neural_network_model(x)
    
    #Using cross entropy as cost function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    
    #Using Gradient Descent as optimizer
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    
    #Number of epochs (iterations)
    n_epochs = 100
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        # Train
        for epoch in range(n_epochs):
            _,epoch_loss = sess.run([train_step, cross_entropy], feed_dict={x: x_train, y: y_train})
            print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)

        # Test
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
        
        # Computing accuracy based on the correct tensor
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        #print(tf.argmax(prediction,1).eval({x: x_test, y: y_test}))
        #print(y.eval({x: x_test, y: y_test}))
        print('Accuracy:', accuracy.eval({x: x_test, y: y_test}))

        #from sklearn.metrics import accuracy_score
        #print accuracy_score(y_test, predictions_dt)
        
        
train_neural_network(x)



