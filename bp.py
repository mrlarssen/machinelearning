import tensorflow as tf
import pandas as pd
import numpy as np

np.set_printoptions(threshold='nan', suppress=True)

    
n_nodes_hl1 = 100
n_nodes_hl2 = 200
n_nodes_hl3 = 200
n_nodes_hl4 = 100

#Input tensor placeholder
x = tf.placeholder('float', [None, 0])
#Output tensor placeholder
y = tf.placeholder('float', [None, 0])

class BP:
    
    def __init__(self,x_train,y_train,x_test, y_test,xn,yn):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.n_nodes_input = xn
        self.n_classes = yn
        x = tf.placeholder('float', [None, self.n_nodes_input])
        y = tf.placeholder('float', [None, self.n_classes])

        
    def model(self, data):
        #Initialize weights and biases in the hidden layers 
        hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([self.n_nodes_input, n_nodes_hl1])),
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
    
        output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl4, self.n_classes])),
                          'biases': tf.Variable(tf.random_normal([self.n_classes]))
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
    def run(self):
        prediction = self.model(x)
        
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
                _,epoch_loss = sess.run([train_step, cross_entropy], feed_dict={x: self.x_train, y: self.y_train})
                print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)
                
            
            # Test
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y,1))
            
            # Computing accuracy based on the correct tensor
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', accuracy.eval({x: self.x_test, y: self.y_test}))

    
