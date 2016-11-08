import tensorflow as tf
import pandas as pd
import numpy as np

np.set_printoptions(threshold='nan', suppress=True)
tf.reset_default_graph()

#Number of input nodes - 17 features
n_nodes_input = 2

#Number of nodes in each hidden layer
n_nodes_hl1 = 100
n_nodes_hl2 = 100
n_nodes_hl3 = 100

#Number of classes classes - 1,0
n_classes = 2

#Mapping
X_ = [[0,0], [0,1], [1,0], [1,1]]
Y_ = [[0,1], [1,0], [1,0], [0,1]] # One-hot, [1,0] = False, [0,1] = True

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
    
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))
                     }
    
    #input * weights + bias for each layer 
    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.sigmoid(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.sigmoid(l2)
    
    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.sigmoid(l3)
    
    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
    
    return output

#Training our neural network model
def train_neural_network(x):
    prediction = neural_network_model(x)
    
    #Using cross entropy as cost function
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction, y))
    tf.scalar_summary('cost_function', cross_entropy)
    #Using Gradient Descent as optimizer
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    tf.scalar_summary('accuracy', accuracy)
    
    #Number of epochs (iterations)
    n_epochs = 100
    
    #Merge all summaries
    summary_op = tf.merge_all_summaries()
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        writer = tf.train.SummaryWriter('/Users/nicholaslarssen/Development/Python/deeplearning/logs/', sess.graph)

        # Train
        for epoch in range(n_epochs):
            _,epoch_loss, summary = sess.run([train_step, cross_entropy, summary_op], feed_dict={x: X_, y: Y_})
            print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)
            
            writer.add_summary(summary, epoch)
            
        # Test
        print('Accuracy:', accuracy.eval({x:X_, y: Y_}))


train_neural_network(x)
