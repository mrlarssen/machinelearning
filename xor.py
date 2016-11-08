import tensorflow as tf
import numpy as np

tf.reset_default_graph()

#Mapping
X_ = [[0,0], [0,1], [1,0], [1,1]]
Y_ = [[1,0], [0,1], [0,1], [1,0]] # One-hot, [1,0] = False, [0,1] = True

n_nodes_input = 2
n_nodes_hl1 = 20
n_classes = 2

x = tf.placeholder("float", [None, n_nodes_input])
y = tf.placeholder("float", [None, n_classes])

#Defining our neural network model
def neural_network_model(data):

    #Initialize weights and biases in the hidden layers 
    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([n_nodes_input, n_nodes_hl1])),
                      'biases': tf.Variable(tf.zeros([n_nodes_hl1]))
                    }
    
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
                      'biases': tf.Variable(tf.zeros([n_classes]))
                    }
    
    #input * weights + bias for each layer 
    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)
    
    output = tf.add(tf.matmul(l1, output_layer['weights']), output_layer['biases'])
    
    return output


def train_neural_network(d):
    prediction = neural_network_model(d)
    
    #Using cross entropy as cost function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    
    #Using the Adam algorithm as optimizer
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
    #Number of epochs (iterations)
    n_epochs = 1000
    
    #For plotting in tensorboard
    tf.scalar_summary('cost_function', cross_entropy)
    tf.scalar_summary('accuracy', accuracy)
    summary_op = tf.merge_all_summaries()
    
    with tf.Session() as sess:
        writer = tf.train.SummaryWriter('/Users/nicholaslarssen/Development/Python/deeplearning/logs/', sess.graph)
        sess.run(tf.initialize_all_variables())

        # Train
        for epoch in range(n_epochs):
            _,epoch_loss,summary = sess.run([train_step, cross_entropy, summary_op], feed_dict={x: X_, y: Y_})
            print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)

            #Logging for tensorboard
            writer.add_summary(summary, epoch)

        # Test
        print('Accuracy:', accuracy.eval({x:X_, y:Y_}))

train_neural_network(x)
    
    
    