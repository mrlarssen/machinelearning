import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#Number of input nodes, images are 28x28 px
n_nodes_input = 784

#Number of nodes in each hidden layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

#Number of classes classes, 0-9
n_classes = 10

#Size of batch, number of images
batch_size = 100

#Matrix: height x width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

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
    
    #input_data * weights + bias for each layer 
    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    
    return output

#Training our neural network model
def train_neural_network(x):
    prediction = neural_network_model(x)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
    
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    
    n_epochs = 100
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        #Train
        for epoch in range(n_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                batch_x,batch_y = mnist.train.next_batch(batch_size)
                #print(batch_x.shape)
                #print(batch_y.shape)
                _,c = sess.run([train_step, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)
        
        #Test
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)