import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#Number of input nodes, images are 28x28 px
n_nodes_input = 784

#Number of nodes in each hidden layer
n_nodes_hl1 = 100
#n_nodes_hl2 = 100
#n_nodes_hl3 = 100
#n_nodes_hl4 = 100

#Number of classes classes, 0-9
n_classes = 10

#Size of batch, number of images
batch_size = 100

#Matrix: height x width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


#Initialize weights and biases in the hidden layers 
hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([n_nodes_input, n_nodes_hl1])),
                  'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))
                }
                
output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
                  'biases': tf.Variable(tf.random_normal([n_classes]))
                 }
    
#input_data * weights + bias for each layer 
l1 = tf.add(tf.matmul(x, hidden_layer_1['weights']), hidden_layer_1['biases'])
l1 = tf.nn.relu(l1)

prediction = tf.add(tf.matmul(l1, output_layer['weights']), output_layer['biases'])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))

train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

n_epochs = 100
#summary_op = tf.merge_all_summaries()
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        batch_count = int(mnist.train.num_examples / batch_size)
        
        for i in range(batch_count):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            _,c = sess.run([train_step, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            epoch_loss += c

        print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)
        

    #Test
    correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print('Accuracy:', accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))