import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

np.set_printoptions(threshold='nan', suppress=True)


scaler = StandardScaler()

df = pd.read_csv('/Users/nicholaslarssen/Development/Python/deeplearning/churn.csv')

col_names = df.columns.tolist()

# Isolate target data
churn_result = df['Churn?']
Y = np.where(churn_result == 'True.', 1, 0)

train_expected = Y[:2500]
test_expected = Y[2500:]

# We don't need these columns
to_drop = ['State','Area Code','Phone','Churn?']
churn_feature_space = df.drop(to_drop,axis=1)

yes_no_cols = ["Int'l Plan", "VMail Plan"]
churn_feature_space[yes_no_cols] = churn_feature_space[yes_no_cols] == 'yes'

#Pull out features for future use
features = churn_feature_space.columns

input_data = churn_feature_space.as_matrix().astype(np.float)
X = scaler.fit_transform(input_data)

#Y = pd.DataFrame(X)

train_data = X[:2500]
test_data = X[2500:]

print "Feature space holds %d observations and %d features" % X.shape
print "Unique target labels:", np.unique(res)

#Number of input nodes - 17
n_nodes_input = 17

#Number of nodes in each hidden layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

#Number of classes classes - 1,0
n_classes = 1

#Matrix: height x width
x = tf.placeholder('float', [None, n_nodes_input])
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
    
    #input_data * weights + bias for each layer 
    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.sigmoid(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.sigmoid(l2)
    
    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.sigmoid(l3)
    
    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
    #output = tf.nn.sigmoid(output)
    
    return output

#Training our neural network model
def train_neural_network(x):
    prediction = neural_network_model(x)

    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction, y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    
    n_epochs = 20
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        #Train
        for epoch in range(n_epochs):
            _,epoch_loss = sess.run([train_step, cross_entropy], feed_dict={x: train_data, y: np.transpose([train_expected])})
            print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)

        #Test
        pred = tf.select(tf.greater_equal(prediction, 0.5), tf.fill([833,1],1.0), tf.fill([833,1], 0.0))
        #correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        correct = tf.equal(pred, y)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_data, y:np.transpose([test_expected])}))
        #print(pred.eval({x: test_data}))
        #print(y.eval({x: test_data, y:np.transpose([test_expected])}))
    

train_neural_network(x)



