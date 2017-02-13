import tensorflow as tf
import pandas as pd
import numpy as np

np.set_printoptions(threshold='nan', suppress=True)

""" 

---------- PRE PROCESSING START ----------

"""

df_main = pd.read_csv('/Users/nicholaslarssen/Development/Python/deeplearning/churn.csv')

col_names = df_main.columns.tolist()

# Isolate target data
churn_result = df_main['Churn?']
df_churn_result = pd.DataFrame(churn_result)

# Convert Boolean values to Binary values
Y = np.where(churn_result == 'True.', 1, 0)
df_Y = pd.DataFrame(Y)

# Extract first 2500 rows as train texpected
train_expected = Y[:2500]
df_train_expected = pd.DataFrame(train_expected)

# Extract last 833 rows as test expected
test_expected = Y[2500:]
df_test_expected = pd.DataFrame(test_expected)

# Dropping unneccessary columns and creating our feature space
to_drop = ['State','Area Code','Phone','Churn?']
df_churn_feature_space = df_main.drop(to_drop,axis=1)

# Convert yes/no to Boolean values
yes_no_cols = ["Int'l Plan", "VMail Plan"]
df_churn_feature_space[yes_no_cols] = df_churn_feature_space[yes_no_cols] == 'yes'

#Feature space as matrix
X = df_churn_feature_space.as_matrix().astype(np.float)

#Normalizing each feature in X within a range of -1.0 to 1.0
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
df_X = pd.DataFrame(X)

# Extract first 2500 rows as train input
train_data = X[:2500]
df_train_data = pd.DataFrame(train_data)

# Extract last 833 rows as test input
test_data = X[2500:]
df_test_data = pd.DataFrame(test_data)

print "Feature space holds %d observations and %d features" % X.shape
print "Unique target labels:", np.unique(Y)

""" 

---------- PRE PROCESSING END ----------

"""

#Number of input nodes - 17 features
n_nodes_input = 17

#Number of nodes in each hidden layer
n_nodes_hl1 = 100
n_nodes_hl2 = 100
n_nodes_hl3 = 100

#Number of classes classes - 1,0
n_classes = 1

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
    
    #Using Gradient Descent as optimizer
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    
    #Number of epochs (iterations)
    n_epochs = 100
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        # Train
        for epoch in range(n_epochs):
            _,epoch_loss = sess.run([train_step, cross_entropy], feed_dict={x: train_data, y: np.transpose([train_expected])})
            print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)

        # Test
        # If output from sigmoid is greater than 0.5, then return 1 else 0
        pred = tf.select(tf.greater_equal(prediction, 0.5), tf.fill([833,1],1.0), tf.fill([833,1],0.0))
        
        # Tensor of True/False corresponding to correct or not correct prediction 
        correct = tf.equal(pred, y)
        #correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
        
        print(tf.shape(pred).eval({ x: test_data, y: np.transpose([test_expected]) }))
        
        # Computing accuracy based on the correct tensor
        #accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        #print('Accuracy:', accuracy.eval({x: test_data, y:np.transpose([test_expected])}))
        #print(pred.eval({x: test_data}))
        #print(y.eval({x: test_data, y:np.transpose([test_expected])}))
    

train_neural_network(x)



#Pull out  for future use
#features = df_churn_feature_space.columns
