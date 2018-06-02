import tensorflow as tf
import pickle
import numpy as np

#Load data from pickle file
pickle_file = 'sentiment_set.pickle'

with open(pickle_file, 'rb') as f:
  data_list = pickle.load(f)
  train_x = data_list[0]
  train_y = data_list[1]
  test_x = data_list[2]
  test_y = data_list[3]
  del data_list
  #print('Training set', train_x.shape, train_y.shape)
  #print('Test set', test_x.shape, test_y.shape)

n_input_size = len(train_x[0])
n_classes = 2
batch_size = 100
#Define the structure of deep neural network
nodes = [n_input_size, 1024, 128, 128, 512, n_classes]

#28*28 = 784
x = tf.placeholder('float',[None,n_input_size])
y = tf.placeholder('float')

#defining layers in dictionary for easy access, not flexible defenition
def neural_net_model(data):
    layer = []
    for i in range(0,len(nodes)-1):
        layer_var = {'weights':tf.Variable(tf.random_normal([nodes[i], nodes[i+1]])), 'biases': tf.Variable(tf.random_normal([nodes[i+1]]))}
        layer.append(layer_var)

    layer_data = data
    for i in range(0,len(layer)-1):
        layer_data = tf.nn.relu(tf.add(tf.matmul(layer_data,layer[i]['weights']),layer[i]['biases']))

    return tf.matmul(layer_data, layer[len(layer)-1]['weights']) + layer[len(layer)-1]['biases']

def train_neural_net(x):
    prediction = neural_net_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost) #default lr=0.001

    n_epochs = 20
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(n_epochs):
            epoch_loss = 0
            """for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)"""
            index = 0
            while index < len(train_x):
                start = index
                end = index + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer,cost], feed_dict = {x:batch_x, y:batch_y})
                epoch_loss = epoch_loss + c
                index = index + batch_size
            print('Epoch',epoch,'completed out of',n_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

train_neural_net(x)
