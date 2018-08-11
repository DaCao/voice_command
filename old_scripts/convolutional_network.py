import tensorflow as tf
import numpy as np
import pickle
import re

# import
with open('training_v2.pickle', 'rb') as handle:
    parsed_df = pickle.load(handle)

# parameters
num_classes = 2
batch_size = 100
num_users = len(parsed_df.keys())
val_portion = 0.2
chunk_size = 40
n_chunks = 400
rnn_size = 128
hm_epochs = 100

# training and testing
training = {}
testing = {}
for key in parsed_df.keys():
    s = np.random.binomial(1, val_portion, 1)
    if s>0:# validation
        testing[key] = parsed_df[key]
    else:
        training[key] = parsed_df[key]

# placeholders
x = tf.placeholder('float', [None, training[key][2].shape[0], training[key][2].shape[1]])
y = tf.placeholder('float',[None,num_classes])

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,num_classes])),
             'biases':tf.Variable(tf.random_normal([num_classes]))}

    x_reshaped = tf.reshape(x,[-1,chunk_size,n_chunks])
    x_unstacked = tf.unstack(x_reshaped,axis = 1)

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = tf.nn.static_rnn(lstm_cell, x_unstacked, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output


