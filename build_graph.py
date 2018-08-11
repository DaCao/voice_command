import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn
import numpy as np
import re

class Graph(object):
    def __init__(self):

        self.numFeatures =
        self.nClasses =  # num_labels
        self.batchSize =
        self.numTimeSteps =
        self.nHidden =

        ####Learning Parameters
        self.learningRate = 0.001
        self.momentum = 0.9
        self.nEpochs = 300
        self.Size = 4

    def build_graph(self):

        #### Graph input
        inputX = tf.placeholder(tf.float32, shape=(self.batchSize, self.numFeatures, self.numTimeSteps))

        # Prep input data to fit requirements of rnn.bidirectional_rnn
        #  Reshape to 2-D tensor (nTimeSteps*Size, nfeatures)
        inputXrs = tf.reshape(inputX, [-1, self.numFeatures])

        #  Split to get a list of 'n_steps' tensors of shape (_size, n_hidden)
        inputList = tf.split(axis=0, num_or_size_splits=self.maxTimeSteps, value=inputXrs) # so feeding batch_item by batch_item...

        targetIxs = tf.placeholder(tf.int64)
        targetVals = tf.placeholder(tf.int32)
        targetShape = tf.placeholder(tf.int64)
        targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
        seqLengths = tf.placeholder(tf.int32, shape=self.Size)

        #### Weights & biases
        stddev = np.sqrt(2.0 / (2 * self.nHidden))
        truncated_normal = tf.truncated_normal([2, self.nHidden], stddev=stddev) # https://www.tensorflow.org/api_docs/python/tf/truncated_normal
    	weightsOutH1, biasesOutH1 = tf.Variable(truncated_normal), tf.Variable(tf.zeros([self.nHidden]))
        weightsOutH2, biasesOutH2 = tf.Variable(truncated_normal), tf.Variable(tf.zeros([self.nHidden]))

        half_normal = tf.truncated_normal([self.nHidden, self.nClasses], stddev=np.sqrt(2.0 / self.nHidden))
        weightsClasses = tf.Variable(half_normal)
        biasesClasses = tf.Variable(tf.zeros([self.nClasses]))

        ####Network
        forwardH1 = rnn_cell.LSTMCell(self.nHidden, use_peepholes=True, state_is_tuple=True)
        backwardH1 = rnn_cell.LSTMCell(self.nHidden, use_peepholes=True, state_is_tuple=True)
        fbH1, _, _ = bidirectional_rnn(forwardH1, backwardH1, inputList, dtype=tf.float32, scope='BDLSTM_H1')
        # Output Tensor, fbH1,  shaped: [batch_size, max_time, layers_output]  we have 2 layers, so layers_output = 2


        """
        Yes the outputs of the two directions are concatenated on the last dimension,
         so to get all forward outputs out[:, :, :hidden_size] and backwards out[:, :, hidden_size:]
        """

        print("building fbH1rs ")
    	fbH1rs = [tf.reshape(t, [self.Size, 2, self.nHidden]) for t in fbH1]

        print("building outH1 ")
        # for each bathc item in fbH1rs, dim = [max_time, layers_output]

        #                    [self.Size, 2, self.nHidden]  [2, self.nHidden]  [self.nHidden]
        outH1 = [tf.reduce_sum(tf.multiply(t, weightsOutH1), axis=1) + biasesOutH1 for t in fbH1rs]


        print("building logits ")
        #     [self.nHidden]     [self.nHidden, self.nClasses]       [self.nClasses]
        logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in outH1]

        print("len(outH1) %d"% len(outH1))

        ####Optimizing
        print("building loss")
        logits3d = tf.stack(logits)
        loss = tf.reduce_mean(ctc.ctc_loss(logits3d, targetY, seqLengths))  # a vector of 4 elements
        out = tf.identity(loss, 'ctc_loss_mean')
        optimizer = tf.train.MomentumOptimizer(self.learningRate, momentum).minimize(loss)



        return 0