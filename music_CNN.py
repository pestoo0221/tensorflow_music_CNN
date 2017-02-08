#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Jidan Zhong 
# 2017- Feb - 02
import pandas as pd
import tensorflow as tf
import random
import os
import tarfile
import numpy as np
import pandas as pd
import time
from tensorflow.python.ops import control_flow_ops
import pickle

##################################
##      Load input data
# f = open('data.pckl', 'wb')
# pickle.dump([x_train_new, y_train_new, x_test_new, y_test_new], f,protocol=-1)
# f.close()
################################# 
f = open('data.pckl', 'rb')
[x_train_new, y_train_new, x_test_new, y_test_new] = pickle.load(f)
f.close()

# x_train_new: list of arrays: 8496 samples, each sample is 40X217 
# y_train_new: list: 0:2831 -- label 0 - Alternative; 2832:5663 -- label 1 - Hiphop; 5664:8495 -- label 2 - Rock; 
# x_test_new: list of arrays: 348 samples, each sample is 40X217 
# y_test_new: list: 0:115 -- label 0 - Alternative; 116:231 -- label 1 - Hiphop; 232:348 -- label 2 - Rock; 

#######
# process data: training and testing data

y_train_f = np.eye(3)[y_train_new]
y_test_f = np.eye(3)[y_test_new] #class_output
#####################################################
def batch_norm(x, n_out, phase_train, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def variable_summaries(var):                ###################################################################ADDDDDED########
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='VALID') #SAME
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides = [1,2,2,1], padding='VALID')#SAME

################################################

################################################

# numFeatures is the number of features in our input data.
width = 40
height = 217 # 434
flat = width * height
class_output = 3

with tf.Graph().as_default():
    with tf.name_scope('input'):
        # 'None' means TensorFlow shouldn't expect a fixed number in that dimension
        # X = tf.placeholder(tf.float32, [None, flat], name='x-input') # Iris has 4 features, so X is a tensor to hold our data.
        X = tf.placeholder(tf.float32, shape=[None, width, height], name='x-input')
        yGold = tf.placeholder(tf.float32, [None, class_output],name ='y-input') # This will be our correct answers matrix for 3 classes.
    
    with tf.name_scope('input_reshape'):   ###################################################################ADDDDDED########
		x_image = tf.reshape(X,[-1,width,height,1])
		tf.summary.image('input', x_image, 4) 
    with tf.name_scope('model'):
        # first layer
        with tf.name_scope('layer1'):
            with tf.name_scope('weights'):
                W_conv1 = weight_variable([5,5,1,12]) #Randomly sample from a normal distribution with standard deviation .01
                variable_summaries(W_conv1)

            with tf.name_scope('bias'):    
                b_conv1 = bias_variable([12])
                variable_summaries(b_conv1)

            with tf.name_scope('conv1'):
                convolve1 = conv2d(x_image, W_conv1) + b_conv1
                h_conv1 = tf.nn.relu(convolve1)
            with tf.name_scope('pool1'):
                h_pool1 = max_pool_2x2(h_conv1)
                tf.summary.histogram('activations', h_pool1)

        with tf.name_scope('layer2'):
            with tf.name_scope('weights'):
                W_conv2 = weight_variable([5,5,12,24]) #Randomly sample from a normal distribution with standard deviation .01
                variable_summaries(W_conv2)

            with tf.name_scope('bias'):    
                b_conv2 = bias_variable([24])
                variable_summaries(b_conv2)

            with tf.name_scope('conv2'):
                convolve2 = conv2d(h_pool1, W_conv2) + b_conv2
                h_conv2 = tf.nn.relu(convolve2) #######
                tf.summary.histogram('activations', h_conv2)
            with tf.name_scope('pool2'):
                h_pool2 = max_pool_2x2(h_conv2)
                tf.summary.histogram('activations', h_pool2)
        
        with tf.name_scope('layer3'):
            with tf.name_scope('weights'):
                W_conv3 = weight_variable([5,5,24,48]) #Randomly sample from a normal distribution with standard deviation .01
                variable_summaries(W_conv2)

            with tf.name_scope('bias'):    
                b_conv3 = bias_variable([48])
                variable_summaries(b_conv2)

            with tf.name_scope('conv2'):
                convolve3 = conv2d(h_pool2, W_conv3) + b_conv3
                h_conv3 = convolve3 # tf.nn.relu(convolve2)
                tf.summary.histogram('activations', h_conv2)
            with tf.name_scope('pool2'):
                h_pool3 = max_pool_2x2(h_conv3)
                tf.summary.histogram('activations', h_pool3)

        with tf.name_scope('Batch_norm') as scope:
            bn_train = tf.placeholder(tf.bool) #Boolean value to guide batchnorm #sess.run(train_step,feed_dict={x:X_train[batch_ind], y_: y_train[batch_ind], keep_prob: dropout, bn_train : True})
            a_bn1 = batch_norm(h_pool3,48,bn_train, 'bn2')
            h_bn1 = tf.nn.relu(a_bn1)
        k = 1 # 7
        kernel = 48 #24
        poolwidth = 23 #51
        N1 = 500#1024   
        N2 = 100 # I dont really need this layer
        # N3 = 100
        with tf.name_scope('layer3_fc'):
            with tf.name_scope('weights'):
                W_fc1 = weight_variable([k*poolwidth*kernel,N1]) # fully connected layer: (((40-5+1)/2)-5+1)/2 = 7,  (((434-5+1)/2)-5+1)/2 = 104 element; 24 feature maps ; total 
                variable_summaries(W_fc1)
            with tf.name_scope('bias'):    
                b_fc1 = bias_variable([N1]) # need  biases
                variable_summaries(b_fc1)
            with tf.name_scope('fully_connected'):
                layer2_matrix = tf.reshape(h_bn1,[-1,k*poolwidth*kernel]) 
                # layer2_matrix = tf.reshape(h_pool2,[-1,7*105*24]) # flatten layer2
                matmul1_fc1 = tf.matmul(layer2_matrix,W_fc1) + b_fc1 #tf.nn.softmax(tf.matmul(layer2_matrix,W_fc1) + b_fc1)
                h_fc1 = tf.nn.relu(matmul1_fc1)
                tf.summary.histogram('activations', h_fc1)

        with tf.name_scope('layer4_fc'):
            with tf.name_scope('weights'):
                W_hfc1 = weight_variable([N1,N2]) # fully connected layer: (((40-5+1)/2)-5+1)/2 = 7,  (((434-5+1)/2)-5+1)/2 = 104 element; 24 feature maps ; total 
                variable_summaries(W_hfc1)
            with tf.name_scope('bias'):    
                b_hfc1 = bias_variable([N2]) # need  biases
                variable_summaries(b_hfc1)
            with tf.name_scope('fully_connected'):
                hlayer2_matrix = tf.reshape(h_fc1,[-1,N1]) 
                # layer2_matrix = tf.reshape(h_pool2,[-1,7*105*24]) # flatten layer2
                matmul1_hfc1 = tf.matmul(hlayer2_matrix,W_hfc1) + b_hfc1 #tf.nn.softmax(tf.matmul(layer2_matrix,W_fc1) + b_fc1)
                h_hfc1 = tf.nn.relu(matmul1_hfc1)
                tf.summary.histogram('activations', h_hfc1)

        with tf.name_scope('Dropout'):
            # Optional phase for reducing overfitting - Dropout        
            keep_prob = tf.placeholder(tf.float32)
            layer3_drop = tf.nn.dropout(h_hfc1, keep_prob)

        with tf.name_scope('Output_layer'):
            with tf.name_scope('weights'):
               # final layer: fully connected ; readout layer
                W_fc2 = weight_variable([N2,class_output])
                variable_summaries(W_fc2)
            with tf.name_scope('bias'):    
                b_fc2 = bias_variable([class_output]) 
                variable_summaries(b_fc2)
            with tf.name_scope('softmax'):
                y_conv = tf.nn.softmax(tf.matmul(layer3_drop,W_fc2) + b_fc2)


    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, yGold))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        learningRate =0.0001
        training = tf.train.AdamOptimizer(learningRate).minimize(loss)
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(yGold, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()    
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    start = time.time()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess: # the first one means if the device doesnt exist, it can automatically appoint an existing device; 2nd means it will show the log infor for parameters and operations are on which device
        train_writer = tf.summary.FileWriter('/home/jidan/test/train', sess.graph)
        test_writer = tf.summary.FileWriter('/home/jidan/test/test', sess.graph)
        sess.run(init,feed_dict={bn_train.name: True})

        for Epoch in range(40):
            tot = len(x_train_new)
            batch = 118
            random.seed(Epoch)
            lis = range(tot)
            random.shuffle(lis)
            for n in range(0,tot,batch):

                x_train_final = [x_train_new[f] for f in lis[n:n+batch]] + [x_train_new[tot+f] for f in lis[n:n+batch]] + [x_train_new[tot*2+f] for f in lis[n:n+batch]]
                y_train_final = [y_train_f[f] for f in lis[n:n+batch]] + [y_train_f[tot+f] for f in lis[n:n+batch]] + [y_train_f[tot*2+f] for f in lis[n:n+batch]]

                summary, acc, ls, trainstep,prediction = sess.run([merged, accuracy, loss, training,y_conv],feed_dict={X: x_train_final, yGold: y_train_final, keep_prob: 0.6, bn_train: True}) #, bn_train: True
                train_writer.add_summary(summary, n)
                print 'epoch %d, step %d, loss %g, acc %f ' %(Epoch,n,ls,acc)
            
            summary, acc = sess.run([merged, accuracy], feed_dict={X: x_test_new, yGold: y_test_f, keep_prob: 1.0, bn_train: False}) #, bn_train: False
            test_writer.add_summary(summary, Epoch) 
            print 'epoch %d, testing acc %f ' %(Epoch,acc)

            if Epoch % 10 == 0:
                saver.save(sess, '/media/truecrypt1/Research/model', global_step=Epoch)
        train_writer.close()
        test_writer.close()

    end = time.time()-start
    print 'time spent running %f' %(end)