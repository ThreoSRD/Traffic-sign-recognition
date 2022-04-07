#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:38:23 2018

@author: Wilkins
"""

import tensorflow as tf
import input_data

#%%
IMG_W = 32
IMG_H = 32
N_CLASSES = 62
BATCH_SIZE = 50
learning_rate = 0.0001
MAX_STEP = 1000

#%%
def conv(layer_name, inData,in_channels, out_channels, kernel_size=[3,3], stride=[1,1,1,1], stddev=0.1):
    ''' Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name:   e.g. conv1, conv2...
        inData:		input tensor, [batch_size, height, width, channels]
        out_channels:	number of output channels (or convolution kernels)
        kernel_size:	the size of convolution kernel, VGG paper used: [3, 3]
        stride:		a list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        stddev:
    Returns:
        4D tensor
    '''
    with tf.variable_scope(layer_name):
        weight  = tf.Variable(tf.truncated_normal([kernel_size[0],kernel_size[1],in_channels, out_channels],stddev=stddev),name='weight')
        biase   = tf.Variable(tf.constant(0.1,shape=[out_channels]),name='biase')
        
        inData = tf.nn.conv2d(inData, weight, stride, padding='SAME', name='conv')
        #inData = tf.nn.bias_add(inData, biases, name='bias_add')
        inData = tf.nn.relu(inData+biase, name='relu')
        return inData
    
def pool(layer_name, x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True):
	'''Pooling op
	Args:
		x:		input tensor
		kernel:	pooling kernel, VGG paper used [1,2,2,1], the size of kernel is 2x2
		stride:	stride size, VGG paper used [1,2,2,1]
		padding:
		is_max_pool: boolen
					if True: use max pooling
					else: use avg pooling
	'''
	if is_max_pool:
		output = tf.nn.max_pool(x, ksize=kernel, strides=stride, padding='SAME', name=layer_name)
	else:
		output = tf.nn.avg_pool(x, ksize=kernel, strides=stride, padding='SAME', name=layer_name)
	return output

#%%
x = tf.placeholder(tf.float32,[None,IMG_W,IMG_H,3],name='x')
#x_inputs=tf.reshape(-1,32,32,1)
y = tf.placeholder(tf.float64,[None,N_CLASSES],name='y')
keep_pro=tf.placeholder(tf.float32)

l1 = conv('conv1',x, in_channels=3, out_channels=36,kernel_size=[5,5])
p1 = pool('p1',l1,is_max_pool=True)

l2 = conv('conv2',p1, in_channels=36, out_channels=72,kernel_size=[5,5])
p2 = pool('p2',l2,is_max_pool=True)

with tf.name_scope('fc1'):
    flat_image=tf.reshape(p2,[-1,8*8*72])
    w3  = tf.Variable(tf.truncated_normal([8*8*72,1024],stddev=0.1),name='w3')
    b3  = tf.Variable(tf.constant(0.1,shape=[1024]),name='b3')
    f1  = tf.nn.relu(tf.matmul(flat_image,w3)+b3)
    d1  = tf.nn.dropout(f1,keep_prob=keep_pro)
    
with tf.name_scope('fc2'):
    w4  = tf.Variable(tf.truncated_normal([1024,N_CLASSES],stddev=0.1),name='w4')
    b4  = tf.Variable(tf.constant(0.1,shape=[N_CLASSES]),name='b4')
    result = tf.nn.softmax(tf.matmul(d1,w4)+b4)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=result))
train= tf.train.AdamOptimizer(learning_rate).minimize(loss)

deal_result = tf.equal(tf.argmax(result,1),tf.argmax(y,1))
acc = tf.reduce_mean(tf.cast(deal_result,tf.float32))


#%%
if __name__ == '__main__':

    train_data_dir  ='./data/Training/'
    test_data_dir   ='./data/Testing/'
    saver = tf.train.Saver(tf.global_variables())
    
    train_images,train_labels   = input_data.get_files(train_data_dir)
    test_images,test_labels     = input_data.get_files(test_data_dir) 
    n_batch=len(train_images)//50
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        train_images_batch_t,train_labels_batch_t   = input_data.get_batch(sess,train_images,train_labels,IMG_W,IMG_H,BATCH_SIZE,400)

        test_images_batch_t,test_labels_batch_t     = input_data.get_batch(sess,test_images,test_labels,IMG_W,IMG_H,2520,2520)

        for step in range(MAX_STEP):
            for batch in range(n_batch):

                train_images_batch,train_labels_batch=sess.run([train_images_batch_t,train_labels_batch_t])
                _, loss_value = sess.run([train, loss], 
                   feed_dict = {x:train_images_batch, y:train_labels_batch,keep_pro:0.5})
            if step % 5 == 0:
                test_images_batch,test_labels_batch=sess.run([test_images_batch_t,test_labels_batch_t])
                accuracy = sess.run([acc], 
                    feed_dict = {x: test_images_batch, y:test_labels_batch,keep_pro:1.0})[0]             
                print("Step %d, train loss = %.2f, val accuracy = %.4f%%"%(step,loss_value,accuracy*100))
            if step % 1000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = './log/train/model.ckpt'
                saver.save(sess, checkpoint_path, global_step=step)

#%%
                
