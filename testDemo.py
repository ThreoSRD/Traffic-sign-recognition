# -*- coding: utf-8 -*-  

import os
import tensorflow as tf
import input_data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt # plt 用于显示图片
#import matplotlib.image as mpimg # mpimg 用于读取图片

#%%
IMG_W = 32
IMG_H = 32
N_CLASSES = 62
test_data_dir='./data/test/'

#%%
def mkdir(path):  
  
    folder = os.path.exists(path)  
  
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹  
        os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径 
        


#%%
if __name__ == '__main__':
    with tf.Graph().as_default() as g:
        x_input =tf.placeholder(tf.float32,[None,IMG_W,IMG_H,3],name='x')
        y_input =tf.placeholder(tf.float64,[None,N_CLASSES],name='y')
        with tf.name_scope('conv1'):
            w1  = tf.Variable(tf.truncated_normal([5,5,3,36],stddev=0.1),name='w1')
            b1  = tf.Variable(tf.constant(0.1,shape=[36]),name='b1')
            l1  = tf.nn.relu(tf.nn.conv2d(x_input,w1,strides=[1,1,1,1],padding='SAME')+b1)
            p1  = tf.nn.max_pool(l1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        
        with tf.name_scope('conv2'):
            w2  = tf.Variable(tf.truncated_normal([5,5,36,72],stddev=0.1),name='w2')
            b2  = tf.Variable(tf.constant(0.1,shape=[72]),name='b2')
            l2  = tf.nn.relu(tf.nn.conv2d(p1,w2,strides=[1,1,1,1],padding='SAME')+b2)
            p2  = tf.nn.max_pool(l2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
            
        with tf.name_scope('fc1'):
            flat_image=tf.reshape(p2,[-1,8*8*72])
            w3  = tf.Variable(tf.truncated_normal([8*8*72,1024],stddev=0.1),name='w3')
            b3  = tf.Variable(tf.constant(0.1,shape=[1024]),name='b3')
            f1  = tf.nn.relu(tf.matmul(flat_image,w3)+b3)
        #    d1  = tf.nn.dropout(f1,keep_prob=0.5)
            
        with tf.name_scope('fc2'):
            w4  = tf.Variable(tf.truncated_normal([1024,N_CLASSES],stddev=0.1),name='w4')
            b4  = tf.Variable(tf.constant(0.1,shape=[N_CLASSES]),name='b4')
            result = tf.nn.softmax(tf.matmul(f1,w4)+b4)
        #    result = tf.nn.softmax(tf.matmul(d1,w4)+b4)
        deal_result = tf.equal(tf.argmax(result,1),tf.argmax(y_input,1))
        acc = tf.reduce_mean(tf.cast(deal_result,tf.float32))
        
        test_images     = input_data.get_filesList(test_data_dir) 
        saver = tf.train.Saver()
        with tf.Session() as sess:
            evaluate_img,evaluate_labels=input_data.get_files('./data/Testing/') 
            images, labels = input_data.get_batch(sess,evaluate_img,evaluate_labels,IMG_W,IMG_H,2520,2520)
            sess.run(tf.global_variables_initializer())
            # load neural network model
            ckpt = tf.train.get_checkpoint_state('./log/train/')
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
                
            for imgPath in test_images:
                test_images_batch_t     = input_data.read_SingleImg(imgPath,IMG_W,IMG_H)
                img = sess.run(test_images_batch_t)
                prediction = sess.run(result, feed_dict={x_input:img})
                
                prediction = np.argmax(prediction,1)
                
#                imgShow = mpimg.imread(imgPath)
                imgShow = Image.open(imgPath)
                imgName = imgPath.split('/')[-1]
                
                plt.imshow(imgShow) # 显示图片
                plt.axis('off') # 不显示坐标轴
                plt.show()
                mkdir('./classify/'+str(prediction[0])+'/')
                imgShow.save('./classify/'+str(prediction[0])+'/'+imgName)
                print('Image name: ',imgName,end='    ')
                print('Path:',imgPath)
                print('The prediction is:    ',prediction[0])
            
            
            i,l = sess.run([images,labels])
            correct = sess.run(acc, feed_dict={x_input:i, y_input:l})
            print('Total testing samples: 2520')
            print('Average accuracy: %.2f%%' %(correct*100))
#%%
                