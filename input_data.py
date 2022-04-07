import os
import tensorflow as tf
import numpy as np
import PIL as Image

def get_files(file_dir):
    directories=[files for files in os.listdir(file_dir)
    if os.path.isdir(os.path.join(file_dir, files))]  
    labels = []
    images = []
    for files in directories:
        data_dir = os.path.join(file_dir, files)
        file_names = [os.path.join(data_dir, f) 
                      for f in os.listdir(data_dir) 
                      if f.endswith(".jpeg")]
        for f in file_names:
            images.append(f)
            labels.append(int(files))
    temp=np.array([images,labels])
    temp=temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
#    labels_vector=[]
#    #对标签进行独热编码
#    for labels in label_list:
#        vector=np.zeros(62)
#        vector[labels]=1
#        labels_vector.append(vector)   
    return image_list,label_list

def get_batch(sess,image,label,width,hight,batch_size,capacity):
    # image, label: 要生成batch的图像和标签list
    # image_W, image_H: 图片的宽高
    # batch_size: 每个batch有多少张图片
    # capacity: 队列容量
    # return: 图像和标签的batch
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    input_queue=tf.train.slice_input_producer([image,label])
    
    label=input_queue[1]
    image_contents=tf.read_file(input_queue[0])

    image = tf.image.convert_image_dtype(tf.image.decode_png(image_contents, channels=3), tf.float32)
    image = tf.image.resize_images(image, [width, hight], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)   # 标准化数据
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,   # 线程
                                              capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    label_batch=tf.one_hot(label_batch,62)
    image_batch = tf.cast(image_batch, tf.float32)
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord=coord)
    return image_batch,label_batch

def get_filesList(file_dir):

    file_names = [os.path.join(file_dir, f) 
						for f in os.listdir(file_dir) 
						if f.endswith(".jpeg")]

    image_list = list(file_names)

    return image_list

def read_SingleImg(img_path, image_W, image_H):

    img_path = tf.cast(img_path, tf.string)

    image_contents = tf.read_file(img_path)
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_images(image, [image_W, image_H], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image_batch = tf.cast([image], tf.float32)
    
    return image_batch