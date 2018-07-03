# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:19:40 2018

用CNN模型对实现MNIST手写数字分类

@author: Ruosi Liang

"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data    

# 读取MNIST数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

 
#定义一个卷积神经网络
x = tf.placeholder("float", shape=[None, 784]) # 图片的大小，28x28=784
y= tf.placeholder("float", shape=[None, 10])  # 需要识别的数字为0-9, 共10个数字
#把x变成一个4d向量，其第2、第3维对应图片的宽、高，
    #最后一维代表图片的颜色通道数: 1表示黑白, 3表示彩色。
x_image = tf.reshape(x, [-1,28,28,1]) 
 
 
#权重初始化
# 权重使用的truncated_normal进行初始化,stddev标准差定义为 0.1
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
 
    
# 偏置初始化为常量0.1
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
 
#卷积 : stride=1，
  # 0边距, 卷积之后和原图大小一样
  # 输入图片参数x和权重W
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


 # 最大化池
 # 池化核函数为 [1, 2, 2, 1]
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
 
#第一层卷积
#卷积核为5x5, 卷积的权重张量形状是[5, 5, 1, 32]，
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32]) # 每一个输出通道都有一个对应的偏置量。
#x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

#第一层最大池化
h_pool1 = max_pool_2x2(h_conv1)
 


#第二层卷积
#卷积核为5x5, 卷积的权重张量形状是[5, 5, 32, 64]，
W_conv2 = weight_variable([5, 5, 32, 64]) # 每一个输出通道都有一个对应的偏置量。
b_conv2 = bias_variable([64])

#ReLU激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#第二层最大池化
h_pool2 = max_pool_2x2(h_conv2)


# 全连接第一层
# 加入一个有1024个神经元的全连接层，用于处理整个图片。
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# 运算, ReLU激活
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
 


#Dropout 防止过拟合
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 

#输出层（softmax）
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) #使用softmax分类器
 
#训练和评估模型
cross_entropy = -tf.reduce_sum(y*tf.log(y_conv)) # 交叉熵损失函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #梯度下降, 使用交叉熵损失函数
prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, "float"))


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# 训练2000次
for i in range(2000):
  batch = mnist.train.next_batch(100) #使用SGD
  #每100次输出一下准确度
  if i%100 == 0: 
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y: batch[1], keep_prob: 1.0})
    print("第%d步, 训练集准确度 %.9f"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
 
 # 输出测试集得到准确度
print("测试集准确度 %.9f"%accuracy.eval(feed_dict={
    x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))