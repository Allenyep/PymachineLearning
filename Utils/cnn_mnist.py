# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('mnist_data',one_hot=True)


input_x = tf.placeholder(tf.float32,[None,28 * 28]) / 255
output_y = tf.placeholder(tf.int32,[None,10])

input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])

#测试集
test_x = mnist.test.images[:3000]
test_y = mnist.test.labels[:3000]

#构建卷积神经网络
#第一层卷积,产生[28,28,32]
conv1 = tf.layers.conv2d(inputs=input_x_images,filters=32,kernel_size=[5,5],strides=1,padding='same',activation=tf.nn.relu)

#第一层池化,形状[14,14,32]
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)

#第二层卷积,[14 * 14 * 32],输出[14 * 14 * 64]
conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[5,5],strides=1,padding='same',activation=tf.nn.relu)

#第二层池化,形状[14,14,64],输出[7, 7 ,64]
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)

#平坦化flat
flat = tf.reshape(pool2,[-1,7 * 7 * 64])

#1024神经元全连接层
dense = tf.layers.dense(inputs=flat,units=1024,activation=tf.nn.relu)

#Dropout丢弃 rate=0.5
dropout = tf.layers.dropout(inputs=dense,rate=0.5)

#十个神经元全连接层=>独热码
logits = tf.layers.dense(inputs=dropout,units=10)

#计算误差(计算交叉熵(Cross entropy)再用Softmax计算百分百概率))
loss=tf.losses.softmax_cross_entropy(onehot_labels=output_y,logits=logits)

#用Adam优化器最小化误差,学习率0.001
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

#计算预测值和实际值匹配程度,返回(accuracy,update_op),会创建两个局部变量
accuracy = tf.metrics.accuracy(labels=tf.argmax(output_y,axis=1),predictions=tf.argmax(logits,axis=1))[1]

#创建会话
sess = tf.Session()

init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

sess.run(init)

for i in range(20000):
    batch = mnist.train.next_batch(50)
    train_loss, train_op_ =sess.run([loss, train_op],{input_x:batch[0],output_y:batch[1]})
    if i % 100 == 0:
        test_accuracy = sess.run(accuracy,{input_x:test_x,output_y:test_y})
        print("Step= %d ,Train= %.4f ,Test accuracy= %.2f")%(i,train_loss,test_accuracy)

#打印20预测值和真实值 对
test_output = sess.run(logits,{input_x:test_x[:20]})
inferenced_y = np.argmax(test_output,1)

print(inferenced_y,'Inferenced numbers')

print(np.argmax(test_y[:20],1),'Real numbers')



