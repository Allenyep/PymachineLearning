# -*- coding: UTF-8 -*-

#使用梯度下降解决线性回归

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

points_num = 100
vectors = []

#用numpy的正太随机分布函数生产100个点
#y = 0.1 * x + 0.2
for i in xrange(points_num):
    x1 = np.random.normal(0.0, 0.66)
    y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0, 0.04)
    vectors.append([x1,y1])

x_data = [v[0] for v in vectors]
y_data = [v[1] for v in vectors]

plt.plot(x_data,y_data,'r*',label="original data")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()

# 构建线性回归模型
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

#定义损失函数cost function
loss = tf.reduce_mean(tf.square(y-y_data))

#使用梯度下降优化器来优化损失函数
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#创建会话
sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

for step in xrange(20):
    sess.run(train)
    print("Step=%d,Loss=%f,Weight=%f,Bias=%f")\
            %(step,sess.run(loss),sess.run(W),sess.run(b))

#绘制图像
plt.plot(x_data,y_data,'r*',label="original data")
plt.title("Linear Regression using Gradient Descent")
plt.plot(x_data,sess.run(W) * x_data + sess.run(b),label="Fitted line")
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

sess.close()
