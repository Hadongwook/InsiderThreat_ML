import pandas as pd
import numpy as np
import tensorflow as tf

train_set = pd.read_csv('C:/Users/lab/PycharmProjects/Insider Threat/test/trainset.csv', sep=',')
cols = train_set.columns.tolist()
colsx = cols[1:-1]
colsy = cols[-1:]
print(cols)
x_data = train_set[colsx]
y_data = train_set[colsy]

print(x_data.shape)
#print(y_data)

X = tf.placeholder("float", [None, 55])
Y = tf.placeholder("float", [None, 1])
W = tf.Variable(tf.random_uniform([55, 1], -1.0, 1.0))

h = tf.matmul(X, W)
hypothesis = tf.div(1., 1. + tf.exp(-h))

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
learning_rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(0, 201):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 20 == 0:
            print(step, ": ", sess.run(cost, feed_dict={X: x_data, Y: y_data}))
            print("H:", sess.run(hypothesis, feed_dict={X: x_data}))


"""
    a = sess.run(hypothesis, feed_dict={X: [[1,0,1,1,4,1,5,1,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]]})
    print("a: ", a, sess.run(tf.arg_max(a, 1))) # arg_max a 의 제일 큰 값을 1 로 만든다.

    b = sess.run(hypothesis, feed_dict={X: [[1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]]})
    print("b :", b, sess.run(tf.arg_max(b, 1)))
"""