import numpy as np
import pandas as pd
import tensorflow as tf
import dataprocess as dp  #데이터 처리를 위한 함수를 담은 파이썬 파일

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, W, B, time_step_size):
    XT = tf.transpose(X, [1, 0, 2])
    XR = tf.reshape(XT, [-1, time_step_size])
    X_split = tf.split(0, time_step_size, XR)

    lstm = tf.nn.rnn_cell.BasicLSTMCell(time_step_size, forget_bias=1.0, state_is_tuple=True)
    outputs, _state = tf.nn.rnn(lstm, X_split, dtype=tf.float32)

    return tf.matmul(outputs[-1], W) + B, lstm.state_size

char_rdic = list('0123456')
char_dic = {w: i for i, w in enumerate(char_rdic)}

# 트레이닝 세트를 만듬 1월부터 2월까지의 행위
user = pd.read_csv('C:/Users/lab/InsiderThreat/daily_work/daily_f_u989.csv', sep=',')
user = user.fillna(method='ffill', axis=1, downcast={'6.0': '6'})
train_set = pd.DataFrame(user[0:40])

cols = train_set.columns.tolist()
cols = cols[3:-1] + cols[-1:]
train_set = train_set[cols]

train_set = np.array(train_set, dtype='f')
x_data = np.empty([1, 54, 7], dtype='f')
sample = np.empty([1, 54], dtype=int)
for seq in train_set:
    temp_x = dp.set_xdata(seq)    #make 2d numpy array
    temp_y = [int(i) for i in seq]
    x_data = np.vstack((x_data, temp_x[None]))
    sample = np.vstack((sample, temp_y[0:]))
x_data = np.delete(x_data, 0, 0)
sample = np.delete(sample, 0, 0)

print("X: ", type(x_data[0][0][0]))
print("Y: ", sample)
#sequence = train_set[0]
#x_data = dp.set_xdata(sequence)
#sample = [int(i) for i in sequence]
print(sample[0,])

X = tf.placeholder('float', [None, 54, 7])
Y = tf.placeholder('int64', [None, 54])

output_size = len(char_dic)
rnn_size = output_size
time_step_size = 54
batch_size = 1

W = init_weights([time_step_size, 7])
B = init_weights([7])

py_x, state_size = model(X,W,B, time_step_size)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001,0.9).minimize(cost)
predic_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(100):
        for start, end in zip(range(0, len(x_data), batch_size), range(batch_size, len(x_data)+1, batch_size)):
            sess.run(train_op, feed_dict={X:x_data[start:end], Y: sample[start:end]})


"""
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)
state = tf.zeros([batch_size, rnn_cell.state_size])
XT = tf.transpose(X, [1, 0, 2])
XR = tf.reshape(XT, [-1, 54])
X_split = tf.split(0, time_step_size, XR)
outputs, state = tf.nn.rnn(rnn_cell, X_split, state)

logit = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
targets = tf.reshape(sample[0,], [-1])
weights = tf.ones([time_step_size * batch_size])

loss = tf.nn.seq2seq.sequence_loss_by_example([logit], [targets], [weights])
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(XR), feed_dict={X: x_data[0:1]})
    for i in range(100):
        sess.run(train_op, feed_dict={X: x_data, Y: sample})
        if i % 20 == 0:
            result = sess.run(tf.arg_max(logit, 1))
            #print('%r, %r' %(result, [char_rdic[t] for t in result]))
        print('%r, %r' % (result, [char_rdic[t] for t in result]))

"""